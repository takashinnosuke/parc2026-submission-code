"""
PARC 2026 - Modal Cloud GPU 上での pi0.5 (Physical Intelligence, PaliGemma 2B +
Action Expert 300M) Sequential Finetuning 学習スクリプト
=================================================================================
train_modal_pi0_sequential.py (pi0版) をベースに、pi0.5 (pi05) に切り替えたもの。
pi0.5 は pi0 と同じ Flow Matching アクションヘッドを持ちつつ、
Knowledge Insulation（VLMバックボーンの知識を破壊的忘却から保護する学習手法）を
導入しており、破壊的忘却への耐性が高い。GPUメモリ要件は pi0 とほぼ同等
（PaliGemma 2B + Action Expert 300M、bf16でL4 24GBに収まる）。

  単一ステージ: lerobot/libero_plus （講座提供データ、Track1のドメイン変動に直結）

ベースは lerobot/pi05_libero_base（LeRobotチーム公式のLIBERO適応済みpi0.5）。
ライセンスは Gemma（PaliGemmaバックボーン由来、レポートでの開示が必要 — 使用自体は
禁止されていない、pi0と同じ扱い）。

学習率等は pi0.5 公式チェックポイントの config.json 既定値
（optimizer_lr=2.5e-5, scheduler_warmup_steps=1000, scheduler_decay_steps=30000）を踏襲する。

中間チェックポイントは save_freq 毎に検知して CPU 側で LoRA マージ + HF アップロードし、
credit切れ・早期停止時にも直近の重みが必ず残るようにする（train_modal_pi0_sequential.py
で見つかったチェックポイント検出の競合状態バグを修正済み: ログ行出現直後は
ディレクトリ書き込みが完了していないことがあるため、最大60秒リトライする）。
"""

import modal

parc_train_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install(
        "git", "wget", "curl", "zip", "unzip", "cmake", "build-essential", "gcc", "g++", "clang", "python3-dev",
        "libgl1-mesa-glx", "libgl1", "libglfw3", "libglew-dev", "libegl1", "libosmesa6", "libosmesa6-dev",
        "libsm6", "libxext6", "libxrender-dev", "libglib2.0-0", "libmagickwand-dev", "ffmpeg",
        "patchelf", "libglu1-mesa-dev", "mesa-utils"
    )
    .run_commands(
        "pip install --upgrade pip setuptools wheel",
        "git clone --depth 1 --branch v0.6.0 https://github.com/huggingface/lerobot.git /workspace/lerobot",
        "pip install -e '/workspace/lerobot[training,pi,peft]'",
        "pip install --no-deps wandb hydra-core omegaconf einops imageio draccus rerun-sdk gymnasium",
        "pip install 'mujoco==3.1.6' 'robosuite==1.4.1' 'bddl==1.0.1' 'easydict>=1.9' Wand scikit-image pandas 'gym==0.26.2' pyserial deepdiff num2words future matplotlib",
    )
    .env({
        "MUJOCO_GL": "egl",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/workspace/lerobot/src",
    })
)

app = modal.App("parc2026-pi05-sequential-finetuner")
hf_secret = modal.Secret.from_name("huggingface-secret")
wandb_secret = modal.Secret.from_name("wandb-secret")


def _sanitize_config(cfg_path):
    import json
    with open(cfg_path, "r", encoding="utf-8") as f:
        c = json.load(f)
    for k in ["rtc_config", "compile_model", "compile_mode"]:
        c.pop(k, None)
    c["type"] = "pi05"
    # 既定の dtype="float32" だと PaliGemma 2B + Action Expert 300M をfp32でロードし、
    # L4 24GBでOOMする（pi0で実測済みの同種の問題）。bf16に固定してメモリを半減させる。
    c["dtype"] = "bfloat16"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(c, f, indent=2)


def _run_lerobot_train(
    policy_path, dataset_repo_id, steps, lora_r, lora_alpha, output_dir, job_name, batch_size=8,
    save_freq=500, checkpoint_sink=None,
):
    """lerobot-train をサブプロセスで実行する。checkpoint_sink が指定されていれば、
    「Checkpoint policy after step N」のログを検知するたびに callback(step, ckpt_dir) を
    同期呼び出しする（GPU上で学習サブプロセスが動き続けたまま、CPU側で中間マージ・
    HFアップロードを行うことを想定）。credit切れ・早期停止のいずれでも、直近の
    アップロード済みチェックポイントを必ず使える状態にするための安全策。"""
    import os
    import re
    import subprocess
    import time

    warmup = max(1, min(1000, steps // 5))
    decay_lr = 2.5e-6

    cmd = [
        "lerobot-train",
        f"--policy.path={policy_path}",
        "--policy.push_to_hub=false",
        "--policy.input_features=null",
        "--policy.output_features=null",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        # pi05既定はSTATE/ACTIONともQUANTILES正規化だが、lerobot/libero_plusの
        # 配布済みstatsにはq01/q99が含まれておらず学習開始直後にValueErrorで
        # 落ちる（augment_dataset_quantile_stats.pyでの再計算は他者所有リポジトリへの
        # push権限が無く現実的でない）。pi0と同じMEAN_STDに強制して回避する。
        '--policy.normalization_mapping={"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}',
        "--policy.dtype=bfloat16",
        "--policy.gradient_checkpointing=true",
        "--optimizer.type=adamw",
        "--optimizer.lr=2.5e-5",
        "--policy.optimizer_lr=2.5e-5",
        f"--policy.scheduler_decay_lr={decay_lr}",
        f"--policy.scheduler_warmup_steps={warmup}",
        f"--policy.scheduler_decay_steps={steps}",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        "--num_workers=4",
        "--seed=42",
        "--save_checkpoint=true",
        f"--save_freq={save_freq}",
        "--log_freq=50",
        "--wandb.enable=true",
        "--wandb.project=PARC2026",
        f"--wandb.notes={job_name} steps={steps} batch={batch_size} r={lora_r}",
        "--peft.method_type=LORA",
        f"--peft.r={lora_r}",
        f"--peft.lora_alpha={lora_alpha}",
    ]
    print(f"\nExecuting: {' '.join(cmd)}\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace/lerobot/src"
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    ckpt_re = re.compile(r"Checkpoint policy after step (\d+)")
    for line in process.stdout:
        print(line, end="", flush=True)
        if checkpoint_sink is not None:
            m = ckpt_re.search(line)
            if m:
                step_num = int(m.group(1))
                ckpt_dir = os.path.join(output_dir, "checkpoints", f"{step_num:06d}", "pretrained_model")
                # ログ行はディレクトリ書き込み完了前に出ることがあるため、
                # 数秒〜数十秒のリトライ待ちを入れる（さもないと無言でスキップされる）。
                found = False
                for attempt in range(30):
                    if os.path.isdir(ckpt_dir) and os.path.exists(os.path.join(ckpt_dir, "config.json")):
                        found = True
                        break
                    time.sleep(2)
                if found:
                    print(f"[checkpoint_sink] step {step_num}: 検出 -> {ckpt_dir}", flush=True)
                    try:
                        checkpoint_sink(step_num, ckpt_dir)
                    except Exception as e:
                        print(f"[checkpoint_sink] step {step_num} 保存失敗（学習は継続）: {e}", flush=True)
                else:
                    print(f"[checkpoint_sink] step {step_num}: {ckpt_dir} が60秒待っても見つからずスキップ", flush=True)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"lerobot-train exited with code {process.returncode} (job={job_name})")


def _find_final_checkpoint_dir(output_dir, steps):
    import glob
    import os
    candidates = sorted(glob.glob(os.path.join(output_dir, "checkpoints", "*", "pretrained_model")))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {output_dir}/checkpoints/*/pretrained_model")
    return candidates[-1]


def _merge_lora_into_base(base_dir, adapter_dir, merged_out_dir, device=None):
    """LoRAアダプタをベース重みにマージし、次段階の --policy.path として使える
    スタンドアロンな model.safetensors を書き出す。device="cpu" を指定すると、
    GPU上で学習サブプロセスが動き続けたまま（VRAM競合を避けて）中間マージできる。"""
    import json
    import os
    import shutil
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from peft import PeftModel

    _sanitize_config(f"{base_dir}/config.json")
    if device is not None:
        with open(f"{base_dir}/config.json", "r+", encoding="utf-8") as f:
            c = json.load(f)
            c["device"] = device
            f.seek(0)
            json.dump(c, f, indent=2)
            f.truncate()
    base_policy = PI05Policy.from_pretrained(base_dir)
    merged = PeftModel.from_pretrained(base_policy, adapter_dir)
    merged = merged.merge_and_unload()

    shutil.rmtree(merged_out_dir, ignore_errors=True)
    merged.save_pretrained(merged_out_dir)

    # merge_and_unload 後の save_pretrained はベースの config.json をそのまま書き出す
    # ため、実際に学習したアダプタ側の config.json（正しい入力特徴量定義）で上書きする。
    # use_peft は必ず false にする（pi0版と同じ理由: 次段が既存アダプタと誤認識しないため）。
    with open(f"{adapter_dir}/config.json", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    adapter_cfg["use_peft"] = False
    adapter_cfg.pop("pretrained_path", None)
    with open(f"{merged_out_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(adapter_cfg, f, indent=2)

    for fname in os.listdir(adapter_dir):
        if fname.startswith("policy_preprocessor") or fname.startswith("policy_postprocessor"):
            shutil.copy2(os.path.join(adapter_dir, fname), os.path.join(merged_out_dir, fname))

    return merged_out_dir


@app.function(
    image=parc_train_image,
    gpu="h100",
    secrets=[hf_secret, wandb_secret],
    timeout=86400,  # 24h(Modal上限)。
)
def train_sequential(
    stage1_steps: int = 11000,
    stage2_steps: int = 0,
    lora_r: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 8,
    resume_from: str = "",
    step_offset: int = 0,
):
    import time
    from huggingface_hub import HfApi, snapshot_download

    repo_id = "nosuke113/parc2026-policy"

    single_stage = stage2_steps <= 0
    print("=== [PARC2026 pi0.5 Finetuning: libero_plus" +
          (" only]" if single_stage else " -> libero]"))
    print(f"Stage 1 steps: {stage1_steps} | Stage 2 steps: {stage2_steps if not single_stage else '(skipped)'} | "
          f"LoRA r={lora_r} alpha={lora_alpha} | batch_size={batch_size} | "
          f"resume_from={resume_from or '(none)'} | step_offset={step_offset}")

    if resume_from:
        # Modal preemption等で中断した場合、途中でHFへ自動アップロード済みの
        # マージ済みチェックポイント(LoRA未装備・スタンドアロン)から再開する。
        # ベースからやり直すより速く、既に学習済みの分を無駄にしない。
        print(f"\n[Stage 1] Downloading resume checkpoint {repo_id}/{resume_from} ...")
        snap_dir = snapshot_download(repo_id=repo_id, allow_patterns=f"{resume_from}/*")
        base_dir = f"{snap_dir}/{resume_from}"
    else:
        print("\n[Stage 1] Downloading base lerobot/pi05_libero_base (Gemma license)...")
        base_dir = snapshot_download(repo_id="lerobot/pi05_libero_base")
    _sanitize_config(f"{base_dir}/config.json")

    intermediate_prefix = f"pi05_single/libero_plus_b{batch_size}_r{lora_r}_step"
    api = HfApi()

    def _checkpoint_sink(local_step_num, ckpt_dir):
        # 学習サブプロセスはGPUを使い続けているため、中間マージはCPUに固定して
        # VRAM競合によるOOMを避ける（多少遅いが、credit切れ・早期停止への保険として
        # 常に直近のアップロード済み重みを残すことを優先する）。
        # step_offsetを足した「累積step数」でラベル付けし、resume前のアップロードと
        # 衝突・混同しないようにする。
        step_num = local_step_num + step_offset
        print(f"\n[checkpoint_sink] step {step_num} (local={local_step_num}): CPU上でマージしてHFへアップロード中...")
        merged = _merge_lora_into_base(
            base_dir, ckpt_dir, f"/tmp/intermediate_merge_{step_num}", device="cpu"
        )
        api.upload_folder(
            folder_path=merged,
            repo_id=repo_id,
            path_in_repo=f"{intermediate_prefix}{step_num}",
            commit_message=f"EXP_007: pi0.5 libero_plus intermediate checkpoint step={step_num} (auto-saved, may be superseded)",
        )
        print(f"[checkpoint_sink] step {step_num}: アップロード完了 -> {intermediate_prefix}{step_num}")

    stage1_output = f"/tmp/stage1_output_{int(time.time())}"
    print(f"\n[Stage 1] Training on lerobot/libero_plus for {stage1_steps} steps "
          f"(save_freq=2500, 中間チェックポイントを都度HFへ自動保存)...")
    _run_lerobot_train(
        policy_path=base_dir,
        dataset_repo_id="lerobot/libero_plus",
        steps=stage1_steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        output_dir=stage1_output,
        job_name="parc2026_pi05_seq_stage1_libero_plus",
        batch_size=batch_size,
        save_freq=2500,
        checkpoint_sink=_checkpoint_sink,
    )
    stage1_ckpt = _find_final_checkpoint_dir(stage1_output, stage1_steps)
    print(f"[Stage 1] Final checkpoint: {stage1_ckpt}")

    print("\n[Stage 1] Merging final LoRA adapter into base weights...")
    stage1_merged = _merge_lora_into_base(base_dir, stage1_ckpt, "/tmp/stage1_merged")
    print(f"[Stage 1] Merged model ready at: {stage1_merged}")

    if single_stage:
        # 破壊的忘却(catastrophic forgetting)リスクを避けるため、Track1に最も直結する
        # libero_plus 単体に全ステップを集中投下する（Stage 2 はスキップ）。
        # pi0.5 は Knowledge Insulation により破壊的忘却耐性自体も pi0 より高い。
        final_merged = stage1_merged
        stage2_steps = 0
    else:
        stage2_output = f"/tmp/stage2_output_{int(time.time())}"
        print(f"\n[Stage 2] Training on lerobot/libero for {stage2_steps} steps (starting from Stage 1 weights)...")
        _run_lerobot_train(
            policy_path=stage1_merged,
            dataset_repo_id="lerobot/libero",
            steps=stage2_steps,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            output_dir=stage2_output,
            job_name="parc2026_pi05_seq_stage2_libero",
            batch_size=batch_size,
        )
        stage2_ckpt = _find_final_checkpoint_dir(stage2_output, stage2_steps)
        print(f"[Stage 2] Final checkpoint: {stage2_ckpt}")

        print("\n[Stage 2] Merging final LoRA adapter into Stage 1 weights...")
        final_merged = _merge_lora_into_base(stage1_merged, stage2_ckpt, "/tmp/final_merged")
        print(f"Final merged model ready at: {final_merged}")

    total_steps_reached = stage1_steps + step_offset
    hf_path = f"pi05_sequential/total{total_steps_reached}_stage2_{stage2_steps}_b{batch_size}_r{lora_r}_final"
    print(f"\nUploading final pi0.5 model to Hugging Face ({repo_id})...")
    api.upload_folder(
        folder_path=final_merged,
        repo_id=repo_id,
        path_in_repo=hf_path,
        commit_message=(
            f"EXP_007: pi0.5 Sequential Finetuning libero_plus (total_steps={total_steps_reached}) -> "
            f"libero({stage2_steps}), LoRA r={lora_r} alpha={lora_alpha}, batch_size={batch_size}, merged"
        ),
    )
    print("Hugging Face upload complete.")

    return {
        "status": "SUCCESS",
        "stage1_steps": stage1_steps,
        "step_offset": step_offset,
        "total_steps_reached": total_steps_reached,
        "stage2_steps": stage2_steps,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "batch_size": batch_size,
        "hf_path": hf_path,
    }


@app.local_entrypoint()
def main(
    stage1_steps: int = 11000,
    stage2_steps: int = 0,
    lora_r: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 8,
    resume_from: str = "",
    step_offset: int = 0,
):
    import json
    print("Launching pi0.5 Sequential Finetuning (libero_plus) on Modal L4 GPU")
    res = train_sequential.remote(
        stage1_steps=stage1_steps,
        stage2_steps=stage2_steps,
        lora_r=lora_r,
        resume_from=resume_from,
        step_offset=step_offset,
        lora_alpha=lora_alpha,
        batch_size=batch_size,
    )
    print("Training Job Completed:")
    print(json.dumps(res, indent=2))
