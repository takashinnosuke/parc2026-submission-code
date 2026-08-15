"""
PARC 2026 - Modal Cloud GPU 上での pi0 (Physical Intelligence, 3.3B) 2段階
Sequential Finetuning 学習スクリプト
=================================================================================
train_modal_sequential.py (SmolVLA版) と同じ Sequential Finetuning 戦略を pi0 に
適用する。SOTA調査 (03_Experiments/EXP_20260802_Model_SOTA_Survey.md) では pi0 の
LIBERO成功率報告値が95.2%とSmolVLA(82.4-90.1%)を上回るため、モデル自体の上限を
引き上げる狙い。

  Stage 1: lerobot/libero_plus （講座提供データ、Track1のドメイン変動に直結）
  Stage 2: lerobot/libero （Apache-2.0、汎化力強化）

ベースは lerobot/pi0_libero_base（LeRobotチーム公式のLIBERO適応済みpi0、
smolvla_libero_plus に相当する開始点）。ライセンスは Gemma（PaliGemmaバックボーン由来、
レポートでの開示が必要 — 使用自体は禁止されていない）。

pi0 は PaliGemma バックボーンを外部リポジトリ参照ではなく model.safetensors に
内包しているため（configuration_pi0.py: get_gemma_config()でアーキテクチャを
ローカル構築）、SmolVLA で必要だった外部VLM重みのオフラインキャッシュ同梱は不要。

学習率等は pi0 公式チェックポイントの config.json 既定値
（optimizer_lr=2.5e-5, scheduler_warmup_steps=1000）を踏襲する
（SmolVLAの1e-4は大きすぎる可能性があるため）。
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
        "pip install -e '/workspace/lerobot[training,pi0,peft]'",
        "pip install --no-deps wandb hydra-core omegaconf einops imageio draccus rerun-sdk gymnasium",
        "pip install 'mujoco==3.1.6' 'robosuite==1.4.1' 'bddl==1.0.1' 'easydict>=1.9' Wand scikit-image pandas 'gym==0.26.2' pyserial deepdiff num2words future matplotlib",
    )
    .env({
        "MUJOCO_GL": "egl",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/workspace/lerobot/src",
    })
)

app = modal.App("parc2026-pi0-sequential-finetuner")
hf_secret = modal.Secret.from_name("huggingface-secret")
wandb_secret = modal.Secret.from_name("wandb-secret")


def _sanitize_config(cfg_path):
    import json
    with open(cfg_path, "r", encoding="utf-8") as f:
        c = json.load(f)
    for k in ["rtc_config", "compile_model", "compile_mode"]:
        c.pop(k, None)
    c["type"] = "pi0"
    # 既定の dtype="float32" だと 4B パラメータをfp32でロードし、L4 24GBでも
    # batch_size=2 未満でOOMする（実測: batch=2/8 いずれも約22GB使用でOOM）。
    # bf16 に固定してメモリを半減させる。
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
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from peft import PeftModel

    _sanitize_config(f"{base_dir}/config.json")
    if device is not None:
        with open(f"{base_dir}/config.json", "r+", encoding="utf-8") as f:
            c = json.load(f)
            c["device"] = device
            f.seek(0)
            json.dump(c, f, indent=2)
            f.truncate()
    base_policy = PI0Policy.from_pretrained(base_dir)
    merged = PeftModel.from_pretrained(base_policy, adapter_dir)
    merged = merged.merge_and_unload()

    shutil.rmtree(merged_out_dir, ignore_errors=True)
    merged.save_pretrained(merged_out_dir)

    # merge_and_unload 後の save_pretrained はベースの config.json をそのまま書き出す
    # ため、実際に学習したアダプタ側の config.json（正しい入力特徴量定義）で上書きする。
    # use_peft は必ず false にする（SmolVLA版と同じ理由: 次段が既存アダプタと誤認識しないため）。
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
    gpu="l4",
    secrets=[hf_secret, wandb_secret],
    timeout=64800,
)
def train_sequential(
    stage1_steps: int = 5000,
    stage2_steps: int = 5000,
    lora_r: int = 8,
    lora_alpha: int = 16,
    batch_size: int = 8,
):
    import time
    from huggingface_hub import HfApi, snapshot_download

    single_stage = stage2_steps <= 0
    print("=== [PARC2026 pi0 Finetuning: libero_plus" +
          (" only]" if single_stage else " -> libero]"))
    print(f"Stage 1 steps: {stage1_steps} | Stage 2 steps: {stage2_steps if not single_stage else '(skipped)'} | "
          f"LoRA r={lora_r} alpha={lora_alpha} | batch_size={batch_size}")

    print("\n[Stage 1] Downloading base lerobot/pi0_libero_base (Gemma license)...")
    base_dir = snapshot_download(repo_id="lerobot/pi0_libero_base")
    _sanitize_config(f"{base_dir}/config.json")

    repo_id = "nosuke113/parc2026-policy"
    intermediate_prefix = f"pi0_single/libero_plus_b{batch_size}_r{lora_r}_step"
    api = HfApi()

    def _checkpoint_sink(step_num, ckpt_dir):
        # 学習サブプロセスはGPUを使い続けているため、中間マージはCPUに固定して
        # VRAM競合によるOOMを避ける（多少遅いが、credit切れ・早期停止への保険として
        # 常に直近のアップロード済み重みを残すことを優先する）。
        print(f"\n[checkpoint_sink] step {step_num}: CPU上でマージしてHFへアップロード中...")
        merged = _merge_lora_into_base(
            base_dir, ckpt_dir, f"/tmp/intermediate_merge_{step_num}", device="cpu"
        )
        api.upload_folder(
            folder_path=merged,
            repo_id=repo_id,
            path_in_repo=f"{intermediate_prefix}{step_num}",
            commit_message=f"EXP_006: pi0 libero_plus intermediate checkpoint step={step_num} (auto-saved, may be superseded)",
        )
        print(f"[checkpoint_sink] step {step_num}: アップロード完了 -> {intermediate_prefix}{step_num}")

    stage1_output = f"/tmp/stage1_output_{int(time.time())}"
    print(f"\n[Stage 1] Training on lerobot/libero_plus for {stage1_steps} steps "
          f"(save_freq=500, 中間チェックポイントを都度HFへ自動保存)...")
    _run_lerobot_train(
        policy_path=base_dir,
        dataset_repo_id="lerobot/libero_plus",
        steps=stage1_steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        output_dir=stage1_output,
        job_name="parc2026_pi0_seq_stage1_libero_plus",
        batch_size=batch_size,
        save_freq=500,
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
            job_name="parc2026_pi0_seq_stage2_libero",
            batch_size=batch_size,
        )
        stage2_ckpt = _find_final_checkpoint_dir(stage2_output, stage2_steps)
        print(f"[Stage 2] Final checkpoint: {stage2_ckpt}")

        print("\n[Stage 2] Merging final LoRA adapter into Stage 1 weights...")
        final_merged = _merge_lora_into_base(stage1_merged, stage2_ckpt, "/tmp/final_merged")
        print(f"Final merged model ready at: {final_merged}")

    hf_path = f"pi0_sequential/stage1_{stage1_steps}_stage2_{stage2_steps}_b{batch_size}_r{lora_r}_final"
    print(f"\nUploading final pi0 model to Hugging Face ({repo_id})...")
    api.upload_folder(
        folder_path=final_merged,
        repo_id=repo_id,
        path_in_repo=hf_path,
        commit_message=(
            f"EXP_006: pi0 Sequential Finetuning libero_plus({stage1_steps}) -> "
            f"libero({stage2_steps}), LoRA r={lora_r} alpha={lora_alpha}, batch_size={batch_size}, merged"
        ),
    )
    print("Hugging Face upload complete.")

    return {
        "status": "SUCCESS",
        "stage1_steps": stage1_steps,
        "stage2_steps": stage2_steps,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "batch_size": batch_size,
        "hf_path": hf_path,
    }


@app.local_entrypoint()
def main(
    stage1_steps: int = 5000,
    stage2_steps: int = 5000,
    lora_r: int = 8,
    lora_alpha: int = 16,
    batch_size: int = 8,
):
    import json
    print("Launching pi0 Sequential Finetuning (libero_plus -> libero) on Modal L4 GPU")
    res = train_sequential.remote(
        stage1_steps=stage1_steps,
        stage2_steps=stage2_steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        batch_size=batch_size,
    )
    print("Training Job Completed:")
    print(json.dumps(res, indent=2))
