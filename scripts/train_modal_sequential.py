"""
PARC 2026 - Modal Cloud GPU 上での 2段階 Sequential Finetuning 学習スクリプト
=================================================================================
LIBERO原論文 (Liu et al., NeurIPS 2023, arXiv:2306.03310) の知見⑤
「複雑な生涯学習アルゴリズムより、大規模事前学習ベースモデルへの逐次
ファインチューニング(Sequential Finetuning)が順転移で最も高い成功率を記録した」
に基づき、以下の2段階でLoRA学習を行う。

  Stage 1: lerobot/libero_plus （講座提供データ・Track1の実タスク名(_table_2, _light_15等)
            が示すドメイン変動＝背景/照明/テクスチャシフトに直結するメインデータ）
  Stage 2: lerobot/libero （Apache-2.0。spatial/object/goal/10 の4スイート統合、457ファイル。
            未公開タスクへの汎化力強化）

Stage 1 で学習した LoRA アダプタをベース重みにマージしてから Stage 2 の出発点とする
（lerobot-train は複数データセットの同時ブレンド指定 (--dataset.repo_id のリスト) を
 `NotImplementedError` として明示的に未実装としているため、2段階を1コンテナ内で
 直列に実行する構成にした）。

重要: lerobot/libero_plus はライセンスタグ未設定だが、本コンペの講座提供データとして
使用可能（RULES.md 9節）。lerobot/libero は確認済み Apache-2.0。
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
        "pip install -e '/workspace/lerobot[training,smolvla,peft]'",
        "pip install --no-deps wandb hydra-core omegaconf einops imageio draccus rerun-sdk gymnasium",
        "pip install 'mujoco==3.1.6' 'robosuite==1.4.1' 'bddl==1.0.1' 'easydict>=1.9' Wand scikit-image pandas 'gym==0.26.2' pyserial deepdiff num2words future matplotlib",
    )
    .env({
        "MUJOCO_GL": "egl",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/workspace/lerobot/src",
    })
)

app = modal.App("parc2026-sequential-finetuner")
hf_secret = modal.Secret.from_name("huggingface-secret")
wandb_secret = modal.Secret.from_name("wandb-secret")


def _sanitize_config(cfg_path):
    import json
    with open(cfg_path, "r", encoding="utf-8") as f:
        c = json.load(f)
    for k in ["rtc_config", "compile_model", "compile_mode"]:
        c.pop(k, None)
    c["type"] = "smolvla"
    c["empty_cameras"] = 0
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(c, f, indent=2)


def _run_lerobot_train(policy_path, dataset_repo_id, steps, lora_r, lora_alpha, output_dir, job_name, batch_size=1):
    import os
    import subprocess

    cmd = [
        "lerobot-train",
        f"--policy.path={policy_path}",
        "--policy.push_to_hub=false",
        "--policy.input_features=null",
        "--policy.output_features=null",
        "--policy.empty_cameras=0",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        "--optimizer.type=adamw",
        "--optimizer.lr=1e-4",
        "--policy.optimizer_lr=1e-4",
        "--policy.scheduler_decay_lr=3e-5",
        "--policy.scheduler_warmup_steps=300",
        f"--policy.scheduler_decay_steps={steps}",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        "--num_workers=4",
        "--seed=42",
        "--save_checkpoint=true",
        "--save_freq=1000",
        "--log_freq=100",
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
    for line in process.stdout:
        print(line, end="", flush=True)
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


def _merge_lora_into_base(base_dir, adapter_dir, merged_out_dir):
    """LoRAアダプタをベース重みにマージし、次段階の --policy.path として使える
    スタンドアロンな model.safetensors を書き出す。"""
    import json
    import os
    import shutil
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from peft import PeftModel

    _sanitize_config(f"{base_dir}/config.json")
    base_policy = SmolVLAPolicy.from_pretrained(base_dir)
    merged = PeftModel.from_pretrained(base_policy, adapter_dir)
    merged = merged.merge_and_unload()

    shutil.rmtree(merged_out_dir, ignore_errors=True)
    merged.save_pretrained(merged_out_dir)

    # merge_and_unload 後の save_pretrained はベースリポジトリの config.json を
    # そのまま書き出すため、実際に学習したアダプタ側の config.json
    # (observation.images.front/wrist 等の正しい入力特徴量定義) で上書きする。
    # ただし use_peft は必ず false にする: アダプタ側の config.json をそのまま
    # 使うと "use_peft": true が残り、次段の lerobot-train がこのディレクトリを
    # 「既存アダプタの続き」と誤認識して adapter_config.json を探しに行き失敗する
    # （merge_and_unload 後はアダプタが本体に統合済みで、もう別ファイルは存在しない）。
    with open(f"{adapter_dir}/config.json", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    adapter_cfg["use_peft"] = False
    adapter_cfg.pop("pretrained_path", None)
    with open(f"{merged_out_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(adapter_cfg, f, indent=2)

    # 正規化パイプライン (policy_preprocessor.json / policy_postprocessor.json とその
    # 統計量 .safetensors) は merge_and_unload().save_pretrained() では出力されない。
    # これが無いと次段の lerobot-train が ProcessorMigrationError で止まるため、
    # アダプタ側チェックポイントから引き継ぐ。
    for fname in os.listdir(adapter_dir):
        if fname.startswith("policy_preprocessor") or fname.startswith("policy_postprocessor"):
            shutil.copy2(os.path.join(adapter_dir, fname), os.path.join(merged_out_dir, fname))

    return merged_out_dir


@app.function(
    image=parc_train_image,
    gpu="l4",
    secrets=[hf_secret, wandb_secret],
    timeout=14400,
)
def train_sequential(
    stage1_steps: int = 5000,
    stage2_steps: int = 5000,
    lora_r: int = 8,
    lora_alpha: int = 16,
    batch_size: int = 1,
):
    import time
    from huggingface_hub import HfApi, snapshot_download

    print("=== [PARC2026 Sequential Finetuning: libero_plus -> libero] ===")
    print(f"Stage 1 steps: {stage1_steps} | Stage 2 steps: {stage2_steps} | "
          f"LoRA r={lora_r} alpha={lora_alpha} | batch_size={batch_size}")

    # --- Stage 1: lerobot/libero_plus (講座提供データ、Track1のドメイン変動に直結) ---
    print("\n[Stage 1] Downloading base lerobot/smolvla_libero_plus...")
    base_dir = snapshot_download(repo_id="lerobot/smolvla_libero_plus")
    _sanitize_config(f"{base_dir}/config.json")

    stage1_output = f"/tmp/stage1_output_{int(time.time())}"
    print(f"\n[Stage 1] Training on lerobot/libero_plus for {stage1_steps} steps...")
    _run_lerobot_train(
        policy_path=base_dir,
        dataset_repo_id="lerobot/libero_plus",
        steps=stage1_steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        output_dir=stage1_output,
        job_name="parc2026_seq_stage1_libero_plus",
        batch_size=batch_size,
    )
    stage1_ckpt = _find_final_checkpoint_dir(stage1_output, stage1_steps)
    print(f"[Stage 1] Final checkpoint: {stage1_ckpt}")

    print("\n[Stage 1] Merging LoRA adapter into base weights for Stage 2 starting point...")
    stage1_merged = _merge_lora_into_base(base_dir, stage1_ckpt, "/tmp/stage1_merged")
    print(f"[Stage 1] Merged model ready at: {stage1_merged}")

    # --- Stage 2: lerobot/libero (Apache-2.0, spatial+object+goal+10 統合、汎化力強化) ---
    stage2_output = f"/tmp/stage2_output_{int(time.time())}"
    print(f"\n[Stage 2] Training on lerobot/libero for {stage2_steps} steps (starting from Stage 1 weights)...")
    _run_lerobot_train(
        policy_path=stage1_merged,
        dataset_repo_id="lerobot/libero",
        steps=stage2_steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        output_dir=stage2_output,
        job_name="parc2026_seq_stage2_libero",
        batch_size=batch_size,
    )
    stage2_ckpt = _find_final_checkpoint_dir(stage2_output, stage2_steps)
    print(f"[Stage 2] Final checkpoint: {stage2_ckpt}")

    print("\n[Stage 2] Merging final LoRA adapter into Stage 1 weights...")
    final_merged = _merge_lora_into_base(stage1_merged, stage2_ckpt, "/tmp/final_merged")
    print(f"Final merged model ready at: {final_merged}")

    repo_id = "nosuke113/parc2026-policy"
    hf_path = f"sequential/stage1_{stage1_steps}_stage2_{stage2_steps}_b{batch_size}_r{lora_r}"
    print(f"\nUploading final sequentially-finetuned model to Hugging Face ({repo_id})...")
    api = HfApi()
    api.upload_folder(
        folder_path=final_merged,
        repo_id=repo_id,
        path_in_repo=hf_path,
        commit_message=(
            f"EXP_005: Sequential Finetuning libero_plus({stage1_steps}) -> "
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
    batch_size: int = 1,
):
    import json
    print("Launching Sequential Finetuning (libero_plus -> libero) on Modal L4 GPU")
    res = train_sequential.remote(
        stage1_steps=stage1_steps,
        stage2_steps=stage2_steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        batch_size=batch_size,
    )
    print("Training Job Completed:")
    print(json.dumps(res, indent=2))
