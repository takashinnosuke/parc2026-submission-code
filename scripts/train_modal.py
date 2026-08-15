"""
PARC 2026 - Modal Cloud GPU 上での運営公式 100% 準拠 SmolVLA LoRA 学習スクリプト
=================================================================================
運営公式サンプル (parc2026_colab_trainer.ipynb) と完全一致する lerobot-train コマンド
およびハイパーパラメータでベースライン学習を実行する。

重要: LeRobot v0.6.0 + [peft] extras が必須（v0.4.0 には --peft.* 引数が存在しない）
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
        # Step 1: pip upgrade
        "pip install --upgrade pip setuptools wheel",
        # Step 2: LeRobot v0.6.0 を最初にインストール（torch/torchvision の依存解決を LeRobot に委ねる）
        "git clone --depth 1 --branch v0.6.0 https://github.com/huggingface/lerobot.git /workspace/lerobot",
        "pip install -e '/workspace/lerobot[training,smolvla,peft]'",
        # Step 3: 残りのパッケージを追加（torch は既にインストール済みなのでスキップ）
        "pip install --no-deps wandb hydra-core omegaconf einops imageio draccus rerun-sdk gymnasium",
        "pip install 'mujoco==3.1.6' 'robosuite==1.4.1' 'bddl==1.0.1' 'easydict>=1.9' Wand scikit-image pandas 'gym==0.26.2' pyserial deepdiff num2words future matplotlib",
    )
    .env({
        "MUJOCO_GL": "egl",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/workspace/lerobot/src",
    })
)

app = modal.App("parc2026-official-lerobot-trainer")
hf_secret = modal.Secret.from_name("huggingface-secret")
wandb_secret = modal.Secret.from_name("wandb-secret")


@app.function(
    image=parc_train_image,
    gpu="l4",
    secrets=[hf_secret, wandb_secret],
    timeout=7200,
)
def train_official_smolvla(steps: int = 3000, lora_r: int = 8, lora_alpha: int = 16):
    import os
    import sys
    import json
    import subprocess
    from huggingface_hub import HfApi, snapshot_download

    print(f"=== [PARC2026 Official LeRobot v0.6.0 SmolVLA LoRA Baseline Fine-Tuning] ===")
    print(f"Target Steps: {steps} | LoRA rank: {lora_r} | LoRA alpha: {lora_alpha}")

    # lerobot バージョン確認
    try:
        import lerobot
        print(f"LeRobot version: {getattr(lerobot, '__version__', 'unknown')}")
    except Exception:
        pass

    # lerobot/smolvla_libero_plus の config.json サニタイズ
    print("📥 Downloading and sanitizing lerobot/smolvla_libero_plus config.json...")
    local_policy_dir = snapshot_download(repo_id="lerobot/smolvla_libero_plus")
    cfg_path = os.path.join(local_policy_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            c = json.load(f)
        # draccus が認識しないフィールドを除去
        for k in ["rtc_config", "compile_model", "compile_mode"]:
            c.pop(k, None)
        c["type"] = "smolvla"
        c["empty_cameras"] = 0
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(c, f, indent=2)

    import time
    output_dir = f"/tmp/official_smolvla_output_{int(time.time())}"

    # ================================================================
    # 運営公式 Colab ノートブック (parc2026_colab_trainer.ipynb L108-L139)
    # と 100% 完全一致する lerobot-train コマンド
    # ================================================================
    cmd = [
        "lerobot-train",
        f"--policy.path={local_policy_dir}",
        "--policy.push_to_hub=false",
        "--policy.input_features=null",
        "--policy.output_features=null",
        "--policy.empty_cameras=0",
        # ===== 運営公式: Vision Encoder 凍結 & Expert-only 学習 =====
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        # ===== 運営公式: Optimizer & Scheduler =====
        "--optimizer.type=adamw",
        "--optimizer.lr=1e-4",
        "--policy.optimizer_lr=1e-4",
        "--policy.scheduler_decay_lr=3e-5",
        "--policy.scheduler_warmup_steps=300",
        f"--policy.scheduler_decay_steps={steps}",
        # ===== 運営公式: データセット & 出力 =====
        "--dataset.repo_id=lerobot/libero_plus",
        f"--output_dir={output_dir}",
        "--job_name=parc2026_smolvla_lora",
        f"--steps={steps}",
        "--batch_size=1",
        "--num_workers=0",
        "--seed=42",
        "--save_checkpoint=true",
        "--save_freq=1000",
        "--log_freq=100",
        # ===== 運営公式: W&B ログ =====
        "--wandb.enable=true",
        "--wandb.project=PARC2026",
        f"--wandb.notes=EXP_003 SmolVLA LoRA r={lora_r} alpha={lora_alpha} steps={steps}",
        # ===== 運営公式: LoRA/PEFT 設定 (v0.6.0 以降で有効) =====
        "--peft.method_type=LORA",
        f"--peft.r={lora_r}",
        f"--peft.lora_alpha={lora_alpha}",
    ]

    print(f"\n[1/2] Executing Official lerobot-train Command:")
    print(f"  {' '.join(cmd)}\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace/lerobot/src"

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()

    if process.returncode != 0:
        print(f"\n⚠️ lerobot-train exited with code {process.returncode}")
        return {"status": "FAILED", "returncode": process.returncode, "steps": steps}

    repo_id = "nosuke113/parc2026-policy"
    print(f"\n[2/2] Uploading official baseline model to Hugging Face ({repo_id})...")
    try:
        api = HfApi()
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            commit_message=f"EXP_003: Official SmolVLA LoRA Baseline (r={lora_r}, alpha={lora_alpha}, steps={steps}, lerobot=v0.6.0)",
        )
        print("✅ Hugging Face へのモデル自動同期が完了しました！")
    except Exception as e:
        print(f"⚠️ HF 上書き同期ステータス: {e}")

    return {
        "status": "SUCCESS",
        "steps": steps,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "output_dir": output_dir,
    }


@app.local_entrypoint()
def main(steps: int = 3000, lora_r: int = 8, lora_alpha: int = 16):
    import json
    print(f"🚀 Launching Official SmolVLA LoRA Baseline Training on Modal L4 GPU")
    print(f"   LeRobot: v0.6.0 | Steps: {steps} | LoRA rank: {lora_r} | LoRA alpha: {lora_alpha}")
    res = train_official_smolvla.remote(steps=steps, lora_r=lora_r, lora_alpha=lora_alpha)
    print("🎉 Training Job Completed:")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
