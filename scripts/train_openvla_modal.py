"""
PARC 2026 - Modal Cloud GPU 上での OpenVLA (7B) モデル ファインチューニング & HF 自動保存スクリプト (EXP_003)
目的: OpenVLA (7B) を Modal L4 GPU (24GB VRAM) で LoRA ファインチューニングし、
     W&B Project PARC2026 連携および Hugging Face (nosuke113/parc2026-openvla-policy) へ自動保存する。
"""

import modal

parc_openvla_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git", "wget", "curl", "zip", "unzip", "cmake", "build-essential", "gcc", "g++", "clang", "python3-dev",
        "libgl1-mesa-glx", "libgl1", "libglfw3", "libglew-dev", "libegl1", "libosmesa6", "libosmesa6-dev",
        "libsm6", "libxext6", "libxrender-dev", "libglib2.0-0", "libmagickwand-dev", "ffmpeg"
    )
    .pip_install(
        "torch", "torchvision", "torchaudio", "diffusers", "pyserial", "deepdiff", "torchcodec", "num2words",
        "mujoco==3.1.6", "robosuite==1.4.1", "bddl==1.0.1", "easydict>=1.9",
        "huggingface_hub", "wandb", "peft", "datasets", "accelerate", "transformers",
        "hydra-core", "omegaconf", "einops", "av", "imageio", "draccus", "rerun-sdk", "gymnasium"
    )
    .run_commands(
        "git clone --branch v0.4.0 https://github.com/huggingface/lerobot.git /workspace/lerobot && pip install --no-deps -e /workspace/lerobot"
    )
    .env({
        "MUJOCO_GL": "egl",
        "PYTHONUNBUFFERED": "1",
        "WANDB_PROJECT": "PARC2026",
    })
)

app = modal.App("parc2026-openvla-trainer")
hf_secret = modal.Secret.from_name("huggingface-secret")
wandb_secret = modal.Secret.from_name("wandb-secret")


@app.function(
    image=parc_openvla_image,
    gpu="l4",
    secrets=[hf_secret, wandb_secret],
    timeout=86400,
)
def train_openvla():
    import os
    import subprocess
    import sys

    print("=== [PARC2026 Modal OpenVLA 7B Fine-Tuning Pipeline] ===")

    # LeRobot groot モジュールパッチ無害化
    policies_init = "/workspace/lerobot/src/lerobot/policies/__init__.py"
    factory_py = "/workspace/lerobot/src/lerobot/policies/factory.py"
    try:
        if os.path.exists(policies_init):
            with open(policies_init, "r", encoding="utf-8") as f:
                lines = f.readlines()
            with open(policies_init, "w", encoding="utf-8") as f:
                for line in lines:
                    if "groot" in line:
                        continue
                    f.write(line)
        if os.path.exists(factory_py):
            with open(factory_py, "r", encoding="utf-8") as f:
                content = f.read()
            content = content.replace(
                "from lerobot.policies.groot.configuration_groot import GrootConfig",
                "class GrootConfig: pass"
            )
            with open(factory_py, "w", encoding="utf-8") as f:
                f.write(content)
        print("✅ LeRobot パッチ無害化完了")
    except Exception as patch_e:
        print(f"⚠️ パッチログ: {patch_e}")

    train_script = "/workspace/lerobot/examples/training/train_policy.py"
    if not os.path.exists(train_script):
        for root, dirs, files in os.walk("/workspace/lerobot"):
            for file in files:
                if file == "train_policy.py":
                    train_script = os.path.join(root, file)
                    break

    target_hf_repo = "nosuke113/parc2026-openvla-policy"

    cmd = [
        sys.executable, train_script,
        "policy=openvla",
        "dataset_repo_id=lerobot/libero",
        "dataset.video_backend=pyav",
        "training.offline_steps=3000",
        "training.batch_size=8",
        "eval.batch_size=1",
        "eval.n_episodes=0",
        "use_peft=true",
        "peft_rank=32",
        "wandb.enable=true",
        "wandb.project=PARC2026",
        "wandb.notes=EXP_003_OpenVLA_7B_LoRA_FineTuning_on_Modal_L4",
        "output_dir=/tmp/outputs/openvla",
        "save_model=true",
        "save_freq=1000",
    ]

    print(f"🚀 Running OpenVLA 7B Fine-Tuning Command:\n{' '.join(cmd)}\n")

    env = os.environ.copy()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )

    for line in iter(process.stdout.readline, ''):
        print(line, end='', flush=True)

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"OpenVLA Training failed with returncode {process.returncode}")

    print("🎉 OpenVLA Training Finished Successfully!")

    # HF 自動同期
    from huggingface_hub import HfApi
    target_dir = None
    for candidate in ["/tmp/outputs/openvla", "outputs/train", "/tmp/outputs", "outputs"]:
        if os.path.exists(candidate):
            target_dir = candidate
            break

    if target_dir:
        print(f"📤 Uploading OpenVLA 7B checkpoints from '{target_dir}' to Hugging Face '{target_hf_repo}'...")
        api = HfApi()
        api.create_repo(repo_id=target_hf_repo, exist_ok=True, private=False)
        api.upload_folder(
            folder_path=target_dir,
            repo_id=target_hf_repo,
            commit_message="EXP_003: OpenVLA 7B LoRA fine-tuned checkpoints on Modal L4"
        )
        print(f"✅ Hugging Face Upload Complete: https://huggingface.co/{target_hf_repo}")

    return {"status": "SUCCESS", "hf_repo": target_hf_repo}


@app.local_entrypoint()
def main():
    print("🚀 Launching OpenVLA (7B) Fine-Tuning Task on Modal L4 GPU...")
    res = train_openvla.remote()
    print("🎉 Modal Task Returned:", res)


if __name__ == "__main__":
    main()
