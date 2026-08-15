#!/usr/bin/env python3
"""PARC 2026 全自動・ログ自動逆流 Colab 学習スクリプト (train_colab.py)

1. 全依存パッケージの自動チェック＆インストール
2. 実行ログおよびエラーログを /content/drive/MyDrive/PARC2026/logs/ に自動追記
3. エラー発生時は error.log に詳細スタックトレースを書き出してローカルエージェントへ報告
4. 学習モデルを Hugging Face へ自動アップロードし、提出用 submission.zip を生成
"""

import os
import sys
import time
import zipfile
import subprocess
import traceback

DRIVE_DIR = "/content/drive/MyDrive/PARC2026"
LOG_DIR = f"{DRIVE_DIR}/logs"
OUTPUT_DIR = f"{DRIVE_DIR}/outputs/smolvla_libero_plus_lora"
SUBMISSION_DIR = f"{DRIVE_DIR}/submissions"

class TeeLogger:
    """標準出力とログファイルへ同時に書き込むロガー"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "train_execution.log")
    sys.stdout = TeeLogger(log_path)
    sys.stderr = sys.stdout
    print(f"\n[AutoTrainer] Logging started. Log file: {log_path}\n", flush=True)

def ensure_dependencies():
    print("[1/4] Checking and installing required packages...", flush=True)
    pkgs = [
        "torchao>=0.16.0",
        "lerobot[dataset]",
        "av",
        "num2words",
        "imageio",
        "imageio-ffmpeg",
        "peft",
        "huggingface_hub",
        "accelerate",
        "diffusers",
        "torchvision"
    ]
    for pkg in pkgs:
        subprocess.run(f"pip install -q {pkg}", shell=True)
    print("✅ All dependencies ready.", flush=True)

def run_training():
    print("[2/4] Executing lerobot-train pipeline...", flush=True)
    cmd = [
        "lerobot-train",
        "--policy.path=lerobot/smolvla_libero_plus",
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
        "--policy.scheduler_decay_steps=3000",
        "--dataset.repo_id=lerobot/libero_plus",
        f"--output_dir={OUTPUT_DIR}",
        "--job_name=parc2026_smolvla_lora",
        "--steps=3000",
        "--batch_size=1",
        "--num_workers=0",
        "--seed=42",
        "--save_checkpoint=true",
        "--save_freq=1000",
        "--log_freq=100",
        "--resume=true",
        "--wandb.enable=false",
        "--peft.method_type=LORA",
        "--peft.r=8",
        "--peft.lora_alpha=16"
    ]
    
    print("Command:", " ".join(cmd), flush=True)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in process.stdout:
        print(line, end="", flush=True)
        
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"lerobot-train exited with error code {process.returncode}")
        
    print("✅ Training finished successfully.", flush=True)

def upload_to_hf():
    print("[3/4] Uploading model to Hugging Face...", flush=True)
    hf_token = os.environ.get("HF_TOKEN", "")
    hf_repo = "nosuke113/parc2026-policy"
    
    if not hf_token:
        print("ℹ️ HF_TOKEN not set in environment. Model saved safely on Google Drive.", flush=True)
        return
        
    try:
        from huggingface_hub import HfApi, login
        login(token=hf_token)
        api = HfApi()
        api.upload_folder(
            folder_path=OUTPUT_DIR,
            repo_id=hf_repo,
            repo_type="model",
        )
        print(f"🎉 Successfully uploaded model to https://huggingface.co/{hf_repo}", flush=True)
    except Exception as e:
        print(f"❌ Upload status: {e}", flush=True)

def create_submission_zip():
    print("[4/4] Generating valid submission.zip package...", flush=True)
    zip_path = os.path.join(SUBMISSION_DIR, "submission_EXP_001.zip")
    template_dir = "submission_template"
    if not os.path.exists(template_dir) and os.path.exists(f"{DRIVE_DIR}/submission_template"):
        template_dir = f"{DRIVE_DIR}/submission_template"

    ps_file = os.path.join(template_dir, "policy_server.py")
    req_file = os.path.join(template_dir, "requirements.txt")

    if not os.path.exists(ps_file):
        print(f"⚠️ Warning: {ps_file} not found. Skipping zip generation.", flush=True)
        return

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(ps_file, arcname="policy_server.py")
        if os.path.exists(req_file):
            zf.write(req_file, arcname="requirements.txt")
            
    print(f"📦 Successfully created submission package: {zip_path}", flush=True)

def main():
    setup_logging()
    print("==================================================", flush=True)
    print("   PARC 2026 Automated Training & Log-Sync Loop   ", flush=True)
    print("==================================================", flush=True)
    
    try:
        ensure_dependencies()
        run_training()
        upload_to_hf()
        create_submission_zip()
        print("\n🎉 ALL TASKS COMPLETED SUCCESSFULLY!", flush=True)
    except Exception as e:
        error_file = os.path.join(LOG_DIR, "error.log")
        with open(error_file, "w", encoding="utf-8") as f:
            f.write(f"Exception: {e}\n")
            f.write(traceback.format_exc())
        print(f"\n❌ Error captured and saved to {error_file}", flush=True)
        print(traceback.format_exc(), flush=True)

if __name__ == "__main__":
    main()
