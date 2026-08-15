#!/usr/bin/env python3
"""PARC 2026 Colab Remote Worker (colab_worker.py)

Colab 上で常駐動作し、ローカルの run_remote_colab.py から送信された
設定ファイル (trigger.json) を検出して学習・エラー制御・HF保存を実行する。
"""

import os
import sys
import time
import json
import subprocess
import traceback

DRIVE_DIR = "/content/drive/MyDrive/PARC2026"
TRIGGER_FILE = f"{DRIVE_DIR}/control/trigger.json"
STATUS_FILE = f"{DRIVE_DIR}/control/status.json"
LOG_DIR = f"{DRIVE_DIR}/logs"
OUTPUT_DIR = f"{DRIVE_DIR}/outputs/smolvla_libero_plus_lora"

def ensure_deps():
    print("[Worker] Checking and installing dependencies...", flush=True)
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
    print("[Worker] Dependencies check completed.", flush=True)

def execute_training(cfg):
    print(f"[Worker] Received training command with config:\n{json.dumps(cfg, indent=2)}", flush=True)
    
    base_model = cfg.get("base_model", "lerobot/smolvla_libero_plus")
    dataset_repo = cfg.get("dataset_repo", "lerobot/libero_plus")
    steps = cfg.get("steps", 3000)
    batch_size = cfg.get("batch_size", 1)
    lr = cfg.get("lr", 1e-4)
    hf_token = cfg.get("hf_token", "")
    
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    cmd = [
        "lerobot-train",
        f"--policy.path={base_model}",
        "--policy.push_to_hub=false",
        "--policy.input_features=null",
        "--policy.output_features=null",
        "--policy.empty_cameras=0",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        "--optimizer.type=adamw",
        f"--optimizer.lr={lr}",
        f"--policy.optimizer_lr={lr}",
        f"--policy.scheduler_decay_lr=3e-5",
        f"--policy.scheduler_warmup_steps=300",
        f"--policy.scheduler_decay_steps={steps}",
        f"--dataset.repo_id={dataset_repo}",
        f"--output_dir={OUTPUT_DIR}",
        "--job_name=parc2026_smolvla_lora",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
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
    
    print(f"[Worker] Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    
    if proc.returncode != 0:
        raise RuntimeError(f"lerobot-train process failed with exit code {proc.returncode}")
        
    print("[Worker] Training finished successfully!", flush=True)
    
    if hf_token:
        try:
            from huggingface_hub import HfApi, login
            login(token=hf_token)
            api = HfApi()
            api.upload_folder(
                folder_path=OUTPUT_DIR,
                repo_id=cfg.get("hf_repo", "nosuke113/parc2026-policy"),
                repo_type="model",
            )
            print(f"[Worker] Successfully uploaded model to Hugging Face!", flush=True)
        except Exception as e:
            print(f"[Worker] HF upload warning: {e}", flush=True)

def main():
    os.makedirs(f"{DRIVE_DIR}/control", exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    ensure_deps()
    
    print("[Worker] Colab Worker is running. Waiting for instructions from local cmd...", flush=True)
    last_ts = 0
    
    while True:
        try:
            if os.path.exists(TRIGGER_FILE):
                with open(TRIGGER_FILE, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    
                ts = cfg.get("timestamp", 0)
                if ts > last_ts and cfg.get("action") == "start_training":
                    last_ts = ts
                    print(f"[Worker] New trigger detected from local cmd at {time.ctime(ts)}!", flush=True)
                    execute_training(cfg)
            time.sleep(5)
        except Exception as e:
            err_msg = f"[Worker Error] {e}\n{traceback.format_exc()}"
            print(err_msg, flush=True)
            with open(f"{LOG_DIR}/error.log", "w", encoding="utf-8") as f:
                f.write(err_msg)
            time.sleep(5)

if __name__ == "__main__":
    main()
