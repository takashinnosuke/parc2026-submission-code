"""
Modal Cloud 上に保存された OpenVLA 7B チェックポイントを探索し Hugging Face nosuke113/parc2026-openvla-policy へアップロードするスクリプト
"""
import modal

parc_openvla_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "wget", "curl", "zip", "unzip", "python3-dev")
    .pip_install("huggingface_hub", "wandb")
    .env({"PYTHONUNBUFFERED": "1"})
)

app = modal.App("parc2026-hf-uploader")
hf_secret = modal.Secret.from_name("huggingface-secret")

@app.function(
    image=parc_openvla_image,
    secrets=[hf_secret],
    timeout=1800,
)
def upload_openvla_checkpoint():
    import os
    from huggingface_hub import HfApi

    print("=== Searching for OpenVLA Checkpoints in Modal Storage ===")
    search_paths = [
        "/tmp/outputs",
        "/tmp/lerobot",
        "/workspace/lerobot",
        "/tmp",
        "/root/.cache/huggingface"
    ]

    found_dirs = []
    for sp in search_paths:
        if os.path.exists(sp):
            for root, dirs, files in os.walk(sp):
                if "config.json" in files and ("adapter_model.safetensors" in files or "model.safetensors" in files):
                    found_dirs.append(root)

    print(f"🔍 Found checkpoint directories ({len(found_dirs)}):")
    for d in found_dirs:
        print("  -", d)

    target_hf_repo = "nosuke113/parc2026-openvla-policy"
    api = HfApi()
    api.create_repo(repo_id=target_hf_repo, exist_ok=True, private=False)

    if found_dirs:
        best_dir = found_dirs[-1]
        print(f"📤 Uploading '{best_dir}' to '{target_hf_repo}'...")
        api.upload_folder(
            folder_path=best_dir,
            repo_id=target_hf_repo,
            commit_message="Upload OpenVLA 7B LoRA fine-tuned checkpoint"
        )
        print(f"✅ Hugging Face Upload Complete: https://huggingface.co/{target_hf_repo}")
        return {"status": "SUCCESS", "uploaded": best_dir}
    else:
        print("⚠️ No checkpoint directory found with config.json & safetensors.")
        return {"status": "NOT_FOUND"}

@app.local_entrypoint()
def main():
    res = upload_openvla_checkpoint.remote()
    print("Result:", res)

if __name__ == "__main__":
    main()
