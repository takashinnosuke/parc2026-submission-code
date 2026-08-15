"""Fetch google/paligemma-3b-pt-224 tokenizer files via Modal (which has gated-repo
access already granted) and re-upload them to our own HF repo so the local dev
machine (with no HF token / no gated access) can download them without gating.
"""

import modal

app = modal.App("parc2026-fetch-paligemma-tokenizer")
hf_secret = modal.Secret.from_name("huggingface-secret")
image = modal.Image.debian_slim().pip_install("huggingface_hub>=0.25.0")


@app.function(image=image, secrets=[hf_secret], timeout=600)
def fetch_and_reupload():
    from huggingface_hub import HfApi, snapshot_download

    src_dir = snapshot_download(
        repo_id="google/paligemma-3b-pt-224",
        allow_patterns=["tokenizer*", "special_tokens_map.json", "*.model"],
    )
    print(f"Downloaded to {src_dir}")

    api = HfApi()
    repo_id = "nosuke113/parc2026-policy"
    api.upload_folder(
        folder_path=src_dir,
        repo_id=repo_id,
        path_in_repo="paligemma_tokenizer",
        commit_message="Mirror google/paligemma-3b-pt-224 tokenizer files for offline submission bundling",
    )
    print(f"Uploaded to {repo_id}/paligemma_tokenizer")
    return "done"


@app.local_entrypoint()
def main():
    res = fetch_and_reupload.remote()
    print(res)
