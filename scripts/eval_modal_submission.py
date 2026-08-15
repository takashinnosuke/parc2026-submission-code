"""
PARC 2026 - Modal L4 GPU 上で提出物(submission_template)そのものを本番同様の
GPU/EGL環境で評価するスクリプト。

harness/evaluate.py (公式ハーネス) をそのまま呼び出すことで、policy_server.py の
HTTPインターフェース経由での実物理ロールアウトを、ローカルCPU検証環境の10秒
タイムアウト制約を受けずに実施する。

使い方:
    modal run scripts/eval_modal_submission.py --n-episodes 2
"""

import modal

parc_eval_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git", "wget", "curl", "zip", "unzip", "build-essential", "cmake",
        "libgl1-mesa-glx", "libgl1", "libglfw3", "libglew-dev", "libegl1",
        "libosmesa6", "libosmesa6-dev",
        "libsm6", "libxext6", "libxrender-dev", "libglib2.0-0", "libmagickwand-dev",
    )
    # torch/torchvision は先に単独で解決させ、他パッケージの依存解決に巻き込まれて
    # ABIが食い違う組み合わせにならないようにする（ローカルCPU検証で遭遇した
    # 「operator torchvision::nms does not exist」と同種の事故を防ぐ）。
    .pip_install("torch==2.11.0", "torchvision==0.26.0")
    .pip_install(
        "mujoco==3.7.0", "robosuite==1.4.0", "numpy==1.26.4", "gym==0.25.2", "bddl==3.6.0",
        "cloudpickle==3.1.2", "easydict==1.13", "hydra-core==1.3.2", "einops==0.8.2",
        "opencv-python-headless==4.11.0.86",
        "scipy", "pyyaml", "h5py", "Pillow", "termcolor", "tqdm", "matplotlib",
        "requests", "msgpack", "fastapi", "uvicorn", "huggingface_hub", "wand",
        "scikit-image", "pytest",
        "gymnasium==1.0.0",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/sylvestf/LIBERO-plus /LIBERO-plus",
        "git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO /LIBERO",
        "touch /LIBERO-plus/libero/__init__.py /LIBERO-plus/libero/libero/__init__.py",
        "sed -i 's/torch.load(init_states_path)/torch.load(init_states_path, weights_only=False)/' "
        "/LIBERO-plus/libero/libero/benchmark/__init__.py || true",
    )
    .env({
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "PYTHONUNBUFFERED": "1",
        "LIBERO_ROOT": "/LIBERO-plus",
        "PYTHONPATH": "/LIBERO-plus:/repo:/repo/compe",
    })
    .add_local_dir(".", remote_path="/repo", ignore=[
        "**/.git", "**/__pycache__", "**/*.pyc",
        "submission_template", "document",
        "LIBERO-plus", "LIBERO", "venv", ".claude", ".vscode",
    ])
    .add_local_file(
        "submissions/submission_EXP_003.zip",
        remote_path="/repo/submissions/submission_EXP_003.zip",
    )
)

app = modal.App("parc2026-submission-gpu-eval")
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(
    image=parc_eval_image,
    gpu="l4",
    secrets=[hf_secret],
    timeout=1800,
    memory=32768,
    retries=0,
)
def run_submission_eval(n_episodes: int = 2, max_steps: int = 300):
    import os
    import subprocess
    import json

    print("=== [PARC2026 Submission GPU Eval via harness/evaluate.py] ===")

    home = os.path.expanduser("~")
    os.makedirs(os.path.join(home, ".libero"), exist_ok=True)
    with open(os.path.join(home, ".libero", "config.yaml"), "w") as f:
        f.write(
            "benchmark_root: /LIBERO-plus/libero/libero\n"
            "bddl_files: /LIBERO-plus/libero/libero/bddl_files\n"
            "init_states: /LIBERO-plus/libero/libero/init_files\n"
            "datasets: /LIBERO-plus/libero/libero/datasets\n"
            "assets: /LIBERO/libero/libero/assets\n"
        )

    subprocess.run(
        ["python", "-c",
         "from huggingface_hub import hf_hub_download; "
         "hf_hub_download('Sylvest/LIBERO-plus', 'assets.zip', repo_type='dataset', local_dir='/tmp/assets')"],
        check=True,
    )
    subprocess.run(["unzip", "-q", "/tmp/assets/assets.zip", "-d", "/LIBERO-plus/libero/libero"], check=True)

    env = os.environ.copy()
    env["USERSUBMISSION"] = "/repo/submissions/submission_EXP_003.zip"
    env["EVAL_OUTPUT_DIR"] = "/tmp/results"
    env["SERVER_TIMEOUT"] = "300"

    cmd = ["python", "evaluate.py", "--n-episodes", str(n_episodes), "--max-steps", str(max_steps)]
    result = subprocess.run(cmd, cwd="/repo/harness", env=env, capture_output=True, text=True)
    print("--- stdout ---")
    print(result.stdout)
    print("--- stderr (tail) ---")
    print("\n".join(result.stderr.splitlines()[-200:]))

    try:
        parsed = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception:
        parsed = {"raw_stdout_tail": result.stdout[-2000:]}

    return {"returncode": result.returncode, "result": parsed}


@app.local_entrypoint()
def main(n_episodes: int = 2, max_steps: int = 300):
    import json
    print(f"Launching submission GPU eval (n_episodes={n_episodes}, max_steps={max_steps})...")
    res = run_submission_eval.remote(n_episodes=n_episodes, max_steps=max_steps)
    print(json.dumps(res, indent=2, ensure_ascii=False))
