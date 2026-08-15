"""
PARC 2026 - Modal L4 GPU 上で提出物 submission_EXP_008_45000.zip の
ロールアウトを可視化する（policy_server.py を実際に起動し、
scripts/visualize_rollout.py と同等のロジックで1エピソード分の
front/wristカメラGIF + actionsログを取得する）。

ローカルCPU(osmesa)検証は1ステップ約7〜10秒かかり遅すぎるため、
本番同様のGPU(L4)/EGL環境で高速に確認するためのスクリプト。

使い方:
    modal run scripts/eval_modal_visualize.py --max-steps 400
"""

import modal

parc_viz_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git", "wget", "curl", "zip", "unzip", "build-essential", "cmake",
        "libgl1-mesa-glx", "libgl1", "libglfw3", "libglew-dev", "libegl1",
        "libosmesa6", "libosmesa6-dev",
        "libsm6", "libxext6", "libxrender-dev", "libglib2.0-0", "libmagickwand-dev",
    )
    .pip_install("torch==2.11.0", "torchvision==0.26.0")
    .pip_install(
        "mujoco==3.7.0", "robosuite==1.4.0", "numpy==1.26.4", "gym==0.25.2", "bddl==3.6.0",
        "cloudpickle==3.1.2", "easydict==1.13", "hydra-core==1.3.2", "einops==0.8.2",
        "opencv-python-headless==4.11.0.86",
        "scipy", "pyyaml", "h5py", "Pillow", "termcolor", "tqdm", "matplotlib",
        "requests", "msgpack", "fastapi", "uvicorn", "huggingface_hub", "wand",
        "scikit-image", "pytest", "imageio",
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
        "submission_template", "submission_template_45k", "document",
        "LIBERO-plus", "LIBERO", "venv", ".claude", ".vscode",
        "submissions",
    ])
    .add_local_file(
        "submissions/submission_EXP_008_45000.zip",
        remote_path="/repo/submissions/submission_EXP_008_45000.zip",
    )
)

app = modal.App("parc2026-visualize-gpu")
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(
    image=parc_viz_image,
    gpu="l4",
    secrets=[hf_secret],
    timeout=1800,
    memory=32768,
    retries=0,
)
def run_visualize(task: str, max_steps: int = 400, timeout_sec: float = 120.0):
    import os
    import subprocess
    import time

    print("=== [PARC2026 GPU Rollout Visualization] ===")

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

    sub_dir = "/tmp/sub45k_gpu"
    os.makedirs(sub_dir, exist_ok=True)
    subprocess.run(["unzip", "-q", "/repo/submissions/submission_EXP_008_45000.zip", "-d", sub_dir], check=True)

    print("--- pip install submission requirements ---")
    r = subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], cwd=sub_dir,
                        capture_output=True, text=True)
    print(r.stdout[-3000:])
    print(r.stderr[-3000:])

    print("--- starting policy_server.py ---")
    server_log = open("/tmp/server.log", "w")
    server_proc = subprocess.Popen(
        ["python3", "policy_server.py", "--port", "8000"],
        cwd=sub_dir, stdout=server_log, stderr=subprocess.STDOUT,
    )

    import requests
    healthy = False
    for _ in range(120):
        try:
            resp = requests.get("http://127.0.0.1:8000/health", timeout=3)
            if resp.status_code == 200:
                healthy = True
                break
        except Exception:
            pass
        time.sleep(3)
    print(f"server healthy={healthy}")
    if not healthy:
        server_proc.kill()
        with open("/tmp/server.log") as f:
            return {"error": "server did not become healthy", "server_log": f.read()[-5000:]}

    out_gif = "/tmp/rollout_gpu.gif"
    out_actions = "/tmp/rollout_gpu_actions.txt"
    cmd = [
        "python", "/repo/scripts/visualize_rollout.py",
        "--server-url", "http://127.0.0.1:8000",
        "--task", task,
        "--max-steps", str(max_steps),
        "--timeout", str(timeout_sec),
        "--out", out_gif,
        "--out-actions", out_actions,
    ]
    result = subprocess.run(cmd, cwd="/repo/harness", capture_output=True, text=True)
    print("--- visualize_rollout stdout ---")
    print(result.stdout[-4000:])
    print("--- visualize_rollout stderr (tail) ---")
    print(result.stderr[-3000:])

    server_proc.kill()

    out = {"returncode": result.returncode, "stdout_tail": result.stdout[-3000:]}
    if os.path.exists(out_gif):
        with open(out_gif, "rb") as f:
            out["gif_bytes"] = f.read()
    if os.path.exists(out_actions):
        with open(out_actions, "r") as f:
            out["actions_text"] = f.read()
    return out


@app.local_entrypoint()
def main(
    task: str = "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_table_2",
    max_steps: int = 400,
    timeout_sec: float = 120.0,
    out: str = ".debug_output/rollout_viz/rollout_45000_rtcfix_gpu.gif",
    out_actions: str = ".debug_output/rollout_viz/rollout_45000_rtcfix_gpu_actions.txt",
):
    print(f"Launching GPU rollout visualization (task={task}, max_steps={max_steps})...")
    res = run_visualize.remote(task=task, max_steps=max_steps, timeout_sec=timeout_sec)
    if "error" in res:
        print("ERROR:", res["error"])
        print(res.get("server_log", ""))
        return
    print(res.get("stdout_tail", ""))
    if "gif_bytes" in res:
        with open(out, "wb") as f:
            f.write(res["gif_bytes"])
        print(f"Saved GIF to {out} ({len(res['gif_bytes'])} bytes)")
    if "actions_text" in res:
        with open(out_actions, "w", encoding="utf-8") as f:
            f.write(res["actions_text"])
        print(f"Saved actions to {out_actions}")
