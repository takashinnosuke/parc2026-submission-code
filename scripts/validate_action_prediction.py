"""
検証用スクリプト: lerobot/libero_plus の検証データ観測をポリシーに入力し、
予測actionとground truth(GT) actionがどれだけ近いかを比較する。

デバッグの切り分けが目的:
- 予測がGTに近い -> パイプライン(画像前処理・カメラキー・正規化)は正しい。
  実ロールアウトでの低成功率は「学習不足」または「open-loop誤差の蓄積」が主因。
- 予測がGTから大きく外れる(ほぼランダム) -> パイプライン自体にバグがある可能性
  (カメラキーの取り違え・画像リサイズ・正規化統計量のミスマッチ等)を疑う。

使い方 (Modal経由、学習と同じ依存環境を使う):
    modal run scripts/validate_action_prediction.py --checkpoint-path <HF repo内のpath> --n-samples 8
"""

import modal

app = modal.App("parc2026-validate-action-prediction")
hf_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install(
        "git", "wget", "curl", "build-essential", "gcc", "g++", "python3-dev",
        "libgl1-mesa-glx", "libgl1", "libglfw3", "libegl1", "libosmesa6", "libosmesa6-dev",
        "libsm6", "libxext6", "libxrender-dev", "libglib2.0-0", "ffmpeg",
    )
    .run_commands(
        "pip install --upgrade pip setuptools wheel",
        "git clone --depth 1 --branch v0.6.0 https://github.com/huggingface/lerobot.git /workspace/lerobot",
        "pip install -e '/workspace/lerobot[training,pi,peft]'",
    )
    .env({"PYTHONPATH": "/workspace/lerobot/src", "PYTHONUNBUFFERED": "1"})
)


@app.function(image=image, gpu="l4", secrets=[hf_secret], timeout=1800)
def validate(checkpoint_path: str, n_samples: int = 8, dataset_repo_id: str = "lerobot/libero_plus"):
    import json
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    print(f"Downloading checkpoint: nosuke113/parc2026-policy/{checkpoint_path}")
    ckpt_dir = snapshot_download(
        repo_id="nosuke113/parc2026-policy",
        allow_patterns=f"{checkpoint_path}/*",
    )
    ckpt_dir = f"{ckpt_dir}/{checkpoint_path}"

    cfg0 = PreTrainedConfig.from_pretrained(ckpt_dir)
    policy_cls = get_policy_class(cfg0.type)
    model = policy_cls.from_pretrained(ckpt_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device=device)
    print(f"Model loaded: {cfg0.type}, device={device}")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=model.config,
        pretrained_path=ckpt_dir,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": device}},
    )

    print(f"Loading dataset: {dataset_repo_id}")
    dataset = LeRobotDataset(repo_id=dataset_repo_id)
    n_total = len(dataset)
    print(f"Dataset has {n_total} frames total")

    rng = np.random.default_rng(42)
    idxs = rng.choice(n_total, size=min(n_samples, n_total), replace=False)

    results = []
    img_keys = [k for k in model.config.input_features if "images" in k]
    print(f"Image keys detected: {img_keys}")

    for idx in idxs:
        sample = dataset[int(idx)]
        gt_action = sample["action"]
        if not torch.is_tensor(gt_action):
            gt_action = torch.tensor(gt_action)

        obs = {}
        for k in img_keys:
            if k in sample:
                obs[k] = sample[k].unsqueeze(0).to(device=device)
        if "observation.state" in sample:
            obs["observation.state"] = sample["observation.state"].unsqueeze(0).to(device=device)
        obs["task"] = sample.get("task", "")
        obs["robot_type"] = ""

        with torch.no_grad():
            obs_p = preprocessor(obs)
            pred_chunk = model.predict_action_chunk(obs_p)
            pred_chunk = postprocessor(pred_chunk)

        pred_first = pred_chunk[0, 0].cpu().numpy()
        gt_np = gt_action.cpu().numpy() if torch.is_tensor(gt_action) else np.asarray(gt_action)
        if gt_np.ndim > 1:
            gt_np = gt_np[0]

        diff = pred_first[: len(gt_np)] - gt_np
        mse = float(np.mean(diff ** 2))
        cos_sim = float(
            np.dot(pred_first[: len(gt_np)], gt_np)
            / (np.linalg.norm(pred_first[: len(gt_np)]) * np.linalg.norm(gt_np) + 1e-8)
        )
        results.append({
            "idx": int(idx),
            "task": str(obs["task"]),
            "pred": pred_first[: len(gt_np)].tolist(),
            "gt": gt_np.tolist(),
            "mse": mse,
            "cos_sim": cos_sim,
        })
        print(f"[{idx}] task={obs['task']!r} mse={mse:.4f} cos_sim={cos_sim:.3f}")
        print(f"    pred={np.array2string(pred_first[: len(gt_np)], precision=3)}")
        print(f"    gt  ={np.array2string(gt_np, precision=3)}")

    mean_mse = float(np.mean([r["mse"] for r in results]))
    mean_cos = float(np.mean([r["cos_sim"] for r in results]))

    # ランダムベースライン(推定): 予測を全てゼロ or ランダム値にした場合のMSE/cos_simと比較する
    random_actions = rng.normal(0, 1, size=(len(results), 7))
    gt_stack = np.array([r["gt"][:7] for r in results])
    random_mse = float(np.mean((random_actions - gt_stack) ** 2))

    print("\n=== SUMMARY ===")
    print(f"n_samples={len(results)}")
    print(f"mean_mse={mean_mse:.4f} (random_baseline_mse~={random_mse:.4f})")
    print(f"mean_cos_sim={mean_cos:.3f} (0=無相関, 1=完全一致)")
    if mean_mse < random_mse * 0.3 and mean_cos > 0.3:
        verdict = "GOOD: 予測はGTに大きく近い。パイプラインは概ね正しい。実ロールアウトの低成功率は学習不足/open-loop誤差蓄積が主因の可能性が高い。"
    elif mean_mse < random_mse * 0.7:
        verdict = "PARTIAL: ある程度GTに近いが弱い。学習継続で改善余地あり。"
    else:
        verdict = "SUSPICIOUS: 予測がほぼランダムベースライン相当。パイプラインのバグ(カメラキー・画像前処理・正規化)を疑うべき。"
    print(f"VERDICT: {verdict}")

    return {
        "mean_mse": mean_mse,
        "mean_cos_sim": mean_cos,
        "random_baseline_mse": random_mse,
        "verdict": verdict,
        "n_samples": len(results),
    }


@app.local_entrypoint()
def main(checkpoint_path: str = "pi05_single/libero_plus_b8_r16_step22000", n_samples: int = 8):
    import json
    res = validate.remote(checkpoint_path=checkpoint_path, n_samples=n_samples)
    print(json.dumps(res, indent=2, ensure_ascii=False))
