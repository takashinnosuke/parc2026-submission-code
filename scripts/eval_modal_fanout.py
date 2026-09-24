"""PARC 2026 - Modal 上でタスク単位に fan-out する並列評価。

Track1 最大の反省点だった「n=3 の比較実験に6〜10時間かかる」を解消するための基盤。
[[03_Experiments/EXP_20260818_Dataset_Audit]] の仮説H1〜H3はいずれも n≥3 の
条件間比較を要求するため、本スクリプトが全仮説検証の前提になる。

## 既存スクリプトとの違い

`scripts/eval_modal.py`（EXP_003期）は以下の問題があり、実験ループに接続されないまま
放置されていた。本スクリプトはそれらを解消する。

| 問題 | 本スクリプトでの解 |
|---|---|
| `@app.function` 1本を `.remote()` 1回＝**逐次実行** | `@app.cls` + `.map()` で**タスク単位に並列化** |
| ロールアウトを独自実装し、**1mm衝突判定が無い**（成功率しか出ない） | `pipeline/rollout.py` の `RolloutExecutor` を**そのまま使う**（衝突判定・軌道ログを含む公式4指標が揃う） |
| 推論も独自実装（`policy.select_action` 直叩き） | **提出物と同じ `submission_template/policy_server.py` の `MyPolicy`** を使う（RTC・グリッパー開放プリアンブル等の修正が入った実物） |
| `c["type"] = "smolvla"` を決め打ち | ポリシーは HF リポジトリ名で指定（π0.5 等に対応） |
| 条件間比較の口が無い | `--configs` で複数条件を並べ、`pipeline/eval_aggregate.py` が比較表を出す |

## fan-out の粒度をタスク単位にした理由

ポリシーのロード（π0.5 は 3.3B）が最も重いので、`@modal.enter()` で**コンテナごとに
1回だけ**行い、`.map()` の各入力＝1タスク（n エピソード）でそのコストを償却する。
エピソード単位まで細かくするとコンテナ数が増えてロードコストが支配的になる。

MuJoCo の EGL コンテキスト解放時に SIGSEGV が起きうるが、`max_inputs` は
`@app.cls` では1しか指定できず（＝タスクごとにモデル再ロード）fan-out の利点を
打ち消す。そのためコンテナは再利用し、落ちた入力だけ `retries` で再試行する。

## 使い方

```bash
# 単一条件
modal run scripts/eval_modal_fanout.py --configs "base=nosuke113/parc2026-policy" --n-episodes 3

# 条件間比較（H1: 重複除去あり/なし）
modal run scripts/eval_modal_fanout.py \
    --configs "full=nosuke113/parc2026-policy@rev_full,dedup=nosuke113/parc2026-policy@rev_dedup" \
    --n-episodes 3

# 集約（ローカル。LIBERO 不要）
python -m pipeline.eval_aggregate results/eval_fanout/*.json --detail
```
"""

import json
import time

import modal

# `scripts/eval_modal_submission.py` で公式harnessの実行実績があるイメージ定義を踏襲する
# （mujoco/robosuite/bddl は本番採点環境と同一ピン）。そこへ提出物の依存を足す。
parc_eval_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git", "wget", "curl", "zip", "unzip", "build-essential", "cmake",
        "libgl1-mesa-glx", "libgl1", "libglfw3", "libglew-dev", "libegl1",
        "libosmesa6", "libosmesa6-dev",
        "libsm6", "libxext6", "libxrender-dev", "libglib2.0-0", "libmagickwand-dev",
    )
    # torch/torchvision は先に単独で解決させ、他パッケージの依存解決に巻き込まれて
    # ABIが食い違う組み合わせにならないようにする。
    .pip_install("torch==2.11.0", "torchvision==0.26.0")
    .pip_install(
        "mujoco==3.7.0", "robosuite==1.4.0", "numpy==1.26.4", "gym==0.25.2", "bddl==3.6.0",
        "cloudpickle==3.1.2", "easydict==1.13", "hydra-core==1.3.2", "einops==0.8.2",
        "opencv-python-headless==4.11.0.86",
        "scipy", "pyyaml", "h5py", "Pillow", "termcolor", "tqdm", "matplotlib",
        "requests", "msgpack", "fastapi", "uvicorn", "huggingface_hub", "wand",
        "scikit-image", "gymnasium==1.0.0",
        # 提出物 (submission_template/requirements.txt) と同じピン
        "draccus==0.10.0", "num2words==0.5.14", "safetensors==0.8.0",
        "tokenizers==0.22.2", "transformers==5.5.4",
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
        "PYTHONPATH": "/LIBERO-plus:/repo",
    })
    .add_local_dir(".", remote_path="/repo", ignore=[
        "**/.git", "**/__pycache__", "**/*.pyc",
        "document", "submissions", "results", "logs",
        "LIBERO-plus", "LIBERO", "venv", ".claude", ".vscode",
        # 8.8GB あるうえ、同梱されていると MyPolicy がこれを読んでしまい
        # `--configs` で指定した条件ごとの重みが**無視される**（条件間比較が
        # 同一重みの二度評価になる）。条件ごとに HF から取得するため除外する。
        # `hf_cache`(21MB, トークナイザのオフラインキャッシュ) と
        # `lerobot`(17MB, vendored) は動作に必要なので残す。
        "submission_template/model_weights",
    ])
)

# 1コンテナが取得してよい重みの上限(GB)。π0.5 のチェックポイント1つが約9.4GBなので
# それを少し上回る値にしておき、リポジトリ全体を引くような指定を弾く。
MAX_WEIGHTS_DOWNLOAD_GB = 15.0

app = modal.App("parc2026-eval-fanout")
hf_secret = modal.Secret.from_name("huggingface-secret")

# 重みのキャッシュ。コンテナごとに 9.35GB を取り直すのは時間も費用も無駄なので、
# Volume に置いて2回目以降は再利用する（学習曲線のように条件×タスクで
# コンテナ数が増えるほど効く）。
weights_volume = modal.Volume.from_name("parc2026-weights", create_if_missing=True)


@app.cls(
    image=parc_eval_image,
    gpu="l4",
    secrets=[hf_secret],
    timeout=3600,
    # π0.5 の bf16 重みは 9.4GB。16GiB で足りる。
    # 32GiB だとメモリ課金が総額の22%を占めていた（実測 $1.13/コンテナ時のうち）。
    memory=16384,
    volumes={"/weights": weights_volume},
    # MuJoCoのEGLコンテキスト解放時にSIGSEGVが起きうる。`max_inputs` は
    # `@app.cls` では1しか許されず、それだとタスクごとにポリシー(3.3B)を
    # 再ロードすることになり fan-out の意味が薄れる。そこでコンテナは再利用し、
    # 落ちた入力だけ再試行して守る。
    retries=modal.Retries(max_retries=2, initial_delay=5.0),
)
class TaskEvaluator:
    """1コンテナ＝1ポリシーで、割り当てられたタスクを順に評価する。"""

    config_label: str = modal.parameter()
    policy_repo: str = modal.parameter()
    n_episodes: int = modal.parameter(default=3)
    max_steps: int = modal.parameter(default=600)
    seed: int = modal.parameter(default=42)
    # 指示文の張り替え（BDDL の language -> 学習データの task 文字列）を有効にするか。
    # 学習と評価で 40 タスク中 30 タスクの指示文が食い違うため既定は有効。
    # 効果を測るときは 0 を指定して同一条件と比べる。
    instruction_map: int = modal.parameter(default=1)
    chunk_size: int = modal.parameter(default=5)   # 提出物既定に合わせる。0で上書きしない

    @modal.enter()
    def setup(self) -> None:
        """LIBERO アセットの配置とポリシーのロード。コンテナごとに1回だけ走る。"""
        import os
        import shutil
        import sys
        import zipfile
        from pathlib import Path

        from huggingface_hub import hf_hub_download, snapshot_download

        sys.path.insert(0, "/repo")
        sys.path.insert(0, "/LIBERO-plus")

        # LIBERO は ~/.libero/config.yaml を見てアセット・BDDL・init を解決する
        home = Path(os.path.expanduser("~"))
        (home / ".libero").mkdir(parents=True, exist_ok=True)
        (home / ".libero" / "config.yaml").write_text(
            "benchmark_root: /LIBERO-plus/libero/libero\n"
            "bddl_files: /LIBERO-plus/libero/libero/bddl_files\n"
            "init_states: /LIBERO-plus/libero/libero/init_files\n"
            "datasets: /LIBERO-plus/libero/libero/datasets\n"
            "assets: /LIBERO/libero/libero/assets\n",
            encoding="utf-8",
        )

        # LIBERO-plus のシーンアセット（著者公式配布、MIT）。
        # robosuite の `xml_path_completion` は LIBERO-plus パッケージ直下の
        # `assets/` を見に行くため、config.yaml の `assets:` 指定では足りず、
        # **物理的に配置する**必要がある。zip の内部レイアウトは版によって
        # ルートが `assets/` だったり1階層深かったりするので、`scenes` ディレクトリを
        # 探してその親を assets として据える（`scripts/eval_modal.py` と同じ方式）。
        asset_zip = hf_hub_download(
            "Sylvest/LIBERO-plus", "assets.zip", repo_type="dataset", local_dir="/tmp/assets"
        )
        staging = Path("/tmp/assets_extracted")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(asset_zip) as z:
            z.extractall(staging)

        assets_dir = Path("/LIBERO-plus/libero/libero/assets")
        scene_parents = [d.parent for d in staging.rglob("scenes") if d.is_dir()]
        if scene_parents:
            shutil.rmtree(assets_dir, ignore_errors=True)
            shutil.copytree(scene_parents[0], assets_dir)
            print(f"[setup] assets を配置: {scene_parents[0]} -> {assets_dir}")
        else:
            # 想定外のレイアウト。次回の調査のため中身を出しておく
            print(f"[setup] 警告: scenes/ が見つからない。展開結果の一部: "
                  f"{[str(x.relative_to(staging)) for x in list(staging.rglob('*'))[:20]]}")
            shutil.copytree(staging, assets_dir, dirs_exist_ok=True)

        probe = assets_dir / "scenes"
        print(f"[setup] assets/scenes 存在={probe.is_dir()} "
              f"サブディレクトリ={[d.name for d in probe.iterdir() if d.is_dir()][:8] if probe.is_dir() else []}")

        # 条件ごとの重みを HF から取得し、`MyPolicy` が読む場所へ据える。
        #
        # `MyPolicy._load_model()` は **`submission_template/model_weights` だけ**を
        # 見に行き、repo_id によるネット越しフォールバックを意図的に持たない
        # （本番はサーバー起動後にネットワークが遮断されるため）。したがって
        # ここで物理的に配置しないと、`--configs` で条件を分けても同じ重みを
        # 評価してしまい、比較が無意味になる。
        weights_dir = Path("/repo/submission_template/model_weights")
        repo, _, subfolder = self.policy_repo.partition("@")
        print(f"[setup] 重みを取得: repo={repo} subfolder={subfolder or '(ルート)'}")

        # --- ダウンロード量の事前確認（必須） ---
        # `nosuke113/parc2026-policy` は中間チェックポイントを119個ぶら下げており
        # **リポジトリ全体で958GB**ある。allow_patterns 無しで snapshot_download すると
        # コンテナごとに全部取りに行き、数時間×コンテナ数ぶんGPUを回し続ける。
        # 2026-08-19 にこれで実費 $8.34 を焼いた。必ず事前に量を測り、閾値を超えたら
        # ダウンロードせずに落とす。
        from huggingface_hub import HfApi

        patterns = [f"{subfolder}/**"] if subfolder else None
        info = HfApi().repo_info(repo, files_metadata=True)
        prefix = f"{subfolder}/" if subfolder else ""
        planned = sum(
            (s.size or 0) for s in info.siblings if s.rfilename.startswith(prefix)
        )
        print(f"[setup] ダウンロード予定量: {planned / 1e9:.2f} GB")
        if planned > MAX_WEIGHTS_DOWNLOAD_GB * 1e9:
            n_ckpt = len({
                "/".join(s.rfilename.split("/")[:-1])
                for s in info.siblings if s.rfilename.endswith("model.safetensors")
            })
            raise RuntimeError(
                f"ダウンロード量 {planned / 1e9:.1f} GB が上限 {MAX_WEIGHTS_DOWNLOAD_GB} GB を超える。"
                f"このリポジトリにはチェックポイントが {n_ckpt} 個ある。"
                f"`--configs 'ラベル={repo}@サブフォルダ'` の形で"
                f"評価したいチェックポイントを1つ指定すること。"
            )

        # Volume 上のキャッシュを見る。あれば取得を丸ごと省略できる。
        cache_key = repo.replace("/", "__") + ("__" + subfolder.replace("/", "__") if subfolder else "")
        cached = Path("/weights") / cache_key
        weights_volume.reload()
        if (cached / "config.json").exists():
            print(f"[setup] キャッシュ命中: {cached}（ダウンロードを省略）")
        else:
            print(f"[setup] キャッシュ無し。取得する: {cached}")
            snapshot = snapshot_download(repo_id=repo, allow_patterns=patterns)
            src = Path(snapshot) / subfolder if subfolder else Path(snapshot)
            tmp = Path("/weights") / (cache_key + ".partial")
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.copytree(src, tmp)
            # 中断時に壊れたキャッシュを掴まないよう、完成してから最終名にする
            shutil.rmtree(cached, ignore_errors=True)
            tmp.rename(cached)
            weights_volume.commit()
            print(f"[setup] キャッシュへ保存: {cached}")

        # `MyPolicy` が見る場所へ**シンボリックリンク**で繋ぐ。実体コピーだと
        # 9.35GB の複製が毎回走るため。
        if weights_dir.is_symlink() or weights_dir.exists():
            if weights_dir.is_symlink():
                weights_dir.unlink()
            else:
                shutil.rmtree(weights_dir, ignore_errors=True)
        weights_dir.symlink_to(cached, target_is_directory=True)
        size_gb = sum(f.stat().st_size for f in cached.rglob("*") if f.is_file()) / 1e9
        self._weights_source = f"{repo}@{subfolder}" if subfolder else repo
        print(f"[setup] 重みを接続: {cached} -> {weights_dir}（{size_gb:.2f} GB）")

        # 提出物と同じポリシー実装を使う（RTC・グリッパー開放プリアンブル等の修正込み）
        # `_INSTRUCTION_MAP` は import 時に確定するので、その前に環境変数を立てる。
        os.environ["PARC_INSTRUCTION_MAP"] = "1" if self.instruction_map else "0"
        if self.chunk_size:
            # 再計画間隔。提出物の既定は5（Track1で0.357を取った設定）。
            os.environ["PARC_CHUNK_SIZE"] = str(self.chunk_size)
        sys.path.insert(0, "/repo/submission_template")
        from policy_server import MyPolicy  # type: ignore[import-not-found]

        self._policy = MyPolicy(repo_id=repo)

        # **ロード完了を待つ。** `MyPolicy.__init__` はバックグラウンドスレッドを
        # 起動して即座に返る（本番はサーバ起動を速くするための設計）。待たずに
        # 評価を始めると `select_action()` が `self.loading` 分岐に入り、
        # **腕7次元すべて 0 の行動を返し続ける**。例外は出ず、600step 完走して
        # 失敗として記録されるため、**成功率が静かに 0 に落ちるだけ**で気づけない。
        # 2026-09-07 の比較実行では、コンテナごとに最初へ回されたタスクが
        # この状態で走り、180本中83本（46%）が腕を1mmも動かさないまま
        # 「失敗」として集計されていた。
        t0 = time.time()
        while self._policy.loading and time.time() - t0 < 1800:
            time.sleep(2.0)
        if self._policy.loading or getattr(self._policy, "model", None) is None:
            raise RuntimeError(
                f"ポリシーのロードが完了しない（{time.time()-t0:.0f}秒経過, "
                f"loading={self._policy.loading}, model={self._policy.model is not None}）。"
                "ゼロ行動で評価が進むのを防ぐためここで停止する。")
        print(f"[setup] 完了（モデルロード待ち {time.time()-t0:.0f}秒）")

    @modal.method()
    def run_task_gif(self, work: dict) -> dict:
        """1タスクを1エピソードだけ回し、**評価と同じループから**GIFを作る。

        `RolloutExecutor(frame_sink=...)` を使うので、描かれる軌道は
        `run_task` が採点する軌道と同一の手順で生成される（別実装で描き直すと、
        見ている絵と採点結果が食い違っても気づけない）。

        Returns:
            `{"gif": bytes, "success": bool, "collided": bool,
              "first_collision_step": int|None, "total_steps": int,
              "arm_zero_ratio": float, "n_frames": int}`
        """
        import io
        import sys
        import traceback

        import numpy as np

        sys.path.insert(0, "/repo")
        from PIL import Image

        from pipeline.config import EvalConfig, PerturbationConfig
        from pipeline.environment import EnvironmentManager
        from pipeline.rollout import RolloutExecutor

        suite, task_name = work["suite"], work["task_id"]
        print(f"[{self.config_label}] GIF suite={suite} task={task_name}")

        frames: list = []
        collision_marks: list = []

        def sink(step, obs, first_collision_step):
            # LIBERO の描画は上下反転して返るので戻す。前方＋手先を横に並べる。
            tiles = []
            for cam in ("agentview", "robot0_eye_in_hand"):
                img = obs.get(f"{cam}_image")
                if img is None:
                    continue
                tiles.append(np.asarray(img)[::-1])
            if not tiles:
                return
            frames.append(np.concatenate(tiles, axis=1).astype(np.uint8))
            collision_marks.append(first_collision_step is not None)

        try:
            eval_config = EvalConfig(
                n_eval_episodes=1,
                max_steps_per_episode=self.max_steps,
                benchmark_name=suite,
                seed=self.seed,
                device="cuda",
            )
            env_manager = EnvironmentManager(eval_config)
            matched = [t for t in env_manager.get_task_infos(suite) if t.name == task_name]
            if not matched:
                raise ValueError(f"タスク {task_name!r} が {suite} に無い")

            executor = RolloutExecutor(env_manager, eval_config, frame_sink=sink)
            res = executor.evaluate_task(
                _PolicyAdapter(self._policy), matched[0], PerturbationConfig()
            )
            ep = res.episodes[0]

            # 衝突が起きたフレーム以降は赤枠で示す（どの時点で当たったかを目で追える）
            pil = []
            for arr, hit in zip(frames, collision_marks):
                im = Image.fromarray(arr).resize(
                    (arr.shape[1] * 2, arr.shape[0] * 2), Image.NEAREST)
                if hit:
                    px = im.load()
                    w, h = im.size
                    for x in range(w):
                        for t in range(3):
                            px[x, t] = (255, 0, 0); px[x, h - 1 - t] = (255, 0, 0)
                    for y in range(h):
                        for t in range(3):
                            px[t, y] = (255, 0, 0); px[w - 1 - t, y] = (255, 0, 0)
                pil.append(im)

            buf = io.BytesIO()
            if pil:
                pil[0].save(buf, format="GIF", save_all=True, append_images=pil[1:],
                            duration=50, loop=0, optimize=True)
            acts = ep.actions
            zr = (sum(1 for a in acts if max(abs(float(v)) for v in a[:6]) < 1e-6)
                  / len(acts)) if len(acts) else 1.0
            return {
                "gif": buf.getvalue(), "success": bool(ep.success),
                "collided": bool(ep.collided),
                "first_collision_step": ep.first_collision_step,
                "total_steps": int(ep.total_steps), "arm_zero_ratio": zr,
                "n_frames": len(pil), "task_name": task_name, "suite": suite,
                "weights_source": self._weights_source,
            }
        except Exception as exc:
            traceback.print_exc()
            return {"gif": b"", "error": f"{type(exc).__name__}: {exc}",
                    "task_name": task_name, "suite": suite}

    @modal.method()
    def run_task(self, work: dict) -> list[dict]:
        """1タスクを n_episodes ぶん評価し、JSON化できるレコード列を返す。

        Args:
            work: `{"suite": "libero_spatial", "task_id": "..."}`。
                **スイートはタスクごとに異なる**（`compe/t1/T1_TASKS.csv` は
                spatial / object / goal が混在する）ため、クラス側の固定値ではなく
                work item で受け取る。
        """
        import sys
        import traceback

        sys.path.insert(0, "/repo")

        from pipeline.config import EvalConfig, PerturbationConfig
        from pipeline.environment import EnvironmentManager
        from pipeline.rollout import RolloutExecutor

        suite, task_name = work["suite"], work["task_id"]
        print(f"[{self.config_label}] suite={suite} task={task_name}")

        try:
            eval_config = EvalConfig(
                n_eval_episodes=self.n_episodes,
                max_steps_per_episode=self.max_steps,
                benchmark_name=suite,
                seed=self.seed,
                device="cuda",
            )
            env_manager = EnvironmentManager(eval_config)
            task_infos = env_manager.get_task_infos(suite)
            matched = [t for t in task_infos if t.name == task_name]
            if not matched:
                raise ValueError(
                    f"タスク {task_name!r} が {suite} に無い（{len(task_infos)} 件中）。"
                    f"候補例: {[t.name for t in task_infos][:3]}"
                )

            executor = RolloutExecutor(env_manager, eval_config)
            task_result = executor.evaluate_task(
                _PolicyAdapter(self._policy), matched[0], PerturbationConfig()
            )
            records = [
                _to_record(ep, self.config_label, self.seed + ep.episode_id)
                for ep in task_result.episodes
            ]
            for r in records:
                r["weights_source"] = self._weights_source
            return records
        except Exception as exc:  # 1タスクの失敗で全体を落とさない
            traceback.print_exc()
            return [{
                "config": self.config_label, "task_name": work.get("task_id", "?"), "episode_id": -1,
                "seed": self.seed, "success": False, "goal_reached": False,
                "collided": False, "first_collision_step": None, "total_steps": 0,
                "elapsed_time_sec": 0.0, "error": f"{type(exc).__name__}: {exc}",
            }]


class _PolicyAdapter:
    """`MyPolicy` を `pipeline.rollout.PolicyInterface` に合わせる薄いアダプタ。

    `MyPolicy.reset()` は `seed` を取らないが、`RolloutExecutor` は
    `reset(instruction=..., seed=...)` で呼ぶため、ここで吸収する。
    """

    def __init__(self, policy) -> None:
        self._policy = policy

    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        # **評価側だけでサンプリングを固定する。** `MyPolicy` はシードを一切扱わず
        # （`seed` で grep して0件）、pi0.5 の Flow Matching は毎回別の乱数で
        # サンプルするため、同一タスク・同一重みでも成功/衝突が反転する。
        # 9/7 に GIF を撮り直した際、記録では衝突していたエピソードが非衝突に、
        # 失敗が成功に変わった。ここで torch を固定すれば提出物を改造せずに
        # 評価だけ再現可能になる（本番の挙動は変わらない）。
        if seed is not None:
            try:
                import torch
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            except Exception as exc:
                print(f"[_PolicyAdapter] シード固定に失敗（再現性なしで続行）: {exc}")
        self._policy.reset(instruction=instruction)

    def get_action(self, obs: dict):
        return self._policy.get_action(obs)


def _to_record(ep, config_label: str, seed: int) -> dict:
    """`EpisodeResult` を JSON 化できる素の dict に落とす。"""
    def arr(xs):
        return [[float(v) for v in x] for x in xs]

    return {
        "config": config_label,
        "task_name": ep.task_name,
        "episode_id": ep.episode_id,
        "seed": seed,
        "success": bool(ep.success),
        "goal_reached": bool(ep.goal_reached),
        "collided": bool(ep.collided),
        "first_collision_step": ep.first_collision_step,
        "total_steps": int(ep.total_steps),
        "elapsed_time_sec": float(ep.elapsed_time_sec),
        "ee_positions": arr(ep.ee_positions),
        "ee_orientations": arr(ep.ee_orientations),
        "joint_positions": arr(ep.joint_positions),
        "actions": arr(ep.actions),
        "rewards": [float(r) for r in ep.rewards],
        # 腕6次元がすべて0だったstepの割合。**1.0 はポリシーが何も出していない印**で、
        # ロード未完了・フォールバック分岐の混入をあとから検出するために残す
        # （成功率だけ見ていると、この状態は「単に下手なポリシー」と区別できない）。
        "arm_zero_ratio": (
            sum(1 for a in ep.actions if max(abs(float(v)) for v in a[:6]) < 1e-6)
            / len(ep.actions)
        ) if len(ep.actions) else 1.0,
        "error": None,
    }


def _parse_configs(spec: str) -> list[tuple[str, str]]:
    """`"label=repo_id,label2=repo_id2"` を [(label, repo_id), ...] に。"""
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"--configs は 'label=repo_id' 形式で指定する: {item!r}")
        label, repo = item.split("=", 1)
        out.append((label.strip(), repo.strip()))
    return out


DEFAULT_TASK_CSV = "compe/t1/T1_TASKS.csv"


def _load_tasks_from_csv(path: str) -> list[dict]:
    """コンペ公式のタスク定義から work item を作る。

    `compe/t1/T1_TASKS.csv` は `task_id` と `suite` を持ち、**スイートはタスクごとに
    異なる**（spatial / object / goal が混在）。なお `libero_spatial` を
    `get_task_infos()` に渡すと LIBERO-plus の摂動バリアント2,402件が返るので、
    ここで評価対象を絞り込むことが必須である（既定で全件を回すと事故になる）。
    """
    import csv
    import pathlib

    rows = list(csv.DictReader(pathlib.Path(path).read_text(encoding="utf-8").splitlines()))
    return [{"suite": r["suite"], "task_id": r["task_id"]} for r in rows]


@app.local_entrypoint()
def main(
    configs: str = "s45000=nosuke113/parc2026-policy@pi05_single/libero_plus_b8_r16_step45000",
    task_csv: str = DEFAULT_TASK_CSV,
    tasks: str = "",
    n_episodes: int = 3,
    max_steps: int = 600,
    seed: int = 42,
    instruction_map: int = 1,
    chunk_size: int = 5,
    out: str = "results/eval_fanout",
) -> None:
    """条件×タスクで fan-out し、結果を JSON に落とす。

    Args:
        configs: `"label=hf_repo_id"` のカンマ区切り。複数指定で条件間比較になる。
        task_csv: 評価タスク定義（既定はコンペ公式の T1 タスク）。
        tasks: `"suite:task_id"` のカンマ区切り。指定すると task_csv より優先する。
        instruction_map: 指示文を学習データの文字列へ張り替えるか（既定 1＝する）。
            0 にすると評価が送る BDDL の文言をそのまま使う。効果を測るときは
            同じ `--out` に対して 1 と 0 で 2 回流すと、ラベルが `_nomap` で
            分かれるので `pipeline/eval_aggregate.py` がそのまま比較表にする。
    """
    import pathlib
    import time

    config_list = _parse_configs(configs)

    if tasks:
        work_items = []
        for item in tasks.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"--tasks は 'suite:task_id' 形式で指定する: {item!r}")
            suite, task_id = item.split(":", 1)
            work_items.append({"suite": suite.strip(), "task_id": task_id.strip()})
    else:
        work_items = _load_tasks_from_csv(task_csv)

    total = len(config_list) * len(work_items) * n_episodes
    print(f"条件 {len(config_list)} × タスク {len(work_items)} × {n_episodes} エピソード "
          f"= {total} ロールアウト")
    for w in work_items:
        print(f"  - [{w['suite']}] {w['task_id']}")

    out_dir = pathlib.Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    all_records: list[dict] = []

    for label, repo in config_list:
        if not instruction_map:
            label = f"{label}_nomap"
        print(f"\n=== 条件 {label} ({repo}) を {len(work_items)} タスクに fan-out ===")
        evaluator = TaskEvaluator(
            config_label=label, policy_repo=repo,
            n_episodes=n_episodes, max_steps=max_steps, seed=seed,
            instruction_map=instruction_map, chunk_size=chunk_size,
        )
        records: list[dict] = []
        for chunk in evaluator.run_task.map(work_items):
            records.extend(chunk)
            done = sum(1 for r in records if r["episode_id"] >= 0)
            print(f"  進捗: {done} エピソード完了 ({time.time() - started:.0f}秒経過)")

        path = out_dir / f"{label}.json"
        path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        print(f"  → {path} ({len(records)} レコード)")
        all_records.extend(records)

    elapsed = time.time() - started
    n_err = sum(1 for r in all_records if r.get("error"))
    print(f"\n=== 完了: {len(all_records)} レコード / {elapsed / 60:.1f}分 "
          f"/ エラー {n_err} 件 ===")
    if n_err:
        first = next(r for r in all_records if r.get("error"))
        print(f"  最初のエラー: [{first['task_name']}] {first['error'][:200]}")
    print(f"集約: python -m pipeline.eval_aggregate {out}/*.json --detail")


@app.function(image=parc_eval_image, timeout=600)
def _list_tasks(benchmark: str) -> list[str]:
    """ベンチマークのタスク名一覧を取得する（ローカルにLIBEROが無いため）。"""
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, "/repo")
    sys.path.insert(0, "/LIBERO-plus")
    home = Path(os.path.expanduser("~"))
    (home / ".libero").mkdir(parents=True, exist_ok=True)
    (home / ".libero" / "config.yaml").write_text(
        "benchmark_root: /LIBERO-plus/libero/libero\n"
        "bddl_files: /LIBERO-plus/libero/libero/bddl_files\n"
        "init_states: /LIBERO-plus/libero/libero/init_files\n"
        "datasets: /LIBERO-plus/libero/libero/datasets\n"
        "assets: /LIBERO/libero/libero/assets\n",
        encoding="utf-8",
    )
    from pipeline.config import EvalConfig
    from pipeline.environment import EnvironmentManager

    infos = EnvironmentManager(EvalConfig(benchmark_name=benchmark)).get_task_infos(benchmark)
    return [t.name for t in infos]


@app.local_entrypoint()
def visualize(
    configs: str = "step45000=nosuke113/parc2026-policy@pi05_valsplit/libero_plus_b8_r16_step45000",
    tasks: str = "",
    max_steps: int = 600,
    seed: int = 42,
    instruction_map: int = 1,
    chunk_size: int = 5,
    out: str = ".debug_output/val_rollout_gif",
):
    """指定タスクのロールアウトを GIF にする。

    `--tasks "suite:task_id,suite:task_id"` で複数指定でき、タスク単位で並列に回る。
    採点と同じ `RolloutExecutor` の中からフレームを取っているので、**GIF に映る
    軌道と、その隣に出る success / collided は同じ1回のロールアウトのもの**である。

    例:
        modal run scripts/eval_modal_fanout.py::visualize \
            --tasks "libero_goal:put_the_bowl_on_the_stove_moved_level1_sample1"
    """
    import pathlib

    config_list = _parse_configs(configs)
    if not tasks.strip():
        raise ValueError("--tasks は必須。'suite:task_id' をカンマ区切りで指定する")
    work_items = []
    for item in tasks.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"--tasks は 'suite:task_id' 形式で指定する: {item!r}")
        suite, tid = item.split(":", 1)
        work_items.append({"suite": suite.strip(), "task_id": tid.strip()})

    out_dir = pathlib.Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, repo in config_list:
        print(f"\n=== 条件 {label} ({repo}) の {len(work_items)} タスクを可視化 ===")
        evaluator = TaskEvaluator(
            config_label=label, policy_repo=repo,
            n_episodes=1, max_steps=max_steps, seed=seed,
            instruction_map=instruction_map, chunk_size=chunk_size,
        )
        for res in evaluator.run_task_gif.map(work_items):
            if res.get("error"):
                print(f"  NG {res['task_name']}: {res['error']}")
                continue
            name = f"{label}__{res['suite']}__{res['task_name'][:80]}.gif"
            path = out_dir / name
            path.write_bytes(res["gif"])
            flag = "成功" if res["success"] else "失敗"
            col = (f"衝突あり(step {res['first_collision_step']})"
                   if res["collided"] else "衝突なし")
            print(f"  {flag} / {col} / {res['total_steps']}step / "
                  f"腕ゼロ率 {res['arm_zero_ratio']:.2f} / {res['n_frames']}フレーム")
            print(f"    -> {path}")
