#!/usr/bin/env python
"""PARC 2026 ローカル詳細CV検証（衝突起因失敗の切り分け付き）。

pipeline/cli.py の標準出力（提出ID・合計時間・track総合スコア・タスク別成功率）に加え、
コミュニティで共有されていた別ツールの出力形式（n_episodes/successes/goals/collisions/
collision_attributable_failures 等のJSON + collision_summary.md + safe_demo_screen）を
再現する。定義は pipeline/collision_analysis.py のdocstring参照。

pipeline.cli.main() は PipelineResult.track_scores（集計済みTaskScoreのみ）しか
保持せず、衝突起因の失敗分析に必要な生の EpisodeResult（goal_reached, collided,
first_collision_step）を捨ててしまうため、EvaluationPipeline内部の
_evaluate_track相当の処理をここで直接組み立てて生のTaskResultを保持する。

使い方（Docker評価コンテナ内、ポリシーサーバーを起動した状態で）:
  python scripts/local_cv_verify_detailed.py --server-url http://localhost:8000 \
      --n-episodes 2 --max-steps 150 --timeout 60 --output-dir results/detailed

ローカルCPU検証時の10秒/リクエスト制限緩和は --timeout で行う
（scripts/local_cv_verify.sh と同じ考え方。本番のGPU前提のハード制約とは別物）。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import EvalConfig, Track, TRACK_BENCHMARKS, TRACK_PERTURBATIONS
from pipeline.environment import EnvironmentManager
from pipeline.rollout import PolicyInterface, RolloutExecutor, TaskResult
from pipeline.scorer import Scorer
from pipeline.collision_analysis import (
    analyze,
    render_collision_summary_markdown,
    write_collision_summary,
    write_safe_demo_screen,
)

logger = logging.getLogger(__name__)


class RandomPolicy:
    """ドライラン用の軽量ポリシー。ポリシーサーバー不要で配線確認ができる。"""

    def __init__(self, action_dim: int = 7):
        self.action_dim = action_dim

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return np.random.uniform(-1, 1, size=self.action_dim).astype(np.float32)

    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        if seed is not None:
            np.random.seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PARC 2026 ローカル詳細CV検証（衝突起因失敗の切り分け付き）"
    )
    parser.add_argument("--server-url", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="ランダムポリシーで配線確認のみ行う")
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--n-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--tasks", nargs="+", default=None, metavar="TASK_ID")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/detailed"))
    parser.add_argument("--submission-id", type=str, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = EvalConfig(
        n_eval_episodes=args.n_episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
    )
    if args.max_tasks is not None:
        config.max_tasks = args.max_tasks
    if args.tasks:
        config.task_ids = args.tasks
    if args.benchmark:
        config.benchmark_name = args.benchmark

    if args.dry_run:
        policy: PolicyInterface = RandomPolicy()
        submission_id = args.submission_id or "dry_run_detailed"
    elif args.server_url:
        from pipeline.remote_policy import RemotePolicyClient

        client = RemotePolicyClient(server_url=args.server_url, timeout_sec=args.timeout)
        client.wait_for_server()
        policy = client
        submission_id = args.submission_id or f"server_{args.server_url.split(':')[-1]}"
    else:
        logging.error("--dry-run または --server-url のいずれかを指定してください。")
        sys.exit(1)

    track = Track.TRACK1
    perturbation = TRACK_PERTURBATIONS[track]
    benchmarks = TRACK_BENCHMARKS[track]
    eval_benchmark = args.benchmark or benchmarks[-1]

    env_manager = EnvironmentManager(config)
    scorer = Scorer()
    executor = RolloutExecutor(env_manager, config, scoring_config=scorer.scoring_config)

    start_time = time.time()

    task_infos = env_manager.get_task_infos(eval_benchmark)
    if config.task_ids:
        available = {t.name for t in task_infos}
        unknown = [n for n in config.task_ids if n not in available]
        if unknown:
            raise RuntimeError(f"--tasks に存在しないタスク: {unknown}. 利用可能: {sorted(available)}")
        wanted = set(config.task_ids)
        task_infos = [t for t in task_infos if t.name in wanted]
    if config.max_tasks is not None:
        task_infos = task_infos[:config.max_tasks]

    task_results: list[TaskResult] = executor.evaluate_tasks(policy, task_infos, perturbation)
    track_score = scorer.score_track(track, eval_benchmark, task_results)

    elapsed = time.time() - start_time

    # --- 標準ブロック（pipeline/cli.py と同じ体裁） ---
    print("\n" + "=" * 60)
    print(f"提出ID: {submission_id}")
    print(f"合計時間: {elapsed:.1f}秒")
    print(f"\n  {track.value}: 総合スコア {track_score.overall_score:.3f}")
    for task_score in track_score.task_scores:
        print(f"    {task_score.task_name}: {task_score.success_rate:.1%}")
    print("=" * 60)

    # --- 衝突分析ブロック（コミュニティ共有ツール互換） ---
    analysis = analyze(task_results)
    print()
    print(json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = write_collision_summary(
        analysis, output_dir / "collision_summary.md",
        submission_id=submission_id, total_elapsed_sec=elapsed,
    )
    print(f"Summary: {summary_path}")
    print(f"Ready: {summary_path}")

    safe_demo_path = write_safe_demo_screen(task_results, output_dir / "safe_demo_screen.json")
    print(f"screening 出力: {safe_demo_path}")

    result_path = output_dir / f"{submission_id}_detailed.json"
    result_path.write_text(
        json.dumps({
            "submission_id": submission_id,
            "total_elapsed_sec": elapsed,
            "track_score": track_score.to_dict(),
            "collision_analysis": analysis.to_dict(),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"詳細結果: {result_path}")


if __name__ == "__main__":
    main()
