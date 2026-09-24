"""fan-out評価の結果を集約し、公式4指標で条件間比較するモジュール。

Modal 上で並列実行した評価（`scripts/eval_modal_fanout.py`）は、コンテナごとに
**JSON化できる素のエピソード記録**を返す。本モジュールはそれを受けて
`pipeline/scorer.py` の `Scorer` に食わせ、実験条件（config）ごとのスコアと
条件間の比較表を作る。

## なぜ record と EpisodeResult を分けるのか

`pipeline/rollout.py` の `EpisodeResult` は numpy 配列を、`TaskInfo` は
`torch.Tensor`（init_states）を保持するため、そのままではプロセス境界を
越えられない。そこで Modal 側は `EpisodeRecord`（純粋なリスト・スカラーのみ）を
返し、ローカルで `EpisodeResult` に復元してから採点する。

なお `Scorer` が `TaskInfo` から参照するのは `.name` だけなので、復元時に
重い `TaskInfo` を作る必要はない（`_TaskInfoRef` で代用する）。

## 公式4指標との対応

| 公式カテゴリ | 本モジュールが出す指標 |
|---|---|
| タスク成功率 | `success_rate`（= goal到達かつ非衝突） |
| 滑らかさ | `avg_cartesian_jerk` / `rms_cartesian_jerk` / `avg_joint_jerk` / `sparc` / `orientation_path_length`（EEF回転総量） |
| 実行効率 | `avg_steps_to_success` / `cartesian_path_length`（軌道総距離） |
| 安全性 | `collision_rate` / `collision_attributable_failure_rate` |

前3者は `Scorer` が、安全性は `pipeline/collision_analysis.py` が算出する。

## 使い方

```bash
python -m pipeline.eval_aggregate results/eval_fanout/*.json --compare
```
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .collision_analysis import analyze as analyze_collisions
from .config import Track
from .rollout import EpisodeResult, TaskResult
from .scorer import Scorer, TrackScore

logger = logging.getLogger(__name__)


@dataclass
class _TaskInfoRef:
    """`Scorer` が参照するのは `.name` だけなので、それだけを持つ軽量な代用。"""

    name: str


@dataclass
class EpisodeRecord:
    """プロセス境界を越えられる、1エピソードの記録。

    `EpisodeResult` と1対1に対応するが、numpy を含まず JSON 化できる。

    Attributes:
        config: 実験条件のラベル。**比較の単位**になる（例 "dedup" / "full"）。
        seed: 再現性のため必ず記録する。
    """

    config: str
    task_name: str
    episode_id: int
    seed: int
    success: bool
    goal_reached: bool
    collided: bool
    first_collision_step: int | None
    total_steps: int
    elapsed_time_sec: float

    # 軌道。滑らかさ・実行効率の指標はここから計算される。
    ee_positions: list[list[float]] = field(default_factory=list)
    ee_orientations: list[list[float]] = field(default_factory=list)
    joint_positions: list[list[float]] = field(default_factory=list)
    actions: list[list[float]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)

    # 失敗時の切り分け用（例外で落ちたのか、単に成功しなかったのか）
    error: str | None = None
    # 実際に評価された重みの出所。条件ラベルと重みが 食い違いになる事故
    # （提出物同梱の重みを読んでしまい、全条件が同一重みになる等）を検知するため
    # 必ず記録する。
    weights_source: str | None = None

    def to_episode_result(self) -> EpisodeResult:
        return EpisodeResult(
            task_name=self.task_name,
            episode_id=self.episode_id,
            success=self.success,
            total_steps=self.total_steps,
            elapsed_time_sec=self.elapsed_time_sec,
            joint_positions=[np.asarray(x, dtype=np.float64) for x in self.joint_positions],
            ee_positions=[np.asarray(x, dtype=np.float64) for x in self.ee_positions],
            ee_orientations=[np.asarray(x, dtype=np.float64) for x in self.ee_orientations],
            actions=[np.asarray(x, dtype=np.float64) for x in self.actions],
            rewards=list(self.rewards),
            collided=self.collided,
            goal_reached=self.goal_reached,
            first_collision_step=self.first_collision_step,
        )


@dataclass
class ConfigScore:
    """1つの実験条件のスコア。"""

    config: str
    n_episodes: int
    n_tasks: int
    n_errors: int
    track_score: TrackScore
    collision: dict[str, Any]
    weights_sources: set[str] = field(default_factory=set)

    @property
    def success_rate(self) -> float:
        return float(self.track_score.overall_metrics.get("mean_success_rate", 0.0))

    def metric(self, name: str) -> float | None:
        """`mean_<name>` で集計された指標を取り出す。"""
        return self.track_score.overall_metrics.get(f"mean_{name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "n_episodes": self.n_episodes,
            "n_tasks": self.n_tasks,
            "n_errors": self.n_errors,
            "track_score": self.track_score.to_dict(),
            "collision": self.collision,
            "weights_sources": sorted(self.weights_sources),
        }


def load_records(paths: list[str | Path]) -> list[EpisodeRecord]:
    """JSON（1ファイル＝レコード配列、またはレコード単体）を読み込む。"""
    records: list[EpisodeRecord] = []
    known = set(EpisodeRecord.__dataclass_fields__)
    for path in paths:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            # 将来スキーマが増えても落ちないよう、未知のキーは捨てる
            records.append(EpisodeRecord(**{k: v for k, v in item.items() if k in known}))
    logger.info("読み込んだエピソード記録: %d 件 (%d ファイル)", len(records), len(paths))
    return records


def build_task_results(records: list[EpisodeRecord]) -> list[TaskResult]:
    """タスクごとに `TaskResult` へまとめる（タスク名順で決定的に並べる）。"""
    by_task: dict[str, list[EpisodeRecord]] = {}
    for r in records:
        by_task.setdefault(r.task_name, []).append(r)
    results = []
    for name in sorted(by_task):
        eps = sorted(by_task[name], key=lambda r: r.episode_id)
        results.append(
            TaskResult(
                task_info=_TaskInfoRef(name=name),  # type: ignore[arg-type]
                episodes=[e.to_episode_result() for e in eps],
            )
        )
    return results


def score_config(
    records: list[EpisodeRecord],
    config: str,
    benchmark_name: str = "libero_spatial",
    track: Track = Track.TRACK1,
) -> ConfigScore:
    """1条件ぶんのレコードを公式4指標で採点する。"""
    subset = [r for r in records if r.config == config]
    if not subset:
        raise ValueError(f"config={config!r} のレコードが無い")
    task_results = build_task_results(subset)
    track_score = Scorer().score_track(track, benchmark_name, task_results)
    return ConfigScore(
        config=config,
        n_episodes=len(subset),
        n_tasks=len(task_results),
        n_errors=sum(1 for r in subset if r.error),
        track_score=track_score,
        collision=analyze_collisions(task_results).to_dict(),
        weights_sources={r.weights_source for r in subset if r.weights_source},
    )


def score_all(records: list[EpisodeRecord], **kwargs: Any) -> list[ConfigScore]:
    """レコードに含まれる全条件を採点する（条件名順）。"""
    return [score_config(records, c, **kwargs) for c in sorted({r.config for r in records})]


# 比較表に出す指標と表示名。公式4カテゴリを1行で見渡せる順に並べる。
COMPARISON_METRICS: list[tuple[str, str, str]] = [
    ("success_rate", "成功率", "{:.1%}"),
    ("collision_rate", "衝突率", "{:.1%}"),
    ("avg_steps_to_success", "平均step", "{:.1f}"),
    ("avg_cartesian_jerk", "平均jerk", "{:.3f}"),
    ("sparc", "SPARC", "{:.3f}"),
    ("cartesian_path_length", "軌道長", "{:.3f}"),
    # 公式の「滑らかさ」は jerk・SPARC に加えて **EEF回転総量** を挙げている
    ("orientation_path_length", "EEF回転", "{:.3f}"),
]


def format_comparison(scores: list[ConfigScore]) -> str:
    """条件間の比較表。n≥3 のアブレーションはこれを見て判断する。"""
    lines: list[str] = []
    add = lines.append

    add("=== 条件間比較（公式4指標） ===")
    header = f"  {'config':<16} {'n_ep':>5} {'tasks':>6}"
    for _, label, _ in COMPARISON_METRICS:
        header += f" {label:>10}"
    header += f" {'errors':>7}"
    add(header)
    add("  " + "-" * (len(header) - 2))

    for s in scores:
        row = f"  {s.config:<16} {s.n_episodes:>5} {s.n_tasks:>6}"
        for key, _, fmt in COMPARISON_METRICS:
            if key == "success_rate":
                value: float | None = s.success_rate
            elif key == "collision_rate":
                value = s.collision.get("collision_rate")
            else:
                value = s.metric(key)
            row += f" {fmt.format(value) if value is not None else '-':>10}"
        row += f" {s.n_errors:>7}"
        add(row)

    if len(scores) >= 2:
        base = scores[0]
        add("")
        add(f"=== {base.config} を基準にした差分 ===")
        for s in scores[1:]:
            diffs = [f"成功率 {(s.success_rate - base.success_rate) * 100:+.1f}pt"]
            b_col, s_col = base.collision.get("collision_rate"), s.collision.get("collision_rate")
            if b_col is not None and s_col is not None:
                diffs.append(f"衝突率 {(s_col - b_col) * 100:+.1f}pt")
            for key, label, _ in COMPARISON_METRICS[2:]:
                bv, sv = base.metric(key), s.metric(key)
                if bv and sv:
                    diffs.append(f"{label} {(sv / bv - 1) * 100:+.1f}%")
            add(f"  {s.config:<16} " + " / ".join(diffs))

    add("")
    sources = {s.config: s.weights_sources for s in scores}
    add("=== 各条件が実際に評価した重み ===")
    for cfg, srcs in sources.items():
        add(f"  {cfg:<16} {sorted(srcs) if srcs else '(未記録)'}")
    distinct = {tuple(sorted(v)) for v in sources.values() if v}
    if len(sources) > 1 and len(distinct) == 1:
        add("  ⚠ 全条件が同一の重みを評価している。条件間比較として無効。")
    add("")
    add("注: n が小さいうちは差分の符号だけを見て結論を出さないこと。")
    add("    Flow Matching 系は試行ごとのばらつきが大きく、n=1〜2 では逆転しうる。")
    return "\n".join(lines)


def format_config_detail(score: ConfigScore) -> str:
    """1条件のタスク別内訳。"""
    lines = [f"=== {score.config}: タスク別 ===",
             f"  {'task':<52} {'成功率':>8} {'step':>8}"]
    for ts in score.track_score.task_scores:
        steps = next((m.value for m in ts.metrics if m.name == "avg_steps_to_success"), None)
        lines.append(
            f"  {ts.task_name[:52]:<52} {ts.success_rate:>7.1%} "
            f"{(f'{steps:.1f}' if steps else '-'):>8}"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="fan-out評価の結果を公式4指標で集約・比較する"
    )
    parser.add_argument("inputs", nargs="+", help="レコードJSON（globパターン可）")
    parser.add_argument("--benchmark", default="libero_spatial")
    parser.add_argument("--detail", action="store_true", help="条件ごとのタスク別内訳も出す")
    parser.add_argument("--json", type=Path, default=None, help="集約結果のJSON出力先")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    paths: list[str] = []
    for pattern in args.inputs:
        matched = glob.glob(pattern)
        paths.extend(matched if matched else [pattern])
    if not paths:
        raise SystemExit("入力ファイルが見つからない")

    records = load_records(paths)
    scores = score_all(records, benchmark_name=args.benchmark)

    print(format_comparison(scores))
    if args.detail:
        for s in scores:
            print()
            print(format_config_detail(s))

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([s.to_dict() for s in scores], indent=2, ensure_ascii=False, default=float),
            encoding="utf-8",
        )
        print(f"\nJSON written to {args.json}")


if __name__ == "__main__":
    main()
