#!/usr/bin/env bash
# PARC 2026 ローカル詳細CV検証（衝突起因失敗の切り分け付き、Omnicampus提出は消費しない）
#
# scripts/local_cv_verify.sh と同じ考え方（本番の10秒/リクエスト制限はGPU前提の
# ハード制約なので、ローカルCPU検証では --timeout を緩めて"方向性"のみ確認する）
# だが、こちらは scripts/local_cv_verify_detailed.py を使い、コミュニティ共有の
# 別ツール互換の collision_summary.md / safe_demo_screen.json / 詳細JSON まで出す。
#
# 使い方:
#   bash scripts/local_cv_verify_detailed.sh [n_episodes] [max_steps] [timeout_sec]
# 例:
#   bash scripts/local_cv_verify_detailed.sh 2 150 60
#
# 出力は ./results/detailed/ 配下（コンテナ内 /workspace/results/detailed を
# ホストにマウントして取り出す）。

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

N_EPISODES="${1:-2}"
MAX_STEPS="${2:-150}"
TIMEOUT_SEC="${3:-60}"
ZIP_PATH="submissions/submission_EXP_003.zip"
HOST_OUT_DIR="$(pwd -W)/results/detailed"
mkdir -p "results/detailed"

echo "[local_cv_verify_detailed] 提出zipを再ビルド中..."
python -c "
import os, zipfile
from pathlib import Path
sub_template = Path('submission_template')
output_zip = Path('$ZIP_PATH')
with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(sub_template):
        if '__pycache__' in root:
            continue
        for f in files:
            if f.endswith('.pyc'):
                continue
            full_path = Path(root) / f
            z.write(full_path, full_path.relative_to(sub_template))
print(f'Rebuilt {output_zip} ({output_zip.stat().st_size / (1024*1024):.1f} MB)')
"

echo "[local_cv_verify_detailed] Docker評価実行 (n_episodes=$N_EPISODES, max_steps=$MAX_STEPS, timeout=${TIMEOUT_SEC}s)..."
MSYS_NO_PATHCONV=1 docker run --rm \
    -v "$(pwd -W)/$ZIP_PATH:/sub.zip" \
    -v "$HOST_OUT_DIR:/workspace/results/detailed" \
    parc2026 \
    bash -c '
        set -euo pipefail
        mkdir -p /tmp/sub
        cd /workspace && python -c "
import zipfile
with zipfile.ZipFile(\"/sub.zip\") as z:
    z.extractall(\"/tmp/sub\")
"
        cd /tmp/sub
        python policy_server.py --port 8000 &
        SERVER_PID=$!
        trap "kill $SERVER_PID 2>/dev/null || true" EXIT

        echo "[local_cv_verify_detailed] ポリシーサーバー起動待機..."
        for i in $(seq 1 120); do
            if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
                echo "[local_cv_verify_detailed] サーバー起動確認 (${i}秒)"
                break
            fi
            sleep 1
        done

        cd /workspace
        python scripts/local_cv_verify_detailed.py \
            --server-url http://localhost:8000 \
            --n-episodes '"$N_EPISODES"' \
            --max-steps '"$MAX_STEPS"' \
            --timeout '"$TIMEOUT_SEC"' \
            --output-dir /workspace/results/detailed
    '

echo "[local_cv_verify_detailed] 完了。結果: results/detailed/"
