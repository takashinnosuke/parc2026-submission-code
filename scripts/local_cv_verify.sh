#!/usr/bin/env bash
# PARC 2026 ローカル事前検証（Omnicampus提出を消費しない手元CV）
#
# 本番の10秒/リクエスト制限はGPU前提の制約であり、ローカルCPU検証機では
# CPU推論速度そのものがボトルネックになるため、この制限を守ろうとすると
# 「本当にタスクを解けているか」を確認する前にタイムアウトしてしまう。
# そこで --timeout を緩め、"正解に近づいているか(方向性)" だけを手元で確認する。
# タイムアウト適合性そのものは別途 GPU 環境（本番 / Modal）で確認すること。
#
# 使い方:
#   bash scripts/local_cv_verify.sh [n_episodes] [max_steps] [timeout_sec]
# 例:
#   bash scripts/local_cv_verify.sh 2 150 60

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

N_EPISODES="${1:-2}"
MAX_STEPS="${2:-150}"
TIMEOUT_SEC="${3:-60}"
ZIP_PATH="submissions/submission_EXP_003.zip"

echo "[local_cv_verify] 提出zipを再ビルド中..."
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

echo "[local_cv_verify] Docker評価実行 (n_episodes=$N_EPISODES, max_steps=$MAX_STEPS, timeout=${TIMEOUT_SEC}s)..."
MSYS_NO_PATHCONV=1 docker run --rm \
    -v "$(pwd -W)/$ZIP_PATH:/sub.zip" \
    parc2026 \
    python evaluate.py /sub.zip \
        --n-episodes "$N_EPISODES" \
        --max-steps "$MAX_STEPS" \
        --timeout "$TIMEOUT_SEC"
