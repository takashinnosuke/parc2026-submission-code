# parc2026-submission-code

[PARC2026](https://matsuolab.github.io/PARC2026/)（Physical AI Robot Challenge 2026）予選 Track1 の提出にあたり、公式配布テンプレート [`matsuolab/PARC2026_pre`](https://github.com/matsuolab/PARC2026_pre) の上に**独自に追加・変更したコードのみ**を収録したリポジトリです。テンプレート自体に含まれるコード（`lerobot`ライブラリ本体、`harness/`等）は再配布せず、差分のみを公開しています。

背景・工夫点の詳細は [PARC2026 予選レポート](../PARC2026_Official_Report.md) を参照してください（主な工夫点: RTC (Real-Time Chunking) の `inference_delay` パラメータ誤りの修正、`chunk_size` の調整、グリッパー初期状態不一致の修正）。

## 使い方

1. 公式テンプレート [`matsuolab/PARC2026_pre`](https://github.com/matsuolab/PARC2026_pre) をclone/セットアップする。
2. 本リポジトリの `scripts/`、`pipeline/cv_evaluator.py` をそのままコピーする。
3. `submission_template/policy_server.py`、`submission_template/requirements.txt` を対応するテンプレート側のファイルに上書きする。
4. `patches/pipeline_and_compe.patch` を、テンプレート側の `pipeline/config.py`、`pipeline/rollout.py`、`pipeline/validator.py`、`compe/t1/register.py` に適用する（`git apply patches/pipeline_and_compe.patch`）。

## 構成

- `scripts/`: 学習（Modal/Colab上でのπ0.5 LoRA学習）、評価、可視化、提出物ビルド用の独自スクリプト群。
- `pipeline/cv_evaluator.py`: ローカルCV評価用の追加モジュール。
- `submission_template/policy_server.py`: RTC統合・グリッパー初期化修正・状態ベクトル構築等を含む提出用ポリシーサーバー実装。
- `submission_template/requirements.txt`: 提出物の依存関係定義。
- `patches/`: 公式テンプレートの既存ファイルに対する小規模な変更差分。

## 学習済み重み

Hugging Face Hub: [`nosuke113/parc2026-policy`](https://huggingface.co/nosuke113/parc2026-policy)（パス: `pi05_sequential/total45000_stage2_0_b8_r16_final`）

## ライセンス

本リポジトリのコードは公式テンプレート [`matsuolab/PARC2026_pre`](https://github.com/matsuolab/PARC2026_pre) の一部を改変したものです。ベースモデルは Google Gemma 利用規約、学習データ `lerobot/libero_plus` は Hugging Face Hub 配布の講座提供データセットです。
