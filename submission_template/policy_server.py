#!/usr/bin/env python3
"""
PARC 2026 Policy Server (Submission Template - Python 3.10 & Non-blocking Fast Server Architecture)
"""

import os
import sys
import typing
import threading
from types import ModuleType
from importlib.machinery import ModuleSpec

# ==============================================================================
# 0点回避・Omnicampus サンドボックス互換プロテクション（必須モンキーパッチ群）
# ==============================================================================

# 0. 学習環境/評価環境の差異対策: 本番はサーバー起動後ネットワーク完全遮断のため、
#    HFライブラリのネット越しフェッチを最初からオフラインモードに固定する。
#    ローカル開発時にネットが通ることで隠れていた「本番だけ失敗する」問題を防ぐ
#    （ローカルでも同じ挙動＝失敗するなら失敗するに揃えることで乖離を無くす）。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 0b. SmolVLA の vlm_model_name ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct") は、
#     学習済み重みは model_weights/ 内のローカル safetensors から読むが、トークナイザ/
#     プロセッサ設定だけは repo_id 名で from_pretrained() 解決される。オフライン化した
#     だけではキャッシュが無いと失敗するため、同梱した軽量キャッシュ（config/tokenizer/
#     processor の json のみ。VLM自体の重みファイルは含まない）を指す。
_HF_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")
if os.path.isdir(_HF_CACHE_DIR):
    os.environ.setdefault("HF_HUB_CACHE", _HF_CACHE_DIR)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _HF_CACHE_DIR)
    os.environ.setdefault("TRANSFORMERS_CACHE", _HF_CACHE_DIR)

# A. Python 3.10 の typing.Self 欠落に対する必須保護
try:
    from typing import Self
except ImportError:
    try:
        from typing_extensions import Self
        typing.Self = Self
        sys.modules['typing'].Self = Self
    except ImportError:
        typing.Self = object
        sys.modules['typing'].Self = object

# A2. Python 3.10 の typing.Unpack 欠落に対する必須保護
#     （lerobot の policies/pretrained.py, factory.py, smolvla/modeling_smolvla.py 等が
#      `from typing import ... Unpack` を直接 import しており、Unpack は Python 3.11 で
#      typing に追加されたため、本番/harness の Python 3.10.12 では ImportError で
#      全ローダーが失敗し、フォールバック(ゼロ)action しか返せなくなる）。
try:
    from typing import Unpack
except ImportError:
    try:
        from typing_extensions import Unpack
        typing.Unpack = Unpack
        sys.modules['typing'].Unpack = Unpack
    except ImportError:
        typing.Unpack = typing.Any
        sys.modules['typing'].Unpack = typing.Any

# B. PyAV (av) 非依存化・未インストール環境での全滅回避モジュールインジェクション
class DummyType(type):
    def __or__(self, other):
        return typing.Union[self, other]
    def __ror__(self, other):
        return typing.Union[other, self]
    def __getattr__(cls, name):
        class SubDummy(metaclass=DummyType):
            pass
        SubDummy.__name__ = f"{cls.__name__}.{name}"
        SubDummy.__qualname__ = f"{cls.__qualname__}.{name}"
        setattr(cls, name, SubDummy)
        return SubDummy

class DummyModule(ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__file__ = "dummy.py"
        self.__spec__ = ModuleSpec(name, loader=None)
    def __getattr__(self, name):
        if name == "__spec__":
            return ModuleSpec(self.__name__, loader=None)
        class DynamicDummy(metaclass=DummyType):
            pass
        DynamicDummy.__name__ = f"{self.__name__}.{name}"
        DynamicDummy.__qualname__ = f"{self.__name__}.{name}"
        setattr(self, name, DynamicDummy)
        return DynamicDummy

if "av" not in sys.modules or getattr(sys.modules["av"], "__spec__", None) is None:
    dummy_av = DummyModule("av")
    dummy_av.open = lambda *a, **k: None
    sys.modules["av"] = dummy_av

# C. transformers / huggingface_hub バージョン不一致に伴うモジュール欠落補填
try:
    import transformers.utils
    if not hasattr(transformers.utils, "torch_compilable_check"):
        transformers.utils.torch_compilable_check = lambda fn: fn
except Exception:
    pass

try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "sync_bucket"):
        huggingface_hub.sync_bucket = lambda *args, **kwargs: None
except Exception:
    pass

try:
    import lerobot.utils.import_utils
    lerobot.utils.import_utils.require_package = lambda *args, **kwargs: None
except Exception:
    pass

# ==============================================================================
# メイン Policy Server ロジック
# ==============================================================================

import time
import json
import argparse
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response

class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, obs: dict) -> np.ndarray:
        pass

    @abstractmethod
    def reset(self, instruction: str = "") -> None:
        pass

class MyPolicy(BasePolicy):
    def __init__(self, repo_id: str = "nosuke113/parc2026-policy"):
        self.repo_id = os.environ.get("POLICY_REPO_ID", repo_id)
        self.instruction = ""
        self.action_queue = []
        # 2026-08-13: chunk_size=20はconfig本来のn_action_steps=10の2倍だった。
        # グリッパー開放プリアンブル導入(下記)によりグリッパー崩壊問題が解消したため、
        # 学習時のn_action_stepsに合わせて10に戻す。
        self.chunk_size = 10
        self.model = None
        self.preprocessor = None
        self.postprocessor = None
        self.device = "cpu"
        self.torch_available = False
        self.loading = True
        # 学習時の画像解像度 (model_weights/config.json の image_features 形状) に合わせる
        # デフォルト値。_load_model 成功後、実際の config から取得できればそちらで上書きする。
        self._image_target_hw = (256, 256)
        self._image_feature_keys = []
        self._rtc_active = False
        self._model_full_chunk_size = self.chunk_size
        self._prev_full_chunk_tensor = None

        # 2026-08-13発見(バグ3): harnessのwarmup処理(env.step(zeros)を最大130回)は
        # gripper action=0を送るが、Pandaグリッパーでは0は「全開と全閉の中間」に対応するため、
        # 学習データでは常に全開(qpos≈0.04)で始まるエピソード先頭が、評価では半開(≈0.02)から
        # 始まってしまう。この学習分布外の観測により、ポリシーはstep0からgripper action=+1
        # (閉)に張り付き、二度と開かずに把持不能になる崩壊挙動が起きていた
        # （A/Bロールアウト可視化で実証済み: .debug_output/rollout_viz/rollout_45k_preopen0.gif
        # vs rollout_45k_preopen25.gif）。harness側は変更できないため、エピソード開始直後、
        # グリッパーが十分開くまで強制的に開放アクションを返すプリアンブルで吸収する。
        self._preamble_active = True
        self._preamble_steps = 0
        self._preamble_max_steps = 25
        self._preamble_open_threshold = 0.035

        print(f"[MyPolicy] Initializing Policy Server asynchronously for fast startup (repo: {self.repo_id})")

        # バックグラウンドスレッドで非同期モデルロード (FastAPI 起動ブロックを 100% 遮断)
        self.loader_thread = threading.Thread(target=self._async_init, daemon=True)
        self.loader_thread.start()

    def _async_init(self):
        try:
            import torch
            self.torch_available = True
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu":
                torch.set_num_threads(os.cpu_count() or 4)
            print(f"[MyPolicy] PyTorch {torch.__version__} loaded. torch.cuda.is_available()={torch.cuda.is_available()}, Device: {self.device}")

            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"[MyPolicy] Detected GPU VRAM: {vram_gb:.2f} GB")
                if vram_gb >= 20:
                    print("[MyPolicy] Using BF16 Full-Precision for NVIDIA L4 (VRAM >= 20GB)")
                    self.torch_dtype = torch.bfloat16
                else:
                    print("[MyPolicy] Using FP16 for Local GPU (VRAM < 20GB)")
                    self.torch_dtype = torch.float16
            else:
                print("[MyPolicy] CUDA not available, using CPU mode")
                self.torch_dtype = torch.float32

            self._load_model(self.repo_id)
        except Exception as e:
            print(f"[MyPolicy] PyTorch/lerobot init notice ({e}). Operating in fallback mode.")
            self.model = None
        finally:
            self.loading = False

    def _clean_config_json(self, ckpt_dir: str):
        config_path = os.path.join(ckpt_dir, "config.json")
        adapter_path = os.path.join(ckpt_dir, "adapter_config.json")

        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                content = content.replace("lerobot\\\\smolvla_libero_plus", "lerobot/smolvla_libero_plus")
                content = content.replace("lerobot\\smolvla_libero_plus", "lerobot/smolvla_libero_plus")
                cfg_dict = json.loads(content)
                
                for k in ["pretrained_revision", "rtc_config", "compile_model", "compile_mode", "model_type"]:
                    cfg_dict.pop(k, None)

                if "type" not in cfg_dict:
                    cfg_dict["type"] = cfg_dict.get("policy_type", "smolvla")

                # 重要: pretrained_path / base_model_name_or_path はネット越し repo ID
                # ("lerobot/smolvla_libero_plus" 等) ではなく、必ず zip 同梱のローカル
                # ディレクトリを指すこと。本番はサーバー起動後ネットワーク完全遮断のため、
                # 誤って repo ID を指すと PEFT ローダーがネット越し取得を試みて失敗する
                # （ローカルはネットが通るため症状が隠れ「手元と採点の乖離」の原因になる）。
                cfg_dict["use_peft"] = True
                cfg_dict["pretrained_path"] = ckpt_dir

                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(cfg_dict, f, indent=2)
                print(f"[MyPolicy] Cleansed config.json at {ckpt_dir}")
            except Exception as e:
                print(f"[MyPolicy] Config cleanse notice ({ckpt_dir}): {e}")

        if os.path.exists(adapter_path):
            try:
                with open(adapter_path, "r", encoding="utf-8") as f:
                    acontent = f.read()
                acontent = acontent.replace("lerobot\\\\smolvla_libero_plus", "lerobot/smolvla_libero_plus")
                acontent = acontent.replace("lerobot\\smolvla_libero_plus", "lerobot/smolvla_libero_plus")
                adict = json.loads(acontent)
                # base_model_name_or_path も同様にローカルディレクトリを指す（上記参照）
                adict["base_model_name_or_path"] = ckpt_dir
                with open(adapter_path, "w", encoding="utf-8") as f:
                    json.dump(adict, f, indent=2)
                print(f"[MyPolicy] Cleansed adapter_config.json at {ckpt_dir}")
            except Exception as e:
                print(f"[MyPolicy] Adapter config cleanse notice ({ckpt_dir}): {e}")

    def _load_model(self, repo_id: str):
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            local_weights = os.path.join(base_dir, "model_weights")

            targets_to_try = []
            if os.path.exists(local_weights):
                for root, dirs, files in os.walk(local_weights):
                    if "config.json" in files and ("model.safetensors" in files or "adapter_model.safetensors" in files):
                        self._clean_config_json(root)
                        targets_to_try.append(root)
                if local_weights not in targets_to_try:
                    self._clean_config_json(local_weights)
                    targets_to_try.append(local_weights)
            # 注意: 本番はサーバー起動後ネットワーク完全遮断のため、repo_id によるネット越し
            # フォールバックは追加しない（zip 同梱の local_weights のみを試す）。
            # ローカル開発時にネットで偶然成功して問題を隠すことを防ぐため。

            print(f"[MyPolicy] Targets to attempt loading: {targets_to_try}")

            for target in targets_to_try:
                if self.model is not None:
                    break
                print(f"[MyPolicy] Attempting to load model from target: '{target}'")

                # Loader 0: config.json の "type" フィールドから動的にポリシークラスを
                # 解決する汎用ローダー（最優先）。SmolVLA・pi0・pi05等、config.jsonの
                # 中身が正しければモデルを差し替えてもコード変更不要で対応できる。
                try:
                    from lerobot.configs.policies import PreTrainedConfig
                    from lerobot.policies.factory import get_policy_class
                    cfg0 = PreTrainedConfig.from_pretrained(target)
                    policy_cls = get_policy_class(cfg0.type)
                    self.model = policy_cls.from_pretrained(target)
                    print(f"[MyPolicy] ✅ Successfully loaded policy using generic get_policy_class("
                          f"'{cfg0.type}') from '{target}'")
                    break
                except Exception as e:
                    print(f"[MyPolicy] Generic policy-class loader notice ({target}): {e}")

                # Loader 1: SmolVLAPolicy Direct API (フォールバック)
                try:
                    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
                    self.model = SmolVLAPolicy.from_pretrained(target)
                    print(f"[MyPolicy] ✅ Successfully loaded policy using SmolVLAPolicy Direct API from '{target}'")
                    break
                except Exception as e:
                    print(f"[MyPolicy] SmolVLAPolicy notice ({target}): {e}")

                # Loader 2: LeRobot PreTrainedConfig + make_policy
                try:
                    from lerobot.configs.policies import PreTrainedConfig
                    from lerobot.policies.factory import make_policy
                    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
                    
                    config = PreTrainedConfig.from_pretrained(target)
                    if hasattr(config, "pretrained_path") and isinstance(config.pretrained_path, str):
                        config.pretrained_path = config.pretrained_path.replace("\\\\", "/").replace("\\", "/")

                    ds_meta = None
                    try:
                        ds_meta = LeRobotDatasetMetadata("lerobot/libero")
                    except Exception:
                        pass
                    
                    rename_map = None
                    try:
                        rename_map = {
                            "observation.images.image": "observation.images.front",
                            "observation.images.image2": "observation.images.wrist"
                        }
                        self.model = make_policy(config, ds_meta=ds_meta, rename_map=rename_map)
                    except Exception:
                        self.model = make_policy(config, ds_meta=ds_meta)

                    print(f"[MyPolicy] ✅ Successfully loaded policy using make_policy from '{target}'")
                    break
                except Exception as e:
                    print(f"[MyPolicy] make_policy notice ({target}): {e}")

                # Loader 3: Direct OpenVLAPolicy API
                try:
                    from lerobot.policies.openvla.modeling_openvla import OpenVLAPolicy
                    self.model = OpenVLAPolicy.from_pretrained(target)
                    print(f"[MyPolicy] ✅ Successfully loaded policy using OpenVLAPolicy Direct API from '{target}'")
                    break
                except Exception as e:
                    print(f"[MyPolicy] OpenVLAPolicy notice ({target}): {e}")

            if self.model is not None:
                if hasattr(self.model, "to"):
                    self.model.to(device=self.device)
                if hasattr(self.model, "eval"):
                    self.model.eval()
                # 実際に読み込んだモデルの学習時解像度・カメラキー名を config から取得する。
                # 学習に使ったデータセットによってキー名が変わる（例: libero_plus由来は
                # "observation.images.front"/"wrist"、lerobot/libero由来は
                # "observation.images.image"/"image2"）ため、ハードコードせず
                # config.input_features の VISUAL 特徴量から動的に検出する。
                self._image_feature_keys = []
                # pi0/pi0.5系は内部の_preprocess_images()が resize_with_pad_torch で
                # config.image_resolution へ自動リサイズする（画像キーが観測に無ければ
                # -1パディングの空カメラとして自動生成するため、front/wristだけ渡せば良い）。
                # このモデルに対して手前で input_features.shape(=学習データセットの生解像度、
                # 例256x256) へ手動リサイズしてしまうと、評価環境の生解像度(128x128)を
                # 一旦256x256へ拡大 → モデル内部で224x224へ再度縮小、という「拡大→縮小」の
                # 二重リサイズになり、学習時（256x256の実写真を224x224へ一度だけ縮小）には
                # 存在しないボケ・情報損失を生む（2026-08-12、把持動作が学習できない一因として発見）。
                # image_resolutionを持つモデルは手動リサイズせず生解像度のまま渡し、
                # モデル内部のresize_with_pad_torchに一度だけ通す。
                self._skip_manual_resize = False
                try:
                    cfg = getattr(self.model, "config", None)
                    if getattr(cfg, "image_resolution", None):
                        self._skip_manual_resize = True
                        print(f"[MyPolicy] モデルは内部で画像を自動リサイズする "
                              f"(image_resolution={cfg.image_resolution})ため、手動リサイズをスキップします。")
                    feats = getattr(cfg, "input_features", None) or getattr(cfg, "image_features", None)
                    if isinstance(feats, dict):
                        for key, feat in feats.items():
                            ftype = str(getattr(feat, "type", ""))
                            shape = getattr(feat, "shape", None)
                            if "VISUAL" not in ftype and not key.startswith("observation.images"):
                                continue
                            self._image_feature_keys.append(key)
                            if not self._skip_manual_resize and shape and len(shape) == 3:
                                h, w = shape[-2], shape[-1]
                                if h and w:
                                    self._image_target_hw = (int(h), int(w))
                except Exception:
                    pass
                print(f"[MyPolicy] 🎉 Policy model successfully loaded and ready for evaluation. "
                      f"(image_target_hw={self._image_target_hw}, image_keys={self._image_feature_keys})")

                # RTC (Real-Time Chunking, Physical Intelligence): チャンク境界での
                # 不連続なアクション変化を抑制し、非対象物体への1mm衝突リスクを下げる狙い。
                # pi0/pi0.5系（config.rtc_config フィールドを持つモデル）のみ有効化。
                # 再学習は不要 — 推論時にconfigへRTCConfigを注入するだけで効く。
                self._rtc_active = False
                self._model_full_chunk_size = self.chunk_size
                self._prev_full_chunk_tensor = None
                try:
                    if hasattr(self.model, "config") and hasattr(self.model.config, "rtc_config") \
                            and hasattr(self.model, "init_rtc_processor"):
                        from lerobot.policies.rtc.configuration_rtc import RTCConfig
                        # execution_horizonは実際の再計画間隔(self.chunk_size)に合わせる
                        # （既定値10のままだと、chunk_sizeを短縮した際にガイダンス窓が
                        # 実行区間より長くなり不整合になるため）。
                        # 2026-08-13修正: max_guidance_weight既定値10.0はRTC原論文のアブレーション
                        # （β=5を超えると利得なし、大きいほどjerky/OODになりやすい）を踏まえ5.0に縮小。
                        self.model.config.rtc_config = RTCConfig(
                            enabled=True, execution_horizon=self.chunk_size, max_guidance_weight=5.0
                        )
                        self.model.init_rtc_processor()
                        self._rtc_active = bool(self.model._rtc_enabled())
                        self._model_full_chunk_size = int(getattr(self.model.config, "chunk_size", self.chunk_size))
                        print(f"[MyPolicy] ✅ RTC (Real-Time Chunking) enabled "
                              f"(model_chunk_size={self._model_full_chunk_size}, exec_horizon={self.model.config.rtc_config.execution_horizon}).")
                except Exception as e:
                    print(f"[MyPolicy] RTC setup notice (running without RTC): {e}")
                    self._rtc_active = False

                # 重要: 学習済みモデルは正規化された入力・正規化された出力actionを前提に
                # 学習されている（model_weights/policy_preprocessor.json 等に統計量が保存
                # されている）。この pre/post-processor を使わずに生の値をそのまま
                # model.select_action() へ渡すと、入力スケールも出力スケールも訓練時と
                # 食い違ったまま推論することになり、精度が大きく劣化する
                # （公式 rollout 実装 lerobot/rollout/inference/sync.py と同じ手順に揃える）。
                try:
                    from lerobot.policies.factory import make_pre_post_processors
                    # 保存済み設定には学習時のデバイス('cuda'等)が焼き込まれているため、
                    # 実行時の実際のデバイス(CPU/GPU)で明示的に上書きする
                    # （公式実装 lerobot/rollout/context.py の preprocessor_overrides と同じ手順）。
                    self.preprocessor, self.postprocessor = make_pre_post_processors(
                        policy_cfg=self.model.config,
                        pretrained_path=target,
                        preprocessor_overrides={"device_processor": {"device": self.device}},
                        postprocessor_overrides={"device_processor": {"device": self.device}},
                    )
                    print("[MyPolicy] ✅ Pre/post-processor pipelines loaded (normalization + tokenizer).")
                except Exception as e:
                    print(f"[MyPolicy] Pre/post-processor loading notice: {e}. "
                          f"Falling back to raw (unnormalized) inference.")
                    self.preprocessor = None
                    self.postprocessor = None
            else:
                print("[MyPolicy] ⚠️ Critical Warning: All model loaders failed. Operating in smooth fallback mode.")

        except Exception as e:
            print(f"[MyPolicy] Critical Error in _load_model: {e}")
            self.model = None

    def reset(self, instruction: str = "") -> None:
        self.instruction = instruction
        self.action_queue = []
        self._prev_full_chunk_tensor = None  # RTC: 新エピソードでは前エピソードのleftoverを引き継がない
        self._preamble_active = True  # 新エピソード開始時は毎回グリッパー開放プリアンブルから
        self._preamble_steps = 0
        if self.model is not None and hasattr(self.model, "reset"):
            try:
                self.model.reset()
            except Exception as e:
                print(f"[MyPolicy] Model reset notice: {e}")
        for proc in (self.preprocessor, self.postprocessor):
            if proc is not None and hasattr(proc, "reset"):
                try:
                    proc.reset()
                except Exception as e:
                    print(f"[MyPolicy] Processor reset notice: {e}")
        print(f"[MyPolicy] Policy reset for task: '{instruction}'")

    def get_action(self, obs: dict) -> np.ndarray:
        if getattr(self, "_preamble_active", False):
            self._preamble_steps += 1
            gripper_qpos = obs.get("robot0_gripper_qpos")
            is_open = True
            if gripper_qpos is not None:
                try:
                    is_open = float(np.asarray(gripper_qpos).reshape(-1)[0]) >= self._preamble_open_threshold
                except Exception:
                    is_open = True
            if is_open or self._preamble_steps > self._preamble_max_steps:
                self._preamble_active = False
                self.action_queue = []  # プリアンブル中に溜まったチャンクは破棄し、以降は通常推論から
            else:
                action = np.zeros(7, dtype=np.float32)
                action[6] = -1.0  # グリッパー開放（-1=全開、+1=全閉。robosuite Panda gripper action convention）
                return action

        if self.action_queue:
            return self.action_queue.pop(0)

        # 非同期ロード中、またはモデル準備中は高速スムーズフォールバック（FastAPIをミリ秒で応答）
        if self.loading or self.model is None or not self.torch_available:
            new_chunk = [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32) for _ in range(self.chunk_size)]
        else:
            new_chunk = self._predict_action_chunk(obs)

        if len(new_chunk) > 1:
            self.action_queue = new_chunk[1:]
        return new_chunk[0]

    def _predict_action_chunk(self, obs: dict) -> list:
        if self.model is not None and self.torch_available:
            try:
                import torch
                
                front_img = None
                wrist_img = None
                robot_state = None
                eef_pos = None
                eef_quat = None
                gripper_qpos = None

                for k, v in obs.items():
                    if isinstance(v, np.ndarray):
                        if k in ["agentview_image", "front_image", "image", "observation.images.front"]:
                            front_img = torch.from_numpy(v)
                        elif k in ["robot0_eye_in_hand_image", "wrist_image", "image2", "observation.images.wrist"]:
                            wrist_img = torch.from_numpy(v)
                        elif k in ["state", "observation.state"]:
                            # 既に組み立て済みの state ベクトルがそのまま来る場合はそれを使う
                            robot_state = torch.from_numpy(v)
                        elif k == "robot0_eef_pos":
                            eef_pos = v
                        elif k == "robot0_eef_quat":
                            eef_quat = v
                        elif k == "robot0_gripper_qpos":
                            gripper_qpos = v

                if robot_state is None and (eef_pos is not None or eef_quat is not None or gripper_qpos is not None):
                    # 重要(2026-08-12発見): model_weights/config.json の observation.state は
                    # eef_pos(3)+eef_quat(4)+gripper_qpos(1)=8次元 ではなく、lerobot本体の
                    # LiberoProcessorStep(lerobot/processor/env_processor.py)が生成する
                    # [eef_pos(3), quat→axis-angle変換後(3), gripper_qpos(2)]=8次元 が正しい。
                    # robosuite の robot0_gripper_qpos は実際には2次元(指ごとの関節位置)であり、
                    # eef_quat(4次元、ノルム1の単位クォータニオン)をそのまま使うのは誤り
                    # （軌道逸脱の定量化でstep10から誤差が蓄積する主因の一つと判明）。
                    parts = []
                    if eef_pos is not None:
                        parts.append(np.asarray(eef_pos, dtype=np.float64).reshape(-1))
                    if eef_quat is not None:
                        # quat (x,y,z,w) -> axis-angle (3,)。lerobotのLiberoProcessorStepと同一の変換。
                        q = np.asarray(eef_quat, dtype=np.float64).reshape(-1)
                        w = np.clip(q[3], -1.0, 1.0)
                        den = np.sqrt(max(1.0 - w * w, 0.0))
                        if den > 1e-10:
                            angle = 2.0 * np.arccos(w)
                            axis = q[:3] / den
                            axis_angle = axis * angle
                        else:
                            axis_angle = np.zeros(3, dtype=np.float64)
                        parts.append(axis_angle)
                    if gripper_qpos is not None:
                        parts.append(np.asarray(gripper_qpos, dtype=np.float64).reshape(-1))
                    state_vec = np.concatenate(parts) if parts else np.zeros(8, dtype=np.float64)
                    if state_vec.shape[0] < 8:
                        state_vec = np.pad(state_vec, (0, 8 - state_vec.shape[0]))
                    else:
                        state_vec = state_vec[:8]
                    robot_state = torch.from_numpy(state_vec)

                if front_img is None:
                    front_img = torch.zeros((128, 128, 3), dtype=torch.uint8)
                if wrist_img is None:
                    wrist_img = torch.zeros((128, 128, 3), dtype=torch.uint8)
                if robot_state is None:
                    robot_state = torch.zeros((8,), dtype=torch.float32)

                if front_img.ndim == 3:
                    front_img = front_img.permute(2, 0, 1).unsqueeze(0)
                if front_img.dtype == torch.uint8:
                    front_img = front_img.float() / 255.0
                else:
                    front_img = front_img.float()

                if wrist_img.ndim == 3:
                    wrist_img = wrist_img.permute(2, 0, 1).unsqueeze(0)
                if wrist_img.dtype == torch.uint8:
                    wrist_img = wrist_img.float() / 255.0
                else:
                    wrist_img = wrist_img.float()

                # 重要(2026-08-12発見): lerobot本体のLiberoProcessorStep
                # (lerobot/processor/env_processor.py)は「HuggingFaceVLA/libero カメラ
                # 向きの慣習」としてH・W両方向に180度反転(torch.flip(dims=[2,3]))した
                # 画像を学習に使っている。この反転をしないまま推論すると、学習時と
                # 上下左右が逆の画像をモデルに見せ続けることになる。
                front_img = torch.flip(front_img, dims=[2, 3])
                wrist_img = torch.flip(wrist_img, dims=[2, 3])

                # 学習時は 256x256 (model_weights/config.json の image_features 参照) だが、
                # 本番/評価環境の観測は 128x128 で届く。リサイズせずに渡すと学習時と評価時で
                # 入力解像度が食い違ったまま推論することになり、精度が大きく劣化しうる。
                # 2026-08-13修正: _skip_manual_resizeが計算されるだけで参照されておらず、
                # pi0/pi0.5系(内部でresize_with_pad_torchにより224へ自動リサイズ)でも
                # ここで128→256への拡大が常に走っていた。結果、256→224への内部縮小と合わせて
                # 「拡大してから縮小」という学習時に存在しない二重リサイズ・ボケが生じていた。
                if not getattr(self, "_skip_manual_resize", False):
                    target_hw = self._image_target_hw
                    if front_img.shape[-2:] != target_hw:
                        front_img = torch.nn.functional.interpolate(
                            front_img, size=target_hw, mode="bilinear", align_corners=False
                        )
                    if wrist_img.shape[-2:] != target_hw:
                        wrist_img = torch.nn.functional.interpolate(
                            wrist_img, size=target_hw, mode="bilinear", align_corners=False
                        )

                if robot_state.ndim == 1:
                    robot_state = robot_state.unsqueeze(0)
                # 実シミュレータ観測 (robot0_joint_pos 等) は numpy float64 (Double) で
                # 送られてくるが、モデル重みは float32 のため明示キャストしないと
                # 「mat1 and mat2 must have the same dtype」で推論が全滅する。
                # (validate_submission.py のスモークテストは float32 ダミー観測を送るため
                # この不一致は実物理ロールアウトでしか顕在化しない — 手元と採点の乖離の一因)
                robot_state = robot_state.float()

                front_img = front_img.to(device=self.device)
                wrist_img = wrist_img.to(device=self.device)
                robot_state = robot_state.to(device=self.device)

                if self.preprocessor is not None and self.postprocessor is not None:
                    # 正規化/言語トークナイズ込みの正規ルート（学習時と同じ前処理・後処理）。
                    # 参照実装: lerobot/rollout/inference/sync.py の get_action() と同一の手順
                    # (preprocessor -> predict_action_chunk -> postprocessor)。
                    # カメラキー名は学習データセットによって異なる（front/wrist もあれば
                    # image/image2 もある）ため、モデルロード時に検出した実際のキー名を使う。
                    img_keys = self._image_feature_keys or ["observation.images.front", "observation.images.wrist"]
                    observation = {
                        img_keys[0]: front_img,
                        "observation.state": robot_state,
                        "task": self.instruction or "",
                        "robot_type": "",
                    }
                    if len(img_keys) > 1:
                        observation[img_keys[1]] = wrist_img
                    with torch.no_grad():
                        observation = self.preprocessor(observation)

                        rtc_kwargs = {}
                        if (
                            self._rtc_active
                            and self._prev_full_chunk_tensor is not None
                            and self._prev_full_chunk_tensor.shape[1] > self.chunk_size
                        ):
                            # 前回チャンクのうち未実行だった残り(leftover)をガイダンスとして渡し、
                            # チャンク境界での不連続なアクション変化を抑える。
                            # 2026-08-13修正(重大): inference_delay は RTC原論文の定義上「観測受領から
                            # 行動が使えるようになるまでの制御ステップ数」であり、本harnessは同期HTTP
                            # (/act の応答を待ってから環境を進める)のため d=0 が正しい。
                            # 誤って self.chunk_size(=10) を渡すと、get_prefix_weights が先頭10ステップを
                            # 前チャンク(1つ前の観測由来)に完全固定(ハードマスク)してしまい、常に
                            # 0.5秒(20Hzで10ステップ)分古い観測で動く準オープンループになっていた。
                            # 「対象物体に向かわずゴール地点にホバリングする」症状の主因と判明。
                            rtc_kwargs = {
                                "prev_chunk_left_over": self._prev_full_chunk_tensor[:, self.chunk_size :, :],
                                "inference_delay": 0,
                                "execution_horizon": self.model.config.rtc_config.execution_horizon,
                            }
                        raw_chunk_tensor = self.model.predict_action_chunk(observation, **rtc_kwargs)
                        if self._rtc_active:
                            self._prev_full_chunk_tensor = raw_chunk_tensor.detach().clone()
                        chunk_tensor = self.postprocessor(raw_chunk_tensor)
                    action_np = chunk_tensor.detach().cpu().numpy().squeeze(0)
                    return [self._sanitize_action(a) for a in action_np[: self.chunk_size]]

                # フォールバック: pre/post-processor が読み込めなかった場合のみ、
                # 正規化なしの生の値で推論する（精度は保証されない安全策）。
                batch = {
                    "observation.images.front": front_img,
                    "observation.images.wrist": wrist_img,
                    "observation.images.image": front_img,
                    "observation.images.image2": wrist_img,
                    "observation.image": front_img,
                    "observation.state": robot_state,
                    "observation.language.tokens": torch.zeros((1, 48), dtype=torch.long, device=self.device),
                    "observation.language.attention_mask": torch.ones((1, 48), dtype=torch.bool, device=self.device),
                }

                with torch.no_grad():
                    if hasattr(self.model, "select_action"):
                        action = self.model.select_action(batch)
                    elif hasattr(self.model, "forward"):
                        out = self.model(batch)
                        action = out.get("action", out) if isinstance(out, dict) else out
                    else:
                        action = self.model(batch)

                if isinstance(action, torch.Tensor):
                    action_np = action.detach().cpu().numpy().squeeze()
                else:
                    action_np = np.array(action, dtype=np.float32).squeeze()

                if action_np.ndim == 1 and action_np.shape[0] == 7:
                    chunk = [self._sanitize_action(action_np)]
                    for _ in range(self.chunk_size - 1):
                        chunk.append(self._sanitize_action(action_np))
                    return chunk
                elif action_np.ndim == 2 and action_np.shape[1] == 7:
                    return [self._sanitize_action(a) for a in action_np[:self.chunk_size]]
            except Exception as e:
                print(f"[MyPolicy] Model inference error: {e}. Falling back to default action.")

        # Default fallback (smooth delta zero action with active gripper)
        chunk = []
        for _ in range(self.chunk_size):
            action = np.zeros(7, dtype=np.float32)
            action[6] = 1.0
            chunk.append(action)
        return chunk

    def _sanitize_action(self, action: np.ndarray) -> np.ndarray:
        act = np.nan_to_num(action.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        act[:6] = np.clip(act[:6], -1.0, 1.0)
        return act

def deserialize_obs(body: bytes) -> dict:
    if not body:
        return {}
    unpacker = msgpack.Unpacker(raw=False)
    unpacker.feed(body)
    raw_obs = unpacker.unpack()

    obs = {}
    for k, v in raw_obs.items():
        if isinstance(v, dict) and "data" in v and "shape" in v:
            dtype = v.get("dtype", "float32")
            arr = np.frombuffer(v["data"], dtype=dtype).reshape(v["shape"])
            obs[k] = arr
        else:
            obs[k] = v
    return obs

def serialize_action(action: np.ndarray) -> bytes:
    action_data = {
        "data": action.astype(np.float32).tobytes(),
        "shape": list(action.shape),
        "dtype": "float32",
    }
    return msgpack.packb(action_data, use_bin_type=True)

app = FastAPI(title="PARC2026 Submission Policy Server")
_policy: Optional[BasePolicy] = None

def set_policy(policy: BasePolicy):
    global _policy
    _policy = policy

@app.get("/health")
def health():
    # RULES.md 仕様: 評価側は /health が 200 を返すまでポーリングし続ける
    # （SERVER_TIMEOUT=120秒が上限）。モデルの非同期ロードが完了する前に 200 を
    # 返してしまうと、評価側が /reset・/act を先に呼び始め、ロード未完了の間は
    # フォールバック(ゼロ)action しか返せず「手元は動くが採点は低スコア」の原因に
    # なるため、ロード完了（成功/失敗いずれか確定）まで 503 を返す。
    if _policy is not None and getattr(_policy, "loading", False):
        return Response(status_code=503, content=b'{"status":"loading"}', media_type="application/json")
    return {"status": "ok"}

@app.post("/reset")
async def reset_policy(request: Request):
    body = await request.body()
    instruction = ""
    if body:
        try:
            data = json.loads(body)
            instruction = data.get("instruction", "")
        except Exception:
            pass
    if _policy:
        _policy.reset(instruction=instruction)
    return {"status": "ok"}

@app.post("/act")
async def act(request: Request):
    body = await request.body()
    obs = deserialize_obs(body)
    if _policy:
        action = _policy.get_action(obs)
    else:
        action = np.zeros(7, dtype=np.float32)
        action[6] = 1.0
    return Response(
        content=serialize_action(action),
        media_type="application/x-msgpack",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    port = int(os.environ.get("PORT", args.port))
    set_policy(MyPolicy())
    print(f"Policy server starting on {args.host}:{port}")
    uvicorn.run(app, host=args.host, port=port, log_level="info")
