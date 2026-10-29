# -*- coding: utf-8 -*-
"""离线 CPU 推理引擎：ONNX int8 分类器，可选长文本分块平均。

设计原则
--------
- 默认不依赖任何第三方包：没装 onnxruntime / tokenizers，或模型文件不存在时，
  `get_engine()` 返回 None，调用方回退到 detector.py 的零依赖引擎。
- 模型只加载一次；线程安全。
- 概率是原始分类器输出，校准放在调用方（training/evaluate 里定阈值）。

用法
----
from model_engine import get_engine
eng = get_engine()
if eng:
    p = eng.predict_proba(text)
"""

from __future__ import annotations

import bisect
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

if getattr(sys, "frozen", False):
    BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
else:
    BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "distilmbert-aigc-onnx"

_ENGINE: Optional["OnnxEngine"] = None
_LOCK = threading.Lock()


class OnnxEngine:
    def __init__(self, model_dir: Path):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_dir = Path(model_dir)
        manifest_path = self.model_dir / "onnx_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)

        with open(manifest_path, encoding="utf-8") as f:
            self.manifest: Dict[str, Any] = json.load(f)

        name = self.manifest.get("quantized") or self.manifest.get("fp32")
        model_path = self.model_dir / name
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        tokenizer_path = self.model_dir / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, min(8, os.cpu_count() or 4))
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.max_length = int(self.manifest.get("max_length", 256))
        # 长文窗口上限：64 × 256 token ≈ 1.6 万 token，覆盖学位论文正文量级
        self.max_windows = max(1, int(self.manifest.get("max_windows", 64)))
        self.cls_id = self.tokenizer.token_to_id("[CLS]")
        self.sep_id = self.tokenizer.token_to_id("[SEP]")

        # 可选校准表：优先按语言（中文正式写作与英文正式文体的分数分布不同）
        def load_table(path: Path) -> Optional[Dict[str, Any]]:
            if not path.exists():
                return None
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if "x" not in data or "y" not in data:
                return None
            return {
                "x": [float(v) for v in data["x"]],
                "y": [float(v) for v in data["y"]],
                "threshold": data.get("threshold_fpr1"),
            }

        self.calibrations: Dict[str, Dict[str, Any]] = {}
        for name in ("zh", "en"):
            for cand in (self.model_dir / f"calibration_{name}.json",
                         self.model_dir.parent / f"calibration_{name}.json"):
                table = load_table(cand)
                if table:
                    self.calibrations[name] = table
                    break

        # 通用回退表
        self.calibration: Optional[Dict[str, Any]] = None
        for cand in (self.model_dir / "calibration.json",
                     self.model_dir.parent / "calibration.json"):
            self.calibration = load_table(cand)
            if self.calibration:
                break

    # ------------------------------------------------------------------
    def _windows(self, text: str) -> List[List[int]]:
        enc = self.tokenizer.encode(text)
        ids = enc.ids
        if len(ids) <= self.max_length:
            return [ids]

        # 去掉首尾 special tokens，再按窗口切；长文最多取 8 块，避免延迟失控
        body = ids
        if body and body[0] == self.cls_id:
            body = body[1:]
        if body and body[-1] == self.sep_id:
            body = body[:-1]

        inner = max(self.max_length - 2, 32)
        total = len(body)
        if total <= inner:
            return [[self.cls_id] + body + [self.sep_id]]

        # 窗口覆盖全文；超出上限时改为等距抽样，而不是只取开头
        stride = max(inner // 4, 1)
        step = inner - stride
        starts = list(range(0, total - inner + 1, step))
        if not starts or starts[-1] != total - inner:
            starts.append(total - inner)
        if len(starts) > self.max_windows:
            last = total - inner
            if self.max_windows == 1:
                starts = [0]
            else:
                starts = [round(last * k / (self.max_windows - 1))
                          for k in range(self.max_windows)]

        chunks: List[List[int]] = []
        for start in starts:
            chunks.append([self.cls_id] + body[start:start + inner] + [self.sep_id])
        return chunks

    def predict_proba(self, text: str) -> float:
        import numpy as np

        windows = self._windows(text or "")
        if not windows:
            return 0.5

        probs: List[float] = []
        for ids in windows:
            feed = {"input_ids": np.array([ids], dtype=np.int64)}
            if "attention_mask" in self.input_names:
                feed["attention_mask"] = np.ones((1, len(ids)), dtype=np.int64)
            logits = self.session.run(None, feed)[0][0]
            # 数值稳定 softmax
            m = float(max(logits))
            exps = np.exp(logits - m)
            probs.append(float(exps[1] / exps.sum()))

        return sum(probs) / len(probs)

    def _pick_calibration(self, text: str):
        """按文本语言选校准曲线：中文占比 ≥15% 用中文表，否则英文表。"""
        if self.calibrations:
            cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
            lang = "zh" if cjk / max(len(text), 1) >= 0.15 else "en"
            table = self.calibrations.get(lang)
            if table:
                return table, lang
        return self.calibration, None

    def _calibrate(self, p: float,
                   table: Optional[Dict[str, Any]] = None) -> float:
        table = table or self.calibration
        if not table:
            return p
        xs, ys = table["x"], table["y"]
        if p <= xs[0]:
            return ys[0]
        if p >= xs[-1]:
            return ys[-1]
        i = bisect.bisect_left(xs, p)
        x0, x1 = xs[i - 1], xs[i]
        y0, y1 = ys[i - 1], ys[i]
        t = (p - x0) / (x1 - x0) if x1 > x0 else 0.0
        return y0 + (y1 - y0) * t

    def predict(self, text: str) -> Dict[str, Any]:
        """返回原始分数与校准后概率（有条形校准表时）。"""
        table, lang = self._pick_calibration(text)
        raw = self.predict_proba(text)
        cal = self._calibrate(raw, table)
        out: Dict[str, Any] = {
            "raw_probability": round(raw, 4),
            "probability": round(cal, 4),
            "percent": round(cal * 100.0, 1),
        }
        if lang:
            out["calibration_lang"] = lang
        if table and table.get("threshold") is not None:
            out["threshold"] = round(float(table["threshold"]), 4)
            out["calibrated"] = True
        return out

    def info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "engine": "onnx-int8",
            "model": self.manifest.get("base_model", ""),
            "max_length": self.max_length,
            "calibrated": bool(self.calibrations or self.calibration),
        }
        if self.calibrations:
            info["calibration_langs"] = sorted(self.calibrations)
            info["thresholds"] = {
                k: round(float(v["threshold"]), 4)
                for k, v in self.calibrations.items()
                if v.get("threshold") is not None
            }
        if self.calibration and self.calibration.get("threshold") is not None:
            info["threshold"] = round(float(self.calibration["threshold"]), 4)
        return info


def get_engine(model_dir: Optional[str] = None) -> Optional[OnnxEngine]:
    """加载并缓存引擎；任何依赖/文件缺失都返回 None（调用方回退）。"""
    global _ENGINE
    path = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    if _ENGINE is not None and Path(getattr(_ENGINE, "model_dir", "")) == path:
        return _ENGINE

    with _LOCK:
        if _ENGINE is not None and Path(getattr(_ENGINE, "model_dir", "")) == path:
            return _ENGINE
        try:
            _ENGINE = OnnxEngine(path)
        except Exception:
            _ENGINE = None
    return _ENGINE


def available(model_dir: Optional[str] = None) -> bool:
    return get_engine(model_dir) is not None


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    eng = get_engine()
    if not eng:
        print("模型不可用（缺 onnxruntime/tokenizers 或模型文件）")
        raise SystemExit(1)
    print(eng.info())
    for sample in (
        "首先，人工智能技术在教育领域的应用具有重要意义。其次，它显著提升了效率。",
        "昨天下午我在楼下便利店买了瓶水，收银的阿姨正在看一部很老的家庭剧。",
    ):
        print(round(eng.predict_proba(sample), 4), sample[:24])
