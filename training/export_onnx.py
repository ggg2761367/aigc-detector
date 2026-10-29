#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把微调后的分类器导出为 ONNX，并做 int8 动态量化（离线 CPU 部署用）。

用法
----
python training/export_onnx.py --model-dir models/xlmr-aigc

输出
----
models/xlmr-aigc-onnx/
  model.onnx          fp32
  model_int8.onnx     int8（MatMul 动态量化，CPU 首选）
  tokenizer.json 等
  onnx_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def main() -> int:
    ap = argparse.ArgumentParser(description="导出并量化 AI 文本检测模型")
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "distilmbert-aigc"))
    ap.add_argument("--output-dir", default=None,
                    help="默认 <model-dir>-onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--skip-quantize", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir) if args.output_dir else \
        model_dir.parent / (model_dir.name + "-onnx")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()

    dummy = tokenizer("hello world", return_tensors="pt",
                      truncation=True, max_length=32)
    wrapper = LogitsOnly(model).eval()

    fp32_path = out_dir / "model.onnx"
    print("[+] 导出 fp32 ONNX ...")
    torch.onnx.export(
        wrapper,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    # tokenizer 与配置文件
    for name in ("tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "sentencepiece.bpe.model"):
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    quantized = None
    if not args.skip_quantize:
        print("[+] int8 动态量化 ...")
        import tempfile
        from onnxruntime.quantization import QuantType, quantize_dynamic
        int8_path = out_dir / "model_int8.onnx"
        if str(fp32_path).isascii() and str(int8_path).isascii():
            quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
        else:
            # onnxruntime 在 Windows 非 ASCII 路径下写中间文件会失败，走临时目录
            with tempfile.TemporaryDirectory() as td:
                tmp_in = Path(td) / "model.onnx"
                tmp_out = Path(td) / "model_int8.onnx"
                shutil.copy2(fp32_path, tmp_in)
                quantize_dynamic(str(tmp_in), str(tmp_out), weight_type=QuantType.QInt8)
                shutil.copy2(tmp_out, int8_path)
        quantized = int8_path.name

    manifest = {
        "base_model": str(model_dir),
        "max_length": 256,
        "labels": ["human", "machine"],
        "fp32": fp32_path.name,
        "quantized": quantized,
    }
    with open(out_dir / "onnx_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[+] 输出目录: {out_dir}")
    print(f"    fp32: {fp32_path.stat().st_size / 1e6:.1f} MB")
    if quantized:
        print(f"    int8: {(out_dir / quantized).stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
