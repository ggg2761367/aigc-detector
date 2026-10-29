#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在留出集上评测分类器，并按 source / lang / domain 拆开报告。

阈值取 dev 集 FPR=1% 的分位点，再在原样 test 上报告 TPR —— 这是
AI 检测器真正有意义的指标（低误报下能抓到多少）。

用法
----
python training/evaluate.py --model-dir models/xlmr-aigc
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def predict(model, tokenizer, texts: List[str], max_length: int, batch_size: int,
            device: torch.device) -> np.ndarray:
    probs: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(batch, truncation=True, max_length=max_length,
                            padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs.append(torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros(0)


def threshold_at_fpr(y_true: np.ndarray, y_score: np.ndarray,
                     fpr_target: float = 0.01) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    valid = np.where(fpr <= fpr_target)[0]
    if len(valid) == 0:
        return 0.5
    best = valid[np.argmax(tpr[valid])]
    return float(thresholds[best])


def metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    if len(y_true) == 0 or len(set(y_true.tolist())) < 2:
        return {"n": int(len(y_true))}
    preds = (y_score >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= 0.01
    return {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "tpr@1fpr": float(tpr[mask].max()) if mask.any() else 0.0,
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }


def grouped_report(rows: List[dict], probs: np.ndarray, threshold: float,
                   key: str) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        buckets[str(r.get(key, "?"))].append(i)
    out = {}
    for name, idx in sorted(buckets.items()):
        if len(idx) < 30:
            continue
        y = np.array([rows[i]["label"] for i in idx])
        p = probs[idx]
        out[name] = metrics(y, p, threshold)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="评测 AI 文本检测分类器")
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "xlmr-aigc"))
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--output", default=str(ROOT / "models" / "eval_report.json"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    ds = load_dataset("json", data_files={
        "dev": str(data_dir / "dev.jsonl"),
        "test": str(data_dir / "test.jsonl"),
    })

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == "cuda":
        model.half()

    dev_rows = list(ds["dev"])
    test_rows = list(ds["test"])

    dev_probs = predict(model, tokenizer, [r["text"] for r in dev_rows],
                        args.max_length, args.batch_size, device)
    test_probs = predict(model, tokenizer, [r["text"] for r in test_rows],
                         args.max_length, args.batch_size, device)

    dev_y = np.array([r["label"] for r in dev_rows])
    test_y = np.array([r["label"] for r in test_rows])

    threshold = threshold_at_fpr(dev_y, dev_probs, 0.01)

    report = {
        "model_dir": args.model_dir,
        "threshold_dev_fpr1": threshold,
        "overall": metrics(test_y, test_probs, threshold),
        "by_source": grouped_report(test_rows, test_probs, threshold, "source"),
        "by_lang": grouped_report(test_rows, test_probs, threshold, "lang"),
        "by_domain": grouped_report(test_rows, test_probs, threshold, "domain"),
        "by_generator": grouped_report(test_rows, test_probs, threshold, "generator"),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    print(f"[+] 报告: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
