#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 dev 集上拟合 isotonic 校准，并保存运行时可直接插值的概率映射。

为什么需要
----------
分类器输出的是 logit/softmax 原始分数，不是「AI 概率」。同一分数在不同领域
含义不同；直接展示百分比会把 0.53 这类原始分当成 53% 风险，误导用户。

做法
----
- dev 上拟合 IsotonicRegression（单调、无参数假设）
- 取 201 个分位点压成运行时插值表，避免 JSON 膨胀
- 在 dev 上按 FPR=1% 定阈值，再在 test 上报告 TPR

用法
----
python training/calibrate.py --model-dir models/distilmbert-aigc
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import predict, threshold_at_fpr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def ece(y_true: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not mask.any():
            continue
        total += abs(p[mask].mean() - y_true[mask].mean()) * mask.mean()
    return float(total)


def main() -> int:
    ap = argparse.ArgumentParser(description="拟合并保存概率校准表")
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "distilmbert-aigc"))
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--output", default=str(ROOT / "models" / "calibration.json"))
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--knots", type=int, default=201)
    ap.add_argument("--include-raid", action="store_true",
                    help="保留 RAID 扰动样本参与校准（默认剔除：攻击改写会压平高分尾）")
    ap.add_argument("--method", choices=["auto", "isotonic", "platt"], default="auto",
                    help="auto=按 dev ECE 选择；platt 平滑单调，避免高分尾部平台")
    args = ap.parse_args()

    ds = load_dataset("json", data_files={
        "dev": str(Path(args.data_dir) / "dev.jsonl"),
        "test": str(Path(args.data_dir) / "test.jsonl"),
    })
    dev_rows, test_rows = list(ds["dev"]), list(ds["test"])

    if not args.include_raid:
        before = len(dev_rows)
        dev_rows = [r for r in dev_rows if r.get("source") != "raid"]
        print(f"[+] 校准集剔除 RAID {before - len(dev_rows)} 条（避免高分尾被扰动样本压平）",
              file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == "cuda":
        model.half()

    dev_p = predict(model, tokenizer, [r["text"] for r in dev_rows],
                    args.max_length, args.batch_size, device)
    test_p = predict(model, tokenizer, [r["text"] for r in test_rows],
                     args.max_length, args.batch_size, device)
    dev_y = np.array([r["label"] for r in dev_rows])
    test_y = np.array([r["label"] for r in test_rows])

    eps = 1e-6

    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, eps, 1.0 - eps)
        return np.log(p / (1.0 - p))

    iso = IsotonicRegression(out_of_bounds="clip").fit(dev_p, dev_y)
    platt = LogisticRegression(solver="lbfgs").fit(
        _logit(dev_p).reshape(-1, 1), dev_y)

    def iso_apply(p: np.ndarray) -> np.ndarray:
        return iso.predict(p)

    def platt_apply(p: np.ndarray) -> np.ndarray:
        return platt.predict_proba(_logit(p).reshape(-1, 1))[:, 1]

    # isotonic 在高分区会被少量「正式人类文体」压成平台，原始 0.99 可能映射到 0.6；
    # Platt（logistic）平滑单调，用 dev ECE 二选一，避免高分尾塌陷。
    ece_iso_dev = ece(dev_y, iso_apply(dev_p))
    ece_platt_dev = ece(dev_y, platt_apply(dev_p))
    if args.method == "auto":
        method = "platt" if ece_platt_dev <= ece_iso_dev else "isotonic"
    else:
        method = args.method
    print(f"[+] dev ECE isotonic={ece_iso_dev:.4f} platt={ece_platt_dev:.4f} -> {method}",
          file=sys.stderr)

    grid = np.unique(np.concatenate([
        np.quantile(dev_p, np.linspace(0.0, 1.0, args.knots)),
        np.array([0.0, 1.0]),
    ]))
    values = platt_apply(grid) if method == "platt" else iso_apply(grid)

    dev_cal = np.interp(dev_p, grid, values)
    test_cal = np.interp(test_p, grid, values)

    threshold = threshold_at_fpr(dev_y, dev_cal, 0.01)
    fpr, tpr, _ = roc_curve(test_y, test_cal)
    mask = fpr <= 0.01

    report = {
        "method": method,
        "x": [round(float(v), 6) for v in grid],
        "y": [round(float(v), 6) for v in values],
        "threshold_fpr1": round(float(threshold), 6),
        "dev": {
            "auroc_raw": round(float(roc_auc_score(dev_y, dev_p)), 4),
            "auroc_cal": round(float(roc_auc_score(dev_y, dev_cal)), 4),
            "ece_raw": round(ece(dev_y, dev_p), 4),
            "ece_cal": round(ece(dev_y, dev_cal), 4),
        },
        "test": {
            "auroc": round(float(roc_auc_score(test_y, test_cal)), 4),
            "tpr@1fpr": round(float(tpr[mask].max()) if mask.any() else 0.0, 4),
            "ece_raw": round(ece(test_y, test_p), 4),
            "ece_cal": round(ece(test_y, test_cal), 4),
        },
        "alternatives": {
            "isotonic": {
                "dev_ece": round(float(ece_iso_dev), 4),
                "test_ece": round(float(ece(test_y, iso_apply(test_p))), 4),
            },
            "platt": {
                "dev_ece": round(float(ece_platt_dev), 4),
                "test_ece": round(float(ece(test_y, platt_apply(test_p))), 4),
            },
        },
    }

    def fit_bundle(dp, dy, tp, ty, method_pref):
        """在给定子集上拟合并返回可直接运行的插值表。"""
        eps2 = 1e-6

        def lg(p):
            p = np.clip(p, eps2, 1.0 - eps2)
            return np.log(p / (1.0 - p))

        iso2 = IsotonicRegression(out_of_bounds="clip").fit(dp, dy)
        pl2 = LogisticRegression(solver="lbfgs").fit(lg(dp).reshape(-1, 1), dy)

        def f_iso(p):
            return iso2.predict(p)

        def f_platt(p):
            return pl2.predict_proba(lg(p).reshape(-1, 1))[:, 1]

        # 等频分箱校准：箱内 AI 占比，再对箱值做单调化。
        # 直接 isotonic 在「大量并列分数」的窄区间会被局部平均压塌（0.99 → 0.42）；
        # 分箱保留校准含义，又不会破坏高分端的置信度。
        nb = 40
        edges = np.unique(np.quantile(dp, np.linspace(0.0, 1.0, nb + 1)))
        bin_idx = np.clip(np.searchsorted(edges, dp, side="right") - 1,
                          0, len(edges) - 2)
        centers, rates, weights = [], [], []
        for b in range(len(edges) - 1):
            mask = bin_idx == b
            if not mask.any():
                continue
            centers.append(float(np.median(dp[mask])))
            rates.append(float(dy[mask].mean()))
            weights.append(int(mask.sum()))
        centers_arr = np.asarray(centers)
        rates_mono = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
            centers_arr, np.asarray(rates), sample_weight=np.asarray(weights)
        ).predict(centers_arr)

        def f_binned(p):
            pos = np.clip(np.searchsorted(edges, p, side="right") - 1,
                          0, len(centers_arr) - 1)
            return rates_mono[pos]

        if method_pref == "binned":
            m2, f2 = "binned", f_binned
        elif method_pref == "auto":
            m2 = "platt" if ece(dy, f_platt(dp)) <= ece(dy, f_iso(dp)) else "isotonic"
            f2 = f_platt if m2 == "platt" else f_iso
        else:
            m2 = method_pref
            f2 = f_platt if m2 == "platt" else f_iso

        g2 = np.unique(np.concatenate([
            np.quantile(dp, np.linspace(0.0, 1.0, args.knots)),
            np.array([0.0, 1.0]),
        ]))
        v2 = f2(g2)
        dcal2 = np.interp(dp, g2, v2)
        tcal2 = np.interp(tp, g2, v2)
        if m2 == "ppv":
            thr_raw = threshold_at_fpr(dy, dp, 0.01)
            thr2 = float(f2(np.array([thr_raw]))[0])
        else:
            thr2 = threshold_at_fpr(dy, dcal2, 0.01)
        fpr2, tpr2, _ = roc_curve(ty, tcal2)
        mk2 = fpr2 <= 0.01
        return {
            "method": m2,
            "x": [round(float(v), 6) for v in g2],
            "y": [round(float(v), 6) for v in v2],
            "threshold_fpr1": round(float(thr2), 6),
            "n_dev": int(len(dy)),
            "dev": {
                "auroc_cal": round(float(roc_auc_score(dy, dcal2)), 4),
                "ece_cal": round(ece(dy, dcal2), 4),
            },
            "test": {
                "auroc": round(float(roc_auc_score(ty, tcal2)), 4),
                "tpr@1fpr": round(float(tpr2[mk2].max()) if mk2.any() else 0.0, 4),
                "ece_cal": round(ece(ty, tcal2), 4),
            },
        }

    # 语言专用校准：英文正式文体（MAGE）会把全局高分尾拉低，中文必须单独拟合
    dev_lang = np.array([r["lang"] for r in dev_rows])
    test_lang = np.array([r["lang"] for r in test_rows])
    for name in ("zh", "en"):
        dm = (dev_lang == "zh") if name == "zh" else (dev_lang != "zh")
        tm = (test_lang == "zh") if name == "zh" else (test_lang != "zh")
        if int(dm.sum()) < 100 or int(tm.sum()) < 50:
            continue
        # 语言曲线统一用分箱校准：避免 isotonic 在并列分数窄区间被压塌。
        bundle = fit_bundle(dev_p[dm], dev_y[dm], test_p[tm], test_y[tm], "binned")
        out_lang = Path(args.output).with_name(f"calibration_{name}.json")
        with open(out_lang, "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False, indent=2)
        print(f"[+] {out_lang} n_dev={bundle['n_dev']} "
              f"thr={bundle['threshold_fpr1']} ece={bundle['test']['ece_cal']}",
              file=sys.stderr)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[+] 已保存: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
