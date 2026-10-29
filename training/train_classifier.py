#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""微调跨语言编码器做 AI 文本二分类（human vs machine）。

默认 XLM-R base：中英双语覆盖，fp16/bf16 在 8GB 显存下用
batch 4 + 梯度累积 4 + 梯度检查点可跑。

用法
----
python training/prepare_data.py
python training/train_classifier.py --model FacebookAI/xlm-roberta-base

显存不足时：
  --batch-size 2 --grad-accum 8 --max-length 192
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
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= max_fpr
    return float(tpr[mask].max()) if mask.any() else 0.0


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auroc": float(roc_auc_score(labels, probs)),
        "tpr@1fpr": tpr_at_fpr(labels, probs),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="微调 AI 文本检测分类器")
    ap.add_argument("--model", default="distilbert-base-multilingual-cased")
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--output-dir", default=str(ROOT / "models" / "distilmbert-aigc"))
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train", type=int, default=50000,
                    help="0 = 全部；默认随机抽 5 万条，兼顾覆盖与训练时长")
    ap.add_argument("--max-eval", type=int, default=12000,
                    help="0 = 全部 dev；默认随机抽 1.2 万条，控制每轮评估耗时")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = {s: str(data_dir / f"{s}.jsonl") for s in ("train", "dev", "test")}
    for split, path in files.items():
        if not os.path.exists(path):
            print(f"[!] 缺少 {path}，先运行 training/prepare_data.py", file=sys.stderr)
            return 1

    raw = load_dataset("json", data_files=files)
    if args.max_train and len(raw["train"]) > args.max_train:
        raw["train"] = raw["train"].shuffle(seed=args.seed).select(range(args.max_train))
    if args.max_eval and len(raw["dev"]) > args.max_eval:
        raw["dev"] = raw["dev"].shuffle(seed=args.seed).select(range(args.max_eval))

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    encoded = raw.map(
        tokenize, batched=True, remove_columns=["text"],
        desc="tokenize",
    )
    keep = {"input_ids", "attention_mask", "token_type_ids", "label"}
    encoded = encoded.remove_columns([c for c in encoded["train"].column_names if c not in keep])

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    use_cuda = torch.cuda.is_available()
    bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())
    fp16 = bool(use_cuda and not bf16)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        warmup_steps=300,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=use_cuda,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_cuda else None,
        dataloader_num_workers=0,
        disable_tqdm=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="auroc",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["dev"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate(encoded["test"])
    print("[+] test:", metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    report = {
        "base_model": args.model,
        "max_length": args.max_length,
        "test_metrics": {k: v for k, v in metrics.items() if not k.startswith("eval_")},
    }
    with open(os.path.join(args.output_dir, "training_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[+] 已保存到 {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
