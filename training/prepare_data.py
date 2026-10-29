#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把公开检测数据集统一成一份 JSONL 语料。

来源
----
- Hello-SimpleAI/HC3          中英问答，human_answers vs chatgpt_answers
- Hello-SimpleAI/HC3-Chinese  中文问答（同上结构）
- FreedomIntelligence/ChatGPT-Detection-PR-HPPT
                              中文学术摘要：人工原文 vs ChatGPT 润色 / 生成
- yaful/MAGE                  英文多领域，human vs 多个 LLM；含 OOD 改写测试集

输出
----
data/processed/{train,dev,test}.jsonl
每行:
  {"id": "...", "text": "...", "label": 0|1, "lang": "zh|en",
   "domain": "...", "source": "...", "generator": "...",
   "split": "train|dev|test", "group": "..."}

label: 0 = 人类, 1 = 机器参与（生成或润色）。
group 用于防止同一问题/标题的样本跨 split 泄漏。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import hf_hub_download  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed"

MIN_CHARS = 40


def clean(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def split_by_hash(key: str, dev: float = 0.05, test: float = 0.05) -> str:
    """确定性按 group 划分，保证同一问题的不同回答落在同一 split。"""
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < test:
        return "test"
    if bucket < test + dev:
        return "dev"
    return "train"


def row(text: str, label: int, lang: str, domain: str, source: str,
        generator: str, split: str, group: str) -> Dict:
    text = clean(text)
    return {
        "id": digest(f"{source}|{generator}|{text}")[:20],
        "text": text,
        "label": int(label),
        "lang": lang,
        "domain": domain,
        "source": source,
        "generator": generator,
        "split": split,
        "group": digest(group)[:16],
    }


# --------------------------------------------------------------------------
# HC3
# --------------------------------------------------------------------------

def load_hc3(repo: str, lang: str) -> List[Dict]:
    path = hf_hub_download(repo, "all.jsonl", repo_type="dataset")
    rows: List[Dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            question = d.get("question") or d.get("id") or ""
            domain = d.get("source") or "qa"
            group = f"hc3-{lang}-{question}"
            split = split_by_hash(group)
            for answer in d.get("human_answers") or []:
                text = clean(answer)
                if len(text) >= MIN_CHARS:
                    rows.append(row(text, 0, lang, domain, "hc3", "human", split, group))
            for answer in d.get("chatgpt_answers") or []:
                text = clean(answer)
                if len(text) >= MIN_CHARS:
                    rows.append(row(text, 1, lang, domain, "hc3", "chatgpt", split, group))
    return rows


# --------------------------------------------------------------------------
# PR-HPPT（中文学术摘要）
# --------------------------------------------------------------------------

_PRHPPT_FILES = {
    "train": "Chinese_data/polished_abstract/train.json",
    "dev": "Chinese_data/polished_abstract/val.json",
    "test": "Chinese_data/polished_abstract/test.json",
}


def load_prhppt() -> List[Dict]:
    repo = "FreedomIntelligence/ChatGPT-Detection-PR-HPPT"
    rows: List[Dict] = []
    for split, filename in _PRHPPT_FILES.items():
        path = hf_hub_download(repo, filename, repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            text = clean(d.get("text", ""))
            if len(text) < 80:
                continue
            label = int(d.get("label", 0))
            generator = "human" if label == 0 else "chatgpt-polish"
            group = f"prhppt-{d.get('title', text[:40])}"
            rows.append(row(text, label, "zh", "academic", "pr-hppt",
                            generator, split, group))

    # 全生成摘要：作为额外的 AI 样本，按 hash 划入 dev/test
    # 注意：仓库里这个 .jsonl 实际是一个 JSON 数组，不是逐行 JSON
    path = hf_hub_download(
        repo, "Chinese_data/generated_abstract/generated_abstract.jsonl",
        repo_type="dataset",
    )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data", [])
    for d in data:
        text = clean(d.get("text", ""))
        if len(text) < 80:
            continue
        split = split_by_hash(text, dev=0.02, test=0.08)
        rows.append(row(text, 1, "zh", "academic-gen", "pr-hppt",
                        "chatgpt-generated", split, text[:64]))
    return rows


# --------------------------------------------------------------------------
# MAGE（英文多领域）
# --------------------------------------------------------------------------

_MAGE_FILES = (
    ("train", "train.csv", "mage"),
    ("dev", "valid.csv", "mage"),
    ("test", "test.csv", "mage"),
    ("test", "test_ood_set_gpt.csv", "ood-gpt"),
    ("test", "test_ood_set_gpt_para.csv", "ood-gpt-para"),
)


def load_mage() -> List[Dict]:
    rows: List[Dict] = []
    for split, filename, domain in _MAGE_FILES:
        path = hf_hub_download("yaful/MAGE", filename, repo_type="dataset")
        with open(path, encoding="utf-8-sig", newline="") as f:
            for d in csv.DictReader(f):
                text = clean(d.get("text", ""))
                if len(text) < MIN_CHARS:
                    continue
                raw_label = int(d.get("label", 0))
                # MAGE 实际编码：1 = human，0 = machine（已按 *_human 源核对）。
                label = 0 if raw_label == 1 else 1
                generator = clean(d.get("src", "")) or "unknown"
                rows.append(row(text, label, "en", domain, "mage",
                                generator if label == 1 else "human",
                                split, text[:96]))
    return rows


# --------------------------------------------------------------------------
# RAID extra（英文，多生成器 + 对抗改写）
# --------------------------------------------------------------------------

_RAID_LANG = {"german": "de", "czech": "cs"}


def _reservoir_add(bucket: List[Dict], item: Dict, cap: int, seen: int,
                   rng: random.Random) -> None:
    """均匀蓄水池抽样：文件按 domain/source 排序，顺序截断会只拿到前几个 domain。"""
    if len(bucket) < cap:
        bucket.append(item)
        return
    j = rng.randrange(seen)
    if j < cap:
        bucket[j] = item


def load_raid_extra(path: Path, max_rows: int, seed: int) -> List[Dict]:
    """extra.csv 没有 label 列：model == "human" 即人类，其余为机器。

    人类侧蓄水池抽样 max_rows 后取一半；机器侧按 (model, attack) 分层蓄水池，
    保证攻击/生成器组合都有代表。两半分别打乱，避免文件排序带来的 domain 偏置。
    """
    if not path.exists():
        print(f"[!] 跳过 RAID：{path} 不存在", file=sys.stderr)
        return []

    rng = random.Random(seed)
    humans: List[Dict] = []
    human_seen = 0
    machines: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    machine_seen: Dict[Tuple[str, str], int] = defaultdict(int)
    machine_cap = max(100, max_rows // 150)
    domain_count: Counter = Counter()

    with open(path, encoding="utf-8-sig", newline="") as f:
        for d in csv.DictReader(f):
            text = clean(d.get("generation", ""))
            if len(text) < 80:
                continue
            model = clean(d.get("model", "")) or "unknown"
            attack = clean(d.get("attack", "")) or "none"
            domain = clean(d.get("domain", "")) or "unknown"
            domain_count[domain] += 1
            source_id = d.get("source_id") or text[:64]
            label = 0 if model.lower() == "human" else 1
            split = split_by_hash(f"raid-{source_id}", dev=0.05, test=0.10)
            item = row(text, label, _RAID_LANG.get(domain.lower(), "en"),
                       f"raid-{domain}", "raid", f"{model}|{attack}", split,
                       f"raid-{source_id}")

            if label == 0:
                human_seen += 1
                _reservoir_add(humans, item, max_rows, human_seen, rng)
            else:
                key = (model, attack)
                machine_seen[key] += 1
                _reservoir_add(machines[key], item, machine_cap,
                               machine_seen[key], rng)

    print(f"    RAID domains: {dict(domain_count.most_common(12))} "
          f"（共 {len(domain_count)} 个）", file=sys.stderr)

    machine_rows = [r for items in machines.values() for r in items]
    rng.shuffle(humans)
    rng.shuffle(machine_rows)
    half = max_rows // 2
    return humans[:half] + machine_rows[:half]


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------

def dedupe(rows: Iterable[Dict]) -> List[Dict]:
    seen: Dict[str, Dict] = {}
    conflicts = 0
    for r in rows:
        key = r["id"]
        if key in seen:
            if seen[key]["label"] != r["label"]:
                conflicts += 1
            continue
        seen[key] = r
    if conflicts:
        print(f"[!] 去重时发现 {conflicts} 条同文本异标签，已保留首条", file=sys.stderr)
    return list(seen.values())


def cap(rows: List[Dict], max_per_source: int, seed: int) -> List[Dict]:
    """只截断 train，dev/test 全量保留；每来源内正负各占一半配额。"""
    rng = random.Random(seed)
    out: List[Dict] = [r for r in rows if r["split"] != "train"]

    by_source: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        if r["split"] == "train":
            by_source[r["source"]].append(r)

    for source, items in by_source.items():
        if len(items) <= max_per_source:
            out.extend(items)
            continue
        pos = [x for x in items if x["label"] == 1]
        neg = [x for x in items if x["label"] == 0]
        rng.shuffle(pos)
        rng.shuffle(neg)
        take_pos = min(len(pos), max_per_source // 2)
        take_neg = min(len(neg), max_per_source - take_pos)
        if take_pos + take_neg < max_per_source:
            take_pos = min(len(pos), max_per_source - take_neg)
        out.extend(pos[:take_pos])
        out.extend(neg[:take_neg])
    return out


def stats(rows: Sequence[Dict]) -> None:
    print(f"总计 {len(rows)} 条")
    for key in ("source", "lang", "split", "label"):
        counter = Counter(r[key] for r in rows)
        print(f"  {key}: {dict(counter)}")
    cross = Counter((r["source"], r["split"]) for r in rows)
    print("  source×split:", dict(cross))


def main() -> int:
    ap = argparse.ArgumentParser(description="构建中英双语 AI 文本检测语料")
    ap.add_argument("--max-per-source", type=int, default=160000,
                    help="每个来源的最大样本数（默认 160000）")
    ap.add_argument("--raid-extra", default=str(ROOT / "data" / "raw" / "raid" / "extra.csv"),
                    help="RAID extra.csv 路径；不存在则跳过")
    ap.add_argument("--raid-max", type=int, default=60000,
                    help="RAID 采样上限（正负各一半）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[+] 下载 / 解析 HC3-Chinese ...")
    rows: List[Dict] = load_hc3("Hello-SimpleAI/HC3-Chinese", "zh")
    print(f"    {len(rows)} 条")

    print("[+] 下载 / 解析 HC3 英文 ...")
    en_rows = load_hc3("Hello-SimpleAI/HC3", "en")
    print(f"    {len(en_rows)} 条")
    rows.extend(en_rows)

    print("[+] 下载 / 解析 PR-HPPT 中文学术摘要 ...")
    pr = load_prhppt()
    print(f"    {len(pr)} 条")
    rows.extend(pr)

    print("[+] 下载 / 解析 MAGE 英文 ...")
    mage = load_mage()
    print(f"    {len(mage)} 条")
    rows.extend(mage)

    print("[+] 解析 RAID extra（多生成器 + 攻击）...")
    raid = load_raid_extra(Path(args.raid_extra), args.raid_max, args.seed)
    print(f"    {len(raid)} 条")
    rows.extend(raid)

    rows = dedupe(rows)
    rows = cap(rows, args.max_per_source, args.seed)

    for split in ("train", "dev", "test"):
        part = [r for r in rows if r["split"] == split]
        path = OUT_DIR / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[+] {path}: {len(part)} 条")

    stats(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
