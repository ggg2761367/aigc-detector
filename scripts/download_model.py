#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 ONNX 权重下载到 models/distilmbert-aigc-onnx/，零第三方依赖。

权重有 130MB 左右，不适合放进 Git 仓库，所以代码和模型分开分发。
本脚本支持三种来源，按优先级选用：

  1. HuggingFace 仓库（默认，公开仓库无需登录）
  2. 国内镜像：设 HF_ENDPOINT=https://hf-mirror.com 或 --endpoint
  3. 本地目录：--src /path/to/onnx-dir（离线拷贝，可从 Release 压缩包里解压出来）

用法
----
    python scripts/download_model.py --repo your-name/aigc-detector-onnx
    python scripts/download_model.py --repo your-name/aigc-detector-onnx --endpoint https://hf-mirror.com
    python scripts/download_model.py --src D:/downloads/onnx
    python scripts/download_model.py --check          # 只检查本地是否齐备

下载完成后重跑 `python app.py`，页面顶部会显示「神经分类器已加载」。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "models" / "distilmbert-aigc-onnx"

# 少一个文件就跑不起来，或者校准会静默降级，所以这里全部列出
REQUIRED = [
    "onnx_manifest.json",   # 告诉加载器权重文件名与窗口参数
    "model_int8.onnx",      # 真正的 int8 权重，约 130MB
    "tokenizer.json",       # HF tokenizers 的快速分词器
]
OPTIONAL = [
    "tokenizer_config.json",
    "calibration.json",     # 全局 isotonic 校准表
    "calibration_zh.json",  # 中文专用校准表
    "calibration_en.json",  # 英文专用校准表
]

CHUNK = 1 << 20  # 1MB


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def missing(dest: Path) -> list[str]:
    return [f for f in REQUIRED if not (dest / f).is_file()]


def report(dest: Path) -> int:
    miss = missing(dest)
    total = 0
    for f in REQUIRED + OPTIONAL:
        p = dest / f
        if p.is_file():
            total += p.stat().st_size
            print(f"  [ok]   {f:<26} {human(p.stat().st_size):>8}")
        else:
            mark = "缺失" if f in REQUIRED else "可选"
            print(f"  [{mark}] {f:<26} {'-':>8}")
    print(f"\n目录: {dest}")
    print(f"合计: {human(total)}")
    return 1 if miss else 0


def fetch(url: str, out: Path, force: bool = False) -> None:
    if out.is_file() and out.stat().st_size > 0 and not force:
        print(f"  [skip] {out.name} 已存在（{human(out.stat().st_size)}），加 --force 可重下")
        return

    req = urllib.request.Request(url, headers={"User-Agent": "aigc-detector-model-fetcher/1.0"})
    tmp = out.with_suffix(out.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            size = int(resp.headers.get("Content-Length") or 0)
            done = 0
            with open(tmp, "wb") as fh:
                while True:
                    block = resp.read(CHUNK)
                    if not block:
                        break
                    fh.write(block)
                    done += len(block)
                    if size:
                        pct = done * 100 // size
                        sys.stdout.write(f"\r  [get ] {out.name}  {pct:3d}%  {human(done)}/{human(size)}")
                        sys.stdout.flush()
            if size:
                sys.stdout.write("\n")
        if tmp.stat().st_size == 0:
            raise OSError("下载到 0 字节")
        tmp.replace(out)
        print(f"  [done] {out.name}  {human(out.stat().st_size)}")
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        if exc.code == 404:
            raise SystemExit(
                f"[!] 404：{url}\n"
                f"    仓库或文件不存在。检查 --repo 拼写；私有仓库需要先 huggingface-cli login。"
            ) from None
        raise SystemExit(f"[!] HTTP {exc.code}: {url}") from None
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"[!] 网络不可达：{exc.reason}\n"
            f"    国内网络可试 --endpoint https://hf-mirror.com，或改用 --src 从本地压缩包解压。"
        ) from None


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 ONNX 权重（零依赖）")
    ap.add_argument("--repo", help="HuggingFace 仓库，如 your-name/aigc-detector-onnx")
    ap.add_argument("--revision", default="main", help="分支/标签，默认 main")
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
                    help="下载端点，默认读 HF_ENDPOINT")
    ap.add_argument("--src", help="本地目录：直接拷贝，不走网络")
    ap.add_argument("--dest", default=str(DEFAULT_DEST), help="目标目录")
    ap.add_argument("--force", action="store_true", help="已有文件也重下")
    ap.add_argument("--check", action="store_true", help="只检查本地文件是否齐备")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    if args.check:
        print("[*] 本地文件检查：")
        return report(dest)

    if args.src:
        src = Path(args.src)
        if not src.is_dir():
            raise SystemExit(f"[!] 不是目录：{src}")
        print(f"[*] 从本地拷贝：{src} -> {dest}")
        for f in REQUIRED + OPTIONAL:
            p = src / f
            if p.is_file():
                shutil.copy2(p, dest / f)
                print(f"  [copy] {f}  {human(p.stat().st_size)}")
            elif f in REQUIRED:
                raise SystemExit(f"[!] 源目录缺必需文件：{f}")
    elif args.repo:
        base = f"{args.endpoint.rstrip('/')}/{args.repo}/resolve/{args.revision}"
        print(f"[*] 来源：{base}")
        for f in REQUIRED + OPTIONAL:
            try:
                fetch(f"{base}/{f}", dest / f, force=args.force)
            except SystemExit as exc:
                if f in OPTIONAL:
                    print(f"  [warn] {f} 未取到，将跳过（校准会降级为未校准）")
                    continue
                raise
    else:
        ap.print_help()
        print("\n[!] 需要 --repo 或 --src 之一。")
        return 2

    print("\n[*] 结果：")
    return report(dest)


if __name__ == "__main__":
    raise SystemExit(main())
