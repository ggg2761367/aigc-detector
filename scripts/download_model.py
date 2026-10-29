#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 ONNX 权重装到 models/distilmbert-aigc-onnx/，零第三方依赖。

权重约 133MB（压缩后 85MB），不适合放进 Git 仓库，所以代码与权重分开分发。
四种来源，按需选一个：

  1. GitHub Release（默认）—— 下载 zip 并解包，只需要一个 GitHub 账号
  2. HuggingFace 仓库       —— 逐个文件下载，海外网络更快
  3. 本地目录               —— 离线拷贝
  4. 任意 zip 直链          —— 自建 CDN、网盘直链

用法
----
    python scripts/download_model.py                      # GitHub Release（默认）
    python scripts/download_model.py --tag v1.0.0         # 指定版本
    python scripts/download_model.py --url https://.../x.zip
    python scripts/download_model.py --src D:/downloads/onnx
    python scripts/download_model.py --hf-repo user/aigc-detector-onnx
    python scripts/download_model.py --check              # 只检查本地是否齐备

装好后重跑 `python app.py`，页面会显示神经分类器已加载。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "models" / "distilmbert-aigc-onnx"

# ---- 发布配置：换仓库或换版本时，只改这三行 ----
DEFAULT_GH_REPO = "ggg2761367/aigc-detector"
DEFAULT_TAG = "v1.0.0"
DEFAULT_ASSET = "aigc-detector-onnx-v1.zip"

# 少一个就跑不起来，或者校准会静默降级，所以必需项全列出来
REQUIRED = [
    "onnx_manifest.json",   # 权重文件名与窗口参数
    "model_int8.onnx",      # int8 权重，约 130MB
    "tokenizer.json",       # HF tokenizers 分词器
]
OPTIONAL = [
    "tokenizer_config.json",
    "calibration.json",     # 全局 isotonic 校准表
    "calibration_zh.json",  # 中文专用
    "calibration_en.json",  # 英文专用
]

CHUNK = 1 << 20  # 1MB
UA = {"User-Agent": "aigc-detector-model-fetcher/1.1"}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def missing(dest: Path) -> list:
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
    if miss:
        print(f"\n[!] 还缺 {len(miss)} 个必需文件，神经分类器不会启用（启发式模式照常可用）。")
    else:
        print("\n[+] 齐了。运行 python app.py 即可加载神经分类器。")
    return 1 if miss else 0


def resolve_release_url(repo: str, tag: str, asset: str, pinned: bool = False) -> str:
    """找出真正挂着 asset 的那个 Release 的下载地址。

    直接拼 /releases/download/<tag>/<asset> 太脆：tag 大小写不同（v1.0.0 与 V1.0.0
    在 GitHub 上是两个 tag）、后来的版本忘了重传资产，都会 404。所以默认走 API
    按「最新 → 最旧」扫一遍，谁身上有这个资产就用谁；扫不到再退回 tag 拼接。
    """
    direct = f"https://github.com/{repo}/releases/download/{tag}/{asset}"
    if pinned:
        return direct

    api = f"https://api.github.com/repos/{repo}/releases?per_page=30"
    req = urllib.request.Request(api, headers={**UA, "Accept": "application/vnd.github+json"})
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            releases = json.load(r)
    except Exception as exc:                                   # 离线 / 限流 / 无 API
        print(f"  [warn] 查 Release 列表失败（{exc}），按 tag 直连")
        return direct

    for rel in releases or []:
        names = [a.get("name", "") for a in rel.get("assets", [])]
        if asset in names:
            if rel.get("tag_name") != tag:
                print(f"  [info] 资产在 {rel.get('tag_name')} 上（配置里是 {tag}），已自动改用")
            return rel["assets"][names.index(asset)]["browser_download_url"]

    print(f"  [warn] 没有任何 Release 挂着 {asset}，尝试 latest 与 {tag}")
    return f"https://github.com/{repo}/releases/latest/download/{asset}"


def stream_to_file(url: str, out: Path, force: bool = False, label: str = "") -> Path:
    """下载到 out，带进度条。已存在且非 force 时跳过。"""
    if out.is_file() and out.stat().st_size > 0 and not force:
        print(f"  [skip] {label or out.name} 已存在（{human(out.stat().st_size)}），加 --force 重下")
        return out

    req = urllib.request.Request(url, headers=UA)
    tmp = out.with_name(out.name + ".part")
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
                        sys.stdout.write(f"\r  [get ] {label or out.name}  {pct:3d}%  "
                                         f"{human(done)}/{human(size)}")
                        sys.stdout.flush()
            if size:
                sys.stdout.write("\n")
        if tmp.stat().st_size == 0:
            raise OSError("下载到 0 字节")
        tmp.replace(out)
        print(f"  [done] {label or out.name}  {human(out.stat().st_size)}")
        return out
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        if exc.code == 404:
            raise SystemExit(
                f"[!] 404：{url}\n"
                f"    三种可能：Release 还没发布 / tag 写错 / 资产文件名不对。\n"
                f"    确认地址：https://github.com/{DEFAULT_GH_REPO}/releases\n"
                f"    也可以先用离线方式：--src <解压后的目录>"
            ) from None
        raise SystemExit(f"[!] HTTP {exc.code}: {url}") from None
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"[!] 网络不可达：{exc.reason}\n"
            f"    可换 --url 指向可访问的直链，或用 --src 从本地解压。"
        ) from None


def extract_weights(zip_path: Path, dest: Path) -> None:
    """从 zip 里挑出权重文件写进 dest。容忍顶层多一层目录。"""
    dest.mkdir(parents=True, exist_ok=True)
    wanted = set(REQUIRED + OPTIONAL)
    got = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            base = os.path.basename(info.filename)
            if base not in wanted:
                continue
            target = dest / base
            with z.open(info) as src, open(target, "wb") as fh:
                shutil.copyfileobj(src, fh, CHUNK)
            got.append(base)
            print(f"  [unzip] {base:<26} {human(target.stat().st_size):>8}")
    if not got:
        raise SystemExit(
            f"[!] 压缩包里没找到权重文件（期望 {', '.join(REQUIRED)}）。\n"
            f"    结构特殊的话先手动解压，再用 --src <解压目录>。"
        )
    lacked = [f for f in REQUIRED if f not in got]
    if lacked:
        raise SystemExit(f"[!] 压缩包缺必需文件：{', '.join(lacked)}")


def download_and_extract(url: str, dest: Path, force: bool = False) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        zpath = Path(td) / "weights.zip"
        print(f"[*] 来源：{url}")
        stream_to_file(url, zpath, force=True, label="weights.zip")
        if not zipfile.is_zipfile(zpath):
            raise SystemExit(
                "[!] 下载到的不是 zip（可能是网页或落地页）。\n"
                "    直链应该以 .zip 结尾；GitHub 请用 Release 资产地址。"
            )
        print("[*] 解包：")
        extract_weights(zpath, dest)


def copy_from_dir(src: Path, dest: Path) -> None:
    if not src.is_dir():
        raise SystemExit(f"[!] 不是目录：{src}")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[*] 从本地拷贝：{src} -> {dest}")
    for f in REQUIRED + OPTIONAL:
        p = src / f
        if p.is_file():
            shutil.copy2(p, dest / f)
            print(f"  [copy] {f:<26} {human(p.stat().st_size):>8}")
        elif f in REQUIRED:
            raise SystemExit(f"[!] 源目录缺必需文件：{f}")


def from_huggingface(repo: str, revision: str, endpoint: str, dest: Path, force: bool) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    base = f"{endpoint.rstrip('/')}/{repo}/resolve/{revision}"
    print(f"[*] 来源：{base}")
    for f in REQUIRED + OPTIONAL:
        try:
            stream_to_file(f"{base}/{f}", dest / f, force=force)
        except SystemExit:
            if f in OPTIONAL:
                print(f"  [warn] {f} 未取到，跳过（校准降级为未校准）")
                continue
            raise


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 AIGC Detector 的 ONNX 权重（零依赖）")
    ap.add_argument("--src", help="本地目录：直接拷贝，不走网络")
    ap.add_argument("--url", help="zip 直链（自建 CDN / 网盘直链）")
    ap.add_argument("--hf-repo", help="HuggingFace 仓库，如 user/aigc-detector-onnx")
    ap.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
                    help="HF 端点，默认读 HF_ENDPOINT")
    ap.add_argument("--gh-repo", default=DEFAULT_GH_REPO, help=f"GitHub 仓库，默认 {DEFAULT_GH_REPO}")
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help=f"Release 标签，默认自动探测（配置项 {DEFAULT_TAG}）")
    ap.add_argument("--asset", default=DEFAULT_ASSET, help=f"Release 资产名，默认 {DEFAULT_ASSET}")
    ap.add_argument("--dest", default=str(DEFAULT_DEST), help="目标目录")
    ap.add_argument("--force", action="store_true", help="已有文件也重下")
    ap.add_argument("--check", action="store_true", help="只检查本地文件是否齐备")
    args = ap.parse_args()

    dest = Path(args.dest)

    if args.check:
        print("[*] 本地文件检查：")
        return report(dest)

    try:
        if args.src:
            copy_from_dir(Path(args.src), dest)
        elif args.url:
            download_and_extract(args.url, dest, args.force)
        elif args.hf_repo:
            from_huggingface(args.hf_repo, "main", args.hf_endpoint, dest, args.force)
        else:
            url = resolve_release_url(args.gh_repo, args.tag, args.asset,
                                      pinned=args.tag != DEFAULT_TAG)
            print(f"[*] 来源：{url}")
            download_and_extract(url, dest, args.force)
    except KeyboardInterrupt:
        print("\n[!] 已中断。")
        return 130

    print("\n[*] 结果：")
    return report(dest)


if __name__ == "__main__":
    raise SystemExit(main())
