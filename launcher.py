#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AIGC 检测器桌面启动器：选空闲端口、起 HTTP 服务、自动开浏览器。

用法
----
python launcher.py                  # 自动找空闲端口并打开浏览器
python launcher.py --port 9000      # 指定端口
python launcher.py --no-browser     # 只起服务

打包后等价于双击 dist/AIGC-Detector/AIGC-Detector.exe。
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import webbrowser
from http.server import ThreadingHTTPServer

import app


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _pick_port(host: str, preferred: int) -> int:
    """优先用 preferred；被占用则从 preferred 往后找，最后退到系统分配。"""
    for port in range(preferred, min(preferred + 50, 65535)):
        if _port_free(host, port):
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def main() -> int:
    ap = argparse.ArgumentParser(description="AIGC 检测器桌面启动器")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    ap.add_argument("--port", type=int, default=8765,
                    help="首选端口；被占用时自动顺延（默认 8765）")
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    port = _pick_port(args.host, args.port)
    httpd = ThreadingHTTPServer((args.host, port), app.Handler)
    url = f"http://{args.host}:{port}/"

    print("=" * 52)
    print("  AIGC Detector（本地运行）")
    print(f"  打开: {url}")
    print("  按 Ctrl+C 停止")
    print("=" * 52)

    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[+] 已停止。")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
