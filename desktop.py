#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AIGC 检测器桌面窗口版：pywebview 原生窗口 + 本地 HTTP 服务。

与 launcher.py 的区别
--------------------
- launcher.py：控制台程序，起服务后开默认浏览器（适合开发调试）
- desktop.py：无控制台，内嵌 WebView2 窗口，关窗口即退出（打包默认入口）

用法
----
python desktop.py                  # 原生窗口 + 自动选端口
python desktop.py --port 9000
python desktop.py --no-window      # 只起服务（冒烟测试）
python desktop.py --browser        # 不开窗口，退回默认浏览器
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import socket
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
from http.server import ThreadingHTTPServer

# PyInstaller --noconsole 下 sys.stdout/stderr 为 None，先兜底再导入 app，
# 否则 Handler.log_message 的 sys.stderr.write 会抛异常打断请求。
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

# 无控制台时异常无处可看：frozen 下统一写 %TEMP%\aigc-desktop.log
if getattr(sys, "frozen", False):
    _LOG = open(os.path.join(tempfile.gettempdir(), "aigc-desktop.log"),
                "a", encoding="utf-8", buffering=1)
    faulthandler.enable(file=_LOG)

    def _excepthook(exc_type, exc, tb):
        traceback.print_exception(exc_type, exc, tb, file=_LOG)
        _LOG.flush()

    sys.excepthook = _excepthook

import app  # noqa: E402

APP_TITLE = "AIGC 检测器"


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _pick_port(host: str, preferred: int) -> int:
    for port in range(preferred, min(preferred + 50, 65535)):
        if _port_free(host, port):
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def main() -> int:
    ap = argparse.ArgumentParser(description=APP_TITLE)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765,
                    help="首选端口；被占用时自动顺延")
    ap.add_argument("--no-window", action="store_true",
                    help="只起服务，不创建窗口（测试用）")
    ap.add_argument("--browser", action="store_true",
                    help="不开原生窗口，退回默认浏览器")
    args = ap.parse_args()

    port = _pick_port(args.host, args.port)
    httpd = ThreadingHTTPServer((args.host, port), app.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://{args.host}:{port}/"

    try:
        if args.no_window or args.browser:
            if args.browser:
                webbrowser.open(url)
            print(f"{APP_TITLE}: {url}", file=sys.stderr)
            while True:
                time.sleep(3600)
        else:
            import webview

            webview.create_window(
                APP_TITLE, url,
                width=1100, height=800,
                min_size=(760, 560),
            )
            webview.start()
    except KeyboardInterrupt:
        pass
    except Exception as e:  # WebView2 缺失等：退回浏览器，不让程序静默死掉
        print(f"[!] 窗口启动失败，改用浏览器: {e}", file=sys.stderr)
        webbrowser.open(url)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
