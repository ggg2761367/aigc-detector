#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地 AIGC 检测服务 —— 零第三方依赖，`python app.py` 即可运行。

接口
----
GET  /                  -> static/index.html
GET  /static/*          -> 静态资源
GET  /api/health        -> {"ok": true, "calibration": {...}}
POST /api/analyze       -> 请求 {"text": "..."}，返回 detector.analyze() 结果
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

from detector import CALIBRATION, analyze
from document import ExtractError, extract
from model_engine import get_engine
from watermark import visible_text

def _resource_dir() -> str:
    """PyInstaller 打包后静态资源在 _MEIPASS/_internal 下。"""
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = _resource_dir()
STATIC_DIR = os.path.join(BASE_DIR, "static")
MAX_BODY = 16 * 1024 * 1024  # 16 MB：足够整本学位论文/多页 PDF 的文本层


class Handler(SimpleHTTPRequestHandler):
    server_version = "AIGCDetector/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    # ---- 日志：只留一行 ----
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _json(self, obj: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # ---- 静态资源 ----
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            eng = get_engine()
            self._json({
                "ok": True,
                "calibration": CALIBRATION,
                "model": eng.info() if eng else None,
            })
            return
        if path in ("/", ""):
            self.path = "/index.html"
        super().do_GET()

    # ---- 请求体 ----
    def _read_body(self) -> Optional[bytes]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            length = 0

        if length <= 0:
            self._json({"ok": False, "error": "空请求体"}, HTTPStatus.BAD_REQUEST)
            return None
        if length > MAX_BODY:
            self._json({"ok": False, "error": "请求过大（上限 16 MB）"}, 413)
            return None
        return self.rfile.read(length)

    # ---- 文档提取 ----
    def _handle_extract(self, parsed) -> None:
        raw = self._read_body()
        if raw is None:
            return

        qs = parse_qs(parsed.query)
        name = (qs.get("name", [""])[0]
                or unquote(self.headers.get("X-Filename", "")))
        name = name.replace("\\", "/").rsplit("/", 1)[-1]

        try:
            result = extract(raw, name)
        except ExtractError as e:
            self._json({"ok": False, "error": str(e)}, HTTPStatus.BAD_REQUEST)
            return
        except Exception as e:  # 解析器兜底：不要让单个坏文件打挂服务
            self._json({"ok": False, "error": f"解析异常: {e}"},
                       HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._json({"ok": True, "filename": name, **result})

    # ---- 分析接口 ----
    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/extract":
            self._handle_extract(parsed)
            return
        if parsed.path != "/api/analyze":
            self._json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
            return

        raw = self._read_body()
        if raw is None:
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError:
            self._json({"ok": False, "error": "请求体不是 UTF-8"}, HTTPStatus.BAD_REQUEST)
            return
        except json.JSONDecodeError as e:
            self._json({"ok": False, "error": f"JSON 解析失败: {e}"}, HTTPStatus.BAD_REQUEST)
            return

        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str):
            self._json({"ok": False, "error": "缺少 text 字段（字符串）"}, HTTPStatus.BAD_REQUEST)
            return

        try:
            result = analyze(text)
        except Exception as e:  # 兜底：单次分析失败不能把服务打挂
            self._json({"ok": False, "error": f"分析异常: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        # 可选：ONNX 分类器结果单独一路返回，避免与启发式分数混算
        if result.get("ok"):
            try:
                eng = get_engine()
                if eng:
                    result["model"] = {
                        **eng.predict(visible_text(text)),
                        **eng.info(),
                    }
            except Exception as e:  # 模型失败不能把启发式结果一起打挂
                result["model"] = {"error": str(e)}

        self._json(result, HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST)


def main() -> int:
    ap = argparse.ArgumentParser(description="本地 AIGC 文本检测服务")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    ap.add_argument("--port", type=int, default=8765, help="监听端口（默认 8765）")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    if not os.path.isdir(STATIC_DIR):
        print(f"[!] 缺少静态目录: {STATIC_DIR}", file=sys.stderr)
        return 1

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print("=" * 52)
    print("  AIGC Detector（本地运行）")
    print(f"  打开: {url}")
    print(f"  校准: temperature={CALIBRATION['temperature']}  "
          f"min_shrink={CALIBRATION['min_shrink']}")
    print("  Ctrl+C 停止")
    print("=" * 52)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[+] 已停止。")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
