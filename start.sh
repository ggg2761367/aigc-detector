#!/usr/bin/env bash
# AIGC Detector - macOS / Linux 一键启动
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || { echo "[x] 找不到 $PY，请先安装 Python 3.10+"; exit 1; }

"$PY" - <<'EOF'
import sys
if sys.version_info < (3, 10):
    sys.exit("[x] 需要 Python 3.10 及以上，当前 %s" % sys.version.split()[0])
EOF

if [ ! -f "models/distilmbert-aigc-onnx/model_int8.onnx" ]; then
  echo "[!] 未找到 ONNX 权重，将以启发式模式启动。"
  echo "    完整版：python3 scripts/download_model.py --repo 你的用户名/aigc-detector-onnx"
  echo
fi

echo "[*] 启动中：http://127.0.0.1:8765/   （Ctrl+C 停止）"
exec "$PY" launcher.py "$@"
