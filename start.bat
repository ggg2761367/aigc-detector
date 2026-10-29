@echo off
rem ============================================================
rem  AIGC Detector - Windows 一键启动
rem  双击即可；首次使用请先装 Python 3.10+（勾选 Add to PATH）
rem ============================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [x] 没找到 python，请先安装 Python 3.10 或更新版本。
    echo     下载：https://www.python.org/downloads/
    echo     安装时务必勾选 "Add python.exe to PATH"。
    pause
    exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [x] Python 版本过低，需要 3.10 及以上。
    python -c "import sys; print('    当前版本:', sys.version)"
    pause
    exit /b 1
)

if not exist "models\distilmbert-aigc-onnx\model_int8.onnx" (
    echo [!] 没找到神经分类器权重，将以启发式模式启动（功能完整，精度略低）。
    echo     完整版：python scripts\download_model.py --repo 你的用户名/aigc-detector-onnx
    echo.
)

echo [*] 启动中，浏览器会自动打开 http://127.0.0.1:8765/
echo     关闭本窗口即停止服务。
echo.
python launcher.py
pause
