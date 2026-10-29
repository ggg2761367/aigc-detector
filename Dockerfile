# 单机部署：不装任何依赖也能跑（启发式模式）
#   docker build -t aigc-detector .
#   docker run --rm -p 8765:8765 aigc-detector
# 要带神经分类器，把权重目录挂进去：
#   docker run --rm -p 8765:8765 -v "$PWD/models:/app/models:ro" aigc-detector

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_ENDPOINT=https://hf-mirror.com

WORKDIR /app

# 先装依赖，代码改了不会让这层缓存失效
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py detector.py document.py model_engine.py watermark.py ./
COPY static ./static
COPY scripts ./scripts
COPY tests ./tests

# 模型走 volume，镜像里不带 130MB 权重
RUN mkdir -p /app/models

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8765/api/health',timeout=2)"

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8765"]
