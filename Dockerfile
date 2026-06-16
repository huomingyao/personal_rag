# ===================== Stage 1: Builder =====================
FROM python:3.11-slim as builder

WORKDIR /app

# 安装编译依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ===================== Stage 2: Runtime =====================
FROM python:3.11-slim

WORKDIR /app

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

# 从 builder 复制已安装的包
COPY --from=builder /install /usr/local

# 复制应用代码
COPY rag.py .
COPY start_server.py .
COPY requirements.txt .
COPY src/ ./src/
COPY templates/ ./templates/
COPY models/ ./models/

# 创建数据目录
RUN mkdir -p /app/data/knowledge_bases /app/data/vector_stores

# 环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=5000

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/kb/list || exit 1

CMD ["python", "start_server.py"]
