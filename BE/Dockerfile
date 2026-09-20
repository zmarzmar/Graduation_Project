FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 논문 길이 검사용 토크나이저 파일을 빌드 시 받아둔다 — 런타임에 네트워크에 의존하지 않도록
ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken
RUN uv run --no-sync python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
