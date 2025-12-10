# Используем Python 3.11, так как pyproject.toml требует ==3.11.*
FROM python:3.11-slim

# Устанавливаем git и curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Настройки
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=never
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Зависимости
COPY pyproject.toml uv.lock* ./
RUN UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_BUILDS=2 uv sync --frozen --no-install-project --no-dev || uv sync --no-install-project --no-dev

# 2. Код
COPY . .

# 3. Финальная настройка
RUN uv sync --frozen --no-dev || uv sync --no-dev && \
    chmod +x run_docker_entrypoint.sh && \
    mkdir -p data ml/outputs && \
    chmod -R 777 data ml/outputs

ENTRYPOINT ["./run_docker_entrypoint.sh"]
