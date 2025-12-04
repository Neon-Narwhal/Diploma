#!/bin/bash

# Остановить скрипт при любой ошибке
set -e

echo "============================================================"
echo "🚀 ЗАПУСК DIPLOMA BENCHMARK (A100)"
echo "============================================================"

# 1. Проверка/Установка uv
if ! command -v uv &> /dev/null; then
    echo "📦 uv не найден. Устанавливаю..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
else
    echo "✅ uv уже установлен"
fi

# 2. Синхронизация окружения
echo "📦 Установка зависимостей..."
uv sync

# 3. Шаг 1: Получение сырых данных (BigOBench)
RAW_DATA_DIR="data/bigobench"

if [ ! -d "$RAW_DATA_DIR" ]; then
    echo "⚙️ [1/2] Запуск prepare_dataset.py (скачивание исходных данных)..."
    uv run python prepare_dataset.py
    echo "✅ Исходные данные готовы."
else
    echo "✅ Исходные данные уже есть."
fi

# 4. Шаг 2: Маппинг классов
MAPPED_DATA_FILE="data/bigobench_mapped/train.jsonl"

if [ ! -f "$MAPPED_DATA_FILE" ]; then
    echo "⚙️ [2/2] Запуск prepare_mapped_dataset.py (маппинг классов)..."
    uv run python prepare_mapped_dataset.py
    echo "✅ Датасет для обучения готов."
else
    echo "✅ Маппинг уже выполнен."
fi

# 5. Запуск бенчмарка
echo "🔥 ЗАПУСК ОБУЧЕНИЯ..."
echo "Логи пишутся в: training_a100.log"

nohup uv run python ml/experiments/run_full_benchmark.py > training_a100.log 2>&1 &

PID=$!
echo "✅ Процесс запущен! PID: $PID"
echo "Следить за логами: tail -f training_a100.log"
