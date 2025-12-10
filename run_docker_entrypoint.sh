#!/bin/bash
set -e

echo "🐳 ЗАПУСК В КОНТЕЙНЕРЕ..."

# 1. Подготовка данных (внутри контейнера)
if [ ! -d "data/bigobench" ] && [ ! -f "data/bigobench_mapped/train.jsonl" ]; then
    echo "⬇️ Скачивание данных..."
    python prepare_dataset.py
fi

if [ ! -f "data/bigobench_mapped/train.jsonl" ]; then
    echo "⚙️ Маппинг данных..."
    python prepare_mapped_dataset.py
fi

# 2. Обучение
echo "🔥 ЗАПУСК ОБУЧЕНИЯ..."
python ml/experiments/run_full_benchmark.py
