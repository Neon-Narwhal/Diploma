#!/bin/bash

# Остановить скрипт при любой ошибке
set -e

echo "============================================================"
echo "🚀 ЗАПУСК ОКРУЖЕНИЯ ДЛЯ A100 BENCHMARK"
echo "============================================================"

# 1. Проверка/Установка uv
if ! command -v uv &> /dev/null; then
    echo "📦 uv не найден. Устанавливаю..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
else
    echo "✅ uv уже установлен"
fi

# 2. Создание виртуального окружения и установка зависимостей
echo "📦 Установка зависимостей проекта..."
uv sync

# 3. Проверка и скачивание данных
# Проверяем, есть ли данные. Если нет — запускаем скрипт скачивания.
# Предполагаем, что у тебя есть скрипт для скачивания (например, data/download.sh или python скрипт)
# Если его нет, этот шаг можно закомментировать или добавить команду curl/wget
DATA_FILE="data/bigobench_mapped/train.jsonl"

if [ ! -f "$DATA_FILE" ]; then
    echo "⬇️ Данные не найдены. Скачиваю Big-O Bench..."
    # Создаем папки
    mkdir -p data/bigobench_mapped
    
    # ВСТАВЬ СЮДА КОМАНДУ СКАЧИВАНИЯ ИЛИ ЗАПУСК СКРИПТА
    # Например, если есть scripts/download_data.py:
    # uv run python scripts/download_data.py
    
    # ИЛИ (если нужно просто скачать архив):
    # wget https://.../dataset.zip -O data/dataset.zip
    # unzip data/dataset.zip -d data/
    
    echo "⚠️ ВНИМАНИЕ: В скрипте run_a100.sh нужно раскомментировать логику скачивания данных!"
else
    echo "✅ Данные уже на месте"
fi

# 4. Запуск бенчмарка
echo "🔥 ЗАПУСК ОБУЧЕНИЯ НА A100..."
echo "Логи будут писаться в файл: training_a100.log"
echo "Можешь отключиться от SSH, процесс запущен в фоне (nohup)."

# Используем nohup, чтобы обучение не прервалось при разрыве SSH
# uv run запускает python внутри виртуального окружения
nohup uv run python ml/experiments/run_full_benchmark.py > training_a100.log 2>&1 &

# Выводим PID процесса
PID=$!
echo "✅ Процесс запущен! PID: $PID"
echo "Чтобы следить за логами, введи: tail -f training_a100.log"
