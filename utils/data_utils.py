"""Утилиты для работы с данными"""

import torch
from typing import Tuple
from utils.config import ModelConfig


def load_data(file_path: str) -> str:
    """Загрузка данных из файла"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Файл {file_path} не найден. Используем тестовые данные.")
        return "Hello world! This is a test dataset for training language models. " * 100


def get_batch(data: torch.Tensor, config: ModelConfig, split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
    """Получение батча для обучения"""
    n = int(0.9 * len(data))
    train_data = data[:n] if split == "train" else data[n:]
    
    ix = torch.randint(len(train_data) - config.block_size, (config.batch_size,))
    x = torch.stack([train_data[i:i + config.block_size] for i in ix])
    y = torch.stack([train_data[i + 1:i + config.block_size + 1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y


def prepare_data(text: str, tokenizer, config: ModelConfig) -> torch.Tensor:
    """
    Подготовка данных для обучения
    
    Args:
        text: Исходный текст
        tokenizer: Токенизатор для кодирования текста
        config: Конфигурация модели
        
    Returns:
        torch.Tensor: Тензор с токенизированными данными
    """
    # Токенизируем текст
    encoded_data = tokenizer.encode(text)
    data_tensor = torch.tensor(encoded_data, dtype=torch.long)
    
    print(f"📊 Подготовлены данные:")
    print(f"   - Исходный текст: {len(text):,} символов")  
    print(f"   - Токенов: {len(encoded_data):,}")
    print(f"   - Размер тензора: {data_tensor.shape}")
    print(f"   - Устройство: {config.device}")
    print(f"   - Эффективность токенизации: {len(encoded_data)/len(text):.3f} токенов/символ")
    
    return data_tensor
