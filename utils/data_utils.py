"""Утилиты для работы с данными"""

import torch
from typing import Tuple
from utils.config import ModelConfig
import json


def load_data(file_path: str) -> str:
    """Загрузка данных из файла"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    except FileNotFoundError:
        print(f"Файл {file_path} не найден. Используем тестовые данные.")
        return "Hello world! This is a test dataset for training language models. " * 100
    
def load_data_json(file_path: str, sample_size: int = -1) -> list:
    """Загрузка данных из файла форматом jsonl"""
    datas = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if sample_size == 0:
                break
            
            line = line.strip()
            if not line:
                continue
                
            try:
                datas.append(json.loads(line))
                sample_size = sample_size - 1 if sample_size > 0 else -1
            except json.JSONDecodeError as e:
                print(f"Ошибка парсинга строки: {line}")
                print(e)
    
    return datas
    

def get_batch(data: torch.Tensor, config: ModelConfig, split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
    """Получение батча для обучения"""
    n = int(0.9 * len(data))
    train_data = data[:n] if split == "train" else data[n:]
    
    ix = torch.randint(len(train_data) - config.block_size, (config.batch_size,))
    x = torch.stack([train_data[i:i + config.block_size] for i in ix])
    y = torch.stack([train_data[i + 1:i + config.block_size + 1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y


def prepare_data(text: str, tokenizer, config: ModelConfig, split_flag: bool = False) -> torch.Tensor:
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
    
    if split_flag:
        n = int(config.train_val_split * len(data_tensor))
        return data_tensor[:n], data_tensor[n:]  
    
    return data_tensor   

def split_data(data: torch.Tensor, train_val_split: float) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(train_val_split * len(data))
    return data[:n], data[n:]

