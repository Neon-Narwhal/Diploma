"""
Training модули для обучения моделей.

Компоненты:
- UniversalTrainer: универсальный тренер для всех моделей
- BigOBenchDataset: датасет для BigOBench
- ModelRegistry: регистрация и создание моделей
"""

from .universal_trainer import UniversalTrainer
from .bigobench_dataset import BigOBenchDataset, create_dataloaders_from_config
from .model_registry import ModelRegistry

__all__ = [
    'UniversalTrainer',
    'BigOBenchDataset',
    'create_dataloaders_from_config',
    'ModelRegistry',
]
