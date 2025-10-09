"""
Конфигурации моделей для проекта Diploma.

Структура:
- BaseTrainingConfig: базовый конфиг со ВСЕМИ общими параметрами
- GPTConfig: наследует Base + добавляет архитектуру GPT
- TransformerSquaredConfig: наследует Base + добавляет архитектуру T² + SVF

"""

from .base_config import BaseTrainingConfig
from .gpt_config import GPTConfig
from .transformer_squared_config import TransformerSquaredConfig

__all__ = [
    'BaseTrainingConfig',
    'GPTConfig',
    'TransformerSquaredConfig',
]

__version__ = '1.0.0'
