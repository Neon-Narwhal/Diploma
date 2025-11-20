"""
Ядро ML-модуля: базовые абстракции, фабрики, реестры.
"""

from ml.core.base_model import BaseModel
from ml.core.model_config import ModelConfig
from ml.core.model_factory import ModelFactory
from ml.core.registry import ModelRegistry

__all__ = [
    'BaseModel',
    'ModelConfig',
    'ModelFactory',
    'ModelRegistry',
]
