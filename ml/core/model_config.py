"""
Конфигурация для моделей.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ModelConfig:
    """
    Конфигурация модели.
    
    Attributes:
        name: имя модели для идентификации
        type: тип модели (catboost, xgboost, lightgbm, voting, stacking и т.д.)
        params: параметры модели
        optimization: параметры оптимизации (optional)
    """
    name: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    optimization: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Валидация конфига"""
        if not self.name:
            raise ValueError("Model name cannot be empty")
        if not self.type:
            raise ValueError("Model type cannot be empty")
