"""
Базовые классы для конфигов.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import yaml


@dataclass
class BaseConfig:
    """
    Базовый класс для всех конфигов.
    Предоставляет методы для сериализации/десериализации.
    """
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return asdict(self)
    
    def to_yaml(self, path: str):
        """Сохранение в YAML"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Создание из словаря"""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'BaseConfig':
        """Загрузка из YAML"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def update(self, **kwargs):
        """Обновление параметров"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
