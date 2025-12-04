"""
Базовый конфигурационный класс для всех модулей.
"""

import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class BaseConfig:
    """
    Базовый класс конфигурации.
    Поддерживает загрузку из YAML и сериализацию.
    """
    
    @classmethod
    def from_yaml(cls, path: str) -> 'BaseConfig':
        """Загрузка конфигурации из YAML файла"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseConfig':
        """Создание конфига из словаря"""
        # Фильтруем только известные поля
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь"""
        return asdict(self)
    
    def to_yaml(self, path: str) -> None:
        """Сохранение конфига в YAML"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    def validate(self) -> bool:
        """
        Валидация конфигурации.
        Переопределяется в подклассах для специфичной валидации.
        """
        return True
    
    def merge(self, other: 'BaseConfig') -> 'BaseConfig':
        """
        Объединение с другим конфигом.
        Значения из other перезаписывают текущие.
        """
        merged_dict = self.to_dict()
        merged_dict.update(other.to_dict())
        return self.__class__.from_dict(merged_dict)
