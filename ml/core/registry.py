"""
Регистрация типов моделей для фабрики.
"""

from typing import Dict, Type, Callable
from ml.core.base_model import BaseModel


class ModelRegistry:
    """
    Реестр типов моделей.
    Позволяет регистрировать новые типы моделей динамически.
    """
    
    _registry: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, model_type: str, model_class: Type[BaseModel]) -> None:
        """
        Регистрация типа модели.
        
        Args:
            model_type: строковый идентификатор типа (например, 'catboost')
            model_class: класс модели, наследующийся от BaseModel
        """
        if model_type in cls._registry:
            raise ValueError(f"Model type '{model_type}' already registered")
        
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"Model class must inherit from BaseModel")
        
        cls._registry[model_type] = model_class
    
    @classmethod
    def get(cls, model_type: str) -> Type[BaseModel]:
        """
        Получение класса модели по типу.
        
        Args:
            model_type: строковый идентификатор типа
            
        Returns:
            Класс модели
            
        Raises:
            ValueError: если тип не зарегистрирован
        """
        if model_type not in cls._registry:
            available = ', '.join(cls._registry.keys())
            raise ValueError(
                f"Unknown model type: '{model_type}'. "
                f"Available types: {available}"
            )
        
        return cls._registry[model_type]
    
    @classmethod
    def list_types(cls) -> list:
        """Список всех зарегистрированных типов"""
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, model_type: str) -> bool:
        """Проверка регистрации типа"""
        return model_type in cls._registry
