"""
Фабрика для создания моделей из конфига.
"""

from ml.core.base_model import BaseModel
from ml.core.model_config import ModelConfig
from ml.core.registry import ModelRegistry


class ModelFactory:
    """
    Фабрика для создания моделей.
    Использует ModelRegistry для динамического создания моделей по типу.
    """
    
    @staticmethod
    def create(config: ModelConfig) -> BaseModel:
        """
        Создание модели из конфига.
        
        Args:
            config: конфигурация модели
            
        Returns:
            Экземпляр модели
            
        Raises:
            ValueError: если тип модели не зарегистрирован
        """
        model_class = ModelRegistry.get(config.type)
        model = model_class(params=config.params)
        return model
    
    @staticmethod
    def create_from_dict(config_dict: dict) -> BaseModel:
        """
        Создание модели из словаря.
        
        Args:
            config_dict: словарь с полями name, type, params
            
        Returns:
            Экземпляр модели
        """
        config = ModelConfig(**config_dict)
        return ModelFactory.create(config)
