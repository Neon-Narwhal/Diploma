# =============================================================================
# training/model_registry.py
# Registry паттерн для динамической регистрации моделей и конфигов
# =============================================================================

from typing import Dict, Type, Callable, Optional
from models.base_model import BaseLanguageModel
from utils.configs.base_config import BaseTrainingConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry для моделей и их конфигов.
    Позволяет динамически добавлять новые модели без изменения кода.
    """
    
    _models: Dict[str, Type[BaseLanguageModel]] = {}
    _configs: Dict[str, Type[BaseTrainingConfig]] = {}
    _factories: Dict[str, Callable] = {}
    
    @classmethod
    def register_model(
        cls, 
        name: str, 
        model_class: Type[BaseLanguageModel],
        config_class: Type[BaseTrainingConfig],
        factory: Optional[Callable] = None
    ):
        """
        Регистрация новой модели.
        
        Args:
            name: Уникальное имя модели (например, 'gpt', 'transformer_squared')
            model_class: Класс модели
            config_class: Класс конфига для этой модели
            factory: Опциональная фабрика для создания модели (если требуется особая инициализация)
        """
        cls._models[name] = model_class
        cls._configs[name] = config_class
        if factory:
            cls._factories[name] = factory
        
        logger.info(f"[Registry] Зарегистрирована модель: {name}")
    
    @classmethod
    def get_model_class(cls, name: str) -> Type[BaseLanguageModel]:
        """Получить класс модели по имени"""
        if name not in cls._models:
            raise ValueError(f"Модель '{name}' не зарегистрирована. Доступные: {list(cls._models.keys())}")
        return cls._models[name]
    
    @classmethod
    def get_config_class(cls, name: str) -> Type[BaseTrainingConfig]:
        """Получить класс конфига по имени модели"""
        if name not in cls._configs:
            raise ValueError(f"Конфиг для модели '{name}' не зарегистрирован")
        return cls._configs[name]
    
    @classmethod
    def create_model(cls, name: str, config: BaseTrainingConfig) -> BaseLanguageModel:
        """Создать модель по имени и конфигу"""
        if name in cls._factories:
            return cls._factories[name](config)
        else:
            model_class = cls.get_model_class(name)
            return model_class(config)
    
    @classmethod
    def list_models(cls) -> list:
        """Список всех зарегистрированных моделей"""
        return list(cls._models.keys())