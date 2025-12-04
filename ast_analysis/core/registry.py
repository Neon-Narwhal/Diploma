"""
Регистрация AST анализаторов.
"""

from typing import Dict, Type, Optional
from ast_analysis.core.base_analyzer import BaseASTAnalyzer


class ASTAnalyzerRegistry:
    """
    Реестр AST анализаторов.
    Позволяет регистрировать и создавать анализаторы по имени.
    """
    
    _registry: Dict[str, Type[BaseASTAnalyzer]] = {}
    
    @classmethod
    def register(cls, name: str, analyzer_class: Type[BaseASTAnalyzer]) -> None:
        """
        Регистрация анализатора.
        
        Args:
            name: Уникальное имя анализатора
            analyzer_class: Класс анализатора
        """
        if name in cls._registry:
            raise ValueError(f"Analyzer '{name}' already registered")
        
        if not issubclass(analyzer_class, BaseASTAnalyzer):
            raise TypeError(f"Analyzer must inherit from BaseASTAnalyzer")
        
        cls._registry[name] = analyzer_class
    
    @classmethod
    def get(cls, name: str) -> Type[BaseASTAnalyzer]:
        """
        Получение класса анализатора по имени.
        
        Args:
            name: Имя анализатора
        
        Returns:
            Класс анализатора
        """
        if name not in cls._registry:
            available = ', '.join(cls._registry.keys())
            raise ValueError(
                f"Unknown analyzer: '{name}'. "
                f"Available: {available}"
            )
        
        return cls._registry[name]
    
    @classmethod
    def create(cls, name: str, **config) -> BaseASTAnalyzer:
        """
        Создание экземпляра анализатора.
        
        Args:
            name: Имя анализатора
            **config: Параметры конфигурации
        
        Returns:
            Экземпляр анализатора
        """
        analyzer_class = cls.get(name)
        return analyzer_class(name=name, **config)
    
    @classmethod
    def list_analyzers(cls) -> list[str]:
        """Список всех зарегистрированных анализаторов"""
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Проверка регистрации анализатора"""
        return name in cls._registry


def register_analyzer(name: str):
    """
    Декоратор для регистрации анализатора.
    
    Usage:
        @register_analyzer('basic')
        class ASTBasicAnalyzer(BaseASTAnalyzer):
            pass
    """
    def decorator(analyzer_class):
        ASTAnalyzerRegistry.register(name, analyzer_class)
        return analyzer_class
    return decorator
