"""
Пакет анализаторов сложности с автоматической регистрацией.
"""

import importlib
import pkgutil
from typing import Dict, Type, Optional
from pathlib import Path

from ..core.base import BaseComplexityAnalyzer, AnalysisContext
from ..core.enums import ComplexityClass, AnalyzerType
from ..core.result import ComplexityResult, ComplexityMetrics
from ..core.registry import AnalyzerRegistry

__version__ = "0.2.0"

# Глобальный реестр анализаторов
_registry = AnalyzerRegistry()


def register_analyzer(name: str = None, enabled: bool = True):
    """
    Декоратор для регистрации анализаторов.
    
    Использование:
        @register_analyzer('my_analyzer')
        class MyAnalyzer(BaseComplexityAnalyzer):
            pass
    """
    def decorator(cls: Type[BaseComplexityAnalyzer]):
        analyzer_name = name or cls.__name__.lower()
        _registry.register(analyzer_name, cls, enabled=enabled)
        return cls
    return decorator


def _auto_discover_analyzers():
    """
    Автоматически импортирует все модули из analyzers/ для регистрации.
    """
    analyzers_path = Path(__file__).parent / "analyzers"
    
    if not analyzers_path.exists():
        return
    
    # Импортируем все .py файлы из analyzers/
    for _, module_name, _ in pkgutil.iter_modules([str(analyzers_path)]):
        if module_name.startswith('_'):
            continue
        try:
            importlib.import_module(f'.analyzers.{module_name}', package=__name__)
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to import analyzer module {module_name}: {e}")


# Автоматическое обнаружение при импорте
_auto_discover_analyzers()


def create_analyzer(analyzer_type: str, config: Optional[dict] = None) -> BaseComplexityAnalyzer:
    """
    Фабричная функция для создания анализаторов по строковому имени.
    
    Args:
        analyzer_type: Тип анализатора ('ast_advanced', 'ml_predictor', и т.д.)
        config: Опциональная конфигурация
        
    Returns:
        Экземпляр анализатора
        
    Raises:
        ValueError: Если анализатор не зарегистрирован
    """
    analyzer = _registry.create(analyzer_type, config)
    
    if analyzer is None:
        available = ', '.join(_registry.list_analyzers())
        raise ValueError(
            f"Unknown analyzer type: '{analyzer_type}'. "
            f"Available analyzers: {available}"
        )
    
    return analyzer


def get_available_analyzers() -> list:
    """Получение списка зарегистрированных анализаторов."""
    return _registry.list_analyzers()


def get_analyzer_info(analyzer_type: str) -> Dict:
    """Получение информации об анализаторе."""
    return _registry.get_info(analyzer_type)


# Экспорт основных классов и функций
__all__ = [
    # Core classes
    'BaseComplexityAnalyzer',
    'AnalysisContext',
    'ComplexityResult',
    'ComplexityMetrics',
    'ComplexityClass',
    'AnalyzerType',
    
    # Factory functions
    'create_analyzer',
    'get_available_analyzers',
    'get_analyzer_info',
    
    # Decorator
    'register_analyzer',
]
