"""Базовые классы и интерфейсы для всех анализаторов"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# ИМПОРТ из enums (единственный источник истины)
from .enums import AnalyzerType


@dataclass
class AnalysisContext:
    """Контекст анализа"""
    source_code: str
    language: str = 'python'
    timeout: Optional[int] = 30
    max_input_size: int = 10000
    debug_mode: bool = False
    cache_results: bool = True
    metadata: Dict[str, Any] = None


class BaseComplexityAnalyzer(ABC):
    """Базовый класс для всех анализаторов сложности"""
    
    def __init__(self, name: str = None, analyzer_type: AnalyzerType = None):
        """
        ОБНОВЛЕНО: аргументы опциональны для совместимости.
        Подклассы могут вызывать super().__init__() без аргументов.
        """
        self.name: str = name or "unknown_analyzer"
        self.analyzer_type: AnalyzerType = analyzer_type or AnalyzerType.AST
        self.is_initialized: bool = False
        self.config: Dict[str, Any] = {}
        self.cache: Dict[str, Any] = {}
        
    @abstractmethod
    def analyze(self, context: AnalysisContext) -> 'ComplexityResult':
        """Основной метод анализа"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Проверка доступности анализатора"""
        pass
    
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Инициализация анализатора"""
        self.config = config or {}
        self.is_initialized = True
        return True
    
    def cleanup(self) -> None:
        """Очистка ресурсов"""
        self.cache.clear()
    
    def get_supported_languages(self) -> List[str]:
        """Поддерживаемые языки"""
        return ['python']
    
    def validate_input(self, context: AnalysisContext) -> bool:
        """Валидация входных данных"""
        return bool(context.source_code and context.source_code.strip())


class AnalyzerFactory:
    """Фабрика анализаторов"""
    
    _analyzers: Dict[str, type] = {}
    _instances: Dict[str, BaseComplexityAnalyzer] = {}
    
    @classmethod
    def register(cls, name: str, analyzer_class: type) -> None:
        """Регистрация анализатора"""
        cls._analyzers[name] = analyzer_class
    
    @classmethod
    def create(cls, name: str, config: Dict[str, Any] = None) -> Optional[BaseComplexityAnalyzer]:
        """Создание экземпляра анализатора"""
        if name in cls._instances:
            return cls._instances[name]
            
        if name not in cls._analyzers:
            return None
            
        analyzer = cls._analyzers[name]()
        if analyzer.initialize(config):
            cls._instances[name] = analyzer
            return analyzer
        return None
    
    @classmethod
    def get_available_analyzers(cls) -> List[str]:
        """Список доступных анализаторов"""
        return [name for name, analyzer_class in cls._analyzers.items()
                if analyzer_class().is_available()]
