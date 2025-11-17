"""
Реестр анализаторов с поддержкой динамической регистрации.
"""

from typing import Dict, Type, Optional, List
from .base import BaseComplexityAnalyzer


class AnalyzerRegistry:
    """Централизованный реестр всех анализаторов."""
    
    def __init__(self):
        self._analyzers: Dict[str, Type[BaseComplexityAnalyzer]] = {}
        self._metadata: Dict[str, dict] = {}
    
    def register(
        self, 
        name: str, 
        analyzer_class: Type[BaseComplexityAnalyzer],
        enabled: bool = True,
        metadata: Optional[dict] = None
    ):
        """Регистрация анализатора."""
        self._analyzers[name] = analyzer_class
        self._metadata[name] = {
            'enabled': enabled,
            'class': analyzer_class.__name__,
            'module': analyzer_class.__module__,
            **(metadata or {})
        }
    
    def create(self, name: str, config: Optional[dict] = None) -> Optional[BaseComplexityAnalyzer]:
        """Создание экземпляра анализатора."""
        analyzer_class = self._analyzers.get(name)
        
        if analyzer_class is None:
            return None
        
        analyzer = analyzer_class()
        
        if config and hasattr(analyzer, 'initialize'):
            analyzer.initialize(config)
        
        return analyzer
    
    def list_analyzers(self, enabled_only: bool = False) -> List[str]:
        """Список зарегистрированных анализаторов."""
        if enabled_only:
            return [
                name for name, meta in self._metadata.items()
                if meta.get('enabled', True)
            ]
        return list(self._analyzers.keys())
    
    def get_info(self, name: str) -> dict:
        """Информация об анализаторе."""
        return self._metadata.get(name, {})
    
    def is_registered(self, name: str) -> bool:
        """Проверка регистрации."""
        return name in self._analyzers
