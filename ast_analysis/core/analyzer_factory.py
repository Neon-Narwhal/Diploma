"""
Фабрика для создания AST анализаторов.
"""

from typing import Dict, Any, Optional
from ast_analysis.core.base_analyzer import BaseASTAnalyzer
from ast_analysis.core.registry import ASTAnalyzerRegistry


class ASTAnalyzerFactory:
    """
    Фабрика для создания AST анализаторов из конфигурации.
    """
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> BaseASTAnalyzer:
        """
        Создание анализатора из конфига.
        
        Args:
            config: Словарь конфигурации с полями:
                - type: тип анализатора (обязательно)
                - name: имя экземпляра (опционально)
                - params: параметры анализатора (опционально)
        
        Returns:
            Экземпляр анализатора
        """
        analyzer_type = config.get('type')
        if not analyzer_type:
            raise ValueError("Analyzer config must have 'type' field")
        
        # Имя экземпляра (для логирования)
        instance_name = config.get('name', analyzer_type)
        
        # Параметры анализатора
        params = config.get('params', {})
        
        # Создание через registry
        analyzer = ASTAnalyzerRegistry.create(analyzer_type, **params)
        
        # Инициализация
        analyzer.initialize()
        
        return analyzer
    
    @staticmethod
    def create_multiple(configs: list[Dict[str, Any]]) -> list[BaseASTAnalyzer]:
        """
        Создание нескольких анализаторов.
        
        Args:
            configs: Список конфигураций
        
        Returns:
            Список анализаторов
        """
        analyzers = []
        for config in configs:
            try:
                analyzer = ASTAnalyzerFactory.create_from_config(config)
                analyzers.append(analyzer)
            except Exception as e:
                print(f"Warning: Failed to create analyzer from {config}: {e}")
        
        return analyzers
