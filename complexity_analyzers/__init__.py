# complexity_analyzers/__init__.py

"""
Пакет анализаторов сложности алгоритмов.
Регистрирует все доступные анализаторы в фабричной функции.
"""

# Импорты базовых классов
from .core.enums import ComplexityClass, AnalyzerType
from .core.base import BaseComplexityAnalyzer, AnalysisContext
from .core.result import ComplexityResult, ComplexityMetrics

# Импорты основных анализаторов
from .analyzers.ast_advanced import AdvancedASTAnalyzer
from .analyzers.runtime_profiler import RuntimeProfiler
from .analyzers.cfg_analyzer import CFGComplexityAnalyzer
from .analyzers.dynamic_tracer import DynamicComplexityTracer
from .metrics.calculator import UniversalMetricsCalculator
from .analyzers.tools_profiler import ToolsIntegrationAnalyzer
from .analyzers.hybrid_ensemble import HybridComplexityAnalyzer

__version__ = "0.2.0"  # Обновлена версия для CFG v2.0

# Экспорт основных классов
__all__ = [
    'BaseComplexityAnalyzer', 'AnalysisContext', 'ComplexityResult', 'ComplexityMetrics',
    'ComplexityClass', 'AnalyzerType', 'create_analyzer', 'get_available_analyzers',
    'AdvancedASTAnalyzer', 'RuntimeProfiler', 'CFGComplexityAnalyzer',
    'MLComplexityPredictor', 'DynamicComplexityTracer', 'UniversalMetricsCalculator',
    'ToolsIntegrationAnalyzer', 'HybridComplexityAnalyzer'
]

def create_analyzer(analyzer_type: str, config: dict = None) -> BaseComplexityAnalyzer:
    """
    Фабричная функция для создания анализаторов по их строковому имени.
    
    Args:
        analyzer_type: Тип анализатора
        config: Конфигурация анализатора
        
    Returns:
        Экземпляр анализатора
    """
    analyzer_map = {
        'ast_advanced': AdvancedASTAnalyzer,
        'runtime_profiler': RuntimeProfiler,
        'cfg_analyzer': CFGComplexityAnalyzer,  # CFG v2.0
        'ml_predictor': MLComplexityPredictor,
        'dynamic_tracer': DynamicComplexityTracer,
        'metrics_calculator': UniversalMetricsCalculator,
        'tools_integration': ToolsIntegrationAnalyzer,
        'hybrid_ensemble': HybridComplexityAnalyzer
    }
    
    analyzer_class = analyzer_map.get(analyzer_type)
    
    if analyzer_class is None:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}")
    
    analyzer = analyzer_class()
    
    if config and hasattr(analyzer, 'initialize'):
        analyzer.initialize(config)
    
    return analyzer

def get_available_analyzers() -> list:
    """Получение списка доступных анализаторов"""
    analyzer_map = {
        'ast_advanced': AdvancedASTAnalyzer,
        'runtime_profiler': RuntimeProfiler,
        'cfg_analyzer': CFGComplexityAnalyzer,
        'ml_predictor': MLComplexityPredictor,
        'dynamic_tracer': DynamicComplexityTracer,
        'metrics_calculator': UniversalMetricsCalculator,
        'tools_integration': ToolsIntegrationAnalyzer,
        'hybrid_ensemble': HybridComplexityAnalyzer
    }
    return list(analyzer_map.keys())
