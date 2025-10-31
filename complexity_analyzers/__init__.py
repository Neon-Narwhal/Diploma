# complexity_analyzers/__init__.py

"""
Пакет анализаторов сложности алгоритмов.
Регистрирует все доступные анализаторы в фабричной функции.
"""

# Импорты базовых классов
from .base.enums import ComplexityClass, AnalyzerType
from .base.analyzer import BaseComplexityAnalyzer, AnalysisContext
from .base.result import ComplexityResult, ComplexityMetrics

# ИСПРАВЛЕНО: Импорты ВСЕХ основных классов анализаторов
from .ast_analyzers.advanced_analyzer import AdvancedASTAnalyzer
from .runtime.profiler import RuntimeProfiler
from .cfg.analyzer import CFGComplexityAnalyzer
from .ml.predictor import MLComplexityPredictor
from .dynamic.tracer import DynamicComplexityTracer
from .metrics.calculator import UniversalMetricsCalculator
from .tools.profiler_tools import ToolsIntegrationAnalyzer
from .hybrid.ensemble import HybridComplexityAnalyzer

__version__ = "0.1.0"

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
    """
    # ИСПРАВЛЕНО: Добавлены все реальные анализаторы в карту
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
    
    analyzer_class = analyzer_map.get(analyzer_type)
    
    if analyzer_class is None:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}")
    
    analyzer = analyzer_class()
    
    if config and hasattr(analyzer, 'initialize'):
        analyzer.initialize(config)
    
    return analyzer

def get_available_analyzers() -> list:
    """Получение списка доступных анализаторов"""
    # Теперь можно динамически получать ключи из карты
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

