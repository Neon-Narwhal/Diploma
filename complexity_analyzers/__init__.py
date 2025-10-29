"""
Пакет анализаторов сложности алгоритмов

Включает различные методы анализа временной сложности:
- Статический анализ AST
- Профилирование времени выполнения
- Анализ графа потока управления (CFG)
- Машинное обучение
- Динамическая трассировка
- Метрический анализ
- Гибридные подходы
"""

from complexity_analyzers.base.enums import ComplexityClass, AnalyzerType
from complexity_analyzers.base.analyzer import BaseComplexityAnalyzer, AnalysisContext
from complexity_analyzers.base.result import ComplexityResult, ComplexityMetrics

# Основные анализаторы
from complexity_analyzers.ast.advanced_analyzer import AdvancedASTAnalyzer
from complexity_analyzers.runtime.profiler import RuntimeProfiler
from complexity_analyzers.cfg.analyzer import CFGComplexityAnalyzer
from complexity_analyzers.ml.predictor import MLComplexityPredictor
from complexity_analyzers.dynamic.tracer import DynamicComplexityTracer
from complexity_analyzers.metrics.calculator import UniversalMetricsCalculator
from complexity_analyzers.hybrid.ensemble import HybridComplexityAnalyzer, EnsembleFactory

# Версия пакета
__version__ = "0.1.0"

# Экспорт основных классов
__all__ = [
    # Базовые классы
    'ComplexityClass',
    'AnalyzerType', 
    'BaseComplexityAnalyzer',
    'AnalysisContext',
    'ComplexityResult',
    'ComplexityMetrics',
    
    # Анализаторы
    'AdvancedASTAnalyzer',
    'RuntimeProfiler',
    'CFGComplexityAnalyzer',
    'MLComplexityPredictor',
    'DynamicComplexityTracer',
    'UniversalMetricsCalculator',
    'HybridComplexityAnalyzer',
    
    # Фабрики
    'EnsembleFactory',
    
    # Вспомогательные функции
    'create_analyzer',
    'get_available_analyzers',
]

def create_analyzer(analyzer_type: str, config: dict = None) -> BaseComplexityAnalyzer:
    """
    Фабричная функция для создания анализаторов
    
    Args:
        analyzer_type: Тип анализатора ('ast', 'runtime', 'cfg', 'ml', 'dynamic', 'hybrid')
        config: Конфигурация анализатора
    
    Returns:
        Экземпляр анализатора
    """
    analyzer_map = {
        'ast': AdvancedASTAnalyzer,
        'runtime': RuntimeProfiler,
        'cfg': CFGComplexityAnalyzer,
        'ml': MLComplexityPredictor,
        'dynamic': DynamicComplexityTracer,
        'hybrid': HybridComplexityAnalyzer,
    }
    
    if analyzer_type not in analyzer_map:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}")
    
    analyzer_class = analyzer_map[analyzer_type]
    analyzer = analyzer_class()
    
    if config:
        analyzer.initialize(config)
    
    return analyzer

def get_available_analyzers() -> list:
    """Получение списка доступных анализаторов"""
    from complexity_analyzers.base.analyzer import AnalyzerFactory
    return AnalyzerFactory.get_available_analyzers()
