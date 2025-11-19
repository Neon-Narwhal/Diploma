# complexity_analyzers/analyzers/__init__.py

"""
Пакет анализаторов сложности.
Содержит все реализованные анализаторы.
"""

# Базовые анализаторы
from .ast_advanced import AdvancedASTAnalyzer
#from .ast_basic import ASTBasicAnalyzer
from .runtime_profiler import RuntimeProfiler
from .cfg_analyzer import CFGComplexityAnalyzer
from .ml_predictor import MLComplexityPredictor
from .dynamic_tracer import DynamicComplexityTracer
from .hybrid_ensemble import HybridComplexityAnalyzer

# CFG v2.0 компоненты
"""from .cfg_builder import PythonCFGBuilder
from .cfg_data_flow import DataFlowAnalyzer
from .cfg_iterator_analysis import IteratorRangeAnalyzer
from .cfg_library_calls import LibraryCallRecognizer
from .cfg_multi_variable import MultiVariableTracker
from .cfg_complexity_composer import ComplexityComposer"""

# Экспорт
__all__ = [
    # Основные анализаторы
    'AdvancedASTAnalyzer',
    'ASTBasicAnalyzer',
    'RuntimeProfiler',
    'CFGComplexityAnalyzer',
    'MLComplexityPredictor',
    'DynamicComplexityTracer',
    'HybridComplexityAnalyzer',
    
    # CFG v2.0 компоненты
    'CFGBuilder',
    'DataFlowAnalyzer',
    'IteratorRangeAnalyzer',
    'LibraryCallRecognizer',
    'MultiVariableTracker',
    'ComplexityComposer',
]
