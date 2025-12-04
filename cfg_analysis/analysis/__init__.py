"""
Модуль анализа CFG: метрики, циклы и поток данных.
"""

from .complexity import CFGComplexityMetrics
from .loops import LoopAnalyzer
from .flow import DataFlowAnalyzer

__all__ = [
    'CFGComplexityMetrics',
    'LoopAnalyzer',
    'DataFlowAnalyzer',
]
