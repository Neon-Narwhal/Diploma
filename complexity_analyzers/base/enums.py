"""Перечисления и константы для анализаторов сложности"""
from enum import Enum, IntEnum
from typing import Dict, Tuple

class ComplexityClass(Enum):
    """Классы временной сложности с нотацией и порядком"""
    CONSTANT = ("O(1)", 1, "constant")
    LOGARITHMIC = ("O(log n)", 2, "logarithmic") 
    LINEAR = ("O(n)", 3, "linear")
    LINEARITHMIC = ("O(n log n)", 4, "linearithmic")
    QUADRATIC = ("O(n²)", 5, "quadratic")
    CUBIC = ("O(n³)", 6, "cubic")
    POLYNOMIAL = ("O(n^k)", 7, "polynomial")
    EXPONENTIAL = ("O(2^n)", 8, "exponential")
    FACTORIAL = ("O(n!)", 9, "factorial")
    UNKNOWN = ("O(?)", 0, "unknown")
    
    def __init__(self, notation: str, complexity_order: int, name: str):
        self.notation = notation
        self.complexity_order = complexity_order
        self.name = name
    
    def __lt__(self, other):
        if isinstance(other, ComplexityClass):
            return self.complexity_order < other.complexity_order
        return NotImplemented
    
    def __le__(self, other):
        if isinstance(other, ComplexityClass):
            return self.complexity_order <= other.complexity_order
        return NotImplemented
    
    def __gt__(self, other):
        if isinstance(other, ComplexityClass):
            return self.complexity_order > other.complexity_order
        return NotImplemented
    
    def __ge__(self, other):
        if isinstance(other, ComplexityClass):
            return self.complexity_order >= other.complexity_order
        return NotImplemented

class AnalyzerType(Enum):
    """Типы анализаторов"""
    STATIC_AST = "static_ast"
    RUNTIME_PROFILER = "runtime_profiler"
    CFG_ANALYZER = "cfg_analyzer"
    ML_PREDICTOR = "ml_predictor"
    DYNAMIC_TRACER = "dynamic_tracer"
    METRICS_CALCULATOR = "metrics_calculator"
    TOOLS_INTEGRATION = "tools_integration"
    HYBRID_ENSEMBLE = "hybrid_ensemble"

class AnalyzerStatus(Enum):
    """Статусы анализатора"""
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ANALYZING = "analyzing"
    ERROR = "error"
    DISABLED = "disabled"

class ConfidenceLevel(IntEnum):
    """Уровни уверенности"""
    VERY_LOW = 1    # 0.0 - 0.2
    LOW = 2         # 0.2 - 0.4  
    MEDIUM = 3      # 0.4 - 0.6
    HIGH = 4        # 0.6 - 0.8
    VERY_HIGH = 5   # 0.8 - 1.0
    
    @classmethod
    def from_confidence(cls, confidence: float) -> 'ConfidenceLevel':
        """Преобразование числового значения в уровень"""
        if confidence < 0.2:
            return cls.VERY_LOW
        elif confidence < 0.4:
            return cls.LOW
        elif confidence < 0.6:
            return cls.MEDIUM
        elif confidence < 0.8:
            return cls.HIGH
        else:
            return cls.VERY_HIGH

class PatternType(Enum):
    """Типы алгоритмических паттернов"""
    SORTING = "sorting"
    SEARCHING = "searching"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    DIVIDE_CONQUER = "divide_conquer"
    GREEDY = "greedy"
    GRAPH_TRAVERSAL = "graph_traversal"
    RECURSIVE = "recursive"
    ITERATIVE = "iterative"
    BACKTRACKING = "backtracking"
    UNKNOWN = "unknown"

class DataStructureUsage(Enum):
    """Использование структур данных"""
    ARRAY_LIST = "array_list"
    DICTIONARY_HASH = "dictionary_hash"
    SET = "set"
    STACK = "stack"
    QUEUE = "queue"
    HEAP = "heap"
    TREE = "tree"
    GRAPH = "graph"
    LINKED_LIST = "linked_list"

# Константы для маппинга метрик
CYCLOMATIC_TO_COMPLEXITY: Dict[Tuple[int, int], ComplexityClass] = {
    (1, 5): ComplexityClass.CONSTANT,
    (6, 10): ComplexityClass.LINEAR,
    (11, 20): ComplexityClass.QUADRATIC,
    (21, 50): ComplexityClass.CUBIC,
    (51, float('inf')): ComplexityClass.EXPONENTIAL
}

NESTING_TO_COMPLEXITY: Dict[int, ComplexityClass] = {
    0: ComplexityClass.CONSTANT,
    1: ComplexityClass.LINEAR,
    2: ComplexityClass.QUADRATIC,
    3: ComplexityClass.CUBIC,
}

# Пороговые значения
CONFIDENCE_THRESHOLDS = {
    'MIN_ACCEPTABLE': 0.3,
    'GOOD': 0.7,
    'EXCELLENT': 0.9
}

ANALYSIS_TIMEOUTS = {
    'FAST': 5,      # секунд
    'NORMAL': 30,   # секунд  
    'SLOW': 120,    # секунд
    'VERY_SLOW': 300 # секунд
}

# Приоритеты анализаторов
ANALYZER_PRIORITIES = {
    AnalyzerType.STATIC_AST: 1,
    AnalyzerType.METRICS_CALCULATOR: 2,
    AnalyzerType.CFG_ANALYZER: 3,
    AnalyzerType.ML_PREDICTOR: 4,
    AnalyzerType.RUNTIME_PROFILER: 5,
    AnalyzerType.DYNAMIC_TRACER: 6,
    AnalyzerType.TOOLS_INTEGRATION: 7,
    AnalyzerType.HYBRID_ENSEMBLE: 8
}
