"""Перечисления для системы анализа сложности"""
from enum import Enum, IntEnum
from typing import Tuple, Dict


class ComplexityClass(Enum):
    """
    Классы временной сложности с нотацией Big-O.
    
    Формат: (class_name, notation, description, order)
    """
    CONSTANT = ('constant', 'O(1)', 'Константная сложность', 1)
    LOGARITHMIC = ('logarithmic', 'O(logn)', 'Логарифмическая сложность', 2)
    LINEAR = ('linear', 'O(n)', 'Линейная сложность', 3)
    LINEARITHMIC = ('linearithmic', 'O(nlogn)', 'Линеаритмическая сложность', 4)
    QUADRATIC = ('quadratic', 'O(n^2)', 'Квадратичная сложность', 5)
    CUBIC = ('cubic', 'O(n^3)', 'Кубическая сложность', 6)
    POLYNOMIAL = ('polynomial', 'O(n^k)', 'Полиномиальная сложность', 7)
    EXPONENTIAL = ('exponential', 'O(2^n)', 'Экспоненциальная сложность', 8)
    FACTORIAL = ('factorial', 'O(n!)', 'Факториальная сложность', 9)
    UNKNOWN = ('unknown', 'O(?)', 'Неизвестная сложность', 0)
    
    def __init__(self, class_name: str, notation: str, description: str, complexity_order: int):
        self.class_name = class_name
        self.notation = notation
        self.description = description
        self.complexity_order = complexity_order
    
    def to_notation(self) -> str:
        """Возвращает Big-O нотацию"""
        return self.notation
    
    def to_class_name(self) -> str:
        """Возвращает внутреннее имя класса"""
        return self.class_name
    
    @classmethod
    def from_notation(cls, notation: str) -> 'ComplexityClass':
        """Создает из нотации O(...)"""
        for member in cls:
            if member.notation == notation:
                return member
        return cls.UNKNOWN
    
    @classmethod
    def from_class_name(cls, class_name: str) -> 'ComplexityClass':
        """Создает из имени класса"""
        for member in cls:
            if member.class_name == class_name:
                return member
        return cls.UNKNOWN
    
    def __lt__(self, other):
        """Сравнение по порядку сложности"""
        if not isinstance(other, ComplexityClass):
            return NotImplemented
        return self.complexity_order < other.complexity_order
    
    def __str__(self):
        return self.notation
    
    def __repr__(self):
        return f"ComplexityClass.{self.name}"


class AnalyzerType(Enum):
    """Типы анализаторов (ЕДИНАЯ ВЕРСИЯ)"""
    AST = 'ast'
    CFG = 'cfg'
    RUNTIME = 'runtime'
    ML = 'ml'
    HYBRID = 'hybrid'
    METRICS = 'metrics'
    DYNAMIC = 'dynamic'


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
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5
    
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


# Пороговые значения
CONFIDENCE_THRESHOLDS = {
    'MIN_ACCEPTABLE': 0.3,
    'GOOD': 0.7,
    'EXCELLENT': 0.9
}

ANALYSIS_TIMEOUTS = {
    'FAST': 5,
    'NORMAL': 30,
    'SLOW': 120,
    'VERY_SLOW': 300
}

ANALYZER_PRIORITIES = {
    AnalyzerType.AST: 1,
    AnalyzerType.METRICS: 2,
    AnalyzerType.CFG: 3,
    AnalyzerType.ML: 4,
    AnalyzerType.RUNTIME: 5,
    AnalyzerType.DYNAMIC: 6,
    AnalyzerType.HYBRID: 7
}
