"""
Конфигурация для анализа сложности алгоритмов через complexity_analyzers.
Полный аналог static_tests/config.py, но для новой системы анализа.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ComplexityClass(Enum):
    """Классы временной сложности (унифицировано с complexity_analyzers)"""
    CONSTANT = 'constant'       # O(1)
    LOGARITHMIC = 'logarithmic' # O(log n)
    LINEAR = 'linear'           # O(n)
    LINEARITHMIC = 'linearithmic' # O(n log n)
    QUADRATIC = 'quadratic'     # O(n²)
    CUBIC = 'cubic'             # O(n³)
    POLYNOMIAL = 'polynomial'   # O(n^k)
    EXPONENTIAL = 'exponential' # O(2^n)
    FACTORIAL = 'factorial'     # O(n!)
    UNKNOWN = 'unknown'         # O(?)


@dataclass
class AnalyzerConfig:
    """Конфигурация для конкретного анализатора"""
    name: str
    enabled: bool = True
    timeout: int = 60
    config_params: Optional[Dict] = None


# ============= ПАРАМЕТРЫ ПРОЕКТА =============
# Определяем базовые пути проекта
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'complexity_analyzers/results'

# Пути к тестовым данным
DEFAULT_PYTHON_DATASET = DATA_DIR / 'python_data.jsonl'


# ============= ПАРАМЕТРЫ ЗАПУСКА =============
# Настройте эти параметры для запуска анализа

# Входной файл или директория (None = использовать DEFAULT_TEST_DIR)
INPUT_PATH = None  # Например: Path("my_algorithms/")

# Выходная директория (None = использовать RESULTS_DIR)
OUTPUT_DIR = None

# Анализаторы для запуска (None = все включенные)
# Варианты: ['ast_advanced'], ['runtime_profiler'], ['ast_advanced', 'hybrid_ensemble'], None
ANALYZERS_TO_RUN = [
    'ast_advanced',      # Быстрый и работающий
    'cfg_analyzer',      # CFG анализатор  
    'ml_predictor',      # ML предиктор
    'hybrid_ensemble'    # Гибридный ансамбль
    # 'runtime_profiler' - ИСКЛЮЧЕН (медленный)
]

# Максимальное количество файлов (None = все)
MAX_SAMPLES = None  # Для тестирования, установите небольшое число

# Максимальное время на файл (секунды)
MAX_TIME_PER_FILE = 60

# Язык программирования
LANGUAGE = 'python'

# Режим анализа
ANALYSIS_MODE = 'comparison'  # 'single', 'comparison', 'benchmark'

# Минимальная уверенность для принятия результата
MIN_CONFIDENCE = 0.3

# Подробные логи
VERBOSE = True

# Сохранять промежуточные результаты
SAVE_INTERMEDIATE = True

# Форматы вывода
OUTPUT_FORMATS = ['json', 'csv']

# ============= ДОСТУПНЫЕ АНАЛИЗАТОРЫ =============

# Все доступные анализаторы из complexity_analyzers
AVAILABLE_ANALYZERS = [
    'ast_basic',           # Базовый AST
    'ast_advanced',        # Продвинутый AST
    'runtime_profiler',    # Runtime профайлер
    'cfg_analyzer',        # CFG анализатор
    'ml_predictor',        # ML предиктор
    'dynamic_tracer',      # Динамическая трассировка
    'metrics_calculator',  # Метрический анализатор
    'tools_integration',   # Внешние инструменты
    'hybrid_ensemble'      # Гибридный ансамбль
]


# Маппинг для сравнения predicted и true values
COMPLEXITY_MAPPING = {
    # True values (из датасета)
    'constant': 'O(1)',
    'logarithmic': 'O(log n)', 
    'linear': 'O(n)',
    'linearithmic': 'O(n log n)',
    'quadratic': 'O(n²)',
    'cubic': 'O(n³)',
    'exponential': 'O(2^n)',
    'factorial': 'O(n!)'
}

# Реестр анализаторов с конфигурациями
ANALYZERS_REGISTRY = {
    'ast_basic': AnalyzerConfig(
        name='ast_basic',
        enabled=True,
        timeout=30,
        config_params={
            'enable_pattern_detection': True,
            'pattern_types': ['loops', 'recursion', 'conditions']
        }
    ),
    'ast_advanced': AnalyzerConfig(
        name='ast_advanced',
        enabled=True,
        timeout=45,
        config_params={
            'enable_pattern_detection': True,
            'enable_feature_extraction': True,
            'pattern_detectors': ['sorting', 'search', 'dp', 'recursive'],
            'feature_extractors': ['basic', 'complexity', 'textual']
        }
    ),
    'runtime_profiler': AnalyzerConfig(
        name='runtime_profiler',
        enabled=True,
        timeout=120,
        config_params={
            'test_sizes': [10, 50, 100, 500, 1000],
            'iterations_per_size': 3,
            'use_subprocess': True,
            'measure_memory': True,
            'curve_fitting_method': 'least_squares'
        }
    ),
    'cfg_analyzer': AnalyzerConfig(
        name='cfg_analyzer',
        enabled=True,
        timeout=60,
        config_params={
            'include_exception_edges': True,
            'calculate_dominance': False,  # Может быть медленным
            'metrics_to_calculate': ['cyclomatic_complexity', 'nesting_depth']
        }
    ),
    'ml_predictor': AnalyzerConfig(
        name='ml_predictor',
        enabled=True,
        timeout=30,
        config_params={
            'models_to_use': ['random_forest', 'xgboost'],
            'feature_selection': True,
            'confidence_threshold': 0.6
        }
    ),
    'dynamic_tracer': AnalyzerConfig(
        name='dynamic_tracer',
        enabled=False,  # По умолчанию отключен (небезопасный)
        timeout=90,
        config_params={
            'trace_method': 'safe_subprocess',
            'trace_timeout': 10,
            'max_recursion_depth': 100,
            'test_data_types': ['list', 'matrix']
        }
    ),
    'metrics_calculator': AnalyzerConfig(
        name='metrics_calculator',
        enabled=True,
        timeout=30,
        config_params={
            'calculators': ['radon', 'mccabe', 'custom'],
            'include_halstead': True,
            'include_maintainability': True
        }
    ),
    'tools_integration': AnalyzerConfig(
        name='tools_integration',
        enabled=False,  # Требует установки внешних инструментов
        timeout=60,
        config_params={
            'tools': ['py-spy', 'line_profiler', 'memory_profiler'],
            'profile_duration': 5
        }
    ),
    'hybrid_ensemble': AnalyzerConfig(
        name='hybrid_ensemble',
        enabled=True,
        timeout=180,
        config_params={
            'enabled_analyzers': ['ast_advanced', 'runtime_profiler', 'cfg_analyzer', 'ml_predictor'],
            'voting_strategy': 'weighted',
            'weighting_strategy': 'confidence_based',
            'min_analyzers_required': 2
        }
    )
}

# ============= ПРЕДУСТАНОВКИ =============

def apply_preset(preset_name: str):
    """Применить предустановку конфигурации"""
    global ANALYZERS_TO_RUN, ANALYSIS_MODE, MAX_TIME_PER_FILE, OUTPUT_FORMATS, VERBOSE
    
    if preset_name == 'quick_test':
        # Быстрый тест одним анализатором
        ANALYZERS_TO_RUN = ['ast_advanced']
        ANALYSIS_MODE = 'single'
        MAX_TIME_PER_FILE = 15
        OUTPUT_FORMATS = ['json']
        VERBOSE = True
        
    elif preset_name == 'full_comparison':
        # Полное сравнение всех основных анализаторов
        ANALYZERS_TO_RUN = ['ast_advanced', 'runtime_profiler', 'cfg_analyzer', 'ml_predictor', 'hybrid_ensemble']
        ANALYSIS_MODE = 'comparison'
        MAX_TIME_PER_FILE = 120
        OUTPUT_FORMATS = ['json', 'csv']
        VERBOSE = True
        
    elif preset_name == 'production':
        # Продакшн режим - только гибридный анализатор
        ANALYZERS_TO_RUN = ['hybrid_ensemble']
        ANALYSIS_MODE = 'single'
        MAX_TIME_PER_FILE = 60
        OUTPUT_FORMATS = ['json']
        VERBOSE = False
        
    elif preset_name == 'research':
        # Исследовательский режим - все анализаторы
        ANALYZERS_TO_RUN = [name for name, config in ANALYZERS_REGISTRY.items() if config.enabled]
        ANALYSIS_MODE = 'comparison'
        MAX_TIME_PER_FILE = 180
        OUTPUT_FORMATS = ['json', 'csv']
        VERBOSE = True
        
    elif preset_name == 'baseline':
        # Базовый тест
        ANALYZERS_TO_RUN = ['ast_basic', 'metrics_calculator']
        ANALYSIS_MODE = 'comparison'
        MAX_TIME_PER_FILE = 30
        OUTPUT_FORMATS = ['json', 'csv']
        VERBOSE = True

# Применить предустановку (раскомментируй нужную)
# apply_preset('quick_test')
# apply_preset('full_comparison')
# apply_preset('production')
# apply_preset('research')
# apply_preset('baseline')

# ============= ФИЛЬТРЫ И ОГРАНИЧЕНИЯ =============

# Фильтрация файлов
FILE_SIZE_LIMIT = 1 * 1024 * 1024  # 1MB максимум на файл
EXCLUDED_PATTERNS = [
    'test_*', '*_test.py', 'conftest.py',
    '__pycache__', '.git', '.pytest_cache',
    'venv', 'env', '.venv'
]

# Рекурсивный поиск в директориях
RECURSIVE_SEARCH = True

# Параллельная обработка
USE_MULTIPROCESSING = True
MAX_WORKERS = 4

# ============= МАППИНГ РЕЗУЛЬТАТОВ =============

# Маппинг между внутренними классами сложности и отображением
COMPLEXITY_DISPLAY_MAPPING = {
    ComplexityClass.CONSTANT: 'O(1)',
    ComplexityClass.LOGARITHMIC: 'O(log n)',
    ComplexityClass.LINEAR: 'O(n)',
    ComplexityClass.LINEARITHMIC: 'O(n log n)',
    ComplexityClass.QUADRATIC: 'O(n²)',
    ComplexityClass.CUBIC: 'O(n³)',
    ComplexityClass.POLYNOMIAL: 'O(n^k)',
    ComplexityClass.EXPONENTIAL: 'O(2^n)',
    ComplexityClass.FACTORIAL: 'O(n!)',
    ComplexityClass.UNKNOWN: 'O(?)'
}

# ============= ГЕНЕРАЦИЯ ТЕСТОВЫХ ФАЙЛОВ =============

# Примеры алгоритмов для автогенерации тестов
SAMPLE_ALGORITHMS = {
    'bubble_sort': {
        'complexity': ComplexityClass.QUADRATIC,
        'code': '''
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
'''
    },
    'binary_search': {
        'complexity': ComplexityClass.LOGARITHMIC,
        'code': '''
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
'''
    },
    'linear_search': {
        'complexity': ComplexityClass.LINEAR,
        'code': '''
def linear_search(arr, target):
    for i, item in enumerate(arr):
        if item == target:
            return i
    return -1
'''
    },
    'merge_sort': {
        'complexity': ComplexityClass.LINEARITHMIC,
        'code': '''
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
'''
    },
    'fibonacci_recursive': {
        'complexity': ComplexityClass.EXPONENTIAL,
        'code': '''
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)
'''
    }
}

# ============= ЛОГИРОВАНИЕ =============

# Уровень логирования
LOG_LEVEL = 'INFO' if VERBOSE else 'WARNING'

# Сохранение логов в файл
LOG_TO_FILE = True
