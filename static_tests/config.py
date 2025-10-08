"""
Конфигурация для анализа временной сложности кода.
Содержит маппинги между метриками инструментов и классами временной сложности.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ComplexityClass(Enum):
    """Классы временной сложности"""
    CONSTANT = 'constant'      # O(1)
    LOGN = 'logn'             # O(log n)
    LINEAR = 'linear'          # O(n)
    NLOGN = 'nlogn'           # O(n log n)
    QUADRATIC = 'quadratic'    # O(n²)
    CUBIC = 'cubic'           # O(n³)
    EXPONENTIAL = 'exponential' # O(2^n)


@dataclass
class ToolConfig:
    """Конфигурация для конкретного инструмента анализа"""
    name: str
    enabled: bool = True
    threshold_rules: Optional[Dict] = None
    custom_mapper: Optional[callable] = None


# ============= ПАРАМЕТРЫ ПРОЕКТА =============
# Определяем базовые пути проекта
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'static_tests/results'

# Пути к датасетам по умолчанию
DEFAULT_PYTHON_DATASET = DATA_DIR / 'python_data.jsonl'
DEFAULT_JAVA_DATASET = DATA_DIR / 'java_dataset.jsonl'


# ============= ПАРАМЕТРЫ ЗАПУСКА =============
# Настройте эти параметры для запуска анализа

# Входной файл (None = использовать DEFAULT_PYTHON_DATASET или DEFAULT_JAVA_DATASET)
INPUT_FILE = None

# Выходная директория (None = использовать RESULTS_DIR)
OUTPUT_DIR = None

# Инструменты для запуска (None = все включенные)
# Варианты: ['radon'], ['lizard'], ['radon', 'lizard'], None и т.д.
TOOLS_TO_RUN = None  # Измените на нужные инструменты

# Максимальное количество образцов (None = все)
MAX_SAMPLES = None  # Для тестирования, установите None для полного анализа

# Язык программирования ('python' или 'java')
LANGUAGE = 'python'

# Доступные инструменты для запуска
AVAILABLE_TOOLS = ['radon', 'lizard', 'mccabe', 'complexipy', "Halstead", "Maintainability Index", "Nested Block Depth"]

ANALYZER_MODE = 'enhanced'  # Измените на 'baseline' при необходимости

# ============= ПРАВИЛА МАППИНГА =============
# Эвристические правила для маппинга цикломатической сложности в временную
CYCLOMATIC_TO_TIME_COMPLEXITY = {
    'radon': {
        (1, 5): ComplexityClass.CONSTANT,
        (6, 10): ComplexityClass.LINEAR,
        (11, 20): ComplexityClass.QUADRATIC,
        (21, 50): ComplexityClass.CUBIC,
        (51, float('inf')): ComplexityClass.EXPONENTIAL
    },
    'mccabe': {
        (1, 5): ComplexityClass.CONSTANT,
        (6, 10): ComplexityClass.LINEAR,
        (11, 20): ComplexityClass.QUADRATIC,
        (21, 50): ComplexityClass.CUBIC,
        (51, float('inf')): ComplexityClass.EXPONENTIAL
    },
    'lizard': {
        (1, 5): ComplexityClass.CONSTANT,
        (6, 10): ComplexityClass.LINEAR,
        (11, 20): ComplexityClass.QUADRATIC,
        (21, 50): ComplexityClass.CUBIC,
        (51, float('inf')): ComplexityClass.EXPONENTIAL
    }
}

# Эвристические правила для когнитивной сложности
COGNITIVE_TO_TIME_COMPLEXITY = {
    'complexipy': {
        (0, 5): ComplexityClass.CONSTANT,
        (6, 15): ComplexityClass.LINEAR,
        (16, 30): ComplexityClass.QUADRATIC,
        (31, 60): ComplexityClass.CUBIC,
        (61, float('inf')): ComplexityClass.EXPONENTIAL
    },
    'lizard': {
        (0, 5): ComplexityClass.CONSTANT,
        (6, 15): ComplexityClass.LINEAR,
        (16, 30): ComplexityClass.QUADRATIC,
        (31, 60): ComplexityClass.CUBIC,
        (61, float('inf')): ComplexityClass.EXPONENTIAL
    }
}

# Добавляем правила маппинга для Halstead
HALSTEAD_TO_TIME_COMPLEXITY = {
    'halstead': {
        (0, 5): ComplexityClass.CONSTANT,
        (5, 10): ComplexityClass.LINEAR,
        (10, 20): ComplexityClass.QUADRATIC,
        (20, 40): ComplexityClass.CUBIC,
        (40, float('inf')): ComplexityClass.EXPONENTIAL
    }
}

# Добавляем правила для Maintainability Index (обратная корреляция!)
MI_TO_TIME_COMPLEXITY = {
    'mi': {
        (80, 101): ComplexityClass.CONSTANT,    # Высокий MI = простой код
        (60, 80): ComplexityClass.LINEAR,
        (40, 60): ComplexityClass.QUADRATIC,
        (20, 40): ComplexityClass.CUBIC,
        (0, 20): ComplexityClass.EXPONENTIAL
    }
}

# Правила для Nested Block Depth
NBD_TO_TIME_COMPLEXITY = {
    'nbd': {
        (0, 1): ComplexityClass.CONSTANT,
        (1, 2): ComplexityClass.LINEAR,
        (2, 3): ComplexityClass.QUADRATIC,
        (3, 4): ComplexityClass.CUBIC,
        (4, float('inf')): ComplexityClass.EXPONENTIAL
    }
}

# Инструменты и их возможности
TOOLS_REGISTRY = {
    'radon': ToolConfig(
        name='radon',
        enabled=True,
        threshold_rules=CYCLOMATIC_TO_TIME_COMPLEXITY['radon']
    ),
    'mccabe': ToolConfig(
        name='mccabe',
        enabled=True,
        threshold_rules=CYCLOMATIC_TO_TIME_COMPLEXITY['mccabe']
    ),
    'lizard': ToolConfig(
        name='lizard',
        enabled=True,
        threshold_rules=CYCLOMATIC_TO_TIME_COMPLEXITY['lizard']
    ),
    'complexipy': ToolConfig(
        name='complexipy',
        enabled=True,
        threshold_rules=COGNITIVE_TO_TIME_COMPLEXITY['complexipy']
    ),
    'big_o': ToolConfig(
        name='big_o',
        enabled=False,  # Требует запуска кода
    ),
    'trend_profiler': ToolConfig(
        name='trend_profiler',
        enabled=False,  # Требует запуска кода
    ),
    'halstead': ToolConfig(
        name='halstead', 
        enabled=True),
    'mi': ToolConfig(
        name='mi', 
        enabled=True),
    'nbd': ToolConfig(
        name='nbd', 
        enabled=True),
}
