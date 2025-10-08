"""
Пакет для анализа временной сложности кода и оценки метрик качества.

Этот пакет предоставляет инструменты для:
- Анализа кода различными инструментами (Radon, McCabe, Lizard, Complexipy)
- Маппинга метрик сложности в классы временной сложности
- Вычисления метрик качества (precision, recall, f1, accuracy)
- Обработки JSONL датасетов
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Импортируем основные классы для удобного доступа
from .config import (
    ComplexityClass,
    ToolConfig,
    TOOLS_REGISTRY,
    CYCLOMATIC_TO_TIME_COMPLEXITY,
    COGNITIVE_TO_TIME_COMPLEXITY
)

from .analyzers_baseline import (
    BaselineAnalyzer,
    RadonBaselineAnalyzer,
    McCabeBaselineAnalyzer,
    LizardBaselineAnalyzer,
    ComplexipyBaselineAnalyzer,
    McCabeBaselineAnalyzer,
    HalsteadAnalyzer,
    MaintainabilityIndexAnalyzer,
    NestedBlockDepthAnalyzer,
    get_baseline_analyzer,
    BASELINE_ANALYZER_FACTORY,
)

from .analyzers_enhanced import (
    RadonEnhancedAnalyzer,
    LizardEnhancedAnalyzer,
    McCabeEnhancedAnalyzer,
    ComplexipyEnhancedAnalyzer,
    HalsteadEnhancedAnalyzer,
    MIEnhancedAnalyzer,
    NBDEnhancedAnalyzer,
    get_enhanced_analyzer,
    ENHANCED_ANALYZER_FACTORY,
)

from .analyzers import (
    BaseAnalyzer,
    RadonAnalyzer,
    McCabeAnalyzer,
    LizardAnalyzer,
    ComplexipyAnalyzer,
    WilyAnalyzer,
    get_analyzer,
    ANALYZER_FACTORY
)




from .metrics_calculator import MetricsCalculator

from .dataset_processor import DatasetProcessor

# Определяем публичный API пакета
__all__ = [
    # Версия
    '__version__',
    '__author__',
    
    # Конфигурация
    'ComplexityClass',
    'ToolConfig',
    'TOOLS_REGISTRY',
    'CYCLOMATIC_TO_TIME_COMPLEXITY',
    'COGNITIVE_TO_TIME_COMPLEXITY'
    
    # Анализаторы
    "BaselineAnalyzer",
    "RadonBaselineAnalyzer",
    "McCabeBaselineAnalyzer",
    "LizardBaselineAnalyzer",
    "ComplexipyBaselineAnalyzer",
    "McCabeBaselineAnalyzer",
    "HalsteadAnalyzer",
    "MaintainabilityIndexAnalyzer",
    "NestedBlockDepthAnalyzer",
    "get_baseline_analyzer",
    "BASELINE_ANALYZER_FACTORY" 

    "RadonEnhancedAnalyzer",
    "LizardEnhancedAnalyzer",
    "McCabeEnhancedAnalyzer",
    "ComplexipyEnhancedAnalyzer",
    "HalsteadEnhancedAnalyzer",
    "MIEnhancedAnalyzer",
    "NBDEnhancedAnalyzer",
    "get_enhanced_analyzer",
    "ENHANCED_ANALYZER_FACTORY",
    
    # Метрики
    'MetricsCalculator',
    
    # Процессор
    'DatasetProcessor',
]
