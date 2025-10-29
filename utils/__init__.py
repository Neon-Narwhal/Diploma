# =============================================================================
# utils/__init__.py
# Центральный модуль утилит проекта Diploma
# =============================================================================

"""
Утилиты для проекта Diploma.

Структура:
    configs/         - Конфигурации моделей (BaseTrainingConfig, GPTConfig, T²Config)
    logging.py       - Логирование генерации и метрик
    data_utils.py    - Работа с данными (load_data, get_batch, prepare_data)
    config.py        - Legacy ModelConfig (для обратной совместимости)
    
    # Новая функциональность для анализа сложности:
    config/          - Конфигурации анализаторов сложности
    io/              - Утилиты ввода/вывода для анализа

Основные экспорты:
    Конфиги:
        - BaseTrainingConfig: базовый конфиг со всеми общими параметрами
        - GPTConfig: конфиг для GPT модели
        - TransformerSquaredConfig: конфиг для Transformer² модели
    
    Логирование:
        - setup_logging: настройка базового логирования
        - GenerationLogger: детальное логирование генерации
        - MetricsFormatter: форматирование метрик для вывода
    
    Данные:
        - load_data: загрузка текстовых данных
        - get_batch: получение батча для обучения
        - prepare_data: подготовка данных
    
    # Новое - Анализ сложности:
    Конфигурации анализаторов:
        - ConfigManager: менеджер конфигураций анализаторов
        - AnalyzerConfig: базовая конфигурация анализатора
        - get_analyzer_config: получение конфигурации
        - set_preset_configuration: установка предустановок
    
    Файловые операции:
        - read_source_file: чтение исходного кода
        - write_results: запись результатов анализа
        - find_python_files: поиск Python файлов
        - ResultsWriter: класс для записи результатов
        - batch_process_files: пакетная обработка файлов

"""

# =============================================================================
# ВЕРСИЯ
# =============================================================================

__version__ = '1.0.0'
__author__ = 'Diploma Project'


# =============================================================================
# ИМПОРТЫ ИЗ ПОДМОДУЛЕЙ (СУЩЕСТВУЮЩИЕ)
# =============================================================================

# ----------------------------- Конфигурации ---------------------------------
from .configs import (
    # Базовый конфиг
    BaseTrainingConfig,
    
    # Model-specific конфиги
    GPTConfig,
    TransformerSquaredConfig,
)

# ----------------------------- Логирование ----------------------------------
from .logging import (
    setup_logging,
    GenerationLogger,
    GenerationSession,
    GenerationStep,
    MetricsFormatter,
)

# ----------------------------- Работа с данными -----------------------------
try:
    from .data_utils import (
        load_data,
        get_batch,
        prepare_data,
    )
except ImportError:
    # Если data_utils.py еще не создан
    load_data = None
    get_batch = None
    prepare_data = None

# ----------------------------- Legacy конфиг ---------------------------------
try:
    from .config import ModelConfig
except ImportError:
    # Если config.py еще не существует
    ModelConfig = None


# =============================================================================
# НОВЫЕ ИМПОРТЫ - АНАЛИЗ СЛОЖНОСТИ
# =============================================================================

# ----------------------------- Конфигурации анализаторов -------------------
try:
    from .configs.analyzer_configs import (
        ConfigManager,
        AnalyzerConfig,
        ASTAnalyzerConfig,
        RuntimeAnalyzerConfig,
        CFGAnalyzerConfig,
        MLAnalyzerConfig,
        DynamicAnalyzerConfig,
        HybridAnalyzerConfig,
        PresetConfigurations,
        config_manager,
        get_analyzer_config,
        set_preset_configuration
    )
except ImportError:
    # Если модуль анализа сложности еще не создан
    ConfigManager = None
    AnalyzerConfig = None
    ASTAnalyzerConfig = None
    RuntimeAnalyzerConfig = None
    CFGAnalyzerConfig = None
    MLAnalyzerConfig = None
    DynamicAnalyzerConfig = None
    HybridAnalyzerConfig = None
    PresetConfigurations = None
    config_manager = None
    get_analyzer_config = None
    set_preset_configuration = None

# ----------------------------- Файловые операции ---------------------------
try:
    from .io.file_utils import (
        read_source_file,
        write_results,
        write_json,
        read_json,
        write_csv,
        read_csv,
        find_python_files,
        create_directory,
        ResultsWriter,
        batch_process_files,
        json_serializer
    )
except ImportError:
    # Если модуль файловых операций еще не создан
    read_source_file = None
    write_results = None
    write_json = None
    read_json = None
    write_csv = None
    read_csv = None
    find_python_files = None
    create_directory = None
    ResultsWriter = None
    batch_process_files = None
    json_serializer = None


# =============================================================================
# ЭКСПОРТЫ (ДОПОЛНЕННЫЕ)
# =============================================================================

__all__ = [
    # Версия
    '__version__',
    '__author__',
    
    # ===== КОНФИГИ МОДЕЛЕЙ =====
    'BaseTrainingConfig',
    'GPTConfig',
    'TransformerSquaredConfig',
    'ModelConfig',  # Legacy
    
    # ===== ЛОГИРОВАНИЕ =====
    'setup_logging',
    'GenerationLogger',
    'GenerationSession',
    'GenerationStep',
    'MetricsFormatter',
    
    # ===== ДАННЫЕ =====
    'load_data',
    'get_batch',
    'prepare_data',
    
    # ===== НОВОЕ - КОНФИГУРАЦИИ АНАЛИЗАТОРОВ =====
    'ConfigManager',
    'AnalyzerConfig',
    'ASTAnalyzerConfig',
    'RuntimeAnalyzerConfig',
    'CFGAnalyzerConfig',
    'MLAnalyzerConfig',
    'DynamicAnalyzerConfig',
    'HybridAnalyzerConfig',
    'PresetConfigurations',
    'config_manager',
    'get_analyzer_config',
    'set_preset_configuration',
    
    # ===== ФАЙЛОВЫЕ ОПЕРАЦИИ =====
    'read_source_file',
    'write_results',
    'write_json',
    'read_json',
    'write_csv',
    'read_csv',
    'find_python_files',
    'create_directory',
    'ResultsWriter',
    'batch_process_files',
    'json_serializer',
]


# =============================================================================
# ИНФОРМАЦИЯ О МОДУЛЕ (ДОПОЛНЕННАЯ)
# =============================================================================

def get_available_configs():
    """
    Возвращает список доступных конфигураций.
    
    Returns:
        List[str]: Список имен конфигов
    """
    configs = [
        'BaseTrainingConfig',
        'GPTConfig',
        'TransformerSquaredConfig',
    ]
    
    # Добавляем конфиги анализаторов если доступны
    if AnalyzerConfig is not None:
        configs.extend([
            'AnalyzerConfig',
            'ASTAnalyzerConfig',
            'RuntimeAnalyzerConfig',
            'CFGAnalyzerConfig',
            'MLAnalyzerConfig',
            'DynamicAnalyzerConfig',
            'HybridAnalyzerConfig',
        ])
    
    return configs


def get_module_info():
    """
    Возвращает информацию о модуле utils.
    
    Returns:
        Dict[str, Any]: Информация о модуле
    """
    return {
        'version': __version__,
        'author': __author__,
        'available_configs': get_available_configs(),
        'has_logging': True,
        'has_data_utils': load_data is not None,
        'has_legacy_config': ModelConfig is not None,
        # Новые возможности
        'has_analyzer_configs': ConfigManager is not None,
        'has_file_utils': read_source_file is not None,
        'complexity_analysis_available': all([
            ConfigManager is not None,
            read_source_file is not None,
            write_results is not None
        ])
    }


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (ДОПОЛНЕННЫЕ)
# =============================================================================

def create_config(model_type: str, **kwargs):
    """
    Фабрика для создания конфигов по имени модели.
    
    Args:
        model_type: Тип модели ('gpt', 'transformer_squared', 'analyzer', etc.)
        **kwargs: Параметры для конфига
    
    Returns:
        Объект конфига
    
    Example:
        >>> config = create_config('gpt', batch_size=16, learning_rate=3e-4)
        >>> analyzer_config = create_config('analyzer', name='ast_advanced', timeout=30)
    """
    model_type = model_type.lower()
    
    if model_type == 'gpt':
        return GPTConfig(**kwargs)
    elif model_type in ['transformer_squared', 't2', 'transformer2']:
        return TransformerSquaredConfig(**kwargs)
    elif model_type == 'base':
        return BaseTrainingConfig(**kwargs)
    # Новые типы конфигов
    elif model_type == 'analyzer' and AnalyzerConfig is not None:
        return AnalyzerConfig(**kwargs)
    elif model_type == 'ast_analyzer' and ASTAnalyzerConfig is not None:
        return ASTAnalyzerConfig(**kwargs)
    elif model_type == 'runtime_analyzer' and RuntimeAnalyzerConfig is not None:
        return RuntimeAnalyzerConfig(**kwargs)
    elif model_type == 'cfg_analyzer' and CFGAnalyzerConfig is not None:
        return CFGAnalyzerConfig(**kwargs)
    elif model_type == 'ml_analyzer' and MLAnalyzerConfig is not None:
        return MLAnalyzerConfig(**kwargs)
    elif model_type == 'dynamic_analyzer' and DynamicAnalyzerConfig is not None:
        return DynamicAnalyzerConfig(**kwargs)
    elif model_type == 'hybrid_analyzer' and HybridAnalyzerConfig is not None:
        return HybridAnalyzerConfig(**kwargs)
    else:
        available_types = ['gpt', 'transformer_squared', 'base']
        if AnalyzerConfig is not None:
            available_types.extend([
                'analyzer', 'ast_analyzer', 'runtime_analyzer', 
                'cfg_analyzer', 'ml_analyzer', 'dynamic_analyzer', 'hybrid_analyzer'
            ])
        
        raise ValueError(
            f"Неизвестный тип модели: {model_type}. "
            f"Доступные: {', '.join(available_types)}"
        )


# =============================================================================
# НОВЫЕ ФУНКЦИИ ДЛЯ АНАЛИЗА СЛОЖНОСТИ
# =============================================================================

def setup_complexity_analysis(preset: str = 'fast'):
    """
    Быстрая настройка анализа сложности.
    
    Args:
        preset: Предустановка ('fast', 'comprehensive', 'research', 'production')
    
    Returns:
        bool: True если настройка прошла успешно
    
    Example:
        >>> setup_complexity_analysis('fast')
        >>> # Теперь можно использовать анализаторы
    """
    if set_preset_configuration is None:
        raise ImportError("Модули анализа сложности не доступны")
    
    try:
        set_preset_configuration(preset)
        return True
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Ошибка настройки анализа сложности: {e}")
        return False


def quick_analyze_file(file_path: str, analyzer_type: str = 'hybrid'):
    """
    Быстрый анализ одного файла.
    
    Args:
        file_path: Путь к Python файлу
        analyzer_type: Тип анализатора ('ast', 'runtime', 'hybrid', etc.)
    
    Returns:
        Dict или None: Результат анализа
    
    Example:
        >>> result = quick_analyze_file('my_algorithm.py', 'ast')
        >>> print(result['complexity_class'])
    """
    if not all([read_source_file, ConfigManager]):
        raise ImportError("Модули анализа сложности не доступны")
    
    try:
        # Это упрощенная версия - полная реализация будет в complexity_analyzers
        from complexity_analyzers import create_analyzer, AnalysisContext
        
        source_code = read_source_file(file_path)
        analyzer = create_analyzer(analyzer_type)
        context = AnalysisContext(source_code=source_code)
        result = analyzer.analyze(context)
        
        return result.to_dict() if hasattr(result, 'to_dict') else result
        
    except ImportError:
        # Fallback если основные анализаторы не доступны
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Основные анализаторы не доступны, используется заглушка")
        return None
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Ошибка анализа файла {file_path}: {e}")
        return None


# =============================================================================
# ПРОВЕРКА ИМПОРТОВ ПРИ ЗАГРУЗКЕ (ДОПОЛНЕННАЯ)
# =============================================================================

def _check_imports():
    """Проверка успешности импортов при загрузке модуля."""
    import logging
    logger = logging.getLogger(__name__)
    
    # Проверяем существующие конфиги
    if BaseTrainingConfig is None:
        logger.warning("⚠️  BaseTrainingConfig не загружен")
    if GPTConfig is None:
        logger.warning("⚠️  GPTConfig не загружен")
    if TransformerSquaredConfig is None:
        logger.warning("⚠️  TransformerSquaredConfig не загружен")
    
    # Проверяем логирование
    if setup_logging is None:
        logger.warning("⚠️  logging.py не загружен")
    
    # Проверяем data_utils
    if load_data is None:
        logger.debug("ℹ️  data_utils.py не найден (это нормально если файл не создан)")
    
    # Проверяем legacy config
    if ModelConfig is None:
        logger.debug("ℹ️  Legacy config.py не найден (это нормально)")
    
    # Новые проверки
    if ConfigManager is None:
        logger.debug("ℹ️  Конфигурации анализаторов сложности не найдены")
    else:
        logger.info("✅ Модуль анализа сложности доступен")
    
    if read_source_file is None:
        logger.debug("ℹ️  Файловые утилиты анализа не найдены")
    else:
        logger.info("✅ Файловые утилиты анализа доступны")


# Запускаем проверку при импорте (только если нужно для отладки)
# _check_imports()


# =============================================================================
# АЛИАСЫ ДЛЯ УДОБСТВА (ДОПОЛНЕННЫЕ)
# =============================================================================

# Короткие алиасы для конфигов моделей
BaseConfig = BaseTrainingConfig
T2Config = TransformerSquaredConfig

# Новые алиасы для анализа сложности
if ConfigManager is not None:
    AnalyzerConfigManager = ConfigManager
    __all__.append('AnalyzerConfigManager')

# Добавляем в exports
__all__.extend(['BaseConfig', 'T2Config'])

# Функции анализа сложности
__all__.extend([
    'setup_complexity_analysis',
    'quick_analyze_file',
])
