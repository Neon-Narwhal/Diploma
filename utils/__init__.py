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

"""

# =============================================================================
# ВЕРСИЯ
# =============================================================================

__version__ = '1.0.0'
__author__ = 'Diploma Project'


# =============================================================================
# ИМПОРТЫ ИЗ ПОДМОДУЛЕЙ
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
# ЭКСПОРТЫ
# =============================================================================

__all__ = [
    # Версия
    '__version__',
    '__author__',
    
    # ===== КОНФИГИ =====
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
]


# =============================================================================
# ИНФОРМАЦИЯ О МОДУЛЕ
# =============================================================================

def get_available_configs():
    """
    Возвращает список доступных конфигураций.
    
    Returns:
        List[str]: Список имен конфигов
    """
    return [
        'BaseTrainingConfig',
        'GPTConfig',
        'TransformerSquaredConfig',
    ]


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
    }


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def create_config(model_type: str, **kwargs):
    """
    Фабрика для создания конфигов по имени модели.
    
    Args:
        model_type: Тип модели ('gpt', 'transformer_squared')
        **kwargs: Параметры для конфига
    
    Returns:
        Объект конфига
    
    Example:
        >>> config = create_config('gpt', batch_size=16, learning_rate=3e-4)
    """
    model_type = model_type.lower()
    
    if model_type == 'gpt':
        return GPTConfig(**kwargs)
    elif model_type in ['transformer_squared', 't2', 'transformer2']:
        return TransformerSquaredConfig(**kwargs)
    elif model_type == 'base':
        return BaseTrainingConfig(**kwargs)
    else:
        raise ValueError(
            f"Неизвестный тип модели: {model_type}. "
            f"Доступные: 'gpt', 'transformer_squared', 'base'"
        )


# =============================================================================
# ПРОВЕРКА ИМПОРТОВ ПРИ ЗАГРУЗКЕ
# =============================================================================

def _check_imports():
    """Проверка успешности импортов при загрузке модуля."""
    import logging
    logger = logging.getLogger(__name__)
    
    # Проверяем конфиги
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


# Запускаем проверку при импорте (только если нужно для отладки)
# _check_imports()


# =============================================================================
# АЛИАСЫ ДЛЯ УДОБСТВА
# =============================================================================

# Короткие алиасы для конфигов
BaseConfig = BaseTrainingConfig
T2Config = TransformerSquaredConfig

# Добавляем в exports
__all__.extend(['BaseConfig', 'T2Config'])
