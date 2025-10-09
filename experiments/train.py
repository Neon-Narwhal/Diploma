# =============================================================================
# experiments/train.py
# ЕДИНСТВЕННЫЙ файл для обучения ВСЕХ моделей на BigOBench
# =============================================================================

"""
Универсальный скрипт обучения моделей на датасете BigOBench.

Поддерживаемые модели:
- GPT (GPTLikeModel)
- TransformerSquared (Transformer²)
- Любые будущие модели через Registry

Использование:
    # 1. Обучение GPT с дефолтными параметрами
    python experiments/train.py
    
    # 2. Обучение с кастомными параметрами (см. примеры внизу файла)
    # Измените параметры в TRAINING_CONFIG и запустите

Архитектура:
    - Все параметры управляются через конфиги (utils/configs/)
    - Единый UniversalTrainer для всех моделей
    - Registry паттерн для добавления новых моделей
    - Полная интеграция с MLflow
    - Автоматическое логирование и сохранение
"""

import sys
import os
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import torch
from typing import Optional, Dict, Any

# Конфигурации
from utils.configs import GPTConfig, TransformerSquaredConfig

# Модели
from models.gpt_model import GPTLikeModel

# Токенизаторы
from my_tokenizers.bpe_tokenizer import BPETokenizer
from my_tokenizers.char_tokenizer import CharTokenizer

# Training
from training.bigobench_dataset import create_dataloaders_from_config
from training.universal_trainer import UniversalTrainer
from training.model_registry import ModelRegistry

# Утилиты
from utils.logging import setup_logging, MetricsFormatter

import logging


# =============================================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# =============================================================================

# Настраиваем базовое логирование
setup_logging(log_level='INFO', log_file='logs/training.log')
logger = logging.getLogger(__name__)

# Formatter для красивого вывода метрик
formatter = MetricsFormatter()


# =============================================================================
# РЕГИСТРАЦИЯ МОДЕЛЕЙ
# =============================================================================

def register_all_models():
    """
    Регистрация всех доступных моделей в Registry.
    
    Добавьте сюда новые модели по мере их создания.
    """
    logger.info("="*80)
    logger.info("РЕГИСТРАЦИЯ МОДЕЛЕЙ")
    logger.info("="*80)
    
    # 1. Регистрируем GPT
    ModelRegistry.register_model(
        name='gpt',
        model_class=GPTLikeModel,
        config_class=GPTConfig
    )
    logger.info("✅ Зарегистрирована модель: GPT")
    
    # 2. Регистрируем TransformerSquared (если реализован)
    try:
        from models.transformer_squared_model import TransformerSquared
        ModelRegistry.register_model(
            name='transformer_squared',
            model_class=TransformerSquared,
            config_class=TransformerSquaredConfig
        )
        logger.info("✅ Зарегистрирована модель: TransformerSquared")
    except ImportError:
        logger.warning("⚠️  TransformerSquared не найден, используется только GPT")
    
    # 3. Добавьте сюда новые модели:
    # try:
    #     from models.my_new_model import MyNewModel
    #     from utils.configs.my_new_config import MyNewConfig
    #     ModelRegistry.register_model('my_new_model', MyNewModel, MyNewConfig)
    # except ImportError:
    #     pass
    
    logger.info(f"📋 Доступные модели: {ModelRegistry.list_models()}")
    logger.info("="*80 + "\n")


# =============================================================================
# СОЗДАНИЕ ТОКЕНИЗАТОРА
# =============================================================================

def create_tokenizer_from_file(
    data_path: str,
    tokenizer_type: str,
    vocab_size: int,
    max_samples: int = 1000
) -> tuple:
    """Создание и обучение токенизатора на данных из файла."""
    logger.info(f"📚 Создание {tokenizer_type.upper()} токенизатора...")
    
    # Проверяем существование файла
    data_path = Path(data_path)
    if not data_path.exists():
        logger.error(f"❌ Файл не найден: {data_path}")
        logger.error(f"   Текущая директория: {Path.cwd()}")
        logger.error(f"   Абсолютный путь: {data_path.absolute()}")
        
        # Подсказка пользователю
        logger.info("\n💡 Возможные решения:")
        logger.info("   1. Проверьте что файл существует")
        logger.info("   2. Измените data_path в TRAINING_CONFIG")
        logger.info("   3. Скачайте BigOBench датасет:")
        logger.info("      https://github.com/facebookresearch/BigOBench")
        
        raise FileNotFoundError(f"Файл данных не найден: {data_path}")
    
    # Загружаем примеры кода из датасета
    texts = []
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                
                if line.strip():
                    try:
                        data = json.loads(line)
                        code = data.get('query_code', '') or data.get('query_dataclass_code', '')
                        if code and len(code.strip()) > 10:
                            texts.append(code)
                    except json.JSONDecodeError as e:
                        logger.debug(f"Пропущена строка {i}: ошибка парсинга JSON")
                        continue
    except Exception as e:
        logger.error(f"❌ Ошибка чтения файла: {e}")
        raise
    
    if not texts:
        logger.error(f"❌ Не удалось загрузить примеры кода из {data_path}")
        logger.error(f"   Загружено строк: {i+1}, валидных примеров: 0")
        logger.info("\n💡 Проверьте:")
        logger.info("   1. Формат файла должен быть JSONL (JSON Lines)")
        logger.info("   2. Каждая строка должна содержать поле 'query_code' или 'query_dataclass_code'")
        raise ValueError(f"Не удалось загрузить данные из {data_path}")
    
    logger.info(f"   Загружено {len(texts)} примеров кода для обучения токенизатора")
    
    # Объединяем тексты
    combined_text = '\n'.join(texts)
    
    # Создаем токенизатор
    if tokenizer_type == 'bpe':
        tokenizer = BPETokenizer(combined_text, vocab_size)
    elif tokenizer_type == 'char':
        tokenizer = CharTokenizer(combined_text)
    else:
        raise ValueError(f"Неизвестный тип токенизатора: {tokenizer_type}")
    
    actual_vocab_size = tokenizer.get_vocab_size()
    logger.info(f"✅ Токенизатор создан: vocab_size={actual_vocab_size}")
    
    return tokenizer, actual_vocab_size



# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ
# =============================================================================

def train_model(
    model_name: str,
    config_overrides: Optional[Dict[str, Any]] = None,
    force_retrain: bool = False
) -> Dict[str, float]:
    """
    Универсальная функция обучения любой зарегистрированной модели.
    
    Args:
        model_name: Имя модели из registry ('gpt', 'transformer_squared', и т.д.)
        config_overrides: Словарь с переопределениями параметров конфига
        force_retrain: Если True, обучает даже если чекпоинт существует
    
    Returns:
        Словарь с финальными метриками обучения
    
    Example:
        >>> metrics = train_model('gpt', {'batch_size': 16, 'learning_rate': 5e-4})
        >>> print(f"Best val loss: {metrics['best_val_loss']:.4f}")
    """
    logger.info("\n" + "="*80)
    logger.info(f"🚀 ОБУЧЕНИЕ МОДЕЛИ: {model_name.upper()}")
    logger.info("="*80)
    
    # ========================================================================
    # 1. ПРОВЕРКА ДОСТУПНОСТИ МОДЕЛИ
    # ========================================================================
    
    if model_name not in ModelRegistry.list_models():
        available = ', '.join(ModelRegistry.list_models())
        raise ValueError(
            f"Модель '{model_name}' не зарегистрирована. "
            f"Доступные модели: {available}"
        )
    
    # ========================================================================
    # 2. СОЗДАНИЕ КОНФИГА
    # ========================================================================
    
    logger.info(f"⚙️  Создание конфигурации для {model_name}...")
    
    # Получаем класс конфига из registry
    ConfigClass = ModelRegistry.get_config_class(model_name)
    
    # Создаем конфиг с дефолтными значениями
    config = ConfigClass()
    
    # Применяем переопределения (если есть)
    if config_overrides:
        logger.info(f"   Применение переопределений: {config_overrides}")
        config.update(**config_overrides)
    
    # Выводим ключевые параметры
    logger.info(f"\n📊 Конфигурация обучения:")
    logger.info(f"   • Модель: {model_name}")
    logger.info(f"   • Данные: {config.data_path}")
    logger.info(f"   • Токенизатор: {config.tokenizer_type} (vocab={config.vocab_size})")
    logger.info(f"   • Batch size: {config.batch_size} x {config.gradient_accumulation_steps} = {config.effective_batch_size}")
    logger.info(f"   • Learning rate: {config.learning_rate:.2e}")
    logger.info(f"   • Max epochs: {config.max_epochs}")
    logger.info(f"   • Device: {config.device}")
    
    if hasattr(config, 'n_embd'):
        logger.info(f"   • Архитектура: n_embd={config.n_embd}, n_layer={config.n_layer}, n_head={config.n_head}")
    
    # ========================================================================
    # 3. СОЗДАНИЕ ТОКЕНИЗАТОРА
    # ========================================================================
    
    logger.info(f"\n📚 Подготовка токенизатора...")
    
    tokenizer, actual_vocab_size = create_tokenizer_from_file(
        data_path=str(config.data_path),
        tokenizer_type=config.tokenizer_type,
        vocab_size=config.vocab_size,
        max_samples=1000
    )
    
    # Обновляем vocab_size в конфиге
    config.vocab_size = actual_vocab_size
    logger.info(f"   • Vocab size обновлен: {actual_vocab_size}")
    
    # ========================================================================
    # 4. СОЗДАНИЕ DATALOADERS
    # ========================================================================
    
    logger.info(f"\n📦 Создание DataLoaders...")
    
    try:
        train_loader, val_loader = create_dataloaders_from_config(
            data_path=str(config.data_path),
            tokenizer=tokenizer,
            config=config,
            train_split=config.train_split,
            num_workers=config.num_workers,
            filter_missing_complexity=config.filter_missing_complexity
        )
    except Exception as e:
        logger.error(f"❌ Ошибка создания DataLoaders: {e}")
        raise
    
    logger.info(f"   • Train batches: {len(train_loader)}")
    logger.info(f"   • Val batches: {len(val_loader)}")
    logger.info(f"   • Total train samples: {len(train_loader.dataset)}")
    logger.info(f"   • Total val samples: {len(val_loader.dataset)}")
    
    # ========================================================================
    # 5. СОЗДАНИЕ МОДЕЛИ
    # ========================================================================
    
    logger.info(f"\n🧠 Создание модели {model_name}...")
    
    try:
        model = ModelRegistry.create_model(model_name, config)
    except Exception as e:
        logger.error(f"❌ Ошибка создания модели: {e}")
        raise
    
    # Информация о модели
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"   • Класс модели: {model.__class__.__name__}")
    logger.info(f"   • Всего параметров: {total_params:,} (~{total_params/1e6:.2f}M)")
    logger.info(f"   • Обучаемых параметров: {trainable_params:,} (~{trainable_params/1e6:.2f}M)")
    
    if hasattr(config, 'estimated_parameters_millions'):
        logger.info(f"   • Оценка из конфига: {config.estimated_parameters_millions:.2f}M")
    
    # Перемещаем модель на устройство
    model = model.to(config.device)
    
    # Опционально: компиляция модели (PyTorch 2.0+)
    if config.compile_model:
        logger.info("   • Компиляция модели с torch.compile...")
        try:
            model = torch.compile(model)
            logger.info("   ✅ Модель скомпилирована")
        except Exception as e:
            logger.warning(f"   ⚠️  Ошибка компиляции: {e}")
    
    # ========================================================================
    # 6. СОЗДАНИЕ УНИВЕРСАЛЬНОГО ТРЕНЕРА
    # ========================================================================
    
    logger.info(f"\n🏋️  Создание UniversalTrainer...")
    
    try:
        trainer = UniversalTrainer(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
    except Exception as e:
        logger.error(f"❌ Ошибка создания тренера: {e}")
        raise
    
    logger.info("   ✅ UniversalTrainer готов к обучению")
    
    # ========================================================================
    # 7. ЗАПУСК ОБУЧЕНИЯ
    # ========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("🎓 НАЧАЛО ОБУЧЕНИЯ")
    logger.info("="*80 + "\n")
    
    try:
        final_metrics = trainer.train()
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Обучение прервано пользователем (Ctrl+C)")
        logger.info("   Сохранение текущего состояния...")
        # Тренер автоматически сохранит последний чекпоинт
        return {'interrupted': True}
    except Exception as e:
        logger.error(f"\n❌ Ошибка во время обучения: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # ========================================================================
    # 8. ВЫВОД ФИНАЛЬНЫХ РЕЗУЛЬТАТОВ
    # ========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    logger.info("="*80)
    logger.info(f"   • Best validation loss: {final_metrics['best_val_loss']:.4f}")
    logger.info(f"   • Final epoch: {final_metrics['final_epoch']}")
    logger.info(f"   • Total training time: {final_metrics.get('total_time', 'N/A')}")
    logger.info(f"   • Model saved to: checkpoints/")
    logger.info("="*80 + "\n")
    
    return final_metrics


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def compare_configs(config_variations: list) -> Dict[str, Any]:
    """
    Сравнение нескольких конфигураций.
    
    Полезно для поиска лучших гиперпараметров.
    
    Args:
        config_variations: Список словарей с вариациями конфига
    
    Returns:
        Словарь с результатами всех экспериментов
    
    Example:
        >>> variations = [
        ...     {'n_embd': 128, 'n_layer': 4},
        ...     {'n_embd': 256, 'n_layer': 6},
        ...     {'n_embd': 512, 'n_layer': 8},
        ... ]
        >>> results = compare_configs(variations)
    """
    logger.info("\n" + "="*80)
    logger.info("🔬 СРАВНЕНИЕ КОНФИГУРАЦИЙ")
    logger.info("="*80)
    
    results = []
    
    for i, config_override in enumerate(config_variations, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Эксперимент {i}/{len(config_variations)}: {config_override}")
        logger.info(f"{'='*80}")
        
        try:
            metrics = train_model('gpt', config_overrides=config_override)
            results.append({
                'config': config_override,
                'metrics': metrics
            })
        except Exception as e:
            logger.error(f"❌ Ошибка в эксперименте {i}: {e}")
            results.append({
                'config': config_override,
                'error': str(e)
            })
    
    # Выводим итоговую таблицу
    logger.info("\n" + "="*80)
    logger.info("📊 РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    logger.info("="*80)
    
    for i, result in enumerate(results, 1):
        config_str = str(result['config'])
        if 'error' in result:
            logger.info(f"{i}. {config_str}: ERROR - {result['error']}")
        else:
            val_loss = result['metrics']['best_val_loss']
            logger.info(f"{i}. {config_str}: Val Loss = {val_loss:.4f}")
    
    logger.info("="*80 + "\n")
    
    return results


# =============================================================================
# ГЛАВНАЯ ТОЧКА ВХОДА
# =============================================================================

if __name__ == "__main__":
    # Регистрируем все модели
    register_all_models()
    
    # ========================================================================
    # КОНФИГУРАЦИЯ ОБУЧЕНИЯ (ИЗМЕНИТЕ ПОД СВОИ НУЖДЫ)
    # ========================================================================
    
    # Выберите модель для обучения
    MODEL_NAME = 'transformer_squared'  # 'gpt' или 'transformer_squared'
    
    # Переопределения конфига (опционально)
    TRAINING_CONFIG = {
        # Данные
        'data_path': 'data/complexity_labels_full_0-49.jsonl',
        
        # Токенизатор
        'tokenizer_type': 'char',  # 'bpe' или 'char'
        'vocab_size': 5000,
        
        # Архитектура (для GPT/T²)
        'n_embd': 256,
        'n_layer': 6,
        'n_head': 4,
        
        # Обучение
        'batch_size': 8,
        'learning_rate': 3e-4,
        'max_epochs': 1,
        'block_size': 512,
        
        # Оптимизация
        'gradient_accumulation_steps': 4,
        'use_amp': True,
        
        # Сохранение
        'save_every_epochs': 2,
        'early_stopping_patience': 5,
        
        # MLflow
        'experiment_name': f'{MODEL_NAME}_bigobench_training',
    }
    
    # ========================================================================
    # ЗАПУСК ОБУЧЕНИЯ
    # ========================================================================
    
    # Вариант 1: Обучение с переопределениями
    metrics = train_model(MODEL_NAME, config_overrides=TRAINING_CONFIG)
    
    # Вариант 2: Обучение с дефолтными параметрами
    # metrics = train_model('gpt')
    
    # Вариант 3: Обучение TransformerSquared
    # metrics = train_model('transformer_squared')
    
    # Вариант 4: Обучение с другим токенизатором
    # metrics = train_model('gpt', config_overrides={'tokenizer_type': 'char'})
    
    # Вариант 5: Сравнение нескольких конфигураций
