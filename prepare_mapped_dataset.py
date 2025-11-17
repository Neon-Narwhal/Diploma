"""
Подготовка BigOBench датасета с маппингом классов

Запуск:
    python prepare_mapped_dataset.py
"""

from data_loaders import BigOBenchDataset, ComplexityMapper
from data_loaders.loaders import LocalLoader, DatasetWriter, DatasetSplitter
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Пайплайн подготовки с маппингом"""
    
    # Конфигурация
    config = {
        'input_dir': Path('data/bigobench'),  # Данные из prepare_dataset.py
        'output_dir': Path('data/bigobench_mapped'),
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
    }
    
    logger.info("="*60)
    logger.info("ПОДГОТОВКА ДАТАСЕТА С МАППИНГОМ")
    logger.info("="*60)
    logger.info(f"Input: {config['input_dir']}")
    logger.info(f"Output: {config['output_dir']}")
    logger.info("="*60 + "\n")
    
    # Загрузка данных
    logger.info("Загрузка исходных данных...")
    loader = LocalLoader(data_dir=config['input_dir'])
    
    all_samples = []
    for split in ['train', 'val', 'test']:
        samples = loader.load_split(split)
        logger.info(f"  {split}: {len(samples):,} образцов")
        all_samples.extend(samples)
    
    logger.info(f"\nВсего образцов: {len(all_samples):,}\n")
    
    # Маппинг классов
    logger.info("Применение маппинга классов...")
    mapper = ComplexityMapper()
    
    # Статистика до маппинга
    logger.info("\nДО МАППИНГА:")
    from collections import Counter
    time_before = Counter(s['time_complexity'] for s in all_samples)
    space_before = Counter(s['space_complexity'] for s in all_samples)
    logger.info(f"  Классов временной сложности: {len(time_before)}")
    logger.info(f"  Классов пространственной сложности: {len(space_before)}")
    
    # Применение маппинга
    mapped_samples = mapper.map_dataset(all_samples)
    
    # Статистика после маппинга
    logger.info("\nПОСЛЕ МАППИНГА:")
    time_after = Counter(s['time_complexity_mapped'] for s in mapped_samples)
    space_after = Counter(s['space_complexity_mapped'] for s in mapped_samples)
    logger.info(f"  Классов временной сложности: {len(time_after)}")
    logger.info(f"  Классов пространственной сложности: {len(space_after)}")
    
    logger.info("\nРаспределение после маппинга:")
    logger.info(f"\n  ВРЕМЕННАЯ СЛОЖНОСТЬ:")
    for cls, count in sorted(time_after.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(mapped_samples) * 100
        logger.info(f"    {cls:<15} {count:>8,} ({pct:>5.1f}%)")
    
    logger.info(f"\n  ПРОСТРАНСТВЕННАЯ СЛОЖНОСТЬ:")
    for cls, count in sorted(space_after.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(mapped_samples) * 100
        logger.info(f"    {cls:<15} {count:>8,} ({pct:>5.1f}%)")
    
    # Разделение и сохранение
    logger.info("\n\nРазделение на splits...")
    splits = DatasetSplitter.split_by_problems(
        mapped_samples,
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        test_ratio=config['test_ratio']
    )
    
    # Сохранение
    logger.info("Сохранение...")
    writer = DatasetWriter(config['output_dir'])
    
    for split_name, split_samples in splits.items():
        writer.write_jsonl(split_samples, f"{split_name}.jsonl")
    
    # Сохранение метаданных
    metadata = {
        'total_samples': len(mapped_samples),
        'time_classes': sorted(list(time_after.keys())),
        'space_classes': sorted(list(space_after.keys())),
        'time_distribution': dict(time_after),
        'space_distribution': dict(space_after),
        'mapping': {
            'time': mapper.time_mapping,
            'space': mapper.space_mapping
        },
        'splits': {name: len(samples) for name, samples in splits.items()}
    }
    
    writer.write_metadata(metadata)
    
    logger.info("\n" + "="*60)
    logger.info("ГОТОВО!")
    logger.info(f"Датасет сохранен в: {config['output_dir']}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
