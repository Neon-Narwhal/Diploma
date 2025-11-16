"""Unified dataset класс"""
from pathlib import Path
from typing import Optional, List, Dict
import logging

from data_loaders.loaders.loaders import HuggingFaceLoader
from data_loaders.loaders.processors import (
    ComplexityClassFilter, DatasetJoiner, DataValidator
)
from data_loaders.loaders.storage import DatasetWriter, DatasetSplitter

logger = logging.getLogger(__name__)


class BigOBenchDataset:
    """Единый интерфейс для работы с BigOBench"""
    
    def __init__(self, 
                 output_dir: Path = Path("data/bigobench"),
                 cache_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir)
        self.cache_dir = cache_dir
        
        self.loader = HuggingFaceLoader(cache_dir)
        self.writer = DatasetWriter(output_dir)
        
        self.samples = []
    
    def prepare(self, 
                min_samples_per_class: int = 1000,
                validate: bool = True) -> 'BigOBenchDataset':
        """Полный пайплайн подготовки"""
        
        # Загрузка
        logger.info("Загрузка данных из HuggingFace")
        complexity_dataset = self.loader.load_bigobench_complexity_labels(streaming=False)
        solutions_dataset = self.loader.load_bigobench_solutions()
        
        # Построение индекса
        solutions_map = self.loader.build_solutions_index(solutions_dataset)
        
        # Джойн
        logger.info("Джойн complexity labels и solution code")
        joined_samples = DatasetJoiner.join_complexity_and_solutions(
            complexity_dataset, solutions_map
        )
        
        # Фильтрация
        logger.info(f"Фильтрация редких классов (threshold={min_samples_per_class})")
        class_filter = ComplexityClassFilter(min_samples=min_samples_per_class)
        filtered_samples, filter_stats = class_filter.filter(joined_samples)
        
        self._print_filter_stats(filter_stats)
        
        # Валидация
        if validate:
            logger.info("Валидация данных")
            valid_samples, validation_stats = DataValidator.validate_dataset(filtered_samples)
            self._print_validation_stats(validation_stats)
            self.samples = valid_samples
        else:
            self.samples = filtered_samples
        
        return self
    
    def split_and_save(self, 
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15):
        """Разделение и сохранение"""
        
        splits = DatasetSplitter.split_by_problems(
            self.samples, train_ratio, val_ratio, test_ratio
        )
        
        for split_name, split_samples in splits.items():
            self.writer.write_jsonl(split_samples, f"{split_name}.jsonl")
        
        # Метаданные
        metadata = {
            'total_samples': len(self.samples),
            'splits': {name: len(samples) for name, samples in splits.items()}
        }
        self.writer.write_metadata(metadata)
    
    @staticmethod
    def _print_filter_stats(stats: Dict):
        """Вывод статистики фильтрации"""
        print(f"\n{'='*60}")
        print("СТАТИСТИКА ФИЛЬТРАЦИИ")
        print(f"{'='*60}")
        print(f"Всего:                      {stats['total']:>12,}")
        print(f"None time complexity:       {stats['none_time']:>12,}")
        print(f"None space complexity:      {stats['none_space']:>12,}")
        print(f"Filtered (rare time):       {stats['filtered_time']:>12,}")
        print(f"Filtered (rare space):      {stats['filtered_space']:>12,}")
        print(f"{'='*60}")
        print(f"СОХРАНЕНО:                  {stats['kept']:>12,}")
        print(f"{'='*60}\n")
    
    @staticmethod
    def _print_validation_stats(stats: Dict):
        """Вывод статистики валидации"""
        print(f"Валидные:   {stats['valid']:,}")
        print(f"Невалидные: {stats['invalid']:,}")
        if stats['errors']:
            print("Ошибки:")
            for error, count in stats['errors'].most_common():
                print(f"  - {error}: {count}")
