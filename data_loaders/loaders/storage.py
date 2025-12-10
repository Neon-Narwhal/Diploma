"""Сохранение датасетов в различные форматы"""
from pathlib import Path
from typing import List, Dict
from sklearn.model_selection import train_test_split
from collections import Counter
import json
import logging

logger = logging.getLogger(__name__)


class DatasetWriter:
    """Запись датасетов на диск"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_jsonl(self, samples: List[Dict], filename: str):
        """Запись в JSONL формат"""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Сохранено {len(samples)} образцов в {filepath}")
        return filepath
    
    def write_parquet(self, samples: List[Dict], filename: str):
        """Запись в Parquet формат"""
        try:
            import pandas as pd
            
            df = pd.DataFrame(samples)
            filepath = self.output_dir / filename
            df.to_parquet(filepath, index=False, compression='snappy')
            
            logger.info(f"Сохранено {len(samples)} образцов в {filepath}")
            return filepath
        
        except ImportError:
            logger.error("pandas/pyarrow не установлены")
            raise
    
    def write_metadata(self, metadata: Dict, filename: str = "metadata.json"):
        """Запись метаданных"""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Метаданные сохранены в {filepath}")
        return filepath


class DatasetSplitter:
    """Разделение датасета на train/val/test"""
    
    @staticmethod
    def split_by_problems(samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Стратифицированный сплит по time_complexity_mapped
        """
        # Проверка соотношений
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01
        
        # Извлекаем labels для стратификации
        labels = [s.get('time_complexity_mapped', s.get('time_complexity')) for s in samples]
        
        # Логируем распределение ДО сплита
        logger.info("Распределение классов BEFORE split:")
        label_counts = Counter(labels)
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {label:<15} {count:>8,} ({count/len(samples)*100:>5.1f}%)")
        
        # Сначала отделяем train от (val+test)
        train_samples, temp_samples, train_labels, temp_labels = train_test_split(
            samples,
            labels,
            train_size=train_ratio,
            stratify=labels,
            random_state=42
        )
        
        # Теперь делим temp на val и test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_samples, test_samples, val_labels, test_labels = train_test_split(
            temp_samples,
            temp_labels,
            train_size=val_ratio_adjusted,
            stratify=temp_labels,
            random_state=42
        )
        
        # Логируем распределение ПОСЛЕ сплита
        logger.info("\nРаспределение классов AFTER split:")
        for split_name, split_labels in [('train', train_labels), ('val', val_labels), ('test', test_labels)]:
            logger.info(f"\n{split_name.upper()}:")
            split_counts = Counter(split_labels)
            for label, count in sorted(split_counts.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {label:<15} {count:>8,} ({count/len(split_labels)*100:>5.1f}%)")
        
        return {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
