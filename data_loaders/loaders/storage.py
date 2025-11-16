"""Сохранение датасетов в различные форматы"""
from pathlib import Path
from typing import List, Dict
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
    def split_by_problems(
        samples: List[Dict],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> Dict[str, List[Dict]]:
        """Problem-level split для предотвращения leakage"""
        import random
        random.seed(random_seed)
        
        # Группировка по problem_id
        problems_to_solutions = {}
        for sample in samples:
            problem_id = sample['problem_id']
            if problem_id not in problems_to_solutions:
                problems_to_solutions[problem_id] = []
            problems_to_solutions[problem_id].append(sample)
        
        # Разделение задач
        problem_ids = list(problems_to_solutions.keys())
        random.shuffle(problem_ids)
        
        n_problems = len(problem_ids)
        n_train = int(n_problems * train_ratio)
        n_val = int(n_problems * val_ratio)
        
        train_problems = set(problem_ids[:n_train])
        val_problems = set(problem_ids[n_train:n_train + n_val])
        test_problems = set(problem_ids[n_train + n_val:])
        
        # Распределение решений
        splits = {'train': [], 'val': [], 'test': []}
        
        for sample in samples:
            problem_id = sample['problem_id']
            if problem_id in train_problems:
                splits['train'].append(sample)
            elif problem_id in val_problems:
                splits['val'].append(sample)
            else:
                splits['test'].append(sample)
        
        logger.info(f"Split: train={len(splits['train'])}, "
                   f"val={len(splits['val'])}, test={len(splits['test'])}")
        
        return splits
