"""Загрузчик датасета BigO-Bench"""
import json
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BigOBenchSample:
    """Образец из BigO-Bench датасета"""
    code: str
    complexity_class: str
    algorithm_name: str
    language: str = 'python'
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'complexity_class': self.complexity_class,
            'algorithm_name': self.algorithm_name,
            'language': self.language,
            'metadata': self.metadata or {}
        }

class BigOBenchLoader:
    """Загрузчик BigO-Bench датасета"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("datasets/cache/bigobench")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # URL к BigO-Bench репозиторию или API
        self.base_url = "https://api.github.com/repos/your-org/bigobench/contents"
        self.local_dataset_path = Path("datasets/raw/bigobench")
        
        self.samples: List[BigOBenchSample] = []
        self._loaded = False
    
    def load_dataset(self, force_reload: bool = False) -> List[BigOBenchSample]:
        """Загрузка датасета"""
        if self._loaded and not force_reload:
            return self.samples
        
        # Попытка загрузки из локального кеша
        cached_file = self.cache_dir / "bigobench_dataset.json"
        if cached_file.exists() and not force_reload:
            logger.info("Loading BigO-Bench from cache")
            self.samples = self._load_from_cache(cached_file)
        else:
            # Загрузка из источника
            if self.local_dataset_path.exists():
                logger.info("Loading BigO-Bench from local files")
                self.samples = self._load_from_local()
            else:
                logger.info("Loading BigO-Bench from remote")
                self.samples = self._load_from_remote()
            
            # Сохранение в кеш
            self._save_to_cache(cached_file)
        
        self._loaded = True
        logger.info(f"Loaded {len(self.samples)} samples from BigO-Bench")
        return self.samples
    
    def _load_from_cache(self, cache_file: Path) -> List[BigOBenchSample]:
        """Загрузка из кеша"""
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            samples = []
            for item in data:
                sample = BigOBenchSample(
                    code=item['code'],
                    complexity_class=item['complexity_class'],
                    algorithm_name=item['algorithm_name'],
                    language=item.get('language', 'python'),
                    metadata=item.get('metadata', {})
                )
                samples.append(sample)
            
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load from cache: {e}")
            return []
    
    def _load_from_local(self) -> List[BigOBenchSample]:
        """Загрузка из локальных файлов"""
        samples = []
        
        try:
            # Ищем файлы с алгоритмами
            for complexity_dir in self.local_dataset_path.iterdir():
                if not complexity_dir.is_dir():
                    continue
                
                complexity_class = complexity_dir.name
                
                for algorithm_file in complexity_dir.glob("*.py"):
                    try:
                        with open(algorithm_file, 'r', encoding='utf-8') as f:
                            code = f.read()
                        
                        # Извлекаем имя алгоритма из имени файла
                        algorithm_name = algorithm_file.stem
                        
                        # Попытка загрузить метаданные
                        metadata_file = algorithm_file.with_suffix('.json')
                        metadata = {}
                        if metadata_file.exists():
                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                        
                        sample = BigOBenchSample(
                            code=code,
                            complexity_class=complexity_class,
                            algorithm_name=algorithm_name,
                            metadata=metadata
                        )
                        samples.append(sample)
                        
                    except Exception as e:
                        logger.warning(f"Failed to load {algorithm_file}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to load from local files: {e}")
        
        return samples
    
    def _load_from_remote(self) -> List[BigOBenchSample]:
        """Загрузка из Hugging Face"""
        from data_loaders import load_dataset
        
        hf_dataset = load_dataset("your-org/bigobench-dataset", split="train")
        
        samples = []
        for item in hf_dataset:
            sample = BigOBenchSample(
                code=item['code'],
                complexity_class=item['complexity_class'],
                algorithm_name=item['algorithm_name'],
                language=item.get('language', 'python'),
                metadata=item.get('metadata', {})
            )
            samples.append(sample)
        
        return samples

    
    
    def _save_to_cache(self, cache_file: Path):
        """Сохранение в кеш"""
        try:
            data = [sample.to_dict() for sample in self.samples]
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save to cache: {e}")
    
    def get_samples_by_complexity(self, complexity_class: str) -> List[BigOBenchSample]:
        """Получение образцов определенного класса сложности"""
        if not self._loaded:
            self.load_dataset()
        
        return [sample for sample in self.samples 
                if sample.complexity_class == complexity_class]
    
    def get_complexity_distribution(self) -> Dict[str, int]:
        """Распределение образцов по классам сложности"""
        if not self._loaded:
            self.load_dataset()
        
        distribution = {}
        for sample in self.samples:
            complexity = sample.complexity_class
            distribution[complexity] = distribution.get(complexity, 0) + 1
        
        return distribution
    
    def get_algorithm_names(self) -> List[str]:
        """Список имен алгоритмов"""
        if not self._loaded:
            self.load_dataset()
        
        return list(set(sample.algorithm_name for sample in self.samples))
    
    def filter_samples(self, 
                      complexity_classes: Optional[List[str]] = None,
                      algorithm_names: Optional[List[str]] = None,
                      min_code_length: Optional[int] = None,
                      max_code_length: Optional[int] = None) -> List[BigOBenchSample]:
        """Фильтрация образцов по критериям"""
        if not self._loaded:
            self.load_dataset()
        
        filtered_samples = self.samples
        
        if complexity_classes:
            filtered_samples = [s for s in filtered_samples 
                              if s.complexity_class in complexity_classes]
        
        if algorithm_names:
            filtered_samples = [s for s in filtered_samples 
                              if s.algorithm_name in algorithm_names]
        
        if min_code_length:
            filtered_samples = [s for s in filtered_samples 
                              if len(s.code) >= min_code_length]
        
        if max_code_length:
            filtered_samples = [s for s in filtered_samples 
                              if len(s.code) <= max_code_length]
        
        return filtered_samples
    
    def create_train_test_split(self, test_size: float = 0.2, 
                              random_state: Optional[int] = None) -> Dict[str, List[BigOBenchSample]]:
        """Разделение на обучающую и тестовую выборки"""
        if not self._loaded:
            self.load_dataset()
        
        import random
        
        if random_state is not None:
            random.seed(random_state)
        
        # Группируем по классам сложности для стратифицированного разделения
        samples_by_complexity = {}
        for sample in self.samples:
            complexity = sample.complexity_class
            if complexity not in samples_by_complexity:
                samples_by_complexity[complexity] = []
            samples_by_complexity[complexity].append(sample)
        
        train_samples = []
        test_samples = []
        
        for complexity, samples in samples_by_complexity.items():
            random.shuffle(samples)
            
            n_test = int(len(samples) * test_size)
            test_samples.extend(samples[:n_test])
            train_samples.extend(samples[n_test:])
        
        # Перемешиваем итоговые выборки
        random.shuffle(train_samples)
        random.shuffle(test_samples)
        
        return {
            'train': train_samples,
            'test': test_samples
        }
    
    def export_to_format(self, output_path: Path, format_type: str = 'jsonl'):
        """Экспорт датасета в различные форматы"""
        if not self._loaded:
            self.load_dataset()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in self.samples:
                    json.dump(sample.to_dict(), f, ensure_ascii=False)
                    f.write('\n')
        
        elif format_type == 'csv':
            import csv
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['algorithm_name', 'complexity_class', 'code', 'language']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                writer.writeheader()
                for sample in self.samples:
                    writer.writerow({
                        'algorithm_name': sample.algorithm_name,
                        'complexity_class': sample.complexity_class,
                        'code': sample.code.replace('\n', '\\n'),
                        'language': sample.language
                    })
        
        elif format_type == 'json':
            data = [sample.to_dict() for sample in self.samples]
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        logger.info(f"Exported {len(self.samples)} samples to {output_path}")

class BigOBenchIterator:
    """Итератор для BigO-Bench датасета"""
    
    def __init__(self, loader: BigOBenchLoader, batch_size: Optional[int] = None):
        self.loader = loader
        self.batch_size = batch_size
        self.current_index = 0
        
        if not loader._loaded:
            loader.load_dataset()
    
    def __iter__(self):
        self.current_index = 0
        return self
    
    def __next__(self) -> Union[BigOBenchSample, List[BigOBenchSample]]:
        if self.current_index >= len(self.loader.samples):
            raise StopIteration
        
        if self.batch_size is None:
            sample = self.loader.samples[self.current_index]
            self.current_index += 1
            return sample
        else:
            batch = self.loader.samples[self.current_index:self.current_index + self.batch_size]
            self.current_index += len(batch)
            return batch

# Глобальный экземпляр загрузчика
_global_loader = None

def get_bigobench_loader(cache_dir: Optional[Path] = None) -> BigOBenchLoader:
    """Получение глобального экземпляра загрузчика"""
    global _global_loader
    if _global_loader is None:
        _global_loader = BigOBenchLoader(cache_dir)
    return _global_loader

def load_bigobench_dataset(force_reload: bool = False) -> List[BigOBenchSample]:
    """Быстрая загрузка BigO-Bench датасета"""
    loader = get_bigobench_loader()
    return loader.load_dataset(force_reload)
