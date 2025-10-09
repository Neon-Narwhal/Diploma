# =============================================================================
# training/bigobench_dataset.py
# Универсальный Dataset для BigOBench и других JSONL датасетов
# =============================================================================

"""
Универсальный Dataset для работы с BigOBench и любыми JSONL форматами.

Особенности:
- Поддержка любых JSONL файлов (не только BigOBench)
- Гибкая конфигурация через config объект
- Автоматическая фильтрация невалидных примеров
- Эффективная загрузка данных (lazy loading)
- Кастомный collate_fn для переменных длин
- Совместимость с любыми токенизаторами
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Callable, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# УНИВЕРСАЛЬНЫЙ DATASET
# =============================================================================

class BigOBenchDataset(Dataset):
    """
    Универсальный Dataset для BigOBench и других JSONL датасетов.
    
    Поддерживает:
    - Автоматическую фильтрацию по критериям
    - Гибкую настройку через config
    - Любые токенизаторы с методами encode/decode
    - Lazy loading для эффективности памяти
    
    Args:
        data_path: путь к JSONL файлу
        tokenizer: токенизатор с методом encode(text) -> List[int]
        config: объект конфига с параметрами (block_size, etc.)
        code_field: имя поля с кодом ('query_code' или 'query_dataclass_code')
        filter_fn: функция фильтрации (sample) -> bool (None = без фильтрации)
        max_samples: максимальное количество примеров (None = все)
        cache_in_memory: загрузить все в память (быстрее, но требует больше RAM)
    
    Example:
        >>> dataset = BigOBenchDataset(
        ...     data_path='data/dataset.json',
        ...     tokenizer=tokenizer,
        ...     config=config,
        ...     filter_fn=lambda x: x.get('time_complexity_inferred') != 'Unknown'
        ... )
        >>> print(f"Dataset size: {len(dataset)}")
        >>> sample = dataset[0]
        >>> print(sample['input_ids'].shape)
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        config,
        code_field: str = 'query_code',
        filter_fn: Optional[Callable[[Dict], bool]] = None,
        max_samples: Optional[int] = None,
        cache_in_memory: bool = False
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.config = config
        self.code_field = code_field
        self.filter_fn = filter_fn
        self.max_samples = max_samples
        self.cache_in_memory = cache_in_memory
        
        # Валидация
        if not self.data_path.exists():
            raise FileNotFoundError(f"Файл не найден: {self.data_path}")
        
        # Загрузка данных
        logger.info(f"[BigOBenchDataset] Загрузка данных из {self.data_path}...")
        self.samples = self._load_data()
        
        if len(self.samples) == 0:
            raise ValueError(f"Датасет пуст после фильтрации! Проверьте файл: {self.data_path}")
        
        logger.info(f"[BigOBenchDataset] Загружено {len(self.samples)} валидных примеров")
    
    def _load_data(self) -> List[Dict]:
        """Загрузка и фильтрация данных из JSONL файла."""
        samples = []
        skipped = 0
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Проверка лимита
                if self.max_samples and len(samples) >= self.max_samples:
                    break
                
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Применяем фильтр (если есть)
                    if self.filter_fn and not self.filter_fn(data):
                        skipped += 1
                        continue
                    
                    # Проверяем наличие кода
                    code = data.get(self.code_field)
                    if not code:
                        # Пробуем альтернативное поле
                        alt_field = 'query_dataclass_code' if self.code_field == 'query_code' else 'query_code'
                        code = data.get(alt_field)
                    
                    if not code or len(code.strip()) < 10:
                        skipped += 1
                        continue
                    
                    # Если cache_in_memory, сразу токенизируем
                    if self.cache_in_memory:
                        data['_cached_tokens'] = self.tokenizer.encode(code)
                    
                    samples.append(data)
                    
                except json.JSONDecodeError:
                    logger.debug(f"[BigOBenchDataset] Пропущена строка {line_num}: ошибка парсинга JSON")
                    skipped += 1
                except Exception as e:
                    logger.debug(f"[BigOBenchDataset] Ошибка обработки строки {line_num}: {e}")
                    skipped += 1
        
        if skipped > 0:
            logger.info(f"[BigOBenchDataset] Пропущено {skipped} невалидных примеров")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Возвращает один пример из датасета.
        
        Returns:
            Dict с ключами:
                - input_ids: tensor [seq_len]
                - target_ids: tensor [seq_len]
                - metadata: dict с дополнительной информацией
        """
        sample = self.samples[idx]
        
        # Получаем код
        code = sample.get(self.code_field)
        if not code:
            alt_field = 'query_dataclass_code' if self.code_field == 'query_code' else 'query_code'
            code = sample.get(alt_field, '')
        
        # Токенизация (используем кэш если есть)
        if self.cache_in_memory and '_cached_tokens' in sample:
            tokens = sample['_cached_tokens']
        else:
            tokens = self.tokenizer.encode(code)
        
        # Обрезка до block_size
        max_len = self.config.block_size
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        
        # Создаем input и target для next-token prediction
        if len(tokens) > 1:
            input_ids = tokens[:-1]
            target_ids = tokens[1:]
        else:
            input_ids = tokens
            target_ids = tokens
        
        # Metadata (полезно для анализа)
        metadata = {
            'idx': idx,
            'code_length': len(code),
            'token_length': len(tokens),
            'problem_id': sample.get('problem_id', ''),
            'solution_id': sample.get('solution_id', ''),
        }
        
        # Добавляем complexity информацию если есть
        if 'time_complexity_inferred' in sample:
            metadata['time_complexity'] = sample['time_complexity_inferred']
        if 'space_complexity_inferred' in sample:
            metadata['space_complexity'] = sample['space_complexity_inferred']
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'metadata': metadata
        }


# =============================================================================
# CUSTOM COLLATE FUNCTION
# =============================================================================

def bigobench_collate_fn(
    batch: List[Dict[str, Any]],
    pad_token_id: int = 0,
    max_length: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Custom collate function для обработки батчей с переменной длиной.
    
    Паддит последовательности до максимальной длины в батче или max_length.
    
    Args:
        batch: список примеров из __getitem__
        pad_token_id: токен для паддинга
        max_length: максимальная длина (если None, берется max из батча)
    
    Returns:
        Dict с батчами input_ids, target_ids, attention_mask, metadata
    """
    input_ids_list = [item['input_ids'] for item in batch]
    target_ids_list = [item['target_ids'] for item in batch]
    metadata_list = [item['metadata'] for item in batch]
    
    # Определяем максимальную длину
    if max_length is None:
        max_length = max(len(ids) for ids in input_ids_list)
    
    # Паддинг
    batch_size = len(batch)
    padded_input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    padded_target_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    
    for i, (input_ids, target_ids) in enumerate(zip(input_ids_list, target_ids_list)):
        seq_len = len(input_ids)
        padded_input_ids[i, :seq_len] = input_ids
        padded_target_ids[i, :seq_len] = target_ids
        attention_mask[i, :seq_len] = 1
    
    return {
        'input_ids': padded_input_ids,
        'target_ids': padded_target_ids,
        'attention_mask': attention_mask,
        'metadata': metadata_list
    }


# =============================================================================
# ФАБРИКА DATALOADER'ОВ
# =============================================================================

def create_dataloaders_from_config(
    data_path: str,
    tokenizer,
    config,
    train_split: float = 0.9,
    num_workers: int = 2,
    filter_missing_complexity: bool = True,
    use_dataclass_code: bool = False,
    use_custom_collate: bool = True,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Универсальная фабрика для создания train/val DataLoader'ов из конфига.
    
    Args:
        data_path: путь к JSONL файлу
        tokenizer: токенизатор
        config: объект BaseTrainingConfig (или наследник)
        train_split: доля тренировочных данных
        num_workers: количество workers для DataLoader
        filter_missing_complexity: пропускать примеры без асимптотики
        use_dataclass_code: использовать query_dataclass_code поле
        use_custom_collate: использовать кастомный collate_fn
        **kwargs: дополнительные параметры для DataLoader
    
    Returns:
        (train_loader, val_loader)
    
    Example:
        >>> from utils.configs import GPTConfig
        >>> config = GPTConfig()
        >>> train_loader, val_loader = create_dataloaders_from_config(
        ...     data_path='data/dataset.json',
        ...     tokenizer=tokenizer,
        ...     config=config
        ... )
    """
    logger.info("[DataLoaders] Создание DataLoaders из конфига...")
    
    # Определяем поле с кодом
    code_field = 'query_dataclass_code' if use_dataclass_code else 'query_code'
    
    # Создаем функцию фильтрации
    def filter_fn(sample: Dict) -> bool:
        if not filter_missing_complexity:
            return True
        
        has_time = sample.get('time_complexity_inferred') not in [None, '', 'Unknown', 'UNKNOWN']
        has_space = sample.get('space_complexity_inferred') not in [None, '', 'Unknown', 'UNKNOWN']
        
        return has_time or has_space
    
    # Создаем полный датасет
    full_dataset = BigOBenchDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        config=config,
        code_field=code_field,
        filter_fn=filter_fn if filter_missing_complexity else None,
        cache_in_memory=False  # Для больших датасетов лучше False
    )
    
    # Разделение на train/val
    train_size = int(len(full_dataset) * train_split)
    val_size = len(full_dataset) - train_size
    
    if train_size == 0 or val_size == 0:
        logger.warning(
            f"[DataLoaders] Предупреждение: мало данных "
            f"(train={train_size}, val={val_size})"
        )
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    # Создаем collate_fn
    if use_custom_collate:
        collate_fn = lambda batch: bigobench_collate_fn(
            batch,
            pad_token_id=0,
            max_length=config.block_size
        )
    else:
        collate_fn = None
    
    # Параметры DataLoader из конфига
    dataloader_kwargs = {
        'batch_size': config.batch_size,
        'num_workers': num_workers,
        'pin_memory': config.pin_memory if hasattr(config, 'pin_memory') else False,
        'collate_fn': collate_fn,
        **kwargs
    }
    
    # Создаем DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **dataloader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **dataloader_kwargs
    )
    
    logger.info(f"[DataLoaders] ✅ Создано:")
    logger.info(f"   • Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"   • Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    
    return train_loader, val_loader


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def get_dataset_statistics(dataset: BigOBenchDataset) -> Dict[str, Any]:
    """
    Вычисление статистики датасета.
    
    Args:
        dataset: объект BigOBenchDataset
    
    Returns:
        Словарь со статистикой
    """
    logger.info("[Statistics] Вычисление статистики датасета...")
    
    code_lengths = []
    token_lengths = []
    complexities = {'time': [], 'space': []}
    
    for sample in dataset.samples:
        code = sample.get(dataset.code_field, '')
        code_lengths.append(len(code))
        
        tokens = dataset.tokenizer.encode(code)
        token_lengths.append(len(tokens))
        
        if 'time_complexity_inferred' in sample:
            complexities['time'].append(sample['time_complexity_inferred'])
        if 'space_complexity_inferred' in sample:
            complexities['space'].append(sample['space_complexity_inferred'])
    
    import numpy as np
    
    stats = {
        'total_samples': len(dataset),
        'code_length': {
            'mean': np.mean(code_lengths),
            'std': np.std(code_lengths),
            'min': np.min(code_lengths),
            'max': np.max(code_lengths),
            'median': np.median(code_lengths)
        },
        'token_length': {
            'mean': np.mean(token_lengths),
            'std': np.std(token_lengths),
            'min': np.min(token_lengths),
            'max': np.max(token_lengths),
            'median': np.median(token_lengths)
        },
        'time_complexity_counts': dict(zip(*np.unique(complexities['time'], return_counts=True))) if complexities['time'] else {},
        'space_complexity_counts': dict(zip(*np.unique(complexities['space'], return_counts=True))) if complexities['space'] else {},
    }
    
    return stats


__all__ = [
    'BigOBenchDataset',
    'bigobench_collate_fn',
    'create_dataloaders_from_config',
    'get_dataset_statistics',
]
