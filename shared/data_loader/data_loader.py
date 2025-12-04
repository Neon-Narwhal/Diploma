"""
Универсальный загрузчик данных для всех модулей.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from shared.data_loader.dataset import Dataset, CodeSample


class DataLoader:
    """
    Универсальный загрузчик данных.
    Читает JSONL/JSON файлы и создаёт Dataset объекты.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Словарь с настройками загрузки
                - train_path: путь к train файлу
                - val_path: путь к val файлу
                - test_path: путь к test файлу
                - code_field: название поля с кодом (default: 'code')
                - label_field: название поля с меткой (default: 'label')
                - preprocessing: настройки препроцессинга
        """
        self.config = config
        self.code_field = config.get('code_field', 'code')
        self.label_field = config.get('label_field', 'label')
        
        # Настройки препроцессинга
        preprocessing = config.get('preprocessing', {})
        self.min_code_length = preprocessing.get('min_code_length', 10)
        self.max_code_length = preprocessing.get('max_code_length', None)
        
        # Лимиты по сплитам
        split_limits = preprocessing.get('split_limits', {})
        global_limit = preprocessing.get('max_samples_per_split', None)
        self.train_limit = split_limits.get('train', global_limit)
        self.val_limit = split_limits.get('val', global_limit)
        self.test_limit = split_limits.get('test', global_limit)
    
    @classmethod
    def from_config(cls, config) -> 'DataLoader':
        """
        Создание из конфига эксперимента.
        Поддерживает как dict, так и dataclass с полем data.
        """
        if hasattr(config, 'data'):
            return cls(config.data)
        elif isinstance(config, dict) and 'data' in config:
            return cls(config['data'])
        return cls(config)
    
    def load(self) -> Dataset:
        """Загрузка всех сплитов"""
        print("=" * 60)
        print("ЗАГРУЗКА ДАННЫХ")
        print("=" * 60)
        
        # Загрузка файлов
        train_samples = self._load_file(
            self.config['train_path'], 
            limit=self.train_limit,
            split_name='train'
        )
        val_samples = self._load_file(
            self.config['val_path'], 
            limit=self.val_limit,
            split_name='val'
        )
        test_samples = self._load_file(
            self.config['test_path'], 
            limit=self.test_limit,
            split_name='test'
        )
        
        # Фильтрация
        train_samples = self._filter_samples(train_samples)
        val_samples = self._filter_samples(val_samples)
        test_samples = self._filter_samples(test_samples)
        
        print(f"\nПосле фильтрации:")
        print(f"  Train: {len(train_samples)}")
        print(f"  Val: {len(val_samples)}")
        print(f"  Test: {len(test_samples)}")
        print("=" * 60)
        
        # Создание Dataset
        dataset = Dataset(
            train=[self._to_code_sample(s) for s in train_samples],
            val=[self._to_code_sample(s) for s in val_samples],
            test=[self._to_code_sample(s) for s in test_samples],
            code_field=self.code_field,
            label_field=self.label_field
        )
        
        return dataset
    
    def _load_file(self, path: str, limit: Optional[int], split_name: str) -> List[Dict]:
        """Загрузка одного файла"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        samples = []
        file_ext = path.suffix.lower()
        
        if file_ext == '.jsonl':
            samples = self._load_jsonl(path, limit)
        elif file_ext == '.json':
            samples = self._load_json(path, limit)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        print(f"  {split_name.capitalize()}: загружено {len(samples)} (limit={limit})")
        return samples
    
    def _load_jsonl(self, path: Path, limit: Optional[int]) -> List[Dict]:
        """Чтение JSONL"""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if limit and len(samples) >= limit:
                    break
                if line.strip():
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return samples
    
    def _load_json(self, path: Path, limit: Optional[int]) -> List[Dict]:
        """Чтение JSON"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data[:limit] if limit else data
        elif isinstance(data, dict):
            # Предполагаем структуру {"samples": [...]}
            samples = data.get('samples', [])
            return samples[:limit] if limit else samples
        else:
            raise ValueError(f"Unexpected JSON structure in {path}")
    
    def _filter_samples(self, samples: List[Dict]) -> List[Dict]:
        """Фильтрация образцов по длине кода"""
        filtered = []
        
        for sample in samples:
            code = sample.get(self.code_field, '')
            
            # Проверка наличия кода
            if not code or not code.strip():
                continue
            
            # Проверка длины
            code_len = len(code)
            if code_len < self.min_code_length:
                continue
            if self.max_code_length and code_len > self.max_code_length:
                continue
            
            # Проверка метки (опционально)
            label = sample.get(self.label_field)
            if label and str(label).strip().lower() in ['', 'unknown', 'none', 'null']:
                label = None
            
            filtered.append({
                'code': code,
                'label': str(label).strip() if label else None,
                'metadata': {k: v for k, v in sample.items() 
                           if k not in [self.code_field, self.label_field]}
            })
        
        return filtered
    
    def _to_code_sample(self, sample_dict: Dict) -> CodeSample:
        """Преобразование словаря в CodeSample"""
        return CodeSample(
            code=sample_dict['code'],
            label=sample_dict.get('label'),
            metadata=sample_dict.get('metadata', {})
        )
