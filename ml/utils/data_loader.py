"""
Универсальный загрузчик данных.
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

from ml.features.extractors import ComplexityFeatureExtractor


@dataclass
class StandardizedData:
    """Универсальный контейнер данных"""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    
    label_encoder: LabelEncoder
    feature_names: List[str]
    
    # Опционально
    code_train: Optional[List[str]] = None
    code_val: Optional[List[str]] = None
    code_test: Optional[List[str]] = None


class DataLoader:
    """Универсальный загрузчик для предразделенных данных"""
    
    def __init__(self, data_config: dict):
        self.config = data_config
        self.feature_names_ = None
        
        # Список экстракторов (всегда список!)
        self.extractors = []
        
        feat_config = self.config.get('features', {})
        extractor_type = feat_config.get('type', 'complexity')
        
        # Логика инициализации
        if extractor_type == 'bert':
            from ml.features.bert_extractor import CodeBertExtractor
            self.extractors.append(CodeBertExtractor(
                model_name=feat_config.get('model_name', "jinaai/jina-embeddings-v2-base-code"),
                batch_size=feat_config.get('batch_size', 8),
                max_length=feat_config.get('max_length', 1024),
                device=feat_config.get('device')
            ))
            
        elif extractor_type == 'hybrid':
            from ml.features.bert_extractor import CodeBertExtractor
            # 1. BERT с PCA
            self.extractors.append(CodeBertExtractor(
                model_name=feat_config.get('model_name', "jinaai/jina-embeddings-v2-base-code"),
                batch_size=feat_config.get('batch_size', 8),
                max_length=feat_config.get('max_length', 1024),
                device=feat_config.get('device'),
                
                # Читаем параметры PCA
                use_pca=feat_config.get('use_pca', False),
                n_components=feat_config.get('n_components', 32)
            ))
            # 2. AST + TF-IDF
            self.extractors.append(ComplexityFeatureExtractor(
                max_tfidf_features=500 
            ))
            
        else: # 'complexity' или дефолт
            # ComplexityFeatureExtractor уже импортирован в начале файла
            self.extractors.append(ComplexityFeatureExtractor(
                max_tfidf_features=feat_config.get('max_tfidf_features', 2000)
            ))
    
    @classmethod
    def from_config(cls, experiment_config):
        """Создание из конфига эксперимента"""
        if hasattr(experiment_config, 'data'):
            return cls(experiment_config.data)
        return cls(experiment_config['data'])
    
    def load(self) -> StandardizedData:
        # === КЭШИРОВАНИЕ: проверка и загрузка ===
        cache_path = self._get_cache_path()
        
        if cache_path:
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                print(f"✓ Признаки загружены из кэша: {cache_path.name}")
                print(f"  Train: {cached_data.X_train.shape}")
                print(f"  Val: {cached_data.X_val.shape}")
                print(f"  Test: {cached_data.X_test.shape}")
                return cached_data
            else:
                print(f"Кэш не найден, извлекаем признаки...")
        
        # Читаем настройки лимитов
        limits = self.config.get('preprocessing', {}).get('split_limits', {})
        global_limit = self.config.get('preprocessing', {}).get('max_samples_per_split')
        
        lim_train = limits.get('train', global_limit)
        lim_val = limits.get('val', global_limit)
        lim_test = limits.get('test', global_limit)

        print("Загрузка данных с лимитами:")
        print(f"  Train limit: {lim_train}")
        print(f"  Val limit:   {lim_val}")
        print(f"  Test limit:  {lim_test}")

        # 1. Загрузка файлов с разными лимитами
        train_samples = self._load_jsonl(self.config['train_path'], limit=lim_train)
        val_samples = self._load_jsonl(self.config['val_path'], limit=lim_val)
        test_samples = self._load_jsonl(self.config['test_path'], limit=lim_test)
        
        print(f"  Загружено: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
        
        # 2. Препроцессинг (фильтрация)
        train_samples = self._preprocess(train_samples)
        val_samples = self._preprocess(val_samples)
        test_samples = self._preprocess(test_samples)
        
        # 3. Извлечение признаков и кодирование меток
        print("\nИзвлечение признаков (Train)...")
        X_train, y_train, codes_train, encoder = self._extract_and_encode(
            train_samples, fit_encoder=True, fit_extractor=True
        )
        
        print(f"Извлечение признаков (Val)...")
        X_val, y_val, codes_val, _ = self._extract_and_encode(
            val_samples, label_encoder=encoder, fit_extractor=False
        )
        
        print(f"Извлечение признаков (Test)...")
        X_test, y_test, codes_test, _ = self._extract_and_encode(
            test_samples, label_encoder=encoder, fit_extractor=False
        )
        
        # 4. Удаление константных признаков
        if self.config.get('features', {}).get('remove_constant', True):
            X_train, X_val, X_test = self._remove_constant_features(X_train, X_val, X_test)
            
        print(f"\nИтоговые данные: {X_train.shape[1]} признаков")
        print(f"  Классы: {len(encoder.classes_)}")
        
        data = StandardizedData(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            label_encoder=encoder,
            feature_names=self.feature_names_,
            code_train=codes_train,
            code_val=codes_val,
            code_test=codes_test,
        )
        
        # === КЭШИРОВАНИЕ: сохранение ===
        if cache_path:
            self._save_to_cache(data, cache_path)
            print(f"✓ Признаки сохранены в кэш: {cache_path.name}")
        
        return data

    
    def _load_jsonl(self, path: str, limit: Optional[int] = None) -> List[Dict]:
        """Чтение JSONL файла"""
        samples = []
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if limit and len(samples) >= limit:
                    break
                if line.strip():
                    try:
                        samples.append(json.loads(line))
                    except:
                        continue
        return samples
    
    def _preprocess(self, samples: List[Dict]) -> List[Dict]:
        """Фильтрация примеров"""
        filtered = []
        code_field = self.config.get('code_field', 'code')
        label_field = self.config.get('label_field', 'time_complexity_mapped')
        min_len = self.config.get('preprocessing', {}).get('min_code_length', 10)
        
        for s in samples:
            code = s.get(code_field, '')
            label = s.get(label_field)
            
            if not code or len(code.strip()) < min_len:
                continue
                
            if not label or str(label).strip() in ['', 'Unknown', 'UNKNOWN', 'None', 'null']:
                continue
                
            filtered.append({'code': code, 'label': str(label).strip()})
            
        return filtered

    def _extract_and_encode(self, samples, fit_encoder=False, label_encoder=None, fit_extractor=False):
        """Извлечение X, y и кодирование"""
        codes = [s['code'] for s in samples]
        labels = [s['label'] for s in samples]
        
        # Кодирование меток
        if fit_encoder:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(labels)
        else:
            known_classes = set(label_encoder.classes_)
            mask = [l in known_classes for l in labels]
            
            codes = [c for c, m in zip(codes, mask) if m]
            labels = [l for l, m in zip(labels, mask) if m]
            y = label_encoder.transform(labels)

        # Извлечение признаков (Проход по всем экстракторам)
        features_list = []
        
        for extractor in self.extractors:
            # Управление состоянием (fit только если нужно и есть такая возможность)
            if fit_extractor and hasattr(extractor, 'is_fitted'):
                extractor.is_fitted = False
            
            # Извлекаем DataFrame
            df_part = extractor.extract(codes)
            features_list.append(df_part)
        
        # Объединяем все части
        if features_list:
            full_df = pd.concat(features_list, axis=1)
        else:
            full_df = pd.DataFrame()
        
        if fit_encoder:
            self.feature_names_ = list(full_df.columns)
            
        return full_df.values, y, codes, label_encoder

    def _remove_constant_features(self, X_train, X_val, X_test):
        """Удаление константных признаков"""
        # Добавляем проверку на пустой массив
        if X_train.shape[1] == 0:
            return X_train, X_val, X_test

        std = X_train.std(axis=0)
        mask = std > 1e-6
        
        if not mask.all():
            removed = (~mask).sum()
            # print(f"  Удалено {removed} константных признаков")
            
            # Обновляем список имен признаков
            if self.feature_names_:
                self.feature_names_ = [f for f, m in zip(self.feature_names_, mask) if m]
            
            return X_train[:, mask], X_val[:, mask], X_test[:, mask]
        
        return X_train, X_val, X_test
    
    def _get_cache_path(self) -> Optional[Path]:
        """Получение пути к кэшу"""
        cache_config = self.config.get('feature_cache', {})
        
        if not cache_config.get('enabled', False):
            return None
        
        cache_key = cache_config.get('cache_key')
        if not cache_key:
            return None
        
        cache_dir = Path(cache_config.get('cache_dir', 'data/feature_cache'))
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        return cache_dir / f"{cache_key}.npz"

    def _load_from_cache(self, cache_path: Path) -> Optional[StandardizedData]:
        """Загрузка признаков из кэша"""
        if not cache_path.exists():
            return None
        
        try:
            data = np.load(cache_path, allow_pickle=True)
            
            # Восстанавливаем label_encoder
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            encoder.classes_ = data['label_encoder_classes']
            
            return StandardizedData(
                X_train=data['X_train'],
                y_train=data['y_train'],
                X_val=data['X_val'],
                y_val=data['y_val'],
                X_test=data['X_test'],
                y_test=data['y_test'],
                label_encoder=encoder,
                feature_names=list(data.get('feature_names', [])),
                code_train=list(data.get('code_train', [])),
                code_val=list(data.get('code_val', [])),
                code_test=list(data.get('code_test', []))
            )
        except Exception as e:
            print(f"Ошибка загрузки кэша: {e}")
            return None

    def _save_to_cache(self, data: StandardizedData, cache_path: Path):
        """Сохранение признаков в кэш"""
        try:
            np.savez_compressed(
                cache_path,
                X_train=data.X_train,
                y_train=data.y_train,
                X_val=data.X_val,
                y_val=data.y_val,
                X_test=data.X_test,
                y_test=data.y_test,
                label_encoder_classes=data.label_encoder.classes_,
                feature_names=data.feature_names,
                code_train=data.code_train,
                code_val=data.code_val,
                code_test=data.code_test
            )
        except Exception as e:
            print(f"Ошибка сохранения кэша: {e}")

