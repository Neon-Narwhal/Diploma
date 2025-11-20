"""
Feature engineering pipeline.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from ml.features.extractors import ComplexityFeatureExtractor, TokenFeatureExtractor
from ml.features.transformers import FeatureTransformer
from ml.features.selectors import FeatureSelector


class FeaturePipeline:
    """
    Полный pipeline обработки признаков:
    1. Извлечение (extraction)
    2. Преобразование (transformation)
    3. Отбор (selection)
    """
    
    def __init__(
        self,
        extractors: Optional[List[str]] = None,
        transformer_method: str = 'standard',
        selector_method: Optional[str] = None,
        n_features: Optional[int] = None,
    ):
        """
        Args:
            extractors: список типов извлекателей ('complexity', 'token')
            transformer_method: метод scaling
            selector_method: метод feature selection (None = без отбора)
            n_features: количество признаков для отбора
        """
        self.extractors = extractors or ['complexity']
        self.transformer_method = transformer_method
        self.selector_method = selector_method
        self.n_features = n_features
        
        # Компоненты pipeline
        self.complexity_extractor = None
        self.token_extractor = None
        self.transformer = None
        self.selector = None
        
        self.is_fitted = False
        self.feature_names_ = None
    
    def fit(self, code_samples: List[str], y: Optional[np.ndarray] = None) -> 'FeaturePipeline':
        """Обучение pipeline"""
        # 1. Извлечение признаков
        X = self._extract_features(code_samples)
        
        # 2. Обучение transformer
        self.transformer = FeatureTransformer(self.transformer_method)
        X_transformed = self.transformer.fit_transform(X)
        
        # 3. Обучение selector (если нужно)
        if self.selector_method and y is not None:
            self.selector = FeatureSelector(self.selector_method, self.n_features)
            self.selector.fit(X_transformed, y)
        
        self.is_fitted = True
        return self
    
    def transform(self, code_samples: List[str]) -> np.ndarray:
        """Применение pipeline"""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        
        # 1. Извлечение
        X = self._extract_features(code_samples)
        
        # 2. Трансформация
        X = self.transformer.transform(X)
        
        # 3. Отбор
        if self.selector:
            X = self.selector.transform(X)
        
        return X
    
    def fit_transform(self, code_samples: List[str], y: Optional[np.ndarray] = None) -> np.ndarray:
        """Обучение и применение"""
        return self.fit(code_samples, y).transform(code_samples)
    
    def _extract_features(self, code_samples: List[str]) -> np.ndarray:
        """Извлечение всех типов признаков"""
        dfs = []
        
        if 'complexity' in self.extractors:
            if self.complexity_extractor is None:
                self.complexity_extractor = ComplexityFeatureExtractor()
            df_complexity = self.complexity_extractor.extract(code_samples)
            dfs.append(df_complexity)
        
        if 'token' in self.extractors:
            if self.token_extractor is None:
                self.token_extractor = TokenFeatureExtractor()
            df_token = self.token_extractor.extract(code_samples)
            dfs.append(df_token)
        
        # Объединение всех признаков
        df_all = pd.concat(dfs, axis=1)
        self.feature_names_ = df_all.columns.tolist()
        
        return df_all.values
    
    def get_feature_names(self) -> List[str]:
        """Получение имен признаков"""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted first")
        
        names = self.feature_names_
        
        if self.selector:
            indices = self.selector.get_selected_features()
            names = [names[i] for i in indices]
        
        return names
