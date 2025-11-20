"""
Трансформеры для обработки признаков.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Optional


class FeatureTransformer:
    """
    Универсальный трансформер признаков.
    Поддерживает различные виды scaling.
    """
    
    def __init__(self, method: str = 'standard'):
        """
        Args:
            method: метод scaling ('standard', 'minmax', 'robust')
        """
        self.method = method
        self.scaler = self._create_scaler()
        self.is_fitted = False
    
    def _create_scaler(self):
        """Создание scaler по методу"""
        if self.method == 'standard':
            return StandardScaler()
        elif self.method == 'minmax':
            return MinMaxScaler()
        elif self.method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
    
    def fit(self, X: np.ndarray) -> 'FeatureTransformer':
        """Обучение трансформера"""
        self.scaler.fit(X)
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Применение трансформации"""
        if not self.is_fitted:
            raise RuntimeError("Transformer must be fitted before transform")
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Обучение и трансформация"""
        return self.fit(X).transform(X)
