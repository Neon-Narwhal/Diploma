"""
Feature selection методы.
"""

import numpy as np
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from typing import Optional


class FeatureSelector:
    """
    Отбор наиболее важных признаков.
    """
    
    def __init__(self, method: str = 'mutual_info', n_features: Optional[int] = None):
        """
        Args:
            method: метод отбора ('mutual_info', 'f_classif', 'variance')
            n_features: количество признаков для отбора (если None, все)
        """
        self.method = method
        self.n_features = n_features
        self.selector = None
        self.selected_indices_ = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FeatureSelector':
        """Обучение селектора"""
        if self.n_features is None:
            self.n_features = X.shape[1]
        
        if self.method == 'mutual_info':
            scores = mutual_info_classif(X, y)
        elif self.method == 'f_classif':
            scores, _ = f_classif(X, y)
        elif self.method == 'variance':
            scores = np.var(X, axis=0)
        else:
            raise ValueError(f"Unknown selection method: {self.method}")
        
        # Отбираем top-k признаков
        self.selected_indices_ = np.argsort(scores)[-self.n_features:]
        self.is_fitted = True
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Применение отбора"""
        if not self.is_fitted:
            raise RuntimeError("Selector must be fitted before transform")
        return X[:, self.selected_indices_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Обучение и применение"""
        return self.fit(X, y).transform(X)
    
    def get_selected_features(self) -> np.ndarray:
        """Индексы отобранных признаков"""
        if not self.is_fitted:
            raise RuntimeError("Selector must be fitted first")
        return self.selected_indices_
