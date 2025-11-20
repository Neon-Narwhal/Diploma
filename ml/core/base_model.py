"""
Базовый интерфейс для всех ML-моделей.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseModel(ABC):
    """
    Абстрактный класс для всех ML-моделей.
    Обеспечивает единообразный интерфейс.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """
        Обучение модели.
        
        Args:
            X: признаки (n_samples, n_features)
            y: таргет (n_samples,)
            
        Returns:
            self для chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание классов.
        
        Args:
            X: признаки (n_samples, n_features)
            
        Returns:
            Предсказанные классы (n_samples,)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание вероятностей классов.
        
        Args:
            X: признаки (n_samples, n_features)
            
        Returns:
            Вероятности для каждого класса (n_samples, n_classes)
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> np.ndarray:
        """
        Получение важности признаков.
        
        Returns:
            Важность каждого признака (n_features,)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Сохранение модели на диск.
        
        Args:
            path: путь для сохранения
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> 'BaseModel':
        """
        Загрузка модели с диска.
        
        Args:
            path: путь к модели
            
        Returns:
            self для chaining
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Получение параметров модели"""
        return self.params.copy()
    
    def set_params(self, **params) -> 'BaseModel':
        """Установка параметров модели"""
        self.params.update(params)
        return self
