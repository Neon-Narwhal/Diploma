"""
Ансамбли моделей: Voting, Stacking.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from ml.core.base_model import BaseModel
from ml.core.model_factory import ModelFactory
from ml.core.model_config import ModelConfig


class VotingEnsemble(BaseModel):
    """
    Voting ensemble: комбинирует предсказания нескольких моделей.
    
    Поддерживает:
    - hard voting: голосование по классу
    - soft voting: усреднение вероятностей
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        
        self.models_configs = self.params.get('models', [])
        self.voting = self.params.get('voting', 'soft')  # 'hard' или 'soft'
        self.weights = self.params.get('weights', None)
        
        if self.voting not in ['hard', 'soft']:
            raise ValueError("voting must be 'hard' or 'soft'")
        
        self.models: List[BaseModel] = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'VotingEnsemble':
        """Обучение всех моделей в ансамбле"""
        self.models = []
        
        for model_config in self.models_configs:
            if isinstance(model_config, dict):
                model_config = ModelConfig(**model_config)
            
            model = ModelFactory.create(model_config)
            model.fit(X, y)
            self.models.append(model)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание через voting"""
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before predict")
        
        if self.voting == 'soft':
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
        else:
            # Hard voting
            predictions = np.array([model.predict(X) for model in self.models])
            
            if self.weights is not None:
                # Weighted voting
                weighted_votes = np.zeros((X.shape[0], len(np.unique(predictions))))
                for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
                    for j, class_idx in enumerate(pred):
                        weighted_votes[j, class_idx] += weight
                return np.argmax(weighted_votes, axis=1)
            else:
                # Majority voting
                from scipy.stats import mode
                return mode(predictions, axis=0)[0].flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Предсказание вероятностей через усреднение"""
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before predict_proba")
        
        all_proba = np.array([model.predict_proba(X) for model in self.models])
        
        if self.weights is not None:
            weights = np.array(self.weights).reshape(-1, 1, 1)
            weighted_proba = all_proba * weights
            return np.sum(weighted_proba, axis=0) / np.sum(self.weights)
        else:
            return np.mean(all_proba, axis=0)
    
    def get_feature_importance(self) -> np.ndarray:
        """Усредненная важность признаков по всем моделям"""
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before get_feature_importance")
        
        importances = np.array([model.get_feature_importance() for model in self.models])
        return np.mean(importances, axis=0)
    
    def save(self, path: str) -> None:
        """Сохранение ансамбля"""
        import joblib
        joblib.dump({
            'models': self.models,
            'params': self.params,
            'voting': self.voting,
            'weights': self.weights,
        }, path)
    
    def load(self, path: str) -> 'VotingEnsemble':
        """Загрузка ансамбля"""
        import joblib
        data = joblib.load(path)
        
        self.models = data['models']
        self.params = data['params']
        self.voting = data['voting']
        self.weights = data['weights']
        self.is_fitted = True
        
        return self


# Регистрация в реестре
from ml.core.registry import ModelRegistry

ModelRegistry.register('voting', VotingEnsemble)
