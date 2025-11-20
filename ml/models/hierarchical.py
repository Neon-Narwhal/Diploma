"""
Иерархические стратегии: Cascade, Per-class.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable
from ml.core.base_model import BaseModel
from ml.core.model_factory import ModelFactory
from ml.core.model_config import ModelConfig


class PerClassModel(BaseModel):
    """
    Per-class стратегия: отдельная модель для каждого класса.
    Каждая модель решает binary задачу: класс vs остальные.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        
        self.model_config_template = self.params.get('model_template')
        self.aggregation = self.params.get('aggregation', 'max_confidence')
        
        self.models: Dict[int, BaseModel] = {}
        self.classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PerClassModel':
        """Обучение отдельной модели для каждого класса"""
        self.classes_ = np.unique(y)
        
        for class_idx in self.classes_:
            # Создаем binary таргет: класс vs остальные
            y_binary = (y == class_idx).astype(int)
            
            # Создаем и обучаем модель
            if isinstance(self.model_config_template, dict):
                config = ModelConfig(**self.model_config_template)
            else:
                config = self.model_config_template
            
            model = ModelFactory.create(config)
            model.fit(X, y_binary)
            
            self.models[class_idx] = model
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание через агрегацию результатов всех моделей"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict")
        
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Вероятности для каждого класса"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba")
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))
        
        for i, class_idx in enumerate(self.classes_):
            model = self.models[class_idx]
            # Вероятность положительного класса (этот класс)
            proba[:, i] = model.predict_proba(X)[:, 1]
        
        # Нормализация вероятностей
        proba_sum = proba.sum(axis=1, keepdims=True)
        proba = proba / (proba_sum + 1e-10)
        
        return proba
    
    def get_feature_importance(self) -> np.ndarray:
        """Усредненная важность по всем моделям"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before get_feature_importance")
        
        importances = np.array([
            model.get_feature_importance() 
            for model in self.models.values()
        ])
        return np.mean(importances, axis=0)
    
    def save(self, path: str) -> None:
        """Сохранение всех моделей"""
        import joblib
        joblib.dump({
            'models': self.models,
            'classes': self.classes_,
            'params': self.params,
        }, path)
    
    def load(self, path: str) -> 'PerClassModel':
        """Загрузка всех моделей"""
        import joblib
        data = joblib.load(path)
        
        self.models = data['models']
        self.classes_ = data['classes']
        self.params = data['params']
        self.is_fitted = True
        
        return self


class CascadeModel(BaseModel):
    """
    Каскадная модель: последовательность моделей с условной маршрутизацией.
    
    Например:
    - Level 1: O(1) vs остальные
    - Level 2 (если не O(1)): O(n) vs O(n^2) vs O(log n)
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        
        self.levels_config = self.params.get('levels', [])
        self.models: List[BaseModel] = []
        self.routing_rules: List[Callable] = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CascadeModel':
        """Обучение каскада моделей"""
        current_X = X
        current_y = y
        current_indices = np.arange(len(y))
        
        for level_config in self.levels_config:
            # Создаем модель для уровня
            model_config = ModelConfig(**level_config['model'])
            model = ModelFactory.create(model_config)
            
            # Маппинг классов для этого уровня
            target_mapping = level_config.get('target_mapping')
            if target_mapping:
                # Преобразуем таргет по маппингу
                level_y = self._map_targets(current_y, target_mapping)
            else:
                level_y = current_y
            
            # Обучаем модель
            model.fit(current_X, level_y)
            self.models.append(model)
            
            # Условие маршрутизации для следующего уровня
            condition = level_config.get('condition')
            if condition:
                # Фильтруем данные для следующего уровня
                predictions = model.predict(current_X)
                mask = self._apply_condition(predictions, condition)
                
                current_X = current_X[mask]
                current_y = current_y[mask]
                current_indices = current_indices[mask]
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание через каскад"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict")
        
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        active_mask = np.ones(n_samples, dtype=bool)
        
        for i, (model, level_config) in enumerate(zip(self.models, self.levels_config)):
            if not active_mask.any():
                break
            
            # Предсказание для активных примеров
            level_predictions = model.predict(X[active_mask])
            predictions[active_mask] = level_predictions
            
            # Обновляем маску для следующего уровня
            if i < len(self.models) - 1:
                condition = level_config.get('condition')
                if condition:
                    level_mask = self._apply_condition(level_predictions, condition)
                    active_indices = np.where(active_mask)[0]
                    active_mask[active_indices] = level_mask
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Вероятности — используется последняя модель в каскаде"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba")
        
        # Упрощенная версия: возвращаем вероятности последней модели
        return self.models[-1].predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Важность из первой модели каскада"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before get_feature_importance")
        
        return self.models[0].get_feature_importance()
    
    def _map_targets(self, y: np.ndarray, mapping: Dict) -> np.ndarray:
        """Преобразование таргетов по маппингу"""
        mapped_y = np.zeros_like(y)
        for new_label, old_labels in mapping.items():
            for old_label in old_labels:
                mapped_y[y == old_label] = new_label
        return mapped_y
    
    def _apply_condition(self, predictions: np.ndarray, condition: str) -> np.ndarray:
        """Применение условия маршрутизации"""
        # Простейшая реализация: condition задает класс для фильтрации
        # Например: "if level1 != 0" -> фильтруем все, кроме класса 0
        # Можно расширить для более сложных условий
        return predictions != 0  # Упрощенная версия
    
    def save(self, path: str) -> None:
        """Сохранение каскада"""
        import joblib
        joblib.dump({
            'models': self.models,
            'levels_config': self.levels_config,
            'params': self.params,
        }, path)
    
    def load(self, path: str) -> 'CascadeModel':
        """Загрузка каскада"""
        import joblib
        data = joblib.load(path)
        
        self.models = data['models']
        self.levels_config = data['levels_config']
        self.params = data['params']
        self.is_fitted = True
        
        return self


# Регистрация в реестре
from ml.core.registry import ModelRegistry

ModelRegistry.register('per_class', PerClassModel)
ModelRegistry.register('cascade', CascadeModel)
