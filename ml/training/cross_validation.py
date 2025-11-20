"""
Cross-validation стратегии.
"""

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from typing import Dict, Any, List, Tuple, Optional
from ml.core.base_model import BaseModel
from ml.evaluation.metrics import compute_metrics


class CrossValidator:
    """
    Кросс-валидация для ML-моделей.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        stratified: bool = True,
        shuffle: bool = True,
        random_state: Optional[int] = 42,
    ):
        """
        Args:
            n_folds: количество фолдов
            stratified: использовать StratifiedKFold
            shuffle: перемешивать данные
            random_state: seed для воспроизводимости
        """
        self.n_folds = n_folds
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state
        
        self.splitter = self._create_splitter()
    
    def _create_splitter(self):
        """Создание splitter"""
        if self.stratified:
            return StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
        else:
            return KFold(
                n_splits=self.n_folds,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
    
    def run(
        self,
        model: BaseModel,
        X: np.ndarray,
        y: np.ndarray,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Запуск кросс-валидации.
        
        Args:
            model: модель для обучения
            X: признаки
            y: таргет
            metrics: список метрик для вычисления
            
        Returns:
            Результаты CV: метрики по фолдам и агрегированные
        """
        if metrics is None:
            metrics = ['accuracy', 'f1_macro']
        
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.splitter.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Обучаем модель
            model.fit(X_train, y_train)
            
            # Предсказания
            y_pred = model.predict(X_val)
            
            # Вычисляем метрики
            fold_metrics = compute_metrics(y_val, y_pred, metrics)
            fold_metrics['fold'] = fold_idx
            
            fold_results.append(fold_metrics)
        
        # Агрегация результатов
        aggregated = self._aggregate_results(fold_results, metrics)
        
        return {
            'fold_results': fold_results,
            'mean': aggregated['mean'],
            'std': aggregated['std'],
            'all_folds': fold_results,
        }
    
    def _aggregate_results(
        self,
        fold_results: List[Dict[str, float]],
        metrics: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Агрегация метрик по фолдам"""
        mean_metrics = {}
        std_metrics = {}
        
        for metric in metrics:
            values = [result[metric] for result in fold_results]
            mean_metrics[metric] = np.mean(values)
            std_metrics[metric] = np.std(values)
        
        return {
            'mean': mean_metrics,
            'std': std_metrics,
        }
