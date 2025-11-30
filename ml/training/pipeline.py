"""
Главный ML pipeline для обучения и оценки.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path

from ml.core.base_model import BaseModel
from ml.core.model_factory import ModelFactory
from ml.core.model_config import ModelConfig
from ml.features.pipeline import FeaturePipeline
from ml.training.cross_validation import CrossValidator
from ml.training.optimization import OptunaOptimizer
from ml.training.callbacks import EarlyStopping, LoggingCallback
from ml.evaluation.metrics import compute_metrics
from ml.utils.logger import MLLogger


class MLPipeline:
    """
    Универсальный pipeline для ML:
    1. Feature engineering
    2. Обучение модели
    3. Cross-validation (опционально)
    4. Оптимизация (опционально)
    5. Оценка
    6. Логирование
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        feature_config: Optional[Dict[str, Any]] = None,
        cv_config: Optional[Dict[str, Any]] = None,
        optimization_config: Optional[Dict[str, Any]] = None,
        logger: Optional[MLLogger] = None,
    ):
        """
        Args:
            model_config: конфигурация модели
            feature_config: конфигурация feature engineering
            cv_config: конфигурация cross-validation
            optimization_config: конфигурация оптимизации
            logger: логгер для MLflow/JSON
        """
        self.model_config = model_config
        self.feature_config = feature_config or {}
        self.cv_config = cv_config
        self.optimization_config = optimization_config
        self.logger = logger
        
        # Компоненты
        self.feature_pipeline = None
        self.model = None
        
        # Результаты
        self.results = {}
    
    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        code_samples_train: Optional[List[str]] = None,
        code_samples_test: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Запуск полного pipeline.
        """
        # 1. Feature engineering
        if code_samples_train is not None:
            X_train, X_test = self._prepare_features(
                code_samples_train,
                code_samples_test,
                y_train,
            )
        
        # 2. Оптимизация гиперпараметров
        # Вызываем метод, который обновит self.model_config.params
        self._optimize_hyperparameters(X_train, y_train, X_val, y_val)
        
        # 3. Обучение финальной модели
        print(f"\nОбучение финальной модели с параметрами:")
        print(f"{self.model_config.params}")  # DEBUG: Проверяем, что параметры обновились

        # Создаем НОВЫЙ экземпляр модели с ОБНОВЛЕННЫМИ параметрами
        from ml.core.model_factory import ModelFactory
        self.model = ModelFactory.create(self.model_config)
        
        # Cross-validation results (если включено)
        if self.cv_config and self.cv_config.get('enabled', False):
            cv_results = self._train_with_cv(X_train, y_train)
            self.results['cv_results'] = cv_results
        
        # Финальный fit
        try:
            # Пытаемся передать eval_set (для CatBoost/LGBM/XGBoost)
            if X_val is not None and y_val is not None:
                # Для некоторых моделей (OvR) eval_set не поддерживается и вызовет TypeError
                # Но мы перехватим его ниже, если это kwarg error
                # Однако OvR не принимает eval_set в принципе, так что лучше проверить тип
                
                # Костыль: OvR модели не поддерживают eval_set в fit
                if 'ovr' in self.model_config.type:
                     self.model.fit(X_train, y_train)
                else:
                     self.model.fit(X_train, y_train, eval_set=(X_val, y_val))
            else:
                self.model.fit(X_train, y_train)
        except TypeError as e:
            print(f"Warning: fit() with eval_set failed ({e}), retrying without eval_set...")
            self.model.fit(X_train, y_train)
        
        # 4. Оценка на train
        train_metrics = self._evaluate(X_train, y_train, prefix='train')
        self.results['train_metrics'] = train_metrics
        
        # 5. Оценка на val
        if X_val is not None and y_val is not None:
            val_metrics = self._evaluate(X_val, y_val, prefix='val')
            self.results['val_metrics'] = val_metrics
        
        # 6. Оценка на test
        if X_test is not None and y_test is not None:
            test_metrics = self._evaluate(X_test, y_test, prefix='test')
            self.results['test_metrics'] = test_metrics
        
        # 7. Логирование
        if self.logger:
            self._log_results()
        
        return self.results

    
    def _prepare_features(
        self,
        code_samples_train: List[str],
        code_samples_test: Optional[List[str]],
        y_train: np.ndarray,
    ) -> tuple:
        """Извлечение и обработка признаков"""
        self.feature_pipeline = FeaturePipeline(**self.feature_config)
        
        # Обучение и трансформация train
        X_train = self.feature_pipeline.fit_transform(code_samples_train, y_train)
        
        # Трансформация test
        X_test = None
        if code_samples_test is not None:
            X_test = self.feature_pipeline.transform(code_samples_test)
        
        return X_train, X_test
    
    def _optimize_hyperparameters(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """Оптимизация гиперпараметров"""
        # Проверка enabled
        if not self.optimization_config or not self.optimization_config.get('enabled', False):
            return

        print(f"Запуск оптимизации гиперпараметров ({self.optimization_config.get('n_trials', 10)} trials)...")
        
        from ml.training.optimization import OptunaOptimizer
        
        optimizer = OptunaOptimizer(
            model_config=self.model_config,
            optimization_config=self.optimization_config,
            cv_config=self.cv_config
        )
        
        # Optuna возвращает словарь лучших параметров
        best_params, best_value = optimizer.optimize(
            X_train, y_train, 
            X_val=X_val, 
            y_val=y_val
        )
        
        print(f"✓ Оптимизация завершена. Best score: {best_value:.4f}")
        print(f"  Best params: {best_params}")

        # Обновляем параметры модели В ТЕКУЩЕМ конфиге
        self.model_config.params.update(best_params)
        
        # Сохраняем результаты для отчета
        self.results['optimization'] = {
            'best_params': best_params,
            'best_value': float(best_value),
            'n_trials': self.optimization_config.get('n_trials')
        }

    def _train_with_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Обучение с cross-validation"""
        # Удаляем поля, которые не относятся к CrossValidator
        cv_config = self.cv_config.copy() if isinstance(self.cv_config, dict) else {}
        cv_config.pop('enabled', None)  # Убираем 'enabled'
        cv_config.pop('metrics', None)  # Убираем 'metrics', используем отдельно
        
        # Получаем метрики
        metrics = self.cv_config.get('metrics', ['accuracy', 'f1_macro']) if isinstance(self.cv_config, dict) else ['accuracy', 'f1_macro']
        
        cv = CrossValidator(**cv_config)
        
        # Создаем новую модель для CV
        model = ModelFactory.create(self.model_config)
        
        cv_results = cv.run(
            model=model,
            X=X,
            y=y,
            metrics=metrics,
        )
        
        return cv_results

    
    def _evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        prefix: str = '',
    ) -> Dict[str, float]:
        """Оценка модели"""
        y_pred = self.model.predict(X)
        
        metrics = compute_metrics(
            y_true=y,
            y_pred=y_pred,
            metrics=['accuracy', 'f1_macro', 'f1_micro', 'precision_macro', 'recall_macro'],
        )
        
        # Добавляем prefix к метрикам
        if prefix:
            metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
        
        return metrics
    
    def _log_results(self):
        """Логирование результатов"""
        # Логируем параметры модели
        self.logger.log_params(self.model_config.params)
        
        # Логируем метрики
        if 'train_metrics' in self.results:
            self.logger.log_metrics(self.results['train_metrics'])
        
        if 'test_metrics' in self.results:
            self.logger.log_metrics(self.results['test_metrics'])
        
        if 'cv_results' in self.results:
            cv_metrics = self.results['cv_results']['mean']
            cv_metrics = {f"cv_{k}": v for k, v in cv_metrics.items()}
            self.logger.log_metrics(cv_metrics)
        
        # Логируем важность признаков
        if hasattr(self.model, 'get_feature_importance'):
            importance = self.model.get_feature_importance()
            self.logger.log_artifact('feature_importance', importance)
    
    def save_model(self, path: str):
        """Сохранение модели"""
        if self.model is None:
            raise RuntimeError("Model must be trained before saving")
        
        from pathlib import Path
        
        # Создаем директорию если не существует
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(str(output_path))
        
        if self.logger:
            self.logger.log_model(str(output_path))

