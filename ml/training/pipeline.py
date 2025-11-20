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
        code_samples_train: Optional[List[str]] = None,
        code_samples_test: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Запуск полного pipeline.
        
        Args:
            X_train: признаки для обучения (если уже извлечены)
            y_train: таргет для обучения
            X_test: признаки для теста
            y_test: таргет для теста
            code_samples_train: исходный код для извлечения признаков (train)
            code_samples_test: исходный код для извлечения признаков (test)
            
        Returns:
            Результаты обучения и оценки
        """
        # 1. Feature engineering (если нужно)
        if code_samples_train is not None:
            X_train, X_test = self._prepare_features(
                code_samples_train,
                code_samples_test,
                y_train,
            )
        
        # 2. Оптимизация гиперпараметров (если нужно)
        if self.optimization_config:
            self._optimize_hyperparameters(X_train, y_train)
        
        # 3. Обучение модели
        if self.cv_config:
            # С cross-validation
            cv_results = self._train_with_cv(X_train, y_train)
            self.results['cv_results'] = cv_results
        
        # Финальное обучение на всех данных
        self.model = ModelFactory.create(self.model_config)
        self.model.fit(X_train, y_train)
        
        # 4. Оценка на train
        train_metrics = self._evaluate(X_train, y_train, prefix='train')
        self.results['train_metrics'] = train_metrics
        
        # 5. Оценка на test (если есть)
        if X_test is not None and y_test is not None:
            test_metrics = self._evaluate(X_test, y_test, prefix='test')
            self.results['test_metrics'] = test_metrics
        
        # 6. Логирование
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
        X: np.ndarray,
        y: np.ndarray,
    ):
        """Оптимизация гиперпараметров"""
        optimizer = OptunaOptimizer(**self.optimization_config)
        
        opt_results = optimizer.optimize(
            model_type=self.model_config.type,
            X=X,
            y=y,
            search_space=self.optimization_config.get('search_space', {}),
            metric=self.optimization_config.get('metric', 'accuracy'),
        )
        
        # Обновляем конфиг лучшими параметрами
        self.model_config.params = opt_results['best_params']
        self.results['optimization'] = opt_results
    
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

