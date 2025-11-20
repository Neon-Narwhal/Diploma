"""
Оптимизация гиперпараметров через Optuna.
"""

import optuna
import numpy as np
from typing import Dict, Any, Optional, Callable
from ml.core.base_model import BaseModel
from ml.core.model_factory import ModelFactory
from ml.core.model_config import ModelConfig
from ml.training.cross_validation import CrossValidator


class OptunaOptimizer:
    """
    Оптимизация гиперпараметров через Optuna.
    """
    
    def __init__(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        direction: str = 'maximize',
    ):
        """
        Args:
            n_trials: количество trials
            timeout: таймаут в секундах
            n_jobs: количество параллельных jobs
            direction: 'maximize' или 'minimize'
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.direction = direction
        
        self.study = None
        self.best_params = None
        self.best_value = None
    
    def optimize(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        search_space: Dict[str, Any],
        metric: str = 'accuracy',
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Запуск оптимизации.
        
        Args:
            model_type: тип модели для оптимизации
            X: признаки
            y: таргет
            search_space: пространство поиска параметров
            metric: метрика для оптимизации
            cv_folds: количество фолдов для CV
            
        Returns:
            Лучшие параметры и результаты
        """
        def objective(trial: optuna.Trial) -> float:
            # Генерация параметров из search space
            params = self._suggest_params(trial, search_space)
            
            # Создание модели
            config = ModelConfig(
                name=f"trial_{trial.number}",
                type=model_type,
                params=params,
            )
            model = ModelFactory.create(config)
            
            # Cross-validation оценка
            cv = CrossValidator(n_folds=cv_folds)
            cv_results = cv.run(model, X, y, metrics=[metric])
            
            return cv_results['mean'][metric]
        
        # Создание и запуск study
        self.study = optuna.create_study(direction=self.direction)
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
        )
        
        # Сохранение лучших результатов
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': len(self.study.trials),
            'study': self.study,
        }
    
    def _suggest_params(
        self,
        trial: optuna.Trial,
        search_space: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Генерация параметров из search space.
        
        Search space format:
        {
            'param_name': {
                'type': 'int'/'float'/'categorical',
                'low': ...,
                'high': ...,
                'choices': [...],
            }
        }
        """
        params = {}
        
        for param_name, param_config in search_space.items():
            param_type = param_config['type']
            
            if param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                )
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False),
                )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices'],
                )
        
        return params
    
    def get_best_config(self, model_type: str, name: str = "optimized") -> ModelConfig:
        """Создание конфига с лучшими параметрами"""
        if self.best_params is None:
            raise RuntimeError("Optimization must be run before get_best_config")
        
        return ModelConfig(
            name=name,
            type=model_type,
            params=self.best_params,
        )
