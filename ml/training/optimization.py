"""
Оптимизация гиперпараметров с помощью Optuna.
"""

import numpy as np
import optuna
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from ml.core.model_config import ModelConfig
from ml.core.model_factory import ModelFactory


class OptunaOptimizer:
    """
    Оптимизатор гиперпараметров.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        optimization_config: Dict[str, Any],
        cv_config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            model_config: конфиг модели (базовые параметры)
            optimization_config: конфиг оптимизации (search space, n_trials)
            cv_config: конфиг кросс-валидации (если не передан val сет)
        """
        self.model_config = model_config
        self.optimization_config = optimization_config
        self.cv_config = cv_config

    def optimize(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        X_val: Optional[np.ndarray] = None, 
        y_val: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, Any], float]:
        """
        Запуск процесса оптимизации.
        """
        # Отключаем лишний шум от Optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            # 1. Генерируем параметры для текущей итерации
            params = self._suggest_params(trial)
            
            # 2. Объединяем с базовыми параметрами модели
            # (например, task_type="GPU" должен остаться)
            model_params = self.model_config.params.copy()
            model_params.update(params)
            
            # 3. Создаем временную модель
            temp_config = ModelConfig(
                name=f"trial_{trial.number}",
                type=self.model_config.type,
                params=model_params
            )
            model = ModelFactory.create(temp_config)
            
            # 4. Обучение и оценка
            metric_name = self.optimization_config.get('metric', 'f1_macro')
            
            try:
                if X_val is not None and y_val is not None:
                    # Если есть явная валидация
                    # Передаем eval_set, если модель поддерживает (обернуто в try/except в модели или тут)
                    try:
                        model.fit(X, y, eval_set=(X_val, y_val))
                    except TypeError:
                        model.fit(X, y)
                        
                    score = self._evaluate_metric(model, X_val, y_val, metric_name)
                else:
                    # Если нет валидации, делаем простой сплит внутри
                    X_t, X_v, y_t, y_v = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    model.fit(X_t, y_t)
                    score = self._evaluate_metric(model, X_v, y_v, metric_name)
                    
            except Exception as e:
                # Если параметры плохие и модель упала, возвращаем плохой скор
                # print(f"Trial failed: {e}")
                return 0.0
            
            return score

        # Создаем study
        study = optuna.create_study(direction="maximize")
        
        print(f"Начало оптимизации ({self.optimization_config.get('n_trials')} итераций)...")
        study.optimize(
            objective, 
            n_trials=self.optimization_config.get('n_trials', 10),
            timeout=self.optimization_config.get('timeout', 3600),
            show_progress_bar=True
        )
        
        return study.best_params, study.best_value

    def _suggest_params(self, trial) -> Dict[str, Any]:
        """Генерация параметров на основе search_space из конфига"""
        params = {}
        search_space = (self.optimization_config.get('search_space') or 
                        self.optimization_config.get('params') or 
                        {})
        
        for name, config in search_space.items():
            param_type = config.get('type')
            
            if param_type == 'int':
                params[name] = trial.suggest_int(
                    name, 
                    config['low'], 
                    config['high'], 
                    log=config.get('log', False)
                )
            elif param_type == 'float':
                params[name] = trial.suggest_float(
                    name, 
                    config['low'], 
                    config['high'], 
                    log=config.get('log', False)
                )
            elif param_type == 'categorical':
                params[name] = trial.suggest_categorical(
                    name, 
                    config['choices']
                )
                
        return params

    def _evaluate_metric(self, model, X, y, metric_name):
        """Вычисление метрики"""
        y_pred = model.predict(X)
        
        if metric_name == 'accuracy':
            return accuracy_score(y, y_pred)
        elif metric_name == 'f1_macro':
            return f1_score(y, y_pred, average='macro')
        elif metric_name == 'f1_micro':
            return f1_score(y, y_pred, average='micro')
        else:
            return accuracy_score(y, y_pred)
