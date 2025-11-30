"""
Универсальный wrapper для gradient boosting моделей.
"""

import numpy as np
from typing import Dict, Any, Optional
from ml.core.base_model import BaseModel

class BoostingModel(BaseModel):
    """
    Универсальная обертка для CatBoost, XGBoost, LightGBM.
    """
    
    SUPPORTED_TYPES = ['catboost', 'xgboost', 'lightgbm']
    
    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(params, **kwargs)
        
        self.boosting_type = self.params.get('boosting_type', 'catboost')
        
        if self.boosting_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unknown boosting_type: {self.boosting_type}")
        
        self.model = self._create_model()
    
    def _create_model(self):
        """Создание модели и передача параметров в конструктор"""
        # Копируем параметры, чтобы не менять оригинал
        model_params = self.params.copy()
        model_params.pop('boosting_type', None)
        
        if self.boosting_type == 'catboost':
            from catboost import CatBoostClassifier
            if 'verbose' not in model_params:
                model_params['verbose'] = False
            return CatBoostClassifier(**model_params)
        
        elif self.boosting_type == 'xgboost':
            from xgboost import XGBClassifier
            if 'verbosity' not in model_params:
                model_params['verbosity'] = 0
            return XGBClassifier(**model_params)
        
        elif self.boosting_type == 'lightgbm':
            from lightgbm import LGBMClassifier
            if 'verbose' not in model_params:
                model_params['verbose'] = -1
            return LGBMClassifier(**model_params)

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set=None) -> 'BoostingModel':
        """
        Обучение модели с правильной обработкой eval_set.
        """
        fit_params = {}
        
        # Обработка eval_set в зависимости от библиотеки
        if eval_set is not None:
            if self.boosting_type == 'catboost':
                fit_params['eval_set'] = eval_set
            else:
                # XGBoost и LightGBM требуют список кортежей [(X, y)]
                fit_params['eval_set'] = [eval_set]
                
            # Для XGBoost нужно явно включить verbose, чтобы видеть прогресс (опционально)
            if self.boosting_type == 'xgboost':
                fit_params['verbose'] = False

        # ВАЖНО: Мы НЕ передаем self.params в fit, так как они уже в конструкторе!
        self.model.fit(X, y, **fit_params)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        if self.boosting_type == 'catboost':
            return self.model.get_feature_importance()
        elif self.boosting_type == 'xgboost':
            return self.model.feature_importances_
        elif self.boosting_type == 'lightgbm':
            return self.model.feature_importances_
        return np.array([])
    
    def save(self, path: str) -> None:
        import joblib
        # Для CatBoost лучше использовать встроенный save_model, но joblib универсальнее для враппера
        joblib.dump(self.model, path)
    
    def load(self, path: str) -> 'BoostingModel':
        import joblib
        self.model = joblib.load(path)
        self.is_fitted = True
        return self



# Регистрация моделей в реестре
from ml.core.registry import ModelRegistry

# Регистрируем каждый тип бустинга отдельно
class CatBoostModel(BoostingModel):
    def __init__(self, params=None):
        params = params or {}
        params['boosting_type'] = 'catboost'
        super().__init__(params)

class XGBoostModel(BoostingModel):
    def __init__(self, params=None):
        params = params or {}
        params['boosting_type'] = 'xgboost'
        super().__init__(params)

class LightGBMModel(BoostingModel):
    def __init__(self, params=None):
        params = params or {}
        params['boosting_type'] = 'lightgbm'
        super().__init__(params)

# Регистрация в реестре
ModelRegistry.register('catboost', CatBoostModel)
ModelRegistry.register('xgboost', XGBoostModel)
ModelRegistry.register('lightgbm', LightGBMModel)
