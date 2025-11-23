"""
Универсальный wrapper для gradient boosting моделей.
"""

import numpy as np
from typing import Dict, Any, Optional
from ml.core.base_model import BaseModel


class BoostingModel(BaseModel):
    """
    Универсальная обертка для CatBoost, XGBoost, LightGBM.
    Тип модели определяется параметром boosting_type.
    """
    
    SUPPORTED_TYPES = ['catboost', 'xgboost', 'lightgbm']
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.boosting_type = self.params.pop('boosting_type', 'catboost')
        
        if self.boosting_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unknown boosting_type: {self.boosting_type}. "
                f"Supported: {self.SUPPORTED_TYPES}"
            )
        
        self.model = self._create_model()
    
    def _create_model(self):
        """Создание модели по типу"""
        # Копируем параметры, чтобы не менять исходный словарь
        params = self.params.copy()
        
        # Убираем boosting_type, так как он не нужен конструктору модели
        params.pop('boosting_type', None)
        
        if self.boosting_type == 'catboost':
            from catboost import CatBoostClassifier
            # Если verbose не задан в конфиге, ставим False по умолчанию
            if 'verbose' not in params:
                params['verbose'] = False
            return CatBoostClassifier(**params)
        
        elif self.boosting_type == 'xgboost':
            from xgboost import XGBClassifier
            # Для XGBoost verbosity=0 отключает вывод
            if 'verbosity' not in params:
                params['verbosity'] = 0
            return XGBClassifier(**params)
        
        elif self.boosting_type == 'lightgbm':
            from lightgbm import LGBMClassifier
            # Для LightGBM verbose=-1 отключает вывод
            if 'verbose' not in params:
                params['verbose'] = -1
            return LGBMClassifier(**params)

    
    def fit(self, X: np.ndarray, y: np.ndarray, eval_set=None) -> 'BoostingModel':
        """Обучение модели"""
        fit_params = {}
        
        # Поддержка eval_set для бустингов
        if eval_set is not None:
            if self.boosting_type == 'catboost':
                fit_params['eval_set'] = eval_set
                fit_params['early_stopping_rounds'] = 50
            elif self.boosting_type == 'xgboost':
                fit_params['eval_set'] = [eval_set]
                fit_params['early_stopping_rounds'] = 50
                fit_params['verbose'] = False
            elif self.boosting_type == 'lightgbm':
                fit_params['eval_set'] = eval_set
                """
                fit_params['callbacks'] = [
                    # lgb callbacks if needed
                ]
                """
        
        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        return self

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание классов"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Предсказание вероятностей"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Получение важности признаков"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before get_feature_importance")
        
        if self.boosting_type == 'catboost':
            return np.array(self.model.get_feature_importance())
        else:
            return self.model.feature_importances_
    
    def save(self, path: str) -> None:
        """Сохранение модели"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        if self.boosting_type == 'catboost':
            self.model.save_model(path)
        else:
            import joblib
            joblib.dump(self.model, path)
    
    def load(self, path: str) -> 'BoostingModel':
        """Загрузка модели"""
        if self.boosting_type == 'catboost':
            from catboost import CatBoostClassifier
            self.model = CatBoostClassifier()
            self.model.load_model(path)
        else:
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
