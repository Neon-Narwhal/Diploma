"""
Ансамбли моделей: Voting, OvR Multi-Boosting, Stacking.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import joblib

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
    
    def fit(self, X: np.ndarray, y: np.ndarray, eval_set=None) -> 'VotingEnsemble':
        """Обучение всех моделей в ансамбле"""
        self.models = []
        
        for model_config in self.models_configs:
            if isinstance(model_config, dict):
                model_config = ModelConfig(**model_config)
            
            model = ModelFactory.create(model_config)
            model.fit(X, y, eval_set=eval_set)
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
        joblib.dump({
            'models': self.models,
            'params': self.params,
            'voting': self.voting,
            'weights': self.weights,
        }, path)
    
    def load(self, path: str) -> 'VotingEnsemble':
        """Загрузка ансамбля"""
        data = joblib.load(path)
        
        self.models = data['models']
        self.params = data['params']
        self.voting = data['voting']
        self.weights = data['weights']
        self.is_fitted = True
        
        return self



class OvRBoostingModel(BaseModel):
    """
    One-vs-Rest Multi-Boosting.
    
    Для каждого класса обучается отдельный бинарный бустинг.
    Автоматически вычисляет class_weight для каждого бинарного классификатора.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        
        # Тип базового бустинга
        self.boosting_type = self.params.pop('boosting_type', 'catboost')
        
        # Параметры для базовых моделей
        self.base_params = self.params.copy()
        
        # OvR classifier
        from sklearn.multiclass import OneVsRestClassifier
        self.ovr_classifier = None
    
    def _create_base_estimator(self):
        """Создание базового бустинга"""
        from ml.models.boosting import BoostingModel
        
        params = self.base_params.copy()
        params['boosting_type'] = self.boosting_type
        
        return BoostingModel(params)
    
    def fit(self, X: np.ndarray, y: np.ndarray, eval_set=None) -> 'OvRBoostingModel':
        """Обучение OvR моделей"""
        from sklearn.multiclass import OneVsRestClassifier
        
        print(f"\n[OvR] Обучение {len(np.unique(y))} бинарных {self.boosting_type} моделей...")
        
        # Создаём базовый estimator с auto_class_weights
        base_estimator = self._create_base_estimator()
        
        # Добавляем auto_class_weights в параметры CatBoost/XGBoost
        if self.boosting_type == 'catboost':
            # auto_class_weights встроен в CatBoost
            if 'auto_class_weights' not in self.base_params:
                self.base_params['auto_class_weights'] = 'Balanced'
            base_estimator = self._create_base_estimator()
        
        # OvR wrapper БЕЗ sample_weight
        self.ovr_classifier = OneVsRestClassifier(
            base_estimator,
            n_jobs=1
        )
        
        # Обучение БЕЗ sample_weight (CatBoost сам посчитает веса)
        self.ovr_classifier.fit(X, y)
        
        self.is_fitted = True
        print(f"[OvR] Обучение завершено")
        return self



    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание классов"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict")
        return self.ovr_classifier.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Предсказание вероятностей"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba")
        
        # OvR возвращает decision_function, нормализуем в вероятности
        decision = self.ovr_classifier.decision_function(X)
        
        # Softmax для преобразования в вероятности
        exp_decision = np.exp(decision - np.max(decision, axis=1, keepdims=True))
        proba = exp_decision / np.sum(exp_decision, axis=1, keepdims=True)
        
        return proba
    
    def get_feature_importance(self) -> np.ndarray:
        """Усреднённая важность признаков по всем OvR моделям"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before get_feature_importance")
        
        importances = []
        for estimator in self.ovr_classifier.estimators_:
            if hasattr(estimator, 'get_feature_importance'):
                importances.append(estimator.get_feature_importance())
        
        if importances:
            return np.mean(importances, axis=0)
        else:
            return np.array([])
    
    def save(self, path: str) -> None:
        """Сохранение модели"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        joblib.dump(self.ovr_classifier, path)
    
    def load(self, path: str) -> 'OvRBoostingModel':
        """Загрузка модели"""
        self.ovr_classifier = joblib.load(path)
        self.is_fitted = True
        return self



class StackingModel(BaseModel):
    """
    Stacking ensemble с Ridge meta-model.
    
    Поддерживает:
    - Загрузку pretrained базовых моделей
    - Обучение новых базовых моделей с нуля
    - OOF predictions через CV
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        
        # Пути к pretrained моделям (опционально)
        self.pretrained_paths = self.params.get('pretrained_models', [])
        
        # Конфиги базовых моделей (если не используем pretrained)
        self.base_configs = self.params.get('base_models', [])
        
        # Meta-model тип
        self.meta_model_type = self.params.get('meta_model', 'ridge')
        
        # CV параметры
        self.cv_folds = self.params.get('cv', 5)
        
        # Stacking classifier
        from sklearn.ensemble import StackingClassifier
        self.stacking_classifier = None
    
    def _load_pretrained_models(self) -> List[Any]:
        """Загрузка pretrained моделей с агрессивной очисткой параметров"""
        models = []
        
        for path in self.pretrained_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Pretrained model not found: {path}")
            
            print(f"[Stacking] Загрузка pretrained модели: {path.name}")
            wrapper = joblib.load(path)
            
            # === FIX: Агрессивная очистка параметров ===
            # 1. Очистка в обертке BoostingModel
            if hasattr(wrapper, 'params'):
                for param in ['early_stopping_rounds', 'eval_set', 'eval_metric']:
                    if param in wrapper.params:
                        print(f"  [Fix] Удаление '{param}' из wrapper params")
                        del wrapper.params[param]
            
            # 2. Очистка внутри самого объекта модели (XGBClassifier/LGBMClassifier)
            if hasattr(wrapper, 'model'):
                inner_model = wrapper.model
                
                # Для XGBoost (хранит параметры в self.early_stopping_rounds, self.kwargs и т.д.)
                if hasattr(inner_model, 'early_stopping_rounds'):
                    if inner_model.early_stopping_rounds is not None:
                         print(f"  [Fix] XGBoost: Обнуление early_stopping_rounds")
                         inner_model.early_stopping_rounds = None
                
                # Для LightGBM (хранит в self._other_params или self.get_params())
                if hasattr(inner_model, 'set_params'):
                    # LightGBM/Sklearn API
                    # Пробуем установить early_stopping_rounds=None через set_params
                    try:
                        inner_model.set_params(early_stopping_rounds=None)
                        print(f"  [Fix] LGBM/Sklearn: set_params(early_stopping_rounds=None)")
                    except Exception:
                        pass
            
            # 3. Пересоздание (на всякий случай, если очистка атрибутов не помогла)
            # Это самый надежный способ, если wrapper поддерживает его
            if hasattr(wrapper, '_create_model'):
                try:
                    print(f"  [Fix] Пересоздание внутреннего объекта модели...")
                    wrapper.model = wrapper._create_model()
                except Exception as e:
                    print(f"  [Warning] Не удалось пересоздать модель: {e}")

            models.append((path.stem, wrapper))
        
        return models


    
    def _create_base_models(self) -> List[Any]:
        """Создание новых базовых моделей"""
        models = []
        
        for config in self.base_configs:
            if isinstance(config, dict):
                config = ModelConfig(**config)
            
            model = ModelFactory.create(config)
            models.append((config.name, model))
        
        return models
    
    def _create_meta_model(self):
        """Создание meta-model"""
        if self.meta_model_type == 'ridge':
            from sklearn.linear_model import RidgeClassifier
            return RidgeClassifier(alpha=10.0)  # Сильная регуляризация
        elif self.meta_model_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(C=0.1, max_iter=1000)
        else:
            raise ValueError(f"Unknown meta_model: {self.meta_model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, eval_set=None) -> 'StackingModel':
        """Обучение stacking"""
        from sklearn.ensemble import StackingClassifier
        
        print(f"\n[Stacking] Создание ансамбля...")
        
        # Базовые модели: загружаем pretrained или создаём новые
        if self.pretrained_paths:
            print(f"[Stacking] Используем {len(self.pretrained_paths)} pretrained моделей")
            base_estimators = self._load_pretrained_models()
        else:
            print(f"[Stacking] Обучаем {len(self.base_configs)} новых моделей")
            base_estimators = self._create_base_models()
            
            # Обучаем базовые модели
            for name, model in base_estimators:
                print(f"[Stacking] Обучение базовой модели: {name}")
                if hasattr(model, 'fit'):
                    model.fit(X, y, eval_set=eval_set)
        
        # Meta-model
        meta_model = self._create_meta_model()
        print(f"[Stacking] Meta-model: {self.meta_model_type}")
        
        # Stacking classifier
        self.stacking_classifier = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_model,
            cv=self.cv_folds,
            stack_method='predict_proba',  # Используем вероятности
            n_jobs=1, 
            verbose=1
        )
        
        print(f"[Stacking] Генерация OOF predictions через {self.cv_folds}-fold CV...")
        self.stacking_classifier.fit(X, y)
        
        self.is_fitted = True
        print(f"[Stacking] Обучение завершено")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание классов"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict")
        return self.stacking_classifier.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Предсказание вероятностей"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba")
        return self.stacking_classifier.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Meta-model не имеет feature importance"""
        return np.array([])
    
    def save(self, path: str) -> None:
        """Сохранение модели"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        joblib.dump(self.stacking_classifier, path)
    
    def load(self, path: str) -> 'StackingModel':
        """Загрузка модели"""
        self.stacking_classifier = joblib.load(path)
        self.is_fitted = True
        return self



# ===== WRAPPER КЛАССЫ ДЛЯ РЕГИСТРАЦИИ =====

class OvRCatBoostModel(OvRBoostingModel):
    def __init__(self, params=None):
        params = params or {}
        params['boosting_type'] = 'catboost'
        super().__init__(params)


class OvRXGBoostModel(OvRBoostingModel):
    def __init__(self, params=None):
        params = params or {}
        params['boosting_type'] = 'xgboost'
        super().__init__(params)


class OvRLightGBMModel(OvRBoostingModel):
    def __init__(self, params=None):
        params = params or {}
        params['boosting_type'] = 'lightgbm'
        super().__init__(params)



# ===== РЕГИСТРАЦИЯ В РЕЕСТРЕ =====

from ml.core.registry import ModelRegistry

ModelRegistry.register('voting', VotingEnsemble)
ModelRegistry.register('ovr_catboost', OvRCatBoostModel)
ModelRegistry.register('ovr_xgboost', OvRXGBoostModel)
ModelRegistry.register('ovr_lgbm', OvRLightGBMModel)
ModelRegistry.register('stacking', StackingModel)
