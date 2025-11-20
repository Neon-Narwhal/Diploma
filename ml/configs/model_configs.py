"""
Готовые конфиги для различных моделей.
"""

from ml.core.model_config import ModelConfig


# CatBoost конфиги
CATBOOST_DEFAULT = ModelConfig(
    name="catboost_default",
    type="catboost",
    params={
        'boosting_type': 'catboost',
        'iterations': 1000,
        'depth': 6,
        'learning_rate': 0.03,
        'verbose': False,
    }
)

CATBOOST_FAST = ModelConfig(
    name="catboost_fast",
    type="catboost",
    params={
        'boosting_type': 'catboost',
        'iterations': 100,
        'depth': 4,
        'learning_rate': 0.1,
        'verbose': False,
    }
)


# XGBoost конфиги
XGBOOST_DEFAULT = ModelConfig(
    name="xgboost_default",
    type="xgboost",
    params={
        'boosting_type': 'xgboost',
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.03,
        'verbosity': 0,
    }
)

XGBOOST_FAST = ModelConfig(
    name="xgboost_fast",
    type="xgboost",
    params={
        'boosting_type': 'xgboost',
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.1,
        'verbosity': 0,
    }
)


# LightGBM конфиги
LIGHTGBM_DEFAULT = ModelConfig(
    name="lightgbm_default",
    type="lightgbm",
    params={
        'boosting_type': 'lightgbm',
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.03,
        'verbose': -1,
    }
)


# Search spaces для оптимизации
CATBOOST_SEARCH_SPACE = {
    'iterations': {'type': 'int', 'low': 100, 'high': 2000},
    'depth': {'type': 'int', 'low': 4, 'high': 10},
    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
    'l2_leaf_reg': {'type': 'float', 'low': 1, 'high': 10},
}

XGBOOST_SEARCH_SPACE = {
    'n_estimators': {'type': 'int', 'low': 100, 'high': 2000},
    'max_depth': {'type': 'int', 'low': 3, 'high': 10},
    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
    'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
    'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
}
