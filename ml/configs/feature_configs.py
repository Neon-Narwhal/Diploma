"""
Готовые конфиги для feature engineering.
"""

from ml.configs.experiment import FeatureConfig


# Только complexity метрики
COMPLEXITY_ONLY = FeatureConfig(
    extractors=['complexity'],
    transformer_method='standard',
    selector_method=None,
    n_features=None,
)

# Complexity + token метрики
COMPLEXITY_TOKEN = FeatureConfig(
    extractors=['complexity', 'token'],
    transformer_method='standard',
    selector_method=None,
    n_features=None,
)

# С feature selection
COMPLEXITY_SELECTED = FeatureConfig(
    extractors=['complexity'],
    transformer_method='standard',
    selector_method='mutual_info',
    n_features=50,
)

# Robust scaling для outliers
COMPLEXITY_ROBUST = FeatureConfig(
    extractors=['complexity'],
    transformer_method='robust',
    selector_method=None,
    n_features=None,
)
