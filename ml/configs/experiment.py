"""
Конфигурация эксперимента.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from ml.configs.base import BaseConfig
from ml.core.model_config import ModelConfig


@dataclass
class FeatureConfig(BaseConfig):
    """
    Конфигурация feature engineering.
    """
    extractors: List[str] = field(default_factory=lambda: ['complexity'])
    transformer_method: str = 'standard'
    selector_method: Optional[str] = None
    n_features: Optional[int] = None


@dataclass
class CVConfig(BaseConfig):
    """
    Конфигурация cross-validation.
    """
    enabled: bool = True
    n_folds: int = 5
    stratified: bool = True
    shuffle: bool = True
    random_state: int = 42
    metrics: List[str] = field(default_factory=lambda: ['accuracy', 'f1_macro'])


@dataclass
class OptimizationConfig(BaseConfig):
    """
    Конфигурация оптимизации гиперпараметров.
    """
    enabled: bool = False
    n_trials: int = 100
    timeout: Optional[int] = None
    metric: str = 'accuracy'
    search_space: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig(BaseConfig):
    """
    Главная конфигурация эксперимента.
    Описывает полный эксперимент с одной или несколькими моделями.
    """
    name: str
    description: str = ""
    
    # Модели для обучения
    models: List[Dict[str, Any]] = field(default_factory=list)
    
    # Feature engineering
    features: Optional[Dict[str, Any]] = None
    
    # Cross-validation
    cross_validation: Optional[Dict[str, Any]] = None
    
    # Optimization
    optimization: Optional[Dict[str, Any]] = None
    
    # Evaluation
    evaluation_metrics: List[str] = field(
        default_factory=lambda: ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    )
    
    # Outputs
    save_models: bool = True
    save_predictions: bool = True
    generate_report: bool = True
    
    # MLflow
    mlflow_experiment: Optional[str] = None
    
    def get_feature_config(self) -> FeatureConfig:
        """Получение конфига признаков"""
        if self.features:
            return FeatureConfig.from_dict(self.features)
        return FeatureConfig()
    
    def get_cv_config(self) -> Optional[CVConfig]:
        """Получение конфига CV"""
        if self.cross_validation:
            return CVConfig.from_dict(self.cross_validation)
        return None
    
    def get_optimization_config(self) -> Optional[OptimizationConfig]:
        """Получение конфига оптимизации"""
        if self.optimization:
            return OptimizationConfig.from_dict(self.optimization)
        return None
    
    def get_model_configs(self) -> List[ModelConfig]:
        """Получение конфигов моделей"""
        return [ModelConfig(**model_dict) for model_dict in self.models]
