"""
ML модуль для обучения gradient boosting моделей на задаче классификации сложности кода.
"""

import sys
from pathlib import Path

# Автоматически добавляем корень проекта в путь
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Импортируем модели чтобы зарегистрировать их
import ml.models  # Это триггерит регистрацию

from ml.core.model_factory import ModelFactory
from ml.core.base_model import BaseModel
from ml.training.pipeline import MLPipeline
from ml.experiments.runner import ExperimentRunner
from ml.configs.experiment import ExperimentConfig

__all__ = [
    'ModelFactory',
    'BaseModel',
    'MLPipeline',
    'ExperimentRunner',
    'ExperimentConfig',
]
