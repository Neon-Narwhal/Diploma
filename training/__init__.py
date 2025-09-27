"""Модули для обучения модели"""

try:
    from .trainer import ModelTrainer, estimate_loss
    from .transformer_squared_training import run_comparison_experiment, TransformerSquaredTrainer, run_laptop_test
    __all__ = ['ModelTrainer', 'estimate_loss', 'run_comparison_experiment', 'TransformerSquaredTrainer', 'run_laptop_test']
except ImportError:
    # Если trainer.py еще не создан
    __all__ = []
