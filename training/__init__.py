"""Модули для обучения модели"""

try:
    from .trainer import ModelTrainer, estimate_loss
    __all__ = ['ModelTrainer', 'estimate_loss']
except ImportError:
    # Если trainer.py еще не создан
    __all__ = []
