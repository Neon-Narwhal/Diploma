"""
Callbacks для обучения.
"""

from typing import Dict, Any, Optional


class EarlyStopping:
    """
    Early stopping callback.
    Останавливает обучение если метрика не улучшается.
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0001,
        mode: str = 'max',
    ):
        """
        Args:
            patience: количество эпох без улучшения
            min_delta: минимальное изменение для считания улучшением
            mode: 'max' (выше лучше) или 'min' (ниже лучше)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_value = None
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, metric_value: float) -> bool:
        """
        Проверка условия останова.
        
        Args:
            metric_value: текущее значение метрики
            
        Returns:
            True если нужно остановить обучение
        """
        if self.best_value is None:
            self.best_value = metric_value
            return False
        
        # Проверка улучшения
        if self.mode == 'max':
            improved = metric_value > (self.best_value + self.min_delta)
        else:
            improved = metric_value < (self.best_value - self.min_delta)
        
        if improved:
            self.best_value = metric_value
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
            return True
        
        return False
    
    def reset(self):
        """Сброс состояния"""
        self.best_value = None
        self.counter = 0
        self.should_stop = False


class LoggingCallback:
    """
    Callback для логирования метрик во время обучения.
    """
    
    def __init__(self, logger=None):
        """
        Args:
            logger: логгер (MLflow, json или любой другой)
        """
        self.logger = logger
    
    def __call__(self, step: int, metrics: Dict[str, Any]):
        """
        Логирование метрик на каждом шаге.
        
        Args:
            step: номер шага/эпохи
            metrics: словарь с метриками
        """
        if self.logger:
            self.logger.log_metrics(metrics, step=step)
