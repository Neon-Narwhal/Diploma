# complexity_analyzers/metrics/base.py

"""Базовые классы для метрических калькуляторов"""
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseMetricsCalculator(ABC):
    """Базовый класс для калькуляторов метрик"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def calculate(self, source_code: str) -> Dict[str, Any]:
        """Вычисление метрик"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Проверка доступности калькулятора"""
        pass
