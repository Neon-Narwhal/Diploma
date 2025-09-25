"""Базовый класс для токенизаторов"""

from abc import ABC, abstractmethod
from typing import List


class BaseTokenizer(ABC):
    """Базовый класс для всех токенизаторов"""
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Кодирование текста в токены"""
        pass
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Декодирование токенов в текст"""
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Получение размера словаря"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Сохранение токенизатора"""
        pass
