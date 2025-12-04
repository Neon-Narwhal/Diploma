"""
Структура датасета для всех модулей.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class CodeSample:
    """Единичный образец кода"""
    code: str
    label: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Dataset:
    """
    Универсальный датасет для анализа кода.
    Используется всеми модулями (ast, cfg, runtime).
    """
    train: List[CodeSample]
    val: List[CodeSample]
    test: List[CodeSample]
    
    # Метаданные датасета
    name: Optional[str] = None
    code_field: str = "code"
    label_field: str = "label"
    
    def __post_init__(self):
        """Валидация после создания"""
        if not self.train:
            raise ValueError("Training set cannot be empty")
    
    @property
    def train_codes(self) -> List[str]:
        """Список кодов для обучения"""
        return [s.code for s in self.train]
    
    @property
    def train_labels(self) -> List[str]:
        """Список меток для обучения"""
        return [s.label for s in self.train if s.label is not None]
    
    @property
    def val_codes(self) -> List[str]:
        """Список кодов для валидации"""
        return [s.code for s in self.val]
    
    @property
    def val_labels(self) -> List[str]:
        """Список меток для валидации"""
        return [s.label for s in self.val if s.label is not None]
    
    @property
    def test_codes(self) -> List[str]:
        """Список кодов для теста"""
        return [s.code for s in self.test]
    
    @property
    def test_labels(self) -> List[str]:
        """Список меток для теста"""
        return [s.label for s in self.test if s.label is not None]
    
    def size(self) -> Dict[str, int]:
        """Размеры сплитов"""
        return {
            'train': len(self.train),
            'val': len(self.val),
            'test': len(self.test),
            'total': len(self.train) + len(self.val) + len(self.test)
        }
    
    def summary(self) -> str:
        """Краткая сводка о датасете"""
        sizes = self.size()
        return (
            f"Dataset: {self.name or 'unnamed'}\n"
            f"  Train: {sizes['train']}\n"
            f"  Val: {sizes['val']}\n"
            f"  Test: {sizes['test']}\n"
            f"  Total: {sizes['total']}"
        )
