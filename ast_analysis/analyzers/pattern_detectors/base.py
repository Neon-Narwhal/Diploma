import ast
from abc import ABC, abstractmethod
from typing import Dict, Any

class PatternDetector(ABC):
    """Базовый класс для детекторов паттернов"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def detect(self, tree: ast.AST) -> Dict[str, Any]:
        """Обнаружение паттерна"""
        pass
