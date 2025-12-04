"""
Генераторы входных данных для динамического анализа.
"""

import random
import string
from typing import Any, List, Dict, Union

class InputGenerator:
    """
    Генератор тестовых данных разного размера N.
    """
    
    def generate(self, input_type: str, n: int) -> Any:
        """
        Генерация данных размера N.
        """
        method = getattr(self, f"_gen_{input_type}", None)
        if not method:
            raise ValueError(f"Unknown input type: {input_type}")
        return method(n)
    
    def _gen_list_int(self, n: int) -> List[int]:
        """Список случайных чисел"""
        return [random.randint(-1000, 1000) for _ in range(n)]
    
    def _gen_list_int_sorted(self, n: int) -> List[int]:
        """Отсортированный список (для binary search)"""
        return sorted(self._gen_list_int(n))

    def _gen_list_str(self, n: int) -> List[str]:
        """Список случайных строк"""
        return [''.join(random.choices(string.ascii_letters, k=5)) for _ in range(n)]
    
    def _gen_int(self, n: int) -> int:
        """Просто число N"""
        return n
        
    def _gen_matrix(self, n: int) -> List[List[int]]:
        """Матрица N x N"""
        real_n = int(n ** 0.5)
        if real_n < 1: real_n = 1
        return [[random.randint(0, 100) for _ in range(real_n)] for _ in range(real_n)]

    def infer_input_type(self, code: str) -> str:
        """
        Эвристика для определения типа входных данных по коду.
        """
        if "def " not in code:
            return "list_int"
            
        try:
            # Простая текстовая эвристика по именам аргументов
            func_def = code.split("def ")[1].split("(")[0]
            # Берем сигнатуру аргументов
            args_part = code.split("def ")[1].split("(")[1].split(")")[0]
            
            if "matrix" in args_part or "grid" in args_part:
                return "matrix"
            if "arr" in args_part or "nums" in args_part or "lst" in args_part or "A" in args_part:
                return "list_int"
            if "n" in args_part or "k" in args_part or "target" in args_part:
                return "int"
                
        except Exception:
            pass
            
        return "list_int" # Default fallback
