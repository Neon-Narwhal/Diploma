"""
Базовый класс для AST анализаторов.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time

# Импортируем из result.py вместо определения здесь
from ast_analysis.core.result import ASTAnalysisResult


class BaseASTAnalyzer(ABC):
    """
    Базовый класс для всех AST анализаторов.
    Определяет интерфейс для анализа Python кода через AST.
    """
    
    def __init__(self, name: str, **config):
        """
        Args:
            name: Имя анализатора
            **config: Параметры конфигурации
        """
        self.name = name
        self.config = config
        self.is_initialized = False
    
    @abstractmethod
    def analyze(self, code: str) -> ASTAnalysisResult:
        """
        Анализ одного фрагмента кода.
        
        Args:
            code: Исходный код Python
        
        Returns:
            Результат анализа
        """
        pass
    
    def batch_analyze(self, codes: list[str]) -> list[ASTAnalysisResult]:
        """
        Батчевый анализ кодов.
        По умолчанию последовательно вызывает analyze для каждого кода.
        Может быть переопределён для оптимизации.
        
        Args:
            codes: Список исходных кодов
        
        Returns:
            Список результатов
        """
        results = []
        for code in codes:
            result = self.analyze(code)
            results.append(result)
        return results
    
    def initialize(self) -> bool:
        """
        Инициализация анализатора.
        Может быть переопределён для подготовки ресурсов.
        """
        self.is_initialized = True
        return True
    
    def cleanup(self):
        """Очистка ресурсов"""
        pass
    
    def validate_code(self, code: str) -> bool:
        """
        Валидация кода перед анализом.
        
        Args:
            code: Исходный код
        
        Returns:
            True если код валиден
        """
        if not code or not code.strip():
            return False
        
        # Проверка минимальной длины
        min_length = self.config.get('min_code_length', 1)
        if len(code) < min_length:
            return False
        
        return True
    
    def _safe_analyze(self, code: str) -> ASTAnalysisResult:
        """
        Обёртка для безопасного анализа с обработкой ошибок.
        
        Args:
            code: Исходный код
        
        Returns:
            Результат анализа
        """
        start_time = time.time()
        
        # Валидация
        if not self.validate_code(code):
            return ASTAnalysisResult.from_error(
                error="Invalid code",
                analyzer_name=self.name
            )
        
        try:
            # Вызов реального анализа
            result = self.analyze(code)
            result.processing_time = time.time() - start_time
            return result
        
        except SyntaxError as e:
            return ASTAnalysisResult.from_error(
                error=f"SyntaxError: {str(e)}",
                analyzer_name=self.name
            )
        
        except Exception as e:
            return ASTAnalysisResult.from_error(
                error=f"Error: {str(e)}",
                analyzer_name=self.name
            )
    
    def get_feature_names(self) -> list[str]:
        """
        Получение списка имён извлекаемых признаков.
        Должно быть переопределено в подклассах.
        """
        return []
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
