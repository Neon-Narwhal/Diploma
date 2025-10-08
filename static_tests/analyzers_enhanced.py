# complexity_analyzer/analyzers_enhanced.py
"""
МОДИФИЦИРОВАННЫЕ (Enhanced) реализации анализаторов.
Добавлен AST-анализ циклов для улучшения предсказания временной сложности.

Улучшения:
1. Анализ глубины вложенности циклов (loop depth)
2. Подсчет общего количества циклов
3. Различение for/while циклов
4. Умный маппинг с учетом структуры кода
"""

import ast
from typing import Dict, Optional, Any
import logging

from analyzers_baseline import (
    RadonBaselineAnalyzer,
    LizardBaselineAnalyzer,
    McCabeBaselineAnalyzer,
    ComplexipyBaselineAnalyzer,
    HalsteadAnalyzer,
    MaintainabilityIndexAnalyzer,
    NestedBlockDepthAnalyzer,
    BaselineAnalyzer
)
from config import ComplexityClass

logger = logging.getLogger(__name__)


class LoopAnalyzer(ast.NodeVisitor):
    """
    AST Visitor для анализа циклов.
    Подсчитывает:
    - Максимальную глубину вложенности циклов
    - Общее количество циклов
    - Типы циклов (for/while)
    """
    
    def __init__(self):
        self.max_depth = 0
        self.current_depth = 0
        self.total_loops = 0
        self.loop_types = {'for': 0, 'while': 0}
        self.loops_at_depth = {}  # {depth: count}
    
    def visit_For(self, node):
        self.total_loops += 1
        self.loop_types['for'] += 1
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        
        # Подсчитываем циклы на каждом уровне глубины
        self.loops_at_depth[self.current_depth] = \
            self.loops_at_depth.get(self.current_depth, 0) + 1
        
        self.generic_visit(node)
        self.current_depth -= 1
    
    def visit_While(self, node):
        self.total_loops += 1
        self.loop_types['while'] += 1
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        
        self.loops_at_depth[self.current_depth] = \
            self.loops_at_depth.get(self.current_depth, 0) + 1
        
        self.generic_visit(node)
        self.current_depth -= 1


class RecursionAnalyzer(ast.NodeVisitor):
    """
    Анализатор рекурсивных вызовов функций.
    """
    
    def __init__(self):
        self.function_names = set()
        self.recursive_calls = []
        self.current_function = None
    
    def visit_FunctionDef(self, node):
        old_function = self.current_function
        self.current_function = node.name
        self.function_names.add(node.name)
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_Call(self, node):
        if self.current_function and isinstance(node.func, ast.Name):
            if node.func.id == self.current_function:
                self.recursive_calls.append(self.current_function)
        self.generic_visit(node)


class EnhancedAnalyzerMixin:
    """
    Mixin для добавления AST-анализа к baseline анализаторам.
    Добавляет анализ циклов и улучшенный маппинг.
    """
    
    def _analyze_code_structure(self, source_code: str) -> Dict[str, Any]:
        """
        Комплексный анализ структуры кода через AST.
        
        Returns:
            {
                'max_loop_depth': int,
                'total_loops': int,
                'for_loops': int,
                'while_loops': int,
                'loops_at_depth': dict,
                'has_recursion': bool,
                'recursive_functions': list
            }
        """
        try:
            tree = ast.parse(source_code)
            
            # Анализ циклов
            loop_analyzer = LoopAnalyzer()
            loop_analyzer.visit(tree)
            
            # Анализ рекурсии
            recursion_analyzer = RecursionAnalyzer()
            recursion_analyzer.visit(tree)
            
            return {
                'max_loop_depth': loop_analyzer.max_depth,
                'total_loops': loop_analyzer.total_loops,
                'for_loops': loop_analyzer.loop_types['for'],
                'while_loops': loop_analyzer.loop_types['while'],
                'loops_at_depth': loop_analyzer.loops_at_depth,
                'has_recursion': len(recursion_analyzer.recursive_calls) > 0,
                'recursive_functions': list(set(recursion_analyzer.recursive_calls))
            }
        except Exception as e:
            logger.debug(f"AST structure analysis failed: {e}")
            return {
                'max_loop_depth': 0,
                'total_loops': 0,
                'for_loops': 0,
                'while_loops': 0,
                'loops_at_depth': {},
                'has_recursion': False,
                'recursive_functions': []
            }
    
    def _map_by_loops_enhanced(
        self, 
        loop_depth: int, 
        total_loops: int,
        has_recursion: bool,
        base_metric: float
    ) -> ComplexityClass:
        """
        УЛУЧШЕННЫЙ маппинг на основе структурного анализа кода.
        
        Приоритеты:
        1. Рекурсия → exponential/nlogn
        2. Глубина вложенности циклов → основной фактор
        3. Количество циклов → уточнение
        4. Базовая метрика (CCN/Cognitive) → финальное уточнение
        
        Args:
            loop_depth: Максимальная глубина вложенности
            total_loops: Общее количество циклов
            has_recursion: Наличие рекурсивных вызовов
            base_metric: Базовая метрика инструмента (CCN/Cognitive/etc)
        """
        # ПРИОРИТЕТ 1: Рекурсия
        if has_recursion:
            if loop_depth >= 1:
                return ComplexityClass.EXPONENTIAL  # Рекурсия + циклы
            elif base_metric > 5:
                return ComplexityClass.EXPONENTIAL  # Сложная рекурсия
            else:
                return ComplexityClass.NLOGN  # Простая рекурсия (divide & conquer)
        
        # ПРИОРИТЕТ 2: Глубина вложенности циклов
        if loop_depth >= 4:
            return ComplexityClass.EXPONENTIAL
        elif loop_depth == 3:
            return ComplexityClass.CUBIC  # O(n³)
        elif loop_depth == 2:
            return ComplexityClass.QUADRATIC  # O(n²)
        elif loop_depth == 1:
            # Один уровень циклов - смотрим количество
            if total_loops > 2:
                # Много последовательных циклов
                if base_metric > 8:
                    return ComplexityClass.NLOGN  # Сложная логика внутри
                else:
                    return ComplexityClass.LINEAR  # Простые циклы
            elif total_loops == 1:
                # Один цикл
                if base_metric > 10:
                    return ComplexityClass.NLOGN  # Сложная логика в цикле
                else:
                    return ComplexityClass.LINEAR  # Простой цикл
            else:
                return ComplexityClass.LINEAR
        
        # ПРИОРИТЕТ 3: Нет циклов, нет рекурсии
        if total_loops == 0:
            if base_metric > 10:
                # Сложная логика без циклов (возможно бинпоиск, битовые операции)
                return ComplexityClass.LOGN  # O(log n)
            else:
                return ComplexityClass.CONSTANT  # O(1)
        
        # Fallback
        return ComplexityClass.LINEAR


# ============================================================================
#                         ENHANCED АНАЛИЗАТОРЫ
# ============================================================================

class RadonEnhancedAnalyzer(EnhancedAnalyzerMixin, RadonBaselineAnalyzer):
    """
    ENHANCED: Radon + AST анализ циклов
    """
    
    def __init__(self):
        super().__init__()
        self.name = 'radon_enhanced'
    
    def analyze(self, source_code: str, language: str = 'python') -> Dict[str, Any]:
        """Radon метрики + структурный анализ"""
        metrics = super().analyze(source_code, language)
        
        if 'error' not in metrics or not metrics.get('error'):
            structure = self._analyze_code_structure(source_code)
            metrics.update(structure)
        
        return metrics
    
    def map_to_complexity(self, metrics: Dict[str, Any]) -> ComplexityClass:
        """Enhanced маппинг с учетом циклов"""
        loop_depth = metrics.get('max_loop_depth', 0)
        total_loops = metrics.get('total_loops', 0)
        has_recursion = metrics.get('has_recursion', False)
        cc = metrics.get('cyclomatic_complexity', 1)
        
        return self._map_by_loops_enhanced(loop_depth, total_loops, has_recursion, cc)


class LizardEnhancedAnalyzer(EnhancedAnalyzerMixin, LizardBaselineAnalyzer):
    """
    ENHANCED: Lizard + AST анализ циклов
    """
    
    def __init__(self):
        super().__init__()
        self.name = 'lizard_enhanced'
    
    def analyze(self, source_code: str, language: str = 'python') -> Dict[str, Any]:
        """Lizard метрики + структурный анализ"""
        metrics = super().analyze(source_code, language)
        
        if 'error' not in metrics or not metrics.get('error'):
            structure = self._analyze_code_structure(source_code)
            metrics.update(structure)
        
        return metrics
    
    def map_to_complexity(self, metrics: Dict[str, Any]) -> ComplexityClass:
        """Enhanced маппинг с учетом циклов"""
        loop_depth = metrics.get('max_loop_depth', 0)
        total_loops = metrics.get('total_loops', 0)
        has_recursion = metrics.get('has_recursion', False)
        cc = metrics.get('cyclomatic_complexity', 1)
        
        return self._map_by_loops_enhanced(loop_depth, total_loops, has_recursion, cc)


class McCabeEnhancedAnalyzer(EnhancedAnalyzerMixin, McCabeBaselineAnalyzer):
    """
    ENHANCED: McCabe + AST анализ циклов
    """
    
    def __init__(self):
        super().__init__()
        self.name = 'mccabe_enhanced'
    
    def analyze(self, source_code: str, language: str = 'python') -> Dict[str, Any]:
        """McCabe метрики + структурный анализ"""
        metrics = super().analyze(source_code, language)
        
        if 'error' not in metrics or not metrics.get('error'):
            structure = self._analyze_code_structure(source_code)
            metrics.update(structure)
        
        return metrics
    
    def map_to_complexity(self, metrics: Dict[str, Any]) -> ComplexityClass:
        """Enhanced маппинг с учетом циклов"""
        loop_depth = metrics.get('max_loop_depth', 0)
        total_loops = metrics.get('total_loops', 0)
        has_recursion = metrics.get('has_recursion', False)
        cc = metrics.get('cyclomatic_complexity', 1)
        
        return self._map_by_loops_enhanced(loop_depth, total_loops, has_recursion, cc)


class ComplexipyEnhancedAnalyzer(EnhancedAnalyzerMixin, ComplexipyBaselineAnalyzer):
    """
    ENHANCED: Complexipy + AST анализ циклов
    """
    
    def __init__(self):
        super().__init__()
        self.name = 'complexipy_enhanced'
    
    def analyze(self, source_code: str, language: str = 'python') -> Dict[str, Any]:
        """Complexipy метрики + структурный анализ"""
        metrics = super().analyze(source_code, language)
        
        if 'error' not in metrics or not metrics.get('error'):
            structure = self._analyze_code_structure(source_code)
            metrics.update(structure)
        
        return metrics
    
    def map_to_complexity(self, metrics: Dict[str, Any]) -> ComplexityClass:
        """Enhanced маппинг с учетом циклов"""
        loop_depth = metrics.get('max_loop_depth', 0)
        total_loops = metrics.get('total_loops', 0)
        has_recursion = metrics.get('has_recursion', False)
        cognitive = metrics.get('cognitive_complexity', 0)
        
        return self._map_by_loops_enhanced(loop_depth, total_loops, has_recursion, cognitive)


class HalsteadEnhancedAnalyzer(EnhancedAnalyzerMixin, HalsteadAnalyzer):
    """
    ENHANCED: Halstead + AST анализ циклов
    """
    
    def __init__(self):
        super().__init__()
        self.name = 'halstead_enhanced'
    
    def analyze(self, source_code: str, language: str = 'python') -> Dict[str, Any]:
        """Halstead метрики + структурный анализ"""
        metrics = super().analyze(source_code, language)
        
        if 'error' not in metrics or not metrics.get('error'):
            structure = self._analyze_code_structure(source_code)
            metrics.update(structure)
        
        return metrics
    
    def map_to_complexity(self, metrics: Dict[str, Any]) -> ComplexityClass:
        """Enhanced маппинг с учетом циклов"""
        loop_depth = metrics.get('max_loop_depth', 0)
        total_loops = metrics.get('total_loops', 0)
        has_recursion = metrics.get('has_recursion', False)
        difficulty = metrics.get('halstead_difficulty', 1)
        
        return self._map_by_loops_enhanced(loop_depth, total_loops, has_recursion, difficulty)


class MIEnhancedAnalyzer(EnhancedAnalyzerMixin, MaintainabilityIndexAnalyzer):
    """
    ENHANCED: Maintainability Index + AST анализ циклов
    """
    
    def __init__(self):
        super().__init__()
        self.name = 'mi_enhanced'
    
    def analyze(self, source_code: str, language: str = 'python') -> Dict[str, Any]:
        """MI метрики + структурный анализ"""
        metrics = super().analyze(source_code, language)
        
        if 'error' not in metrics or not metrics.get('error'):
            structure = self._analyze_code_structure(source_code)
            metrics.update(structure)
        
        return metrics
    
    def map_to_complexity(self, metrics: Dict[str, Any]) -> ComplexityClass:
        """Enhanced маппинг с учетом циклов"""
        loop_depth = metrics.get('max_loop_depth', 0)
        total_loops = metrics.get('total_loops', 0)
        has_recursion = metrics.get('has_recursion', False)
        mi = metrics.get('maintainability_index', 100)
        
        # MI обратно коррелирует, преобразуем в "сложность"
        mi_as_complexity = (100 - mi) / 10  # Превращаем в 0-10 шкалу
        
        return self._map_by_loops_enhanced(loop_depth, total_loops, has_recursion, mi_as_complexity)


class NBDEnhancedAnalyzer(EnhancedAnalyzerMixin, NestedBlockDepthAnalyzer):
    """
    ENHANCED: Nested Block Depth + улучшенный анализ
    Отличие: считаем ТОЛЬКО циклы, игнорируем if/with
    """
    
    def __init__(self):
        super().__init__()
        self.name = 'nbd_enhanced'
    
    def analyze(self, source_code: str, language: str = 'python') -> Dict[str, Any]:
        """NBD + фокус на циклах"""
        # Используем структурный анализ вместо baseline
        structure = self._analyze_code_structure(source_code)
        
        return {
            'nested_block_depth': structure['max_loop_depth'],  # Только циклы!
            'total_loops': structure['total_loops'],
            'has_recursion': structure['has_recursion'],
            'raw_output': structure
        }
    
    def map_to_complexity(self, metrics: Dict[str, Any]) -> ComplexityClass:
        """Enhanced маппинг только на основе циклов"""
        loop_depth = metrics.get('nested_block_depth', 0)
        total_loops = metrics.get('total_loops', 0)
        has_recursion = metrics.get('has_recursion', False)
        
        return self._map_by_loops_enhanced(loop_depth, total_loops, has_recursion, loop_depth)


# ============================================================================
#                         ФАБРИКА И ЭКСПОРТ
# ============================================================================

ENHANCED_ANALYZER_FACTORY = {
    'radon_enhanced': RadonEnhancedAnalyzer,
    'lizard_enhanced': LizardEnhancedAnalyzer,
    'mccabe_enhanced': McCabeEnhancedAnalyzer,
    'complexipy_enhanced': ComplexipyEnhancedAnalyzer,
    'halstead_enhanced': HalsteadEnhancedAnalyzer,
    'mi_enhanced': MIEnhancedAnalyzer,
    'nbd_enhanced': NBDEnhancedAnalyzer,
}


def get_enhanced_analyzer(tool_name: str) -> Optional[BaselineAnalyzer]:
    """Получить enhanced анализатор по имени"""
    analyzer_class = ENHANCED_ANALYZER_FACTORY.get(tool_name)
    if analyzer_class:
        return analyzer_class()
    return None
