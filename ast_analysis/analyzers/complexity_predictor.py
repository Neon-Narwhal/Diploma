"""
Предсказатель сложности на основе AST анализа.
"""

from typing import Dict, Any, Tuple
from ast_analysis.core.enums import ComplexityClass


class ComplexityPredictor:
    """
    Предсказатель класса сложности на основе результатов анализа AST.
    Использует эвристические правила из старого анализатора.
    """
    
    def __init__(self):
        self.name = "complexity_predictor"
    
    def predict(self, analysis_data: Dict[str, Any]) -> Tuple[str, float]:
        """
        Предсказание класса сложности.
        
        Args:
            analysis_data: Результаты анализа (циклы, рекурсия, структуры данных)
        
        Returns:
            (predicted_class, confidence) - класс сложности и уверенность
        """
        loop_summary = analysis_data.get('loop_summary', {})
        recursion_summary = analysis_data.get('recursion_summary', {})
        features = analysis_data.get('features', {})
        
        # Проверка рекурсии
        if recursion_summary.get('recursive_functions', 0) > 0:
            return self._predict_from_recursion(recursion_summary)
        
        # Проверка циклов
        if loop_summary.get('total_loops', 0) > 0:
            return self._predict_from_loops(loop_summary, features)
        
        # Нет циклов и рекурсии
        return (ComplexityClass.CONSTANT.value, 0.9)
    
    def _predict_from_loops(self, loop_summary: Dict[str, Any], 
                           features: Dict[str, Any]) -> Tuple[str, float]:
        """Предсказание на основе циклов"""
        max_nesting = loop_summary.get('max_nesting', 0)
        has_log_step = loop_summary.get('has_logarithmic_step', False)
        has_dep_loop = loop_summary.get('has_dependent_inner_loop', False)
        has_sorting = self._has_sorting_operation(features)
        
        # Логика из старого анализатора
        if max_nesting == 0:
            return (ComplexityClass.CONSTANT.value, 0.95)
        
        elif max_nesting == 1:
            if has_log_step:
                # i *= 2 или i //= 2
                return (ComplexityClass.LOGARITHMIC.value, 0.85)
            elif has_sorting:
                # sorted() внутри цикла
                return (ComplexityClass.LINEARITHMIC.value, 0.8)
            else:
                # Обычный цикл
                return (ComplexityClass.LINEAR.value, 0.9)
        
        elif max_nesting == 2:
            if has_log_step:
                # for(N) { while(N/=2) }
                return (ComplexityClass.LINEARITHMIC.value, 0.85)
            elif has_dep_loop:
                # for i in range(n): for j in range(i)
                return (ComplexityClass.QUADRATIC.value, 0.9)
            else:
                # Два независимых вложенных цикла
                return (ComplexityClass.QUADRATIC.value, 0.85)
        
        elif max_nesting >= 3:
            return (ComplexityClass.CUBIC.value, 0.85)
        
        return (ComplexityClass.UNKNOWN.value, 0.3)
    
    def _predict_from_recursion(self, recursion_summary: Dict[str, Any]) -> Tuple[str, float]:
        """Предсказание на основе рекурсии"""
        patterns = recursion_summary.get('recursion_patterns', [])
        
        if not patterns:
            return (ComplexityClass.LINEAR.value, 0.6)
        
        # Берём первую рекурсивную функцию
        pattern = patterns[0]
        recursion_type = pattern.get('recursion_type', 'none')
        estimated = pattern.get('estimated_complexity')
        
        if recursion_type == 'linear':
            return (ComplexityClass.LINEAR.value, 0.8)
        elif recursion_type == 'binary':
            return (ComplexityClass.EXPONENTIAL.value, 0.85)
        elif recursion_type == 'tree':
            return (ComplexityClass.FACTORIAL.value, 0.8)
        
        # Fallback на оценку из рекурсии
        if estimated:
            return (estimated.value, 0.7)
        
        return (ComplexityClass.EXPONENTIAL.value, 0.6)
    
    def _has_sorting_operation(self, features: Dict[str, Any]) -> bool:
        """Проверка наличия операций сортировки"""
        # Ищем вызовы sorted, sort
        # Простая эвристика: если есть много операций
        num_calls = features.get('Call', 0)
        return num_calls > 0  # Упрощено, можно детализировать
