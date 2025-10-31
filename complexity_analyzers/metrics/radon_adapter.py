"""Адаптер для библиотеки Radon"""
from typing import Dict, Any, Optional, List
import logging

from complexity_analyzers.metrics.base import BaseMetricsCalculator

logger = logging.getLogger(__name__)

class RadonAdapter(BaseMetricsCalculator):
    """Адаптер для интеграции с библиотекой Radon"""
    
    def __init__(self):
        super().__init__('radon')
        self._radon_available = self._check_radon_availability()
    
    def _check_radon_availability(self) -> bool:
        """Проверка доступности Radon"""
        try:
            import radon.complexity
            import radon.metrics
            return True
        except ImportError:
            logger.warning("Radon library not available. Install with: pip install radon")
            return False
    
    def is_available(self) -> bool:
        """Проверка доступности адаптера"""
        return self._radon_available
    
    def calculate(self, source_code: str) -> Dict[str, Any]:
        """Вычисление метрик через Radon"""
        if not self._radon_available:
            return {'error': 'Radon not available'}
        
        try:
            from radon.complexity import cc_visit
            from radon.metrics import mi_visit, h_visit
            
            metrics = {}
            
            # Цикломатическая сложность
            cc_results = cc_visit(source_code)
            if cc_results:
                complexities = [result.complexity for result in cc_results]
                metrics['cyclomatic_complexity'] = max(complexities)
                metrics['average_cyclomatic_complexity'] = sum(complexities) / len(complexities)
                metrics['total_functions'] = len(cc_results)
                
                # Детальная информация о функциях
                metrics['function_complexities'] = [
                    {
                        'name': result.name,
                        'complexity': result.complexity,
                        'type': result.letter,
                        'line': result.lineno,
                        'col': result.col_offset,
                        'end_line': result.endline,
                        'end_col': result.end_col_offset
                    }
                    for result in cc_results
                ]
            else:
                metrics['cyclomatic_complexity'] = 1
                metrics['average_cyclomatic_complexity'] = 1
                metrics['total_functions'] = 0
                metrics['function_complexities'] = []
            
            # Индекс сопровождаемости
            try:
                mi_score = mi_visit(source_code, multi=True)
                if isinstance(mi_score, (int, float)):
                    metrics['maintainability_index'] = float(mi_score)
                elif hasattr(mi_score, '__iter__'):
                    # Если возвращается список, берём среднее
                    mi_values = list(mi_score)
                    if mi_values:
                        metrics['maintainability_index'] = sum(mi_values) / len(mi_values)
                    else:
                        metrics['maintainability_index'] = 100.0
                else:
                    metrics['maintainability_index'] = 100.0
                
                # Ранг сопровождаемости
                mi_value = metrics['maintainability_index']
                if mi_value >= 80:
                    metrics['maintainability_rank'] = 'A'
                elif mi_value >= 60:
                    metrics['maintainability_rank'] = 'B'
                elif mi_value >= 40:
                    metrics['maintainability_rank'] = 'C'
                elif mi_value >= 20:
                    metrics['maintainability_rank'] = 'D'
                else:
                    metrics['maintainability_rank'] = 'F'
                    
            except Exception as e:
                logger.warning(f"Failed to calculate maintainability index: {e}")
                metrics['maintainability_index'] = None
                metrics['maintainability_rank'] = None
            
            # Halstead метрики
            try:
                h_results = h_visit(source_code)
                if isinstance(h_results, tuple) and len(h_results) >= 1:
                    total_halstead = h_results[0]
                    
                    if total_halstead:
                        metrics['halstead_difficulty'] = total_halstead.difficulty or 0
                        metrics['halstead_volume'] = total_halstead.volume or 0
                        metrics['halstead_effort'] = total_halstead.effort or 0
                        metrics['halstead_bugs'] = total_halstead.bugs or 0
                        metrics['halstead_time'] = total_halstead.time or 0
                        
                        # Детальные Halstead метрики
                        metrics['halstead_vocabulary'] = getattr(total_halstead, 'vocabulary', 0)
                        metrics['halstead_length'] = getattr(total_halstead, 'length', 0)
                        metrics['halstead_calculated_length'] = getattr(total_halstead, 'calculated_length', 0)
                    else:
                        self._set_default_halstead_metrics(metrics)
                else:
                    self._set_default_halstead_metrics(metrics)
                    
            except Exception as e:
                logger.warning(f"Failed to calculate Halstead metrics: {e}")
                self._set_default_halstead_metrics(metrics)
            
            # Дополнительные метрики на основе сырых данных
            try:
                metrics.update(self._calculate_additional_metrics(source_code))
            except Exception as e:
                logger.warning(f"Failed to calculate additional metrics: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Radon calculation failed: {e}")
            return {'error': str(e)}
    
    def _set_default_halstead_metrics(self, metrics: Dict[str, Any]):
        """Установка значений Halstead метрик по умолчанию"""
        metrics['halstead_difficulty'] = 0
        metrics['halstead_volume'] = 0
        metrics['halstead_effort'] = 0
        metrics['halstead_bugs'] = 0
        metrics['halstead_time'] = 0
        metrics['halstead_vocabulary'] = 0
        metrics['halstead_length'] = 0
        metrics['halstead_calculated_length'] = 0
    
    def _calculate_additional_metrics(self, source_code: str) -> Dict[str, Any]:
        """Дополнительные метрики на основе исходного кода"""
        lines = source_code.split('\n')
        
        # Подсчёт строк
        total_lines = len(lines)
        blank_lines = sum(1 for line in lines if not line.strip())
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        code_lines = total_lines - blank_lines - comment_lines
        
        # Средняя длина строки
        non_empty_lines = [line for line in lines if line.strip()]
        avg_line_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines) if non_empty_lines else 0
        
        return {
            'radon_total_lines': total_lines,
            'radon_blank_lines': blank_lines,
            'radon_comment_lines': comment_lines,
            'radon_code_lines': code_lines,
            'radon_avg_line_length': avg_line_length,
            'radon_comment_ratio': comment_lines / total_lines if total_lines > 0 else 0
        }
    
    def calculate_raw_metrics(self, source_code: str) -> Dict[str, Any]:
        """Получение сырых метрик от Radon без обработки"""
        if not self._radon_available:
            return {}
        
        try:
            from radon.complexity import cc_visit
            from radon.metrics import mi_visit, h_visit
            from radon.raw import analyze
            
            raw_results = {}
            
            # Сырые метрики (LOC, LLOC, SLOC, etc.)
            try:
                raw_metrics = analyze(source_code)
                raw_results['raw'] = {
                    'loc': raw_metrics.loc,  # Lines of Code
                    'lloc': raw_metrics.lloc,  # Logical Lines of Code
                    'sloc': raw_metrics.sloc,  # Source Lines of Code
                    'comments': raw_metrics.comments,
                    'multi': raw_metrics.multi,  # Multi-line strings
                    'blank': raw_metrics.blank,
                    'single_comments': raw_metrics.single_comments
                }
            except Exception as e:
                logger.warning(f"Failed to get raw metrics: {e}")
                raw_results['raw'] = None
            
            # Сырые результаты цикломатической сложности
            try:
                raw_results['complexity'] = cc_visit(source_code)
            except Exception as e:
                logger.warning(f"Failed to get complexity results: {e}")
                raw_results['complexity'] = []
            
            # Сырые Halstead результаты
            try:
                raw_results['halstead'] = h_visit(source_code)
            except Exception as e:
                logger.warning(f"Failed to get Halstead results: {e}")
                raw_results['halstead'] = None
            
            # Сырой индекс сопровождаемости
            try:
                raw_results['maintainability'] = mi_visit(source_code, multi=True)
            except Exception as e:
                logger.warning(f"Failed to get maintainability results: {e}")
                raw_results['maintainability'] = None
            
            return raw_results
            
        except Exception as e:
            logger.error(f"Failed to get raw Radon metrics: {e}")
            return {}
    
    def get_function_level_metrics(self, source_code: str) -> List[Dict[str, Any]]:
        """Получение метрик на уровне функций"""
        if not self._radon_available:
            return []
        
        try:
            from radon.complexity import cc_visit
            
            cc_results = cc_visit(source_code)
            
            function_metrics = []
            for result in cc_results:
                func_metrics = {
                    'name': result.name,
                    'complexity': result.complexity,
                    'type': result.letter,  # F=function, M=method, C=class
                    'line_start': result.lineno,
                    'col_start': result.col_offset,
                    'line_end': result.endline,
                    'col_end': result.end_col_offset,
                    'is_method': result.letter == 'M',
                    'is_function': result.letter == 'F',
                    'is_class': result.letter == 'C'
                }
                
                # Классификация сложности функции
                complexity = result.complexity
                if complexity <= 5:
                    func_metrics['complexity_rank'] = 'A'  # Low
                elif complexity <= 10:
                    func_metrics['complexity_rank'] = 'B'  # Moderate
                elif complexity <= 20:
                    func_metrics['complexity_rank'] = 'C'  # High
                else:
                    func_metrics['complexity_rank'] = 'F'  # Very High
                
                function_metrics.append(func_metrics)
            
            return function_metrics
            
        except Exception as e:
            logger.error(f"Failed to get function-level metrics: {e}")
            return []
    
    def calculate_complexity_distribution(self, source_code: str) -> Dict[str, int]:
        """Распределение функций по уровням сложности"""
        function_metrics = self.get_function_level_metrics(source_code)
        
        distribution = {'A': 0, 'B': 0, 'C': 0, 'F': 0}
        
        for func in function_metrics:
            rank = func.get('complexity_rank', 'A')
            if rank in distribution:
                distribution[rank] += 1
        
        return distribution
