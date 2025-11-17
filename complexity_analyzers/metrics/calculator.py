"""Общий калькулятор метрик сложности"""
import ast
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

from complexity_analyzers.metrics.base import BaseMetricsCalculator
from complexity_analyzers.core.enums import ComplexityClass

@dataclass
class MetricsResult:
    """Результат вычисления метрик"""
    cyclomatic_complexity: Optional[int] = None
    cognitive_complexity: Optional[int] = None
    halstead_difficulty: Optional[float] = None
    halstead_volume: Optional[float] = None
    halstead_effort: Optional[float] = None
    maintainability_index: Optional[float] = None
    lines_of_code: Optional[int] = None
    logical_lines_of_code: Optional[int] = None
    comment_lines: Optional[int] = None
    blank_lines: Optional[int] = None
    nested_depth: Optional[int] = None
    function_count: Optional[int] = None
    class_count: Optional[int] = None
    complexity_density: Optional[float] = None
    raw_metrics: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'cyclomatic_complexity': self.cyclomatic_complexity,
            'cognitive_complexity': self.cognitive_complexity,
            'halstead_difficulty': self.halstead_difficulty,
            'halstead_volume': self.halstead_volume,
            'halstead_effort': self.halstead_effort,
            'maintainability_index': self.maintainability_index,
            'lines_of_code': self.lines_of_code,
            'logical_lines_of_code': self.logical_lines_of_code,
            'comment_lines': self.comment_lines,
            'blank_lines': self.blank_lines,
            'nested_depth': self.nested_depth,
            'function_count': self.function_count,
            'class_count': self.class_count,
            'complexity_density': self.complexity_density,
            'raw_metrics': self.raw_metrics or {}
        }

class UniversalMetricsCalculator:
    """Универсальный калькулятор метрик"""
    
    def __init__(self):
        self.calculators: Dict[str, BaseMetricsCalculator] = {}
        # ИСПРАВЛЕНО: импорт перенесен сюда
        from complexity_analyzers.metrics.custom_metrics import CustomMetricsCalculator
        self.custom_calculator = CustomMetricsCalculator()
        
        # Инициализация адаптеров
        self._initialize_calculators()
    
    def _initialize_calculators(self):
        """Инициализация всех доступных калькуляторов"""
        # ИСПРАВЛЕНО: импорты перенесены внутрь метода
        from complexity_analyzers.metrics.radon_adapter import RadonAdapter
        from complexity_analyzers.metrics.mccabe_adapter import McCabeAdapter
        
        radon_adapter = RadonAdapter()
        if radon_adapter.is_available():
            self.calculators['radon'] = radon_adapter
        
        mccabe_adapter = McCabeAdapter()
        if mccabe_adapter.is_available():
            self.calculators['mccabe'] = mccabe_adapter
        
        self.calculators['custom'] = self.custom_calculator
    
    def calculate_all_metrics(self, source_code: str, 
                            enabled_calculators: Optional[List[str]] = None) -> MetricsResult:
        """Вычисление всех доступных метрик"""
        if enabled_calculators is None:
            enabled_calculators = list(self.calculators.keys())
        
        all_metrics = {}
        
        # Собираем метрики от всех калькуляторов
        for calc_name in enabled_calculators:
            if calc_name in self.calculators:
                try:
                    metrics = self.calculators[calc_name].calculate(source_code)
                    all_metrics[calc_name] = metrics
                except Exception as e:
                    all_metrics[calc_name] = {'error': str(e)}
        
        # Агрегируем результаты
        return self._aggregate_metrics(all_metrics, source_code)
    
    def _aggregate_metrics(self, all_metrics: Dict[str, Dict[str, Any]], 
                          source_code: str) -> MetricsResult:
        """Агрегация метрик от разных калькуляторов"""
        result = MetricsResult(raw_metrics=all_metrics)
        
        # Цикломатическая сложность - берём максимальную
        cc_values = []
        for calc_name, metrics in all_metrics.items():
            if 'cyclomatic_complexity' in metrics:
                cc_values.append(metrics['cyclomatic_complexity'])
        
        if cc_values:
            result.cyclomatic_complexity = max(cc_values)
        
        # Когнитивная сложность
        cognitive_values = []
        for calc_name, metrics in all_metrics.items():
            if 'cognitive_complexity' in metrics:
                cognitive_values.append(metrics['cognitive_complexity'])
        
        if cognitive_values:
            result.cognitive_complexity = max(cognitive_values)
        
        # Halstead метрики - берём из Radon если доступны
        if 'radon' in all_metrics:
            radon_metrics = all_metrics['radon']
            result.halstead_difficulty = radon_metrics.get('halstead_difficulty')
            result.halstead_volume = radon_metrics.get('halstead_volume')
            result.halstead_effort = radon_metrics.get('halstead_effort')
            result.maintainability_index = radon_metrics.get('maintainability_index')
        
        # Базовые метрики кода
        if 'custom' in all_metrics:
            custom_metrics = all_metrics['custom']
            result.lines_of_code = custom_metrics.get('lines_of_code')
            result.logical_lines_of_code = custom_metrics.get('logical_lines_of_code')
            result.comment_lines = custom_metrics.get('comment_lines')
            result.blank_lines = custom_metrics.get('blank_lines')
            result.nested_depth = custom_metrics.get('nested_depth')
            result.function_count = custom_metrics.get('function_count')
            result.class_count = custom_metrics.get('class_count')
        
        # Вычисляемые метрики
        if result.cyclomatic_complexity and result.lines_of_code:
            result.complexity_density = result.cyclomatic_complexity / result.lines_of_code
        
        return result
    
    def get_available_calculators(self) -> List[str]:
        """Получение списка доступных калькуляторов"""
        return list(self.calculators.keys())
    
    def add_calculator(self, name: str, calculator: BaseMetricsCalculator):
        """Добавление нового калькулятора"""
        if calculator.is_available():
            self.calculators[name] = calculator
    
    def remove_calculator(self, name: str):
        """Удаление калькулятора"""
        if name in self.calculators and name != 'custom':  # Не удаляем custom
            del self.calculators[name]

class ComplexityClassifier:
    """Классификатор сложности на основе метрик"""
    
    def __init__(self):
        # Пороговые значения для классификации
        self.cyclomatic_thresholds = {
            (1, 5): ComplexityClass.CONSTANT,
            (6, 10): ComplexityClass.LINEAR,
            (11, 20): ComplexityClass.QUADRATIC,
            (21, 50): ComplexityClass.CUBIC,
            (51, float('inf')): ComplexityClass.EXPONENTIAL
        }
        
        self.cognitive_thresholds = {
            (0, 5): ComplexityClass.CONSTANT,
            (6, 15): ComplexityClass.LINEAR,
            (16, 30): ComplexityClass.QUADRATIC,
            (31, 60): ComplexityClass.CUBIC,
            (61, float('inf')): ComplexityClass.EXPONENTIAL
        }
        
        self.nesting_thresholds = {
            0: ComplexityClass.CONSTANT,
            1: ComplexityClass.LINEAR,
            2: ComplexityClass.QUADRATIC,
            3: ComplexityClass.CUBIC,
        }
    
    def classify_by_cyclomatic(self, cyclomatic_complexity: int) -> ComplexityClass:
        """Классификация по цикломатической сложности"""
        for (min_val, max_val), complexity_class in self.cyclomatic_thresholds.items():
            if min_val <= cyclomatic_complexity < max_val:
                return complexity_class
        return ComplexityClass.EXPONENTIAL
    
    def classify_by_cognitive(self, cognitive_complexity: int) -> ComplexityClass:
        """Классификация по когнитивной сложности"""
        for (min_val, max_val), complexity_class in self.cognitive_thresholds.items():
            if min_val <= cognitive_complexity < max_val:
                return complexity_class
        return ComplexityClass.EXPONENTIAL
    
    def classify_by_nesting(self, nested_depth: int) -> ComplexityClass:
        """Классификация по глубине вложенности"""
        if nested_depth in self.nesting_thresholds:
            return self.nesting_thresholds[nested_depth]
        elif nested_depth >= 4:
            return ComplexityClass.EXPONENTIAL
        else:
            return ComplexityClass.CONSTANT
    
    def classify_by_maintainability(self, maintainability_index: float) -> ComplexityClass:
        """Классификация по индексу сопровождаемости (обратная корреляция)"""
        if maintainability_index >= 80:
            return ComplexityClass.CONSTANT
        elif maintainability_index >= 60:
            return ComplexityClass.LINEAR
        elif maintainability_index >= 40:
            return ComplexityClass.QUADRATIC
        elif maintainability_index >= 20:
            return ComplexityClass.CUBIC
        else:
            return ComplexityClass.EXPONENTIAL
    
    def classify_combined(self, metrics: MetricsResult) -> Dict[str, ComplexityClass]:
        """Комбинированная классификация"""
        classifications = {}
        
        if metrics.cyclomatic_complexity is not None:
            classifications['cyclomatic'] = self.classify_by_cyclomatic(
                metrics.cyclomatic_complexity
            )
        
        if metrics.cognitive_complexity is not None:
            classifications['cognitive'] = self.classify_by_cognitive(
                metrics.cognitive_complexity
            )
        
        if metrics.nested_depth is not None:
            classifications['nesting'] = self.classify_by_nesting(
                metrics.nested_depth
            )
        
        if metrics.maintainability_index is not None:
            classifications['maintainability'] = self.classify_by_maintainability(
                metrics.maintainability_index
            )
        
        return classifications
    
    def get_consensus_classification(self, classifications: Dict[str, ComplexityClass]) -> ComplexityClass:
        """Получение консенсусной классификации"""
        if not classifications:
            return ComplexityClass.UNKNOWN
        
        # Простое мажоритарное голосование
        from collections import Counter
        votes = Counter(classifications.values())
        
        # Если есть ничья, берём более высокую сложность
        most_common = votes.most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            # Ничья - берём максимальную сложность
            return max(classifications.values())
        else:
            return most_common[0][0]

class MetricsAnalyzer:
    """Анализатор метрик с классификацией"""
    
    def __init__(self):
        self.calculator = UniversalMetricsCalculator()
        self.classifier = ComplexityClassifier()
    
    def analyze(self, source_code: str) -> Dict[str, Any]:
        """Полный анализ метрик с классификацией"""
        # Вычисление метрик
        metrics = self.calculator.calculate_all_metrics(source_code)
        
        # Классификация
        classifications = self.classifier.classify_combined(metrics)
        consensus = self.classifier.get_consensus_classification(classifications)
        
        # Оценка уверенности
        confidence = self._calculate_confidence(metrics, classifications)
        
        return {
            'metrics': metrics.to_dict(),
            'classifications': {k: v.notation for k, v in classifications.items()},
            'consensus_complexity': consensus.notation,
            'confidence': confidence,
            'available_calculators': self.calculator.get_available_calculators()
        }
    
    def _calculate_confidence(self, metrics: MetricsResult, 
                            classifications: Dict[str, ComplexityClass]) -> float:
        """Расчёт уверенности в классификации"""
        if not classifications:
            return 0.0
        
        # Базовая уверенность
        base_confidence = 0.7
        
        # Увеличиваем уверенность при согласованности классификаций
        unique_classifications = set(classifications.values())
        if len(unique_classifications) == 1:
            # Все методы дают одинаковый результат
            base_confidence += 0.2
        elif len(unique_classifications) <= 2:
            # Небольшие расхождения
            base_confidence += 0.1
        
        # Увеличиваем уверенность при наличии нескольких метрик
        metrics_count = len([m for m in [
            metrics.cyclomatic_complexity,
            metrics.cognitive_complexity,
            metrics.nested_depth,
            metrics.maintainability_index
        ] if m is not None])
        
        if metrics_count >= 3:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
