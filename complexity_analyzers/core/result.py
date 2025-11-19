"""Классы результатов анализа"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json
from datetime import datetime

# ИМПОРТ из enums (единственный источник истины)
from .enums import ComplexityClass


@dataclass
class ComplexityMetrics:
    """Метрики сложности"""
    time_complexity: ComplexityClass = ComplexityClass.UNKNOWN
    space_complexity: ComplexityClass = ComplexityClass.UNKNOWN
    cyclomatic_complexity: Optional[int] = None
    cognitive_complexity: Optional[int] = None
    nested_depth: Optional[int] = None
    loop_count: Optional[int] = None
    recursive_calls: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'time_complexity': self.time_complexity.to_notation(),
            'space_complexity': self.space_complexity.to_notation(),
            'cyclomatic_complexity': self.cyclomatic_complexity,
            'cognitive_complexity': self.cognitive_complexity,
            'nested_depth': self.nested_depth,
            'loop_count': self.loop_count,
            'recursive_calls': self.recursive_calls
        }


@dataclass
class ComplexityResult:
    """Результат анализа сложности"""
    complexity_class: ComplexityClass
    confidence: float = 0.0
    analyzer_name: str = ""
    analysis_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Детальные метрики
    metrics: ComplexityMetrics = field(default_factory=ComplexityMetrics)
    
    # Специфичные данные анализаторов
    ast_features: Dict[str, Any] = field(default_factory=dict)
    runtime_data: Dict[str, Any] = field(default_factory=dict)
    cfg_metrics: Dict[str, Any] = field(default_factory=dict)
    ml_predictions: Dict[str, Any] = field(default_factory=dict)
    dynamic_traces: Dict[str, Any] = field(default_factory=dict)
    tool_outputs: Dict[str, Any] = field(default_factory=dict)
    
    # Метаданные
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    debug_info: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Проверка валидности результата"""
        return (self.complexity_class != ComplexityClass.UNKNOWN and 
                0.0 <= self.confidence <= 1.0 and 
                not self.errors)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'complexity_class': self.complexity_class.to_notation(),
            'confidence': self.confidence,
            'analyzer_name': self.analyzer_name,
            'analysis_time': self.analysis_time,
            'timestamp': self.timestamp.isoformat(),
            'metrics': self.metrics.to_dict(),
            'ast_features': self.ast_features,
            'runtime_data': self.runtime_data,
            'cfg_metrics': self.cfg_metrics,
            'ml_predictions': self.ml_predictions,
            'dynamic_traces': self.dynamic_traces,
            'tool_outputs': self.tool_outputs,
            'errors': self.errors,
            'warnings': self.warnings
        }
    
    def to_json(self) -> str:
        """Сериализация в JSON"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class ResultAggregator:
    """Агрегатор результатов от нескольких анализаторов"""
    
    def __init__(self):
        self.results: List[ComplexityResult] = []
        self.weights: Dict[str, float] = {}
    
    def add_result(self, result: ComplexityResult, weight: float = 1.0) -> None:
        """Добавление результата"""
        self.results.append(result)
        self.weights[result.analyzer_name] = weight
    
    def aggregate(self) -> ComplexityResult:
        """Агрегация результатов"""
        if not self.results:
            return ComplexityResult(ComplexityClass.UNKNOWN)
        
        # Взвешенное голосование
        complexity_votes = {}
        total_confidence = 0.0
        total_weight = 0.0
        
        for result in self.results:
            if result.is_valid():
                weight = self.weights.get(result.analyzer_name, 1.0)
                vote_power = weight * result.confidence
                
                if result.complexity_class not in complexity_votes:
                    complexity_votes[result.complexity_class] = 0.0
                complexity_votes[result.complexity_class] += vote_power
                
                total_confidence += result.confidence * weight
                total_weight += weight
        
        if not complexity_votes:
            return ComplexityResult(ComplexityClass.UNKNOWN)
        
        # Выбираем класс с максимальным весом
        best_complexity = max(complexity_votes, key=complexity_votes.get)
        avg_confidence = total_confidence / total_weight if total_weight > 0 else 0.0
        
        return ComplexityResult(
            complexity_class=best_complexity,
            confidence=min(avg_confidence, 1.0),
            analyzer_name="aggregated",
            debug_info={
                'individual_results': len(self.results),
                'complexity_votes': {c.to_notation(): v for c, v in complexity_votes.items()}
            }
        )
