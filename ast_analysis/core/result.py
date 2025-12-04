"""
Структуры для результатов AST анализа.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class ASTMetrics:
    """Базовые AST метрики"""
    total_nodes: int = 0
    unique_node_types: int = 0
    max_depth: int = 0
    max_width: int = 0
    
    num_functions: int = 0
    num_classes: int = 0
    num_imports: int = 0
    
    num_loops: int = 0
    num_conditionals: int = 0
    max_nesting_depth: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'total_nodes': self.total_nodes,
            'unique_node_types': self.unique_node_types,
            'max_depth': self.max_depth,
            'max_width': self.max_width,
            'num_functions': self.num_functions,
            'num_classes': self.num_classes,
            'num_imports': self.num_imports,
            'num_loops': self.num_loops,
            'num_conditionals': self.num_conditionals,
            'max_nesting_depth': self.max_nesting_depth,
        }


@dataclass
class ASTAnalysisResult:
    """
    Результат AST анализа кода с предсказанием сложности.
    """
    success: bool
    features: Dict[str, Any]
    
    # Опциональные поля
    error: Optional[str] = None
    processing_time: float = 0.0
    parsed: bool = False
    
    # Предсказание сложности (НОВОЕ)
    prediction: Optional[str] = None  # "O(N)", "O(N^2)", etc.
    confidence: float = 0.0  # 0.0 - 1.0
    prediction_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Метаданные
    analyzer_name: Optional[str] = None
    code_length: Optional[int] = None
    metrics: Optional[ASTMetrics] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь"""
        result = {
            'success': self.success,
            'features': self.features,
            'parsed': self.parsed,
            'processing_time': self.processing_time,
        }
        
        # Предсказание
        if self.prediction:
            result['prediction'] = self.prediction
            result['confidence'] = self.confidence
            result['prediction_metadata'] = self.prediction_metadata
        
        if self.error:
            result['error'] = self.error
        
        if self.analyzer_name:
            result['analyzer_name'] = self.analyzer_name
        
        if self.code_length:
            result['code_length'] = self.code_length
        
        if self.metrics:
            result['metrics'] = self.metrics.to_dict()
        
        if self.metadata:
            result['metadata'] = self.metadata
        
        return result
    
    @classmethod
    def from_error(cls, error: str, analyzer_name: Optional[str] = None) -> 'ASTAnalysisResult':
        """Создание результата с ошибкой"""
        return cls(
            success=False,
            features={},
            error=error,
            parsed=False,
            analyzer_name=analyzer_name
        )
    
    @classmethod
    def from_success(cls, 
                     features: Dict[str, Any],
                     analyzer_name: Optional[str] = None,
                     prediction: Optional[str] = None,
                     confidence: float = 0.0,
                     prediction_metadata: Optional[Dict] = None,
                     **kwargs) -> 'ASTAnalysisResult':
        """Создание успешного результата"""
        return cls(
            success=True,
            features=features,
            parsed=True,
            analyzer_name=analyzer_name,
            prediction=prediction,
            confidence=confidence,
            prediction_metadata=prediction_metadata or {},
            **kwargs
        )


@dataclass
class BatchAnalysisResult:
    """Результат батчевого анализа"""
    results: List[ASTAnalysisResult]
    total_samples: int
    successful: int
    failed: int
    avg_processing_time: float
    
    @classmethod
    def from_results(cls, results: List[ASTAnalysisResult]) -> 'BatchAnalysisResult':
        """Создание из списка результатов"""
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        
        times = [r.processing_time for r in results if r.processing_time > 0]
        avg_time = sum(times) / len(times) if times else 0.0
        
        return cls(
            results=results,
            total_samples=total,
            successful=successful,
            failed=failed,
            avg_processing_time=avg_time
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация"""
        return {
            'total_samples': self.total_samples,
            'successful': self.successful,
            'failed': self.failed,
            'success_rate': self.successful / self.total_samples if self.total_samples > 0 else 0.0,
            'avg_processing_time': self.avg_processing_time,
            'results': [r.to_dict() for r in self.results]
        }
