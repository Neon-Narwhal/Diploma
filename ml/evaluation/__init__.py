"""
Evaluation модуль.
"""

from ml.evaluation.metrics import (
    compute_metrics,
    compute_confusion_matrix,
    compute_classification_report,
    compute_per_class_metrics,
)
from ml.evaluation.comparison import ModelComparison
from ml.evaluation.visualization import ModelVisualizer
from ml.evaluation.report import ReportGenerator

__all__ = [
    'compute_metrics',
    'compute_confusion_matrix',
    'compute_classification_report',
    'compute_per_class_metrics',
    'ModelComparison',
    'ModelVisualizer',
    'ReportGenerator',
]
