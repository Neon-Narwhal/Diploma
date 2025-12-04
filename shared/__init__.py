"""
Shared библиотека общих компонентов.
"""

from shared.configs.base import BaseConfig
from shared.data_loader import Dataset, CodeSample, DataLoader
from shared.evaluation import (
    compute_metrics,
    compute_per_class_metrics,
    compute_confusion_matrix,
    ReportGenerator,
    Evaluator
)
from shared.processing import BatchProcessor, BatchGenerator, ProcessingResult
from shared.utils import Logger, ExperimentLogger, MLflowTracker

__all__ = [
    # Configs
    'BaseConfig',
    
    # Data
    'Dataset',
    'CodeSample',
    'DataLoader',
    
    # Evaluation
    'compute_metrics',
    'compute_per_class_metrics',
    'compute_confusion_matrix',
    'ReportGenerator',
    'Evaluator',
    
    # Processing
    'BatchProcessor',
    'BatchGenerator',
    'ProcessingResult',
    
    # Utils
    'Logger',
    'ExperimentLogger',
    'MLflowTracker',
]
