"""
Вычисление метрик для оценки моделей.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, List, Optional


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: List[str],
    y_pred_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Вычисление набора метрик.
    
    Args:
        y_true: истинные метки
        y_pred: предсказанные метки
        metrics: список метрик для вычисления
        y_pred_proba: вероятности (для ROC-AUC)
        
    Returns:
        Словарь метрик
    """
    results = {}
    
    for metric in metrics:
        if metric == 'accuracy':
            results[metric] = accuracy_score(y_true, y_pred)
        
        elif metric == 'f1_macro':
            results[metric] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        elif metric == 'f1_micro':
            results[metric] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        elif metric == 'f1_weighted':
            results[metric] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        elif metric == 'precision_macro':
            results[metric] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        
        elif metric == 'precision_micro':
            results[metric] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        
        elif metric == 'recall_macro':
            results[metric] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        elif metric == 'recall_micro':
            results[metric] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        
        elif metric == 'roc_auc' and y_pred_proba is not None:
            try:
                results[metric] = roc_auc_score(
                    y_true,
                    y_pred_proba,
                    multi_class='ovr',
                    average='macro',
                )
            except:
                results[metric] = 0.0
    
    return results


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Вычисление confusion matrix"""
    return confusion_matrix(y_true, y_pred)


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> str:
    """Вычисление classification report"""
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """
    Вычисление метрик для каждого класса отдельно.
    
    Returns:
        Словарь {класс: {метрика: значение}}
    """
    classes = np.unique(y_true)
    per_class = {}
    
    for cls in classes:
        # Бинарная маска для класса
        mask = (y_true == cls)
        y_true_binary = mask.astype(int)
        y_pred_binary = (y_pred == cls).astype(int)
        
        per_class[int(cls)] = {
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
            'support': int(mask.sum()),
        }
    
    return per_class
