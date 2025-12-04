"""
Универсальный evaluator для оценки результатов.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from shared.evaluation.metrics import compute_metrics, compute_per_class_metrics, compute_confusion_matrix


class Evaluator:
    """
    Оценщик результатов анализа с классификацией.
    Работает с предсказаниями и ground truth метками.
    """
    
    def __init__(self, metric_names: List[str]):
        """
        Args:
            metric_names: Список метрик для вычисления
        """
        self.metric_names = metric_names
    
    def evaluate(self, 
                 results: List[Dict[str, Any]],
                 include_per_class: bool = True) -> Dict[str, Any]:
        """
        Оценка результатов с предсказаниями.
        
        Args:
            results: Результаты обработки с полями:
                - ground_truth: истинная метка
                - analyzers: {analyzer_name: {prediction: ...}}
            include_per_class: Вычислять per-class метрики
        
        Returns:
            Словарь с метриками и per-class метриками
        """
        # Извлечение предсказаний и ground truth
        y_true, y_pred = self._extract_labels(results)
        
        if len(y_true) == 0 or len(y_pred) == 0:
            return {
                'metrics': {},
                'per_class_metrics': {},
                'error': 'No valid predictions found'
            }
        
        # Вычисление основных метрик
        metrics = compute_metrics(
            y_true=np.array(y_true),
            y_pred=np.array(y_pred),
            metrics=self.metric_names
        )
        
        result = {'metrics': metrics}
        
        # Per-class метрики
        if include_per_class:
            per_class = compute_per_class_metrics(
                y_true=np.array(y_true),
                y_pred=np.array(y_pred)
            )
            result['per_class_metrics'] = per_class
        
        # Confusion matrix
        cm = compute_confusion_matrix(
            y_true=np.array(y_true),
            y_pred=np.array(y_pred)
        )
        result['confusion_matrix'] = cm
        
        return result
    
    def _extract_labels(self, results: List[Dict[str, Any]]) -> tuple[List[str], List[str]]:
        """
        Извлечение ground truth и предсказаний.
        
        Args:
            results: Список результатов
        
        Returns:
            (y_true, y_pred) - списки меток
        """
        y_true = []
        y_pred = []
        
        for result in results:
            # Ground truth
            gt = result.get('ground_truth')
            if not gt:
                continue
            
            # Предсказание (берём из первого анализатора)
            analyzers_results = result.get('analyzers', {})
            if not analyzers_results:
                continue
            
            # Берём первый успешный анализатор
            prediction = None
            for analyzer_name, analyzer_result in analyzers_results.items():
                if analyzer_result.get('success') and analyzer_result.get('prediction'):
                    prediction = analyzer_result['prediction']
                    break
            
            if prediction:
                y_true.append(gt)
                y_pred.append(prediction)
        
        return y_true, y_pred
