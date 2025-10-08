"""
Вычисление метрик качества предсказаний (precision, recall, f1, accuracy).
"""

from typing import Dict, List, Tuple
from collections import Counter
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    classification_report
)

from config import ComplexityClass


class MetricsCalculator:
    """Калькулятор метрик качества классификации"""
    
    def __init__(self, class_labels: List[str] = None):
        """
        Args:
            class_labels: Список меток классов в нужном порядке
        """
        if class_labels is None:
            class_labels = [c.value for c in ComplexityClass]
        self.class_labels = class_labels
    
    def calculate_metrics(
        self, 
        y_true: List[str], 
        y_pred: List[str]
    ) -> Dict[str, any]:
        """
        Вычисляет все метрики классификации
        
        Args:
            y_true: Истинные метки классов
            y_pred: Предсказанные метки классов
            
        Returns:
            Словарь с метриками
        """
        # Базовые метрики
        accuracy = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1 для каждого класса
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, 
            y_pred, 
            labels=self.class_labels,
            zero_division=0
        )
        
        # Взвешенные и макро-усредненные метрики
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, 
            y_pred,
            average='weighted',
            zero_division=0
        )
        
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, 
            y_pred,
            average='macro',
            zero_division=0
        )
        
        # Матрица ошибок
        cm = confusion_matrix(y_true, y_pred, labels=self.class_labels)
        
        # Формируем результат
        result = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': {}
        }
        
        # Метрики по каждому классу
        for i, label in enumerate(self.class_labels):
            result['per_class_metrics'][label] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
        
        # Отчет классификации
        result['classification_report'] = classification_report(
            y_true, 
            y_pred, 
            labels=self.class_labels,
            zero_division=0,
            output_dict=True
        )
        
        return result
    
    def calculate_class_distribution(
        self,
        y_true: List[str],
        y_pred: List[str]
    ) -> Dict[str, Dict]:
        """
        Вычисляет распределение классов в истинных и предсказанных метках
        
        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки
            
        Returns:
            Словарь со статистикой по каждому классу
        """
        true_counter = Counter(y_true)
        pred_counter = Counter(y_pred)
        
        total_samples = len(y_true)
        
        distribution = {}
        for complexity_class in self.class_labels:
            true_count = true_counter.get(complexity_class, 0)
            pred_count = pred_counter.get(complexity_class, 0)
            
            # Подсчет правильных предсказаний для этого класса
            correct_count = sum(
                1 for t, p in zip(y_true, y_pred)
                if t == complexity_class and p == complexity_class
            )
            
            distribution[complexity_class] = {
                'true_count': true_count,
                'predicted_count': pred_count,
                'correct_predictions': correct_count,
                'true_percentage': (true_count / total_samples * 100) if total_samples > 0 else 0,
                'predicted_percentage': (pred_count / total_samples * 100) if total_samples > 0 else 0,
                'accuracy_for_class': (correct_count / true_count * 100) if true_count > 0 else 0
            }
        
        return distribution
    
    def calculate_error_distribution(
        self, 
        y_true: List[str], 
        y_pred: List[str]
    ) -> Dict[str, int]:
        """
        Подсчитывает распределение ошибок по типам
        
        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки
            
        Returns:
            Словарь с количеством каждого типа ошибок
        """
        errors = {}
        for true, pred in zip(y_true, y_pred):
            if true != pred:
                error_type = f"{true}_as_{pred}"
                errors[error_type] = errors.get(error_type, 0) + 1
        
        return errors
    
    def calculate_complexity_distance(
        self, 
        y_true: List[str], 
        y_pred: List[str]
    ) -> Dict[str, float]:
        """
        Вычисляет среднее расстояние между предсказанной и истинной сложностью
        (насколько далеко ошиблись)
        
        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки
            
        Returns:
            Словарь со статистикой расстояний
        """
        complexity_order = {c.value: i for i, c in enumerate(ComplexityClass)}
        
        distances = []
        for true, pred in zip(y_true, y_pred):
            if true in complexity_order and pred in complexity_order:
                distance = abs(complexity_order[true] - complexity_order[pred])
                distances.append(distance)
        
        if not distances:
            return {'mean_distance': 0, 'max_distance': 0}
        
        return {
            'mean_distance': np.mean(distances),
            'median_distance': np.median(distances),
            'max_distance': np.max(distances),
            'std_distance': np.std(distances)
        }
