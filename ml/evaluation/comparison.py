"""
Сравнение результатов нескольких моделей.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from scipy import stats


class ModelComparison:
    """
    Сравнение результатов нескольких моделей.
    """
    
    def __init__(self, results: Dict[str, Dict[str, Any]]):
        """
        Args:
            results: словарь {имя_модели: результаты}
        """
        self.results = results
    
    def create_comparison_table(self, metrics: List[str]) -> pd.DataFrame:
        """
        Создание таблицы сравнения метрик.
        """
        rows = []
        
        for model_name, model_results in self.results.items():
            row = {'model': model_name}
            
            # Определяем источник метрик
            # Приоритет: test_metrics -> cv_results -> train_metrics
            source = None
            prefix = ""
            
            if 'test_metrics' in model_results:
                source = model_results['test_metrics']
                prefix = "test_"
            elif 'cv_results' in model_results:
                source = model_results['cv_results']['mean']
                prefix = "cv_"  # Если метрики в CV имеют префикс
            elif 'train_metrics' in model_results:
                source = model_results['train_metrics']
                prefix = "train_"
            
            if source:
                for metric in metrics:
                    # Пробуем найти метрику с префиксом или без
                    val = source.get(f"{prefix}{metric}") or source.get(metric)
                    row[metric] = val if val is not None else np.nan
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df

    
    def find_best_model(self, metric: str = 'accuracy') -> str:
        """
        Поиск лучшей модели по метрике.
        """
        best_model = None
        best_value = -np.inf
        
        for model_name, model_results in self.results.items():
            value = None
            
            if 'test_metrics' in model_results:
                # Ищем test_metric или metric
                value = model_results['test_metrics'].get(f"test_{metric}") or \
                        model_results['test_metrics'].get(metric)
            elif 'cv_results' in model_results:
                value = model_results['cv_results']['mean'].get(metric)
            
            if value is not None and value > best_value:
                best_value = value
                best_model = model_name
        
        return best_model

    
    def statistical_test(
        self,
        model1: str,
        model2: str,
        metric: str = 'accuracy',
    ) -> Dict[str, float]:
        """
        Статистический тест для сравнения двух моделей.
        Использует t-test на результатах CV.
        
        Args:
            model1: имя первой модели
            model2: имя второй модели
            metric: метрика для сравнения
            
        Returns:
            Результаты t-test
        """
        # Получаем результаты CV для обеих моделей
        results1 = self.results[model1].get('cv_results', {}).get('all_folds', [])
        results2 = self.results[model2].get('cv_results', {}).get('all_folds', [])
        
        if not results1 or not results2:
            raise ValueError("Both models must have CV results for statistical test")
        
        # Извлекаем значения метрики по фолдам
        values1 = [fold[metric] for fold in results1]
        values2 = [fold[metric] for fold in results2]
        
        # T-test
        statistic, p_value = stats.ttest_rel(values1, values2)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_diff': np.mean(values1) - np.mean(values2),
        }
    
    def rank_models(self, metric: str = 'accuracy') -> List[tuple]:
        """Ранжирование моделей"""
        rankings = []
        
        for model_name, model_results in self.results.items():
            value = None
            if 'test_metrics' in model_results:
                value = model_results['test_metrics'].get(f"test_{metric}") or \
                        model_results['test_metrics'].get(metric)
            elif 'cv_results' in model_results:
                value = model_results['cv_results']['mean'].get(metric)
            
            if value is not None:
                rankings.append((model_name, value))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

