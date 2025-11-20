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
        
        Args:
            metrics: список метрик для сравнения
            
        Returns:
            DataFrame с метриками для каждой модели
        """
        rows = []
        
        for model_name, model_results in self.results.items():
            row = {'model': model_name}
            
            # Извлекаем метрики
            for metric in metrics:
                # Проверяем разные источники метрик
                if 'test_metrics' in model_results:
                    row[metric] = model_results['test_metrics'].get(metric, np.nan)
                elif 'train_metrics' in model_results:
                    row[metric] = model_results['train_metrics'].get(metric, np.nan)
                elif 'cv_results' in model_results:
                    cv_mean = model_results['cv_results']['mean']
                    row[metric] = cv_mean.get(metric, np.nan)
                else:
                    row[metric] = np.nan
                
                # Добавляем CV std если есть
                if 'cv_results' in model_results:
                    cv_std = model_results['cv_results']['std']
                    row[f"{metric}_std"] = cv_std.get(metric, np.nan)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def find_best_model(self, metric: str = 'accuracy') -> str:
        """
        Поиск лучшей модели по метрике.
        
        Args:
            metric: метрика для сравнения
            
        Returns:
            Имя лучшей модели
        """
        best_model = None
        best_value = -np.inf
        
        for model_name, model_results in self.results.items():
            # Извлекаем значение метрики
            value = None
            
            if 'test_metrics' in model_results:
                value = model_results['test_metrics'].get(metric)
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
        """
        Ранжирование моделей по метрике.
        
        Args:
            metric: метрика для ранжирования
            
        Returns:
            Список (имя_модели, значение_метрики) отсортированный по убыванию
        """
        rankings = []
        
        for model_name, model_results in self.results.items():
            value = None
            
            if 'test_metrics' in model_results:
                value = model_results['test_metrics'].get(metric)
            elif 'cv_results' in model_results:
                value = model_results['cv_results']['mean'].get(metric)
            
            if value is not None:
                rankings.append((model_name, value))
        
        # Сортируем по убыванию
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
