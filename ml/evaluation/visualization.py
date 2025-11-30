"""
Визуализация результатов.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
from pathlib import Path


class ModelVisualizer:
    """
    Визуализация результатов моделей.
    """
    
    def __init__(self, figsize: tuple = (10, 8)):
        self.figsize = figsize
        sns.set_style('whitegrid')
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ):
        """
        Визуализация confusion matrix.
        
        Args:
            cm: confusion matrix
            class_names: имена классов
            save_path: путь для сохранения
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(
        self,
        importance: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_n: int = 20,
        save_path: Optional[str] = None,
    ):
        """
        Визуализация важности признаков.
        
        Args:
            importance: массив важности
            feature_names: имена признаков
            top_n: количество top признаков для отображения
            save_path: путь для сохранения
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        # Сортируем по важности
        indices = np.argsort(importance)[-top_n:]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.barh(
            range(len(indices)),
            importance[indices],
            align='center',
        )
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_metrics_comparison(
        self,
        comparison_df,
        metrics: List[str],
        save_path: Optional[str] = None,
    ):
        """
        Визуализация сравнения метрик нескольких моделей.
        
        Args:
            comparison_df: DataFrame с результатами сравнения
            metrics: список метрик для визуализации
            save_path: путь для сохранения
        """
        n_metrics = len(metrics)
        if n_metrics == 0:
            print("Warning: No metrics to plot")
            return None
        fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 5, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            # Проверяем наличие std для error bars
            has_std = f"{metric}_std" in comparison_df.columns
            
            if has_std:
                ax.bar(
                    comparison_df['model'],
                    comparison_df[metric],
                    yerr=comparison_df[f"{metric}_std"],
                    capsize=5,
                )
            else:
                ax.bar(comparison_df['model'], comparison_df[metric])
            
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_xlabel('Model')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_cv_boxplots(
        self,
        cv_results: dict,
        metric: str = 'accuracy',
        save_path: Optional[str] = None,
    ):
        """
        Boxplot результатов cross-validation для нескольких моделей.
        
        Args:
            cv_results: словарь {модель: cv_results}
            metric: метрика для визуализации
            save_path: путь для сохранения
        """
        data = []
        labels = []
        
        for model_name, results in cv_results.items():
            fold_values = [fold[metric] for fold in results['all_folds']]
            data.append(fold_values)
            labels.append(model_name)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.boxplot(data, labels=labels)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_xlabel('Model')
        ax.set_title(f'Cross-Validation {metric.title()}')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
