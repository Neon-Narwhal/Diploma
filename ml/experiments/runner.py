"""
Главный runner для запуска экспериментов.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

from ml.configs.experiment import ExperimentConfig
from ml.core.model_config import ModelConfig
from ml.training.pipeline import MLPipeline
from ml.evaluation.comparison import ModelComparison
from ml.evaluation.visualization import ModelVisualizer
from ml.evaluation.report import ReportGenerator
from ml.utils.logger import MLLogger


class ExperimentRunner:
    """
    Запуск экспериментов из конфига.
    Поддерживает:
    - Обучение нескольких моделей
    - Сравнение результатов
    - Генерация отчетов
    - Логирование в MLflow/JSON
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Args:
            config: конфигурация эксперимента
        """
        self.config = config
        self.results = {}
        self.models = {}
    
    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Запуск всех моделей из конфига.
        
        Args:
            X_train: признаки для обучения
            y_train: таргет для обучения
            X_test: признаки для теста
            y_test: таргет для теста
            
        Returns:
            Результаты всех моделей
        """
        print(f"\n{'='*80}")
        print(f"ЭКСПЕРИМЕНТ: {self.config.name}")
        print(f"{self.config.description}")
        print(f"{'='*80}\n")
        
        model_configs = self.config.get_model_configs()
        
        for i, model_config in enumerate(model_configs, 1):
            print(f"\n[{i}/{len(model_configs)}] Обучение модели: {model_config.name}")
            print(f"{'-'*80}")
            
            # Запуск pipeline для модели
            result = self._train_single_model(
                model_config,
                X_train,
                y_train,
                X_test,
                y_test,
            )
            
            self.results[model_config.name] = result
        
        # Сравнение моделей
        if len(model_configs) > 1:
            self._compare_models()
        
        # Генерация отчета
        if self.config.generate_report:
            self._generate_report()
        
        return self.results
    
    def _train_single_model(
        self,
        model_config: ModelConfig,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray],
        y_test: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Обучение одной модели"""
        # Создаем логгер
        logger = MLLogger(
            use_mlflow=True,
            use_json=True,
            json_path=f"ml/outputs/logs/{model_config.name}.json",
            experiment_name=self.config.mlflow_experiment or self.config.name,
        )
        
        # Создаем pipeline
        pipeline = MLPipeline(
            model_config=model_config,
            feature_config=self.config.features,
            cv_config=self.config.cross_validation,
            optimization_config=self.config.optimization,
            logger=logger,
        )
        
        # Запуск
        with logger:
            result = pipeline.run(X_train, y_train, X_test, y_test)
            
            # Сохранение модели
            if self.config.save_models:
                model_path = f"ml/outputs/models/{model_config.name}.pkl"
                pipeline.save_model(model_path)
                print(f"✓ Модель сохранена: {model_path}")
            
            # Сохранение модели в памяти
            self.models[model_config.name] = pipeline.model
        
        return result
    
    def _compare_models(self):
        """Сравнение результатов моделей"""
        print(f"\n{'='*80}")
        print("СРАВНЕНИЕ МОДЕЛЕЙ")
        print(f"{'='*80}\n")
        
        comparison = ModelComparison(self.results)
        
        # Таблица метрик
        metrics = self.config.evaluation_metrics
        comparison_df = comparison.create_comparison_table(metrics)
        
        print(comparison_df.to_string(index=False))
        print()
        
        # Лучшая модель
        best_model = comparison.find_best_model(metric='accuracy')
        print(f"Лучшая модель по accuracy: {best_model}")
        
        # Ранжирование
        rankings = comparison.rank_models(metric='f1_macro')
        print(f"\nРанжирование по f1_macro:")
        for rank, (name, score) in enumerate(rankings, 1):
            print(f"  {rank}. {name}: {score:.4f}")
        
        # Сохранение таблицы
        report_gen = ReportGenerator()
        report_path = report_gen.generate_comparison_report(
            comparison_df,
            filename=f"{self.config.name}_comparison.csv"
        )
        print(f"\n✓ Таблица сравнения сохранена: {report_path}")
        
        # Визуализация
        self._visualize_comparison(comparison_df)
    
    def _visualize_comparison(self, comparison_df):
        """Визуализация сравнения"""
        visualizer = ModelVisualizer()
        
        # График сравнения метрик
        fig = visualizer.plot_metrics_comparison(
            comparison_df,
            metrics=['accuracy', 'f1_macro'],
            save_path=f"ml/outputs/reports/{self.config.name}_comparison.png"
        )
        print(f"✓ График сравнения сохранен")
    
    def _generate_report(self):
        """Генерация финального отчета"""
        print(f"\n{'='*80}")
        print("ГЕНЕРАЦИЯ ОТЧЕТА")
        print(f"{'='*80}\n")
        
        report_gen = ReportGenerator()
        
        # JSON отчет со всеми результатами
        report_path = report_gen.generate_json_report(
            self.results,
            filename=f"{self.config.name}_results.json"
        )
        print(f"✓ JSON отчет сохранен: {report_path}")
