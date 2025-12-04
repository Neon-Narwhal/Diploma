"""
Раннер для запуска AST экспериментов с классификацией.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from ast_analysis.configs.experiment import ASTExperimentConfig
from ast_analysis.processing.pipeline import ASTPipeline
from shared.data_loader import DataLoader
from shared.evaluation import Evaluator, ReportGenerator
from shared.utils import ExperimentLogger


class ASTExperimentRunner:
    """
    Раннер для AST экспериментов с предсказанием сложности.
    """
    
    def __init__(self, config: ASTExperimentConfig):
        """
        Args:
            config: Конфигурация эксперимента
        """
        self.config = config
        self.config.validate()
        
        # Логгер
        self.logger = ExperimentLogger(
            experiment_name=config.name,
            log_dir="logs/ast_analysis",
            console=True,
            json_file=True
        )
        
        # Загрузчик данных
        self.data_loader = DataLoader.from_config(config)
        
        # Пайплайн
        self.pipeline = ASTPipeline(config.to_dict())
        
        # Evaluator
        self.evaluator = Evaluator(config.evaluation_metrics)
        
        # Report generator
        self.report_generator = ReportGenerator(output_dir="outputs/ast_analysis/reports")
        
        self.results = {}
    
    def run(self) -> Dict[str, Any]:
        """
        Запуск эксперимента.
        
        Returns:
            Словарь с результатами
        """
        self.logger.log_start(self.config.to_dict())
        
        try:
            # 1. Загрузка данных
            self.logger.log_stage("Loading data")
            dataset = self.data_loader.load()
            self.logger.info(dataset.summary())
            
            # 2. Обработка через пайплайн
            self.logger.log_stage("Running AST analysis with predictions")
            analysis_results = self.pipeline.process(dataset)
            
            # 3. Оценка результатов (с ground truth)
            self.logger.log_stage("Evaluating predictions")
            evaluation_results = self._evaluate_results(analysis_results)
            
            # 4. Сохранение результатов
            if self.config.save_results:
                self.logger.log_stage("Saving results")
                self._save_results(analysis_results, evaluation_results)
            
            # 5. Генерация отчёта
            if self.config.generate_report:
                self.logger.log_stage("Generating report")
                self._generate_report(evaluation_results)
            
            self.results = {
                'analysis_results': analysis_results,
                'evaluation': evaluation_results,
                'dataset_info': dataset.size()
            }
            
            # Финальная сводка
            self._print_summary(evaluation_results)
            
            self.logger.log_end(self._flatten_metrics(evaluation_results))
            
            return self.results
        
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            raise
    
    def _evaluate_results(self, analysis_results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """Оценка результатов по всем сплитам"""
        evaluation = {}
        
        for split_name in ['train', 'val', 'test']:
            split_results = analysis_results.get(split_name, [])
            
            if not split_results:
                continue
            
            # Вычисляем метрики с per-class
            split_eval = self.evaluator.evaluate(split_results, include_per_class=True)
            
            evaluation[split_name] = split_eval
            
            # Логируем метрики
            metrics = split_eval.get('metrics', {})
            self.logger.info(f"\n{split_name.upper()} Metrics:")
            for name, value in metrics.items():
                self.logger.info(f"  {name}: {value:.4f}")
        
        return evaluation
    
    def _save_results(self, 
                     analysis_results: Dict[str, List[Dict]], 
                     evaluation_results: Dict[str, Dict]):
        """Сохранение результатов"""
        output_path = self.config.output_path
        if not output_path:
            output_path = f"outputs/ast_analysis/{self.config.name}_results.jsonl"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохранение результатов анализа в JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for split_name, split_results in analysis_results.items():
                for result in split_results:
                    result['split'] = split_name
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Results saved to: {output_path}")
        
        # Сохранение метрик отдельно
        metrics_path = output_path.parent / f"{self.config.name}_metrics.json"
        metrics_data = {
            split: {
                'metrics': eval_result.get('metrics', {}),
                'per_class_metrics': eval_result.get('per_class_metrics', {})
            }
            for split, eval_result in evaluation_results.items()
        }
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False, default=self._json_serializer)
        
        self.logger.info(f"Metrics saved to: {metrics_path}")
    
    def _generate_report(self, evaluation_results: Dict[str, Dict]):
        """Генерация отчёта через ReportGenerator"""
        # Подготовка данных для отчёта
        metrics = {
            split: eval_result.get('metrics', {})
            for split, eval_result in evaluation_results.items()
        }
        
        per_class_metrics = {
            split: eval_result.get('per_class_metrics', {})
            for split, eval_result in evaluation_results.items()
        }
        
        confusion_matrices = {
            split: eval_result.get('confusion_matrix')
            for split, eval_result in evaluation_results.items()
        }
        
        # Генерация отчёта
        report_content = self.report_generator.generate_classification_report(
            experiment_name=self.config.name,
            description=self.config.description,
            config=self.config.to_dict(),
            metrics=metrics,
            per_class_metrics=per_class_metrics,
            confusion_matrices=confusion_matrices
        )
        
        # Сохранение
        report_path = self.report_generator.save_report(
            content=report_content,
            filename=f"{self.config.name}_report.md"
        )
        
        self.logger.info(f"Report saved to: {report_path}")
    
    def _print_summary(self, evaluation_results: Dict[str, Dict]):
        """Печать финальной сводки"""
        print("\n" + "=" * 60)
        print("CLASSIFICATION SUMMARY")
        print("=" * 60)
        
        for split_name, eval_result in evaluation_results.items():
            metrics = eval_result.get('metrics', {})
            print(f"\n{split_name.upper()}:")
            
            # Основные метрики
            for metric in ['f1_macro', 'recall_macro', 'accuracy']:
                if metric in metrics:
                    print(f"  {metric}: {metrics[metric]:.4f}")
    
    def _flatten_metrics(self, evaluation_results: Dict[str, Dict]) -> Dict[str, float]:
        """Преобразование метрик в плоский словарь"""
        flat_metrics = {}
        for split_name, eval_result in evaluation_results.items():
            metrics = eval_result.get('metrics', {})
            for metric_name, metric_value in metrics.items():
                flat_metrics[f"{split_name}_{metric_name}"] = float(metric_value)
        return flat_metrics
    
    def _json_serializer(self, obj):
        """Сериализация для JSON"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return str(obj)
