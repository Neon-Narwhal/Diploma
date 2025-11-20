"""
Универсальный логгер для MLflow и JSON.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


class MLLogger:
    """
    Универсальный логгер, поддерживающий MLflow и JSON.
    """
    
    def __init__(
        self,
        use_mlflow: bool = True,
        use_json: bool = True,
        json_path: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        """
        Args:
            use_mlflow: использовать MLflow
            use_json: использовать JSON логирование
            json_path: путь для JSON файла
            experiment_name: имя эксперимента для MLflow
        """
        self.use_mlflow = use_mlflow
        self.use_json = use_json
        self.json_path = json_path or "ml/outputs/logs/experiment.json"
        self.experiment_name = experiment_name
        
        # Хранилище для JSON
        self.json_data = {
            'params': {},
            'metrics': {},
            'artifacts': {},
        }
        
        # Инициализация MLflow
        if self.use_mlflow:
            self._init_mlflow()
    
    def _init_mlflow(self):
        """Инициализация MLflow"""
        try:
            import mlflow
            
            if self.experiment_name:
                mlflow.set_experiment(self.experiment_name)
            
            mlflow.start_run()
            self.mlflow = mlflow
            
        except ImportError:
            print("MLflow not installed, disabling MLflow logging")
            self.use_mlflow = False
    
    def log_params(self, params: Dict[str, Any]):
        """
        Логирование параметров.
        
        Args:
            params: словарь параметров
        """
        # MLflow
        if self.use_mlflow:
            self.mlflow.log_params(params)
        
        # JSON
        if self.use_json:
            self.json_data['params'].update(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Логирование метрик.
        
        Args:
            metrics: словарь метрик
            step: номер шага (опционально)
        """
        # MLflow
        if self.use_mlflow:
            if step is not None:
                for key, value in metrics.items():
                    self.mlflow.log_metric(key, value, step=step)
            else:
                self.mlflow.log_metrics(metrics)
        
        # JSON
        if self.use_json:
            if step is not None:
                if 'metrics_history' not in self.json_data:
                    self.json_data['metrics_history'] = []
                self.json_data['metrics_history'].append({
                    'step': step,
                    **metrics,
                })
            else:
                self.json_data['metrics'].update(metrics)
    
    def log_artifact(self, name: str, artifact: Any):
        """
        Логирование артефакта.
        
        Args:
            name: имя артефакта
            artifact: данные артефакта
        """
        # JSON (сохраняем как массив если это numpy)
        if self.use_json:
            if isinstance(artifact, np.ndarray):
                self.json_data['artifacts'][name] = artifact.tolist()
            else:
                self.json_data['artifacts'][name] = artifact
    
    def log_model(self, model_path: str):
        """
        Логирование модели.
        
        Args:
            model_path: путь к сохраненной модели
        """
        if self.use_mlflow:
            self.mlflow.log_artifact(model_path)
        
        if self.use_json:
            self.json_data['model_path'] = model_path
    
    def save_json(self):
        """Сохранение JSON лога"""
        if not self.use_json:
            return
        
        output_path = Path(self.json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.json_data, f, indent=2)
    
    def end_run(self):
        """Завершение логирования"""
        if self.use_mlflow:
            self.mlflow.end_run()
        
        if self.use_json:
            self.save_json()
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.end_run()
