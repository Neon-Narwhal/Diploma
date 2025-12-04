"""
Интеграция с MLflow для трекинга экспериментов.
"""

from typing import Dict, Any, Optional
import json

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class MLflowTracker:
    """
    Обёртка для MLflow трекинга.
    Автоматически отключается если MLflow не установлен.
    """
    
    def __init__(self,
                 experiment_name: str,
                 run_name: Optional[str] = None,
                 tracking_uri: Optional[str] = None):
        """
        Args:
            experiment_name: Название эксперимента
            run_name: Название run
            tracking_uri: URI для MLflow tracking server
        """
        self.enabled = MLFLOW_AVAILABLE
        
        if not self.enabled:
            print("Warning: MLflow not available, tracking disabled")
            return
        
        self.experiment_name = experiment_name
        self.run_name = run_name
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
    
    def start_run(self, run_name: Optional[str] = None):
        """Начало run"""
        if not self.enabled:
            return
        
        mlflow.start_run(run_name=run_name or self.run_name)
    
    def end_run(self):
        """Окончание run"""
        if not self.enabled:
            return
        
        mlflow.end_run()
    
    def log_params(self, params: Dict[str, Any]):
        """Логирование параметров"""
        if not self.enabled:
            return
        
        # Flatten nested dicts
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Логирование метрик"""
        if not self.enabled:
            return
        
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_artifact(self, filepath: str):
        """Логирование артефакта"""
        if not self.enabled:
            return
        
        mlflow.log_artifact(filepath)
    
    def log_dict(self, data: Dict, filename: str):
        """Логирование словаря как JSON артефакт"""
        if not self.enabled:
            return
        
        mlflow.log_dict(data, filename)
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Рекурсивное выпрямление вложенного словаря"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                # Конвертируем в строку для MLflow
                items.append((new_key, str(v)))
        return dict(items)


class DummyTracker:
    """Заглушка для когда MLflow недоступен"""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def start_run(self, *args, **kwargs):
        pass
    
    def end_run(self):
        pass
    
    def log_params(self, *args, **kwargs):
        pass
    
    def log_metrics(self, *args, **kwargs):
        pass
    
    def log_artifact(self, *args, **kwargs):
        pass
    
    def log_dict(self, *args, **kwargs):
        pass
