"""
Универсальное логирование для всех модулей.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class Logger:
    """
    Унифицированный логгер для всех модулей.
    Поддерживает логи в консоль, JSON файлы и структурированные логи.
    """
    
    def __init__(self,
                 name: str,
                 log_dir: Optional[str] = None,
                 console: bool = True,
                 json_file: bool = False):
        """
        Args:
            name: Имя логгера (обычно имя модуля)
            log_dir: Директория для сохранения логов
            console: Логировать в консоль
            json_file: Сохранять JSON логи
        """
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаём Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # JSON file handler
        self.json_file = None
        if json_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.json_file = self.log_dir / f"{name}_{timestamp}.jsonl"
        
        self.structured_logs = []
    
    def info(self, message: str, **kwargs):
        """Логирование INFO"""
        self.logger.info(message)
        self._log_structured("INFO", message, kwargs)
    
    def warning(self, message: str, **kwargs):
        """Логирование WARNING"""
        self.logger.warning(message)
        self._log_structured("WARNING", message, kwargs)
    
    def error(self, message: str, **kwargs):
        """Логирование ERROR"""
        self.logger.error(message)
        self._log_structured("ERROR", message, kwargs)
    
    def debug(self, message: str, **kwargs):
        """Логирование DEBUG"""
        self.logger.debug(message)
        self._log_structured("DEBUG", message, kwargs)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Логирование метрик"""
        message = f"Metrics: {metrics}"
        if step is not None:
            message = f"Step {step} - {message}"
        
        self.info(message)
        self._log_structured("METRICS", message, {"metrics": metrics, "step": step})
    
    def log_config(self, config: Dict[str, Any]):
        """Логирование конфигурации"""
        self.info(f"Config: {json.dumps(config, indent=2)}")
        self._log_structured("CONFIG", "Configuration loaded", {"config": config})
    
    def _log_structured(self, level: str, message: str, extra: Dict):
        """Сохранение структурированного лога"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
            **extra
        }
        
        self.structured_logs.append(log_entry)
        
        # Сохранение в JSON файл
        if self.json_file:
            with open(self.json_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def get_logs(self) -> list:
        """Получение всех структурированных логов"""
        return self.structured_logs
    
    def save_logs(self, filepath: Optional[str] = None):
        """Сохранение логов в файл"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.log_dir / f"{self.name}_{timestamp}_logs.json"
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.structured_logs, f, indent=2, ensure_ascii=False)
        
        self.info(f"Logs saved to: {filepath}")


class ExperimentLogger(Logger):
    """
    Расширенный логгер для экспериментов.
    Добавляет контекст эксперимента к логам.
    """
    
    def __init__(self,
                 experiment_name: str,
                 run_name: Optional[str] = None,
                 **kwargs):
        super().__init__(name=experiment_name, **kwargs)
        self.experiment_name = experiment_name
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now()
    
    def log_start(self, config: Dict[str, Any]):
        """Логирование начала эксперимента"""
        self.info(f"Starting experiment: {self.experiment_name}")
        self.info(f"Run name: {self.run_name}")
        self.log_config(config)
    
    def log_end(self, results: Dict[str, Any]):
        """Логирование окончания эксперимента"""
        duration = (datetime.now() - self.start_time).total_seconds()
        self.info(f"Experiment completed in {duration:.2f}s")
        self.log_metrics(results)
    
    def log_stage(self, stage_name: str, **kwargs):
        """Логирование стадии эксперимента"""
        self.info(f"Stage: {stage_name}")
        self._log_structured("STAGE", stage_name, kwargs)
