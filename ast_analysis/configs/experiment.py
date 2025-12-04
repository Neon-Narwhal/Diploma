"""
Конфигурация AST эксперимента.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from shared.configs.base import BaseConfig


@dataclass
class ASTExperimentConfig(BaseConfig):
    """
    Конфигурация эксперимента AST анализа.
    """
    name: str
    description: str = ""
    
    # Данные
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Анализаторы
    analyzers: List[Dict[str, Any]] = field(default_factory=list)
    
    # Обработка
    processing: Dict[str, Any] = field(default_factory=dict)
    
    # Оценка
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        'success_rate',
        'parsing_success_rate',
        'avg_processing_time',
        'feature_coverage'
    ])
    
    # Вывод
    save_results: bool = True
    output_path: Optional[str] = None
    generate_report: bool = True
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Получение конфига обработки"""
        defaults = {
            'batch_size': 100,
            'parallel': True,
            'n_workers': 4,
            'timeout_per_sample': 5,
            'skip_on_error': True
        }
        defaults.update(self.processing)
        return defaults
    
    def validate(self) -> bool:
        """Валидация конфигурации"""
        if not self.name:
            raise ValueError("Experiment name is required")
        
        if not self.data:
            raise ValueError("Data config is required")
        
        if not self.analyzers:
            raise ValueError("At least one analyzer must be specified")
        
        # Проверка data config
        required_data_fields = ['train_path', 'val_path', 'test_path']
        for field in required_data_fields:
            if field not in self.data:
                raise ValueError(f"Data config missing required field: {field}")
        
        return True
