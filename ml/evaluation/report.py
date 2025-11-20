"""
Генерация отчетов.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


class ReportGenerator:
    """
    Генерация отчетов о результатах экспериментов.
    """
    
    def __init__(self, output_dir: str = "ml/outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_json_report(
        self,
        results: Dict[str, Any],
        filename: str = "report.json",
    ) -> str:
        """
        Генерация JSON отчета.
        
        Args:
            results: результаты экспериментов
            filename: имя файла
            
        Returns:
            Путь к сохраненному файлу
        """
        output_path = self.output_dir / filename
        
        # Конвертируем numpy массивы в списки для JSON
        results_serializable = self._make_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        return str(output_path)
    
    def generate_comparison_report(
        self,
        comparison_df: pd.DataFrame,
        filename: str = "comparison.csv",
    ) -> str:
        """
        Генерация CSV отчета сравнения моделей.
        
        Args:
            comparison_df: DataFrame с результатами сравнения
            filename: имя файла
            
        Returns:
            Путь к сохраненному файлу
        """
        output_path = self.output_dir / filename
        comparison_df.to_csv(output_path, index=False)
        return str(output_path)
    
    def _make_serializable(self, obj):
        """Конвертация объектов в JSON-serializable формат"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
