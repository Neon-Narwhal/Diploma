"""
Генерация отчетов для всех модулей.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np


class ReportGenerator:
    """
    Генератор отчетов для AST/CFG/Runtime/ML модулей.
    """
    
    def __init__(self, output_dir: str = "outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_classification_report(
        self,
        experiment_name: str,
        description: str,
        config: Dict[str, Any],
        metrics: Dict[str, Dict[str, float]],
        per_class_metrics: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
        confusion_matrices: Optional[Dict[str, np.ndarray]] = None,
    ) -> str:
        """
        Генерация полного отчета о классификации.
        
        Args:
            experiment_name: Название эксперимента
            description: Описание
            config: Конфигурация
            metrics: Метрики по сплитам {split: {metric: value}}
            per_class_metrics: Per-class метрики {split: {class: {metric: value}}}
            confusion_matrices: Confusion matrices {split: matrix}
        
        Returns:
            Markdown текст отчета
        """
        lines = []
        
        # Заголовок
        lines.append(f"# {experiment_name}")
        lines.append("")
        lines.append(f"**Description:** {description}")
        lines.append(f"**Date:** {self._get_timestamp()}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Конфигурация
        lines.append("## Configuration")
        lines.append("")
        lines.extend(self._format_config(config))
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Таблица метрик
        lines.append("## Classification Metrics")
        lines.append("")
        lines.append(self._format_metrics_table(metrics))
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Per-class метрики
        if per_class_metrics:
            lines.append("## Per-Class Performance")
            lines.append("")
            for split_name, class_metrics in per_class_metrics.items():
                lines.append(f"### {split_name.capitalize()} Set")
                lines.append("")
                lines.append(self._format_per_class_table(class_metrics))
                lines.append("")
        
        # Confusion matrices
        if confusion_matrices:
            lines.append("## Confusion Matrices")
            lines.append("")
            for split_name, cm in confusion_matrices.items():
                lines.append(f"### {split_name.capitalize()} Set")
                lines.append("")
                lines.append("```")
                lines.append(str(cm))
                lines.append("```")
                lines.append("")
        
        return "\n".join(lines)
    
    def _format_metrics_table(self, metrics: Dict[str, Dict[str, float]]) -> str:
        """Форматирование таблицы метрик"""
        if not metrics:
            return "*No metrics available*"
        
        # Получаем все метрики
        all_metric_names = set()
        for split_metrics in metrics.values():
            all_metric_names.update(split_metrics.keys())
        
        metric_names = sorted(all_metric_names)
        
        # Заголовок таблицы
        header = "| Split | " + " | ".join(metric_names) + " |"
        separator = "|-------|" + "|".join(["-------"] * len(metric_names)) + "|"
        
        lines = [header, separator]
        
        # Строки данных
        for split_name in ['train', 'val', 'test']:
            if split_name not in metrics:
                continue
            
            split_metrics = metrics[split_name]
            values = [f"{split_metrics.get(m, 0.0):.4f}" for m in metric_names]
            row = f"| {split_name.capitalize()} | " + " | ".join(values) + " |"
            lines.append(row)
        
        return "\n".join(lines)
    
    def _format_per_class_table(self, class_metrics: Dict[str, Dict[str, float]]) -> str:
        """Форматирование per-class метрик"""
        if not class_metrics:
            return "*No per-class metrics available*"
        
        # Заголовок
        header = "| Class | F1 | Precision | Recall | Support |"
        separator = "|-------|-----|-----------|--------|---------|"
        
        lines = [header, separator]
        
        # Строки для каждого класса
        for class_name, metrics in sorted(class_metrics.items()):
            f1 = metrics.get('f1', 0.0)
            precision = metrics.get('precision', 0.0)
            recall = metrics.get('recall', 0.0)
            support = metrics.get('support', 0)
            
            row = f"| {class_name} | {f1:.4f} | {precision:.4f} | {recall:.4f} | {support} |"
            lines.append(row)
        
        return "\n".join(lines)
    
    def _format_config(self, config: Dict[str, Any]) -> List[str]:
        """Форматирование конфигурации"""
        lines = []
        
        # Основные параметры
        if 'analyzers' in config:
            analyzers = [a.get('name', 'unknown') for a in config['analyzers']]
            lines.append(f"- **Analyzers:** {', '.join(analyzers)}")
        
        if 'processing' in config:
            proc = config['processing']
            lines.append(f"- **Processing:** parallel={proc.get('parallel')}, "
                        f"workers={proc.get('n_workers')}, "
                        f"batch_size={proc.get('batch_size')}")
        
        return lines
    
    def _get_timestamp(self) -> str:
        """Получение текущего времени"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def save_report(self, content: str, filename: str) -> Path:
        """Сохранение отчета в файл"""
        report_path = self.output_dir / filename
        report_path.write_text(content, encoding='utf-8')
        return report_path
    
    def generate_json_report(
        self,
        results: Dict[str, Any],
        filename: str = "report.json",
    ) -> Path:
        """Генерация JSON отчета"""
        output_path = self.output_dir / filename
        
        # Конвертируем numpy в списки
        results_serializable = self._make_serializable(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def _make_serializable(self, obj):
        """Конвертация в JSON-serializable формат"""
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