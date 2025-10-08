# complexity_analyzer/dataset_processor.py (ДЛЯ BASELINE)
"""
Обработка JSONL датасета и запуск анализа.
"""

import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
import logging
import numpy as np

from config import TOOLS_REGISTRY, ComplexityClass, ANALYZER_MODE
from analyzers_baseline import get_baseline_analyzer 
from analyzers_enhanced import get_enhanced_analyzer
from metrics_calculator import MetricsCalculator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Кастомный JSON encoder для NumPy типов"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


class DatasetProcessor:
    """Процессор для обработки датасета"""
    
    def __init__(self, tools_to_use: Optional[List[str]] = None):
        # ИСПРАВЛЕНИЕ: сначала определяем tools_to_use
        if tools_to_use is None:
            # Берем все enabled инструменты из TOOLS_REGISTRY
            tools_to_use = [
                name for name, config in TOOLS_REGISTRY.items() 
                if config.enabled
            ]
        
        self.tools = tools_to_use
        
        # Загружаем анализаторы в зависимости от режима
        self.analyzers = {}
        for tool in tools_to_use:  # ← Теперь tools_to_use точно не None
            if ANALYZER_MODE == 'baseline':
                analyzer_name = f'{tool}_baseline'
                analyzer = get_baseline_analyzer(analyzer_name)
            else:  # enhanced
                analyzer_name = f'{tool}_enhanced'
                analyzer = get_enhanced_analyzer(analyzer_name)
            
            if analyzer is None:
                logger.error(f"❌ Failed to load analyzer: {analyzer_name}")
            else:
                logger.info(f"✅ Loaded analyzer: {analyzer.name} (mode: {ANALYZER_MODE})")
                self.analyzers[tool] = analyzer
        
        if not self.analyzers:
            raise ValueError(
                f"No analyzers loaded! Check ANALYZER_MODE ({ANALYZER_MODE}) "
                f"and TOOLS_TO_RUN in config.py"
            )
        
        self.metrics_calculator = MetricsCalculator()
    
    def load_jsonl(self, filepath: Path) -> List[Dict[str, Any]]:
        """Загружает данные из JSONL файла"""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def process_sample(
        self, 
        sample: Dict[str, Any], 
        tool_name: str
    ) -> tuple[Dict[str, Any], float]:
        """Обрабатывает один образец кода"""
        analyzer = self.analyzers.get(tool_name)
        
        if not analyzer:
            logger.error(f"Analyzer not found: {tool_name}")
            return {
                'error': f'Analyzer {tool_name} not found',
                'predicted_complexity': None,
                'true_complexity': sample.get('complexity', 'unknown')
            }, 0.0
        
        try:
            start_time = time.perf_counter()
            
            # Анализ кода
            metrics = analyzer.analyze(sample['src'])
            
            # Проверка на ошибки
            if 'error' in metrics and metrics['error']:
                logger.debug(f"Analysis error for {tool_name}: {metrics['error']}")
            
            # Маппинг в класс сложности
            predicted_complexity = analyzer.map_to_complexity(metrics)
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            return {
                'true_complexity': sample['complexity'],
                'predicted_complexity': predicted_complexity.value,
                'metrics': metrics,
                'problem': sample.get('problem', 'unknown'),
                'tags': sample.get('tags', []),
                'tool': tool_name,
                'execution_time': execution_time
            }, execution_time
            
        except Exception as e:
            logger.error(f"❌ Error processing sample with {tool_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'predicted_complexity': None,
                'true_complexity': sample['complexity'],
                'execution_time': 0.0
            }, 0.0
    
    def _convert_to_native_types(self, obj):
        """Рекурсивно конвертирует numpy типы в нативные Python типы"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_native_types(item) for item in obj)
        else:
            return obj
    
    def process_dataset(
        self, 
        filepath: Path,
        output_dir: Path,
        max_samples: Optional[int] = None
    ) -> Dict[str, Dict]:
        """Обрабатывает весь датасет всеми инструментами"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading dataset from {filepath}")
        dataset = self.load_jsonl(filepath)
        
        if max_samples:
            dataset = dataset[:max_samples]
        
        logger.info(f"Loaded {len(dataset)} samples")
        
        all_results = {}
        
        for tool_name in self.tools:
            logger.info(f"Processing with {tool_name}...")
            
            tool_start_time = time.perf_counter()
            
            tool_results = []
            y_true = []
            y_pred = []
            execution_times = []
            
            for sample in tqdm(dataset, desc=f"Analyzing with {tool_name}"):
                result, exec_time = self.process_sample(sample, tool_name)
                tool_results.append(result)
                execution_times.append(exec_time)
                
                if result['predicted_complexity'] is not None:
                    y_true.append(result['true_complexity'])
                    y_pred.append(result['predicted_complexity'])
            
            tool_end_time = time.perf_counter()
            total_execution_time = tool_end_time - tool_start_time
            
            # Вычисляем статистику по времени выполнения
            valid_times = [t for t in execution_times if t > 0]
            time_statistics = self._calculate_time_statistics(
                execution_times=valid_times,
                total_time=total_execution_time,
                total_samples=len(dataset)
            )
            
            # Вычисляем метрики качества
            logger.info(f"Calculating metrics for {tool_name}...")
            
            if len(y_true) > 0 and len(y_pred) > 0:
                metrics = self.metrics_calculator.calculate_metrics(y_true, y_pred)
                error_dist = self.metrics_calculator.calculate_error_distribution(y_true, y_pred)
                complexity_dist = self.metrics_calculator.calculate_complexity_distance(y_true, y_pred)
                class_distribution = self.metrics_calculator.calculate_class_distribution(y_true, y_pred)
            else:
                logger.error(f"❌ No valid predictions for {tool_name}!")
                metrics = {'accuracy': 0, 'f1_weighted': 0, 'precision_weighted': 0, 'recall_weighted': 0}
                error_dist = {}
                complexity_dist = {'mean_distance': 0}
                class_distribution = {}
            
            results_data = {
                'predictions': tool_results,
                'metrics': self._convert_to_native_types(metrics),
                'error_distribution': error_dist,
                'complexity_distance': self._convert_to_native_types(complexity_dist),
                'class_distribution': self._convert_to_native_types(class_distribution),
                'time_statistics': self._convert_to_native_types(time_statistics),
                'total_samples': len(dataset),
                'valid_predictions': len(y_pred)
            }
            
            all_results[tool_name] = results_data
            
            output_file = output_dir / f"{tool_name}_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            logger.info(f"Results saved to {output_file}")
            if len(y_true) > 0:
                logger.info(
                    f"Accuracy: {metrics['accuracy']:.3f}, "
                    f"F1: {metrics['f1_weighted']:.3f}, "
                    f"Valid predictions: {len(y_pred)}/{len(dataset)}"
                )
            else:
                logger.warning(f"No valid predictions!")
        
        # Сохраняем сводку
        summary_file = output_dir / "summary.json"
        summary = {
            tool: {
                'accuracy': results['metrics'].get('accuracy', 0),
                'f1_weighted': results['metrics'].get('f1_weighted', 0),
                'precision_weighted': results['metrics'].get('precision_weighted', 0),
                'recall_weighted': results['metrics'].get('recall_weighted', 0),
                'mean_complexity_distance': results['complexity_distance'].get('mean_distance', 0),
                'total_execution_time': results['time_statistics']['total_execution_time'],
                'mean_time_per_sample': results['time_statistics']['mean_time_per_sample'],
                'median_time_per_sample': results['time_statistics']['median_time_per_sample'],
                'samples_per_second': results['time_statistics']['samples_per_second'],
                'valid_predictions': results['valid_predictions'],
                'total_samples': results['total_samples']
            }
            for tool, results in all_results.items()
        }
        
        summary = self._convert_to_native_types(summary)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        logger.info(f"Summary saved to {summary_file}")
        
        return all_results
    
    def _calculate_time_statistics(
        self,
        execution_times: List[float],
        total_time: float,
        total_samples: int
    ) -> Dict[str, float]:
        """Вычисляет статистику по времени выполнения"""
        if not execution_times:
            return {
                'total_execution_time': total_time,
                'mean_time_per_sample': 0.0,
                'median_time_per_sample': 0.0,
                'min_time_per_sample': 0.0,
                'max_time_per_sample': 0.0,
                'std_time_per_sample': 0.0,
                'samples_per_second': 0.0,
                'total_samples': total_samples
            }
        
        times_array = np.array(execution_times)
        
        return {
            'total_execution_time': float(total_time),
            'mean_time_per_sample': float(np.mean(times_array)),
            'median_time_per_sample': float(np.median(times_array)),
            'min_time_per_sample': float(np.min(times_array)),
            'max_time_per_sample': float(np.max(times_array)),
            'std_time_per_sample': float(np.std(times_array)),
            'percentile_95_time': float(np.percentile(times_array, 95)),
            'percentile_99_time': float(np.percentile(times_array, 99)),
            'samples_per_second': float(total_samples / total_time if total_time > 0 else 0.0),
            'total_samples': int(total_samples)
        }
