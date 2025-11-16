"""Статистический анализ BigOBench датасета"""
from data_loaders.loaders.loaders import HuggingFaceLoader
from collections import Counter
import json
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class BigOBenchAnalyzer:
    """Анализ статистики BigOBench датасета"""
    
    def __init__(self, output_dir: Path = Path("datasets/metadata")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = HuggingFaceLoader()
    
    def analyze(self, streaming: bool = True) -> Dict:
        """Полный статистический анализ"""
        
        logger.info("Загрузка датасета для анализа")
        dataset = self.loader.load_bigobench_complexity_labels(streaming=streaming)
        
        stats = self._collect_statistics(dataset)
        results = self._compute_results(stats)
        
        # Сохранение
        output_path = self.output_dir / "bigobench_statistics.json"
        self._save_results(results, output_path)
        
        # Вывод
        self._print_results(results, output_path)
        
        return results
    
    def _collect_statistics(self, dataset) -> Dict:
        """Сбор сырой статистики"""
        stats = {
            'time_complexity': Counter(),
            'space_complexity': Counter(),
            'language': Counter(),
            'code_lengths': [],
            'problems': set(),
            'solutions_per_problem': Counter(),
            'time_space_pairs': Counter(),
            'total_samples': 0
        }
        
        print("Обработка образцов...")
        for i, example in enumerate(dataset):
            if i % 50000 == 0:
                print(f"Обработано {i} образцов...")
            
            time_cls = example.get('time_complexity_inferred', 'unknown')
            space_cls = example.get('space_complexity_inferred', 'unknown')
            
            stats['time_complexity'][time_cls] += 1
            stats['space_complexity'][space_cls] += 1
            stats['time_space_pairs'][(time_cls, space_cls)] += 1
            
            # Язык - в complexity_labels нет, ставим python по умолчанию
            lang = example.get('language', 'python')
            stats['language'][lang] += 1
            
            # Длина кода - в complexity_labels нет поля solution_code
            # Поэтому пропускаем или ставим 0
            code = example.get('solution_code', '')
            stats['code_lengths'].append(len(code))
            
            problem_id = example.get('problem_id')
            if problem_id:
                stats['problems'].add(problem_id)
                stats['solutions_per_problem'][problem_id] += 1
            
            stats['total_samples'] += 1
        
        return stats
    
    def _compute_results(self, stats: Dict) -> Dict:
        """Вычисление агрегатов"""
        code_lengths = sorted(stats['code_lengths'])
        n = len(code_lengths)
        
        results = {
            'total_samples': stats['total_samples'],
            'unique_problems': len(stats['problems']),
            'avg_solutions_per_problem': (
                stats['total_samples'] / len(stats['problems']) 
                if stats['problems'] else 0
            ),
            
            'time_complexity_distribution': {
                cls: {
                    'count': count,
                    'percentage': (count / stats['total_samples'] * 100)
                }
                for cls, count in stats['time_complexity'].most_common()
            },
            
            'space_complexity_distribution': {
                cls: {
                    'count': count,
                    'percentage': (count / stats['total_samples'] * 100)
                }
                for cls, count in stats['space_complexity'].most_common()
            },
            
            'language_distribution': dict(stats['language']),
            
            'code_length_stats': {
                'min': code_lengths[0] if code_lengths else 0,
                'max': code_lengths[-1] if code_lengths else 0,
                'median': code_lengths[n // 2] if code_lengths else 0,
                'p25': code_lengths[n // 4] if code_lengths else 0,
                'p75': code_lengths[3 * n // 4] if code_lengths else 0,
                'mean': sum(code_lengths) / n if code_lengths else 0
            },
            
            'top_time_space_pairs': [
                {
                    'time': pair[0],
                    'space': pair[1],
                    'count': count,
                    'percentage': (count / stats['total_samples'] * 100)
                }
                for pair, count in stats['time_space_pairs'].most_common(20)
            ]
        }
        
        return results
    
    def _save_results(self, results: Dict, output_path: Path):
        """Сохранение в JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Результаты сохранены в {output_path}")
    
    def _print_results(self, results: Dict, output_path: Path):
        """Вывод результатов в консоль"""
        print(f"\n{'='*60}")
        print("СТАТИСТИКА BIGOBENCH ДАТАСЕТА")
        print(f"{'='*60}\n")
        
        print(f"Всего образцов: {results['total_samples']:,}")
        print(f"Уникальных задач: {results['unique_problems']:,}")
        print(f"Среднее решений на задачу: {results['avg_solutions_per_problem']:.1f}\n")
        
        # Временная сложность
        self._print_complexity_distribution(
            "ВРЕМЕННАЯ СЛОЖНОСТЬ",
            results['time_complexity_distribution']
        )
        
        # Пространственная сложность
        self._print_complexity_distribution(
            "ПРОСТРАНСТВЕННАЯ СЛОЖНОСТЬ",
            results['space_complexity_distribution']
        )
        
        # Языки
        print("\nЯЗЫКИ ПРОГРАММИРОВАНИЯ:")
        for lang, count in sorted(
            results['language_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            pct = count / results['total_samples'] * 100
            lang_str = str(lang) if lang is not None else 'unknown'
            print(f"{lang_str:<15} {count:>12,} ({pct:>6.2f}%)")
        
        # Длина кода
        print("\nДЛИНА КОДА (символов):")
        stats = results['code_length_stats']
        print(f"  Минимум:  {stats['min']:>8,}")
        print(f"  25%:      {stats['p25']:>8,}")
        print(f"  Медиана:  {stats['median']:>8,}")
        print(f"  Среднее:  {stats['mean']:>8,.0f}")
        print(f"  75%:      {stats['p75']:>8,}")
        print(f"  Максимум: {stats['max']:>8,}")
        
        # Топ пар
        print("\nТОП-10 ПАР (ВРЕМЯ, ПАМЯТЬ):")
        print(f"{'Временная':<15} {'Пространственная':<15} {'Количество':>12} {'Процент':>10}")
        print("-" * 60)
        for pair in results['top_time_space_pairs'][:10]:
            time_str = str(pair['time']) if pair['time'] is not None else 'None'
            space_str = str(pair['space']) if pair['space'] is not None else 'None'
            print(f"{time_str:<15} {space_str:<15} {pair['count']:>12,} {pair['percentage']:>9.2f}%")
        
        print(f"\n{'='*60}")
        print(f"Результаты сохранены в {output_path}")
    
    @staticmethod
    def _print_complexity_distribution(title: str, distribution: Dict):
        """Вывод распределения классов сложности"""
        print(f"\n{title}:")
        print(f"{'Класс':<20} {'Количество':>12} {'Процент':>10}")
        print("-" * 45)
        
        for cls, data in sorted(
            distribution.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        ):
            cls_str = str(cls) if cls is not None else 'None'
            print(f"{cls_str:<20} {data['count']:>12,} {data['percentage']:>9.2f}%")


# Функция для обратной совместимости
def analyze_bigobench() -> Dict:
    """
    Анализ BigOBench датасета
    
    Returns:
        Dict с результатами анализа
    """
    analyzer = BigOBenchAnalyzer()
    return analyzer.analyze()


if __name__ == "__main__":
    analyze_bigobench()
