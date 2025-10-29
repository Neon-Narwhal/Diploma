"""Скрипт для пакетного анализа множества файлов"""
import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from main_runner import create_analyzer, load_config
import sys

sys.path.append(str(Path(__file__).parent.parent))

from complexity_analyzers.base.analyzer import AnalysisContext
from utils.io.file_utils import read_source_file

def find_python_files(directory: Path, recursive: bool = True) -> List[Path]:
    """Поиск Python файлов в директории"""
    if recursive:
        return list(directory.rglob("*.py"))
    else:
        return list(directory.glob("*.py"))

def analyze_single_file(file_path: Path, analyzer, timeout: int) -> Dict[str, Any]:
    """Анализ одного файла"""
    try:
        source_code = read_source_file(str(file_path))
        
        context = AnalysisContext(
            source_code=source_code,
            language='python',
            timeout=timeout,
            debug_mode=False
        )
        
        result = analyzer.analyze(context)
        
        return {
            'file_path': str(file_path),
            'complexity_class': result.complexity_class.notation,
            'confidence': result.confidence,
            'analyzer_name': result.analyzer_name,
            'analysis_time': result.analysis_time,
            'success': result.is_valid(),
            'errors': result.errors,
            'warnings': result.warnings,
            'metrics': result.metrics.to_dict() if result.metrics else None
        }
        
    except Exception as e:
        return {
            'file_path': str(file_path),
            'complexity_class': 'ERROR',
            'confidence': 0.0,
            'analyzer_name': 'N/A',
            'analysis_time': 0.0,
            'success': False,
            'errors': [str(e)],
            'warnings': [],
            'metrics': None
        }

def save_results_csv(results: List[Dict[str, Any]], output_file: Path):
    """Сохранение результатов в CSV"""
    if not results:
        return
    
    fieldnames = [
        'file_path', 'complexity_class', 'confidence', 'analyzer_name',
        'analysis_time', 'success', 'error_count', 'warning_count'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Заголовок
        writer.writerow(fieldnames)
        
        # Данные
        for result in results:
            writer.writerow([
                result['file_path'],
                result['complexity_class'],
                f"{result['confidence']:.3f}",
                result['analyzer_name'],
                f"{result['analysis_time']:.3f}",
                result['success'],
                len(result.get('errors', [])),
                len(result.get('warnings', []))
            ])

def generate_summary_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Генерация сводного отчета"""
    total_files = len(results)
    successful_analyses = sum(1 for r in results if r['success'])
    failed_analyses = total_files - successful_analyses
    
    # Распределение по классам сложности
    complexity_distribution = {}
    confidence_scores = []
    analysis_times = []
    
    for result in results:
        if result['success']:
            complexity = result['complexity_class']
            complexity_distribution[complexity] = complexity_distribution.get(complexity, 0) + 1
            confidence_scores.append(result['confidence'])
            analysis_times.append(result['analysis_time'])
    
    # Статистика
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    avg_analysis_time = sum(analysis_times) / len(analysis_times) if analysis_times else 0
    
    return {
        'summary': {
            'total_files': total_files,
            'successful_analyses': successful_analyses,
            'failed_analyses': failed_analyses,
            'success_rate': successful_analyses / total_files if total_files > 0 else 0,
            'avg_confidence': avg_confidence,
            'avg_analysis_time': avg_analysis_time
        },
        'complexity_distribution': complexity_distribution,
        'top_errors': get_top_errors(results),
        'slowest_analyses': get_slowest_analyses(results)
    }

def get_top_errors(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Получение наиболее частых ошибок"""
    error_counts = {}
    
    for result in results:
        for error in result.get('errors', []):
            error_counts[error] = error_counts.get(error, 0) + 1
    
    # Сортируем по частоте
    sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
    
    return [{'error': error, 'count': count} for error, count in sorted_errors[:5]]

def get_slowest_analyses(results: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """Получение самых медленных анализов"""
    successful_results = [r for r in results if r['success']]
    sorted_results = sorted(successful_results, key=lambda x: x['analysis_time'], reverse=True)
    
    return [
        {
            'file_path': r['file_path'],
            'analysis_time': r['analysis_time'],
            'complexity_class': r['complexity_class']
        }
        for r in sorted_results[:top_n]
    ]

def main():
    """Главная функция пакетного анализа"""
    parser = argparse.ArgumentParser(
        description="Пакетный анализ сложности Python файлов"
    )
    
    parser.add_argument(
        'input_directory',
        help='Директория с Python файлами'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Базовое имя для выходных файлов',
        default='batch_analysis'
    )
    
    parser.add_argument(
        '-a', '--analyzer',
        choices=['hybrid', 'adaptive', 'lightweight', 'comprehensive'],
        default='lightweight',  # Для пакетного анализа лучше быстрый анализатор
        help='Тип анализатора'
    )
    
    parser.add_argument(
        '-t', '--timeout',
        type=int,
        default=10,
        help='Таймаут для анализа одного файла'
    )
    
    parser.add_argument(
        '-j', '--jobs',
        type=int,
        default=4,
        help='Количество параллельных процессов'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Рекурсивный поиск файлов'
    )
    
    parser.add_argument(
        '--config',
        help='Путь к файлу конфигурации'
    )
    
    args = parser.parse_args()
    
    try:
        input_dir = Path(args.input_directory)
        if not input_dir.exists():
            print(f"❌ Директория не найдена: {input_dir}")
            sys.exit(1)
        
        # Поиск Python файлов
        python_files = find_python_files(input_dir, args.recursive)
        
        if not python_files:
            print("❌ Python файлы не найдены")
            sys.exit(1)
        
        print(f"🔍 Найдено {len(python_files)} Python файлов")
        
        # Загрузка конфигурации и создание анализатора
        config = load_config(args.config)
        analyzer = create_analyzer(args.analyzer, config)
        
        if not analyzer:
            print("❌ Не удалось создать анализатор")
            sys.exit(1)
        
        print(f"📊 Анализатор: {args.analyzer}")
        print(f"⚡ Параллельность: {args.jobs} процессов")
        print("⏳ Выполняется анализ...")
        
        # Параллельный анализ файлов
        results = []
        
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            # Запускаем задачи
            future_to_file = {
                executor.submit(analyze_single_file, file_path, analyzer, args.timeout): file_path
                for file_path in python_files
            }
            
            # Собираем результаты
            completed = 0
            for future in as_completed(future_to_file):
                result = future.result()
                results.append(result)
                completed += 1
                
                # Прогресс
                if completed % 10 == 0 or completed == len(python_files):
                    print(f"📈 Обработано: {completed}/{len(python_files)} файлов")
        
        # Сохранение результатов
        output_base = Path(args.output)
        
        # JSON с детальными результатами
        json_file = output_base.with_suffix('.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # CSV с сводной информацией
        csv_file = output_base.with_suffix('.csv')
        save_results_csv(results, csv_file)
        
        # Сводный отчет
        summary = generate_summary_report(results)
        summary_file = output_base.with_name(f"{output_base.stem}_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Вывод сводки
        print("\n" + "="*50)
        print("📊 СВОДКА АНАЛИЗА")
        print("="*50)
        print(f"Всего файлов: {summary['summary']['total_files']}")
        print(f"Успешные анализы: {summary['summary']['successful_analyses']}")
        print(f"Неудачные анализы: {summary['summary']['failed_analyses']}")
        print(f"Коэффициент успеха: {summary['summary']['success_rate']:.1%}")
        print(f"Средняя уверенность: {summary['summary']['avg_confidence']:.1%}")
        print(f"Среднее время анализа: {summary['summary']['avg_analysis_time']:.2f}s")
        
        print("\n📈 Распределение сложности:")
        for complexity, count in summary['complexity_distribution'].items():
            print(f"  {complexity}: {count} файлов")
        
        print(f"\n✅ Результаты сохранены:")
        print(f"  📄 Детали: {json_file}")
        print(f"  📊 Таблица: {csv_file}")
        print(f"  📋 Сводка: {summary_file}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Анализ прерван пользователем")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Ошибка выполнения: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
