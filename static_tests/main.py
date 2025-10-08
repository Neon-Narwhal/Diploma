"""
Главный скрипт для запуска анализа.
Все параметры берутся из config.py
"""

from pathlib import Path
import logging
from typing import Optional, List

from dataset_processor import DatasetProcessor
from config import (
    TOOLS_REGISTRY,
    DEFAULT_PYTHON_DATASET,
    DEFAULT_JAVA_DATASET,
    RESULTS_DIR,
    DATA_DIR,
    ComplexityClass,
    # Параметры запуска из конфига
    INPUT_FILE,
    OUTPUT_DIR,
    TOOLS_TO_RUN,
    MAX_SAMPLES,
    LANGUAGE
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_analysis(
    input_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    tools: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    language: str = 'python'
) -> dict:
    """
    Запускает анализ кода с заданными параметрами.
    
    Args:
        input_path: Путь к JSONL файлу. Если None, используется DEFAULT_PYTHON_DATASET
        output_dir: Директория для сохранения результатов. Если None, используется RESULTS_DIR
        tools: Список инструментов для использования. Если None, используются все включенные
        max_samples: Максимальное количество образцов для обработки
        language: Язык программирования ('python' или 'java')
    
    Returns:
        Словарь с результатами анализа
    """
    # Определяем путь к входному файлу
    if input_path is None:
        if language.lower() == 'python':
            input_path = DEFAULT_PYTHON_DATASET
        elif language.lower() == 'java':
            input_path = DEFAULT_JAVA_DATASET
        else:
            raise ValueError(f"Неизвестный язык: {language}. Используйте 'python' или 'java'")
        
        logger.info(f"Используется датасет по умолчанию: {input_path}")
    else:
        input_path = Path(input_path)
    
    # Определяем директорию для результатов
    if output_dir is None:
        output_dir = RESULTS_DIR / language.lower()
        logger.info(f"Используется директория для результатов по умолчанию: {output_dir}")
    else:
        output_dir = Path(output_dir)
    
    # Проверяем существование входного файла
    if not input_path.exists():
        logger.error(f"Входной файл не найден: {input_path}")
        logger.info(f"Убедитесь, что файл находится в {DATA_DIR}")
        logger.info(f"Ожидаемая структура:")
        logger.info(f"  project_root/")
        logger.info(f"  ├── complexity_analyzer/")
        logger.info(f"  └── data/")
        logger.info(f"      └── {input_path.name}")
        raise FileNotFoundError(f"Файл не найден: {input_path}")
    
    if tools is None:
        from config import TOOLS_TO_RUN
        tools = TOOLS_TO_RUN
    # Инициализируем процессор
    processor = DatasetProcessor(tools_to_use=tools)
    
    # Запускаем обработку
    logger.info("Запуск обработки датасета...")
    results = processor.process_dataset(
        filepath=input_path,
        output_dir=output_dir,
        max_samples=max_samples
    )
    
    logger.info("Обработка завершена!")
    logger.info(f"Результаты сохранены в {output_dir}")
    
    # Выводим краткую статистику
    print("\n" + "="*80)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("="*80)
    
    for tool_name, tool_results in results.items():
        metrics = tool_results['metrics']
        time_stats = tool_results['time_statistics']
        
        print(f"\n{'='*80}")
        print(f"{tool_name.upper()}")
        print(f"{'='*80}")
        
        print(f"\n📊 МЕТРИКИ КАЧЕСТВА:")
        print(f"  Accuracy:              {metrics['accuracy']:.4f}")
        print(f"  F1 (weighted):         {metrics['f1_weighted']:.4f}")
        print(f"  Precision (weighted):  {metrics['precision_weighted']:.4f}")
        print(f"  Recall (weighted):     {metrics['recall_weighted']:.4f}")
        print(f"  Валидных предсказаний: {tool_results['valid_predictions']}/{tool_results['total_samples']}")
        
        print(f"\n⏱️  ВРЕМЕННЫЕ МЕТРИКИ:")
        print(f"  Общее время:           {time_stats['total_execution_time']:.2f} сек")
        print(f"  Среднее время/образец: {time_stats['mean_time_per_sample']*1000:.2f} мс")
        print(f"  Медиана время/образец: {time_stats['median_time_per_sample']*1000:.2f} мс")
        print(f"  Мин время/образец:     {time_stats['min_time_per_sample']*1000:.2f} мс")
        print(f"  Макс время/образец:    {time_stats['max_time_per_sample']*1000:.2f} мс")
        print(f"  Образцов в секунду:    {time_stats['samples_per_second']:.2f}")
        
        # Статистика по классам асимптотики
        print(f"\n🎯 СТАТИСТИКА ПО КЛАССАМ АСИМПТОТИКИ:")
        class_dist = tool_results['class_distribution']
        
        print(f"\n{'Класс':<12} {'Истинных':<10} {'Предсказано':<12} {'Правильно':<10} {'Точность':<10}")
        print("-" * 80)
        
        for complexity_class in [c.value for c in ComplexityClass]:
            if complexity_class in class_dist:
                dist = class_dist[complexity_class]
                print(
                    f"{complexity_class:<12} "
                    f"{dist['true_count']:<10} "
                    f"{dist['predicted_count']:<12} "
                    f"{dist['correct_predictions']:<10} "
                    f"{dist['accuracy_for_class']:.1f}%"
                )
    
    # Сравнительная таблица по скорости
    if len(results) > 1:
        print("\n" + "="*80)
        print("СРАВНЕНИЕ ИНСТРУМЕНТОВ ПО СКОРОСТИ")
        print("="*80)
        print(f"{'Инструмент':<15} {'Общее время':<15} {'Сред. время':<15} {'Образцов/сек':<15}")
        print("-" * 80)
        
        # Сортируем по среднему времени
        sorted_tools = sorted(
            results.items(),
            key=lambda x: x[1]['time_statistics']['mean_time_per_sample']
        )
        
        for tool_name, tool_results in sorted_tools:
            time_stats = tool_results['time_statistics']
            print(
                f"{tool_name:<15} "
                f"{time_stats['total_execution_time']:>12.2f} s  "
                f"{time_stats['mean_time_per_sample']*1000:>12.2f} ms "
                f"{time_stats['samples_per_second']:>14.2f}"
            )
    
    return results


def main():
    """
    Запускает анализ с параметрами из config.py
    """
    print("="*80)
    print("ЗАПУСК АНАЛИЗА СЛОЖНОСТИ КОДА")
    print("="*80)
    print(f"Параметры из config.py:")
    print(f"  Входной файл:     {INPUT_FILE or 'по умолчанию'}")
    print(f"  Выходная папка:   {OUTPUT_DIR or 'по умолчанию'}")
    print(f"  Инструменты:      {TOOLS_TO_RUN or 'все включенные'}")
    print(f"  Макс. образцов:   {MAX_SAMPLES or 'все'}")
    print(f"  Язык:             {LANGUAGE}")
    print("="*80)
    print()
    
    try:
        run_analysis(
            input_path=INPUT_FILE,
            output_dir="static_tests/results/results_advenced",
            tools=TOOLS_TO_RUN,
            max_samples=MAX_SAMPLES,
            language=LANGUAGE
        )
        print("\n✅ Анализ успешно завершен!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Ошибка: {e}")
        print(f"\n💡 Убедитесь, что структура проекта выглядит так:")
        print(f"   project_root/")
        print(f"   ├── complexity_analyzer/")
        print(f"   │   ├── __init__.py")
        print(f"   │   ├── config.py")
        print(f"   │   └── ...")
        print(f"   ├── data/")
        print(f"   │   └── python_dataset.jsonl")
        print(f"   └── run_analysis.py")
        
    except Exception as e:
        print(f"\n❌ Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
