"""
Подготовка BigOBench датасета

Запуск:
    python prepare_dataset.py
"""

from data_loaders import BigOBenchDataset
from pathlib import Path


def main():
    """Основной пайплайн подготовки данных"""
    
    # Конфигурация
    config = {
        'output_dir': Path('data/bigobench'),
        'min_samples_per_class': 100,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'validate': True
    }
    
    print("="*60)
    print("ПОДГОТОВКА BIGOBENCH ДАТАСЕТА")
    print("="*60)
    print(f"Output: {config['output_dir']}")
    print(f"Min samples per class: {config['min_samples_per_class']}")
    print(f"Train/Val/Test: {config['train_ratio']}/{config['val_ratio']}/{config['test_ratio']}")
    print("="*60 + "\n")
    
    # Инициализация
    dataset = BigOBenchDataset(output_dir=config['output_dir'])
    
    # Подготовка
    print("Загрузка и обработка данных...")
    dataset.prepare(
        min_samples_per_class=config['min_samples_per_class'],
        validate=config['validate']
    )
    
    # Сохранение
    print("\nРазделение и сохранение...")
    dataset.split_and_save(
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        test_ratio=config['test_ratio']
    )
    
    print("\n" + "="*60)
    print("ГОТОВО!")
    print(f"Датасет сохранен в: {config['output_dir']}")
    print("="*60)


if __name__ == '__main__':
    main()
