"""
Запуск одной модели с реальными данными BigOBench.
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List

from ml.configs.experiment import ExperimentConfig
from ml.experiments.runner import ExperimentRunner
from ml.features.extractors import ComplexityFeatureExtractor


def load_bigobench_data(
    data_path: str,
    test_split: float = 0.2,
    val_split: float = 0.1,
    random_state: int = 42,
    max_samples: int = 10000,  # Ограничение для теста
    min_class_size: int = 50,  # Минимум примеров на класс
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    """
    Загрузка и подготовка данных из BigOBench для ML.
    """
    import json
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    print(f"\nЗагрузка данных из {data_path}...")
    
    # Загрузка JSONL
    samples = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if len(samples) >= max_samples:
                break
            
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
            except:
                continue
            
            # Получаем код
            code = data.get('code', '')
            
            if not code or len(code.strip()) < 10:
                continue
            
            # Получаем асимптотику
            complexity = (
                data.get('time_complexity_mapped') or 
                data.get('time_complexity')
            )
            
            if not complexity or str(complexity).strip() in ['', 'Unknown', 'UNKNOWN', 'None', 'null']:
                continue
            
            samples.append({
                'code': code,
                'complexity': str(complexity).strip(),
            })
    
    print(f"✓ Загружено {len(samples)} примеров")
    
    if len(samples) == 0:
        raise ValueError("Нет валидных примеров после фильтрации!")
    
    # Извлекаем код и метки
    code_samples = [s['code'] for s in samples]
    complexity_labels = [s['complexity'] for s in samples]
    
    # Фильтруем редкие классы
    from collections import Counter
    class_counts = Counter(complexity_labels)
    
    print(f"\nИсходное распределение классов:")
    for cls, count in class_counts.most_common():
        print(f"  {cls}: {count} примеров")
    
    # Оставляем только классы с достаточным количеством примеров
    valid_classes = {cls for cls, count in class_counts.items() if count >= min_class_size}
    
    if len(valid_classes) < 2:
        print(f"\n⚠ Предупреждение: мало классов после фильтрации, снижаем порог до 10")
        min_class_size = 10
        valid_classes = {cls for cls, count in class_counts.items() if count >= min_class_size}
    
    # Фильтруем примеры
    filtered_samples = []
    filtered_codes = []
    filtered_labels = []
    
    for code, label in zip(code_samples, complexity_labels):
        if label in valid_classes:
            filtered_samples.append({'code': code, 'complexity': label})
            filtered_codes.append(code)
            filtered_labels.append(label)
    
    print(f"\nПосле фильтрации редких классов: {len(filtered_samples)} примеров")
    
    # Кодируем метки в числа
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(filtered_labels)
    
    print(f"\nФинальные классы сложности ({len(label_encoder.classes_)} классов):")
    for i, class_name in enumerate(label_encoder.classes_):
        count = (y == i).sum()
        print(f"  {i}: {class_name} - {count} примеров")
    
    # Извлечение признаков из кода
    print(f"\nИзвлечение признаков из кода...")
    extractor = ComplexityFeatureExtractor()
    X_df = extractor.extract(filtered_codes)
    X = X_df.values
    
    print(f"✓ Извлечено {X.shape[1]} признаков")
    print(f"  Примеры признаков: {list(X_df.columns[:5])}")
    
    # Проверка на константные признаки
    non_const_features = []
    for i, col in enumerate(X_df.columns):
        if X[:, i].std() > 1e-6:  # Не константный
            non_const_features.append(i)
    
    if len(non_const_features) < X.shape[1]:
        print(f"\n⚠ Удалено {X.shape[1] - len(non_const_features)} константных признаков")
        X = X[:, non_const_features]
        print(f"  Осталось {X.shape[1]} признаков")
    
    if X.shape[1] == 0:
        raise ValueError("Все признаки константные! Проверьте экстрактор.")
    
    # Разделение на train/val/test
    X_temp, X_test, y_temp, y_test, code_temp, code_test = train_test_split(
        X, y, filtered_codes,
        test_size=test_split,
        stratify=y,
        random_state=random_state,
    )
    
    val_size_adjusted = val_split / (1 - test_split)
    X_train, X_val, y_train, y_val, code_train, code_val = train_test_split(
        X_temp, y_temp, code_temp,
        test_size=val_size_adjusted,
        stratify=y_temp,
        random_state=random_state,
    )
    
    print(f"\nРазделение данных:")
    print(f"  Train: {X_train.shape[0]} примеров, {X_train.shape[1]} признаков")
    print(f"  Val:   {X_val.shape[0]} примеров")
    print(f"  Test:  {X_test.shape[0]} примеров")
    
    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        code_train, code_val, code_test
    )





def run_single_model(
    config_path: str,
    data_path: str,
    test_split: float = 0.2,
    val_split: float = 0.1,
):
    """
    Запуск одной модели на данных BigOBench.
    
    Args:
        config_path: путь к YAML конфигу
        data_path: путь к JSONL файлу BigOBench
        test_split: доля test данных
        val_split: доля validation данных
        
    Returns:
        Результаты обучения
    """
    # Загрузка данных
    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        code_train, code_val, code_test
    ) = load_bigobench_data(
        data_path=data_path,
        test_split=test_split,
        val_split=val_split,
    )
    
    # Загрузка конфига
    config = ExperimentConfig.from_yaml(config_path)
    
    # Запуск
    runner = ExperimentRunner(config)
    
    # Используем только train и test (val можно использовать для CV внутри)
    results = runner.run(X_train, y_train, X_test, y_test)
    
    return results


# Пример использования
if __name__ == "__main__":
    # Путь к данным BigOBench
    DATA_PATH = "data/bigobench_mapped/train.jsonl"
    
    # Проверка существования файла
    if not Path(DATA_PATH).exists():
        print(f"❌ Файл не найден: {DATA_PATH}")
        exit(1)
    
    # Запуск эксперимента
    results = run_single_model(
        config_path="ml/configs/presets/single_catboost.yaml",
        data_path=DATA_PATH,
        test_split=0.2,
        val_split=0.1,
    )
    
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ")
    print("="*80)
    for model_name, model_results in results.items():
        print(f"\nМодель: {model_name}")
        
        if 'test_metrics' in model_results:
            print("Test метрики:")
            for metric, value in model_results['test_metrics'].items():
                print(f"  {metric}: {value:.4f}")
        
        if 'cv_results' in model_results:
            print("CV метрики:")
            for metric, value in model_results['cv_results']['mean'].items():
                std = model_results['cv_results']['std'][metric]
                print(f"  {metric}: {value:.4f} ± {std:.4f}")
