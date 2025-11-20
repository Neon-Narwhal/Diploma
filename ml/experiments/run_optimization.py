"""
Запуск с оптимизацией гиперпараметров на BigOBench.
"""

import numpy as np
from pathlib import Path

from ml.configs.experiment import ExperimentConfig
from ml.experiments.runner import ExperimentRunner
from ml.experiments.run_single import load_bigobench_data


def run_with_optimization(
    config_path: str,
    data_path: str,
    test_split: float = 0.2,
    val_split: float = 0.1,
):
    """
    Запуск с оптимизацией гиперпараметров на BigOBench.
    
    Args:
        config_path: путь к YAML конфигу (должен содержать optimization секцию)
        data_path: путь к JSONL файлу BigOBench
        test_split: доля test данных
        val_split: доля validation данных
        
    Returns:
        Результаты с оптимизированными параметрами
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
    
    # Проверка что оптимизация включена
    if not config.optimization or not config.optimization.get('enabled'):
        raise ValueError("Optimization must be enabled in config")
    
    # Запуск
    runner = ExperimentRunner(config)
    results = runner.run(X_train, y_train, X_test, y_test)
    
    # Вывод лучших параметров
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    print("="*80)
    
    for model_name, model_results in results.items():
        if 'optimization' in model_results:
            opt_results = model_results['optimization']
            print(f"\n{model_name} - Лучшие параметры:")
            for param, value in opt_results['best_params'].items():
                print(f"  {param}: {value}")
            print(f"  Best score: {opt_results['best_value']:.4f}")
            print(f"  Total trials: {opt_results['n_trials']}")
    
    return results


# Пример использования
if __name__ == "__main__":
    DATA_PATH = "data/bigobench_mapped/train.jsonl"
    
    if not Path(DATA_PATH).exists():
        print(f"❌ Файл не найден: {DATA_PATH}")
        exit(1)
    
    results = run_with_optimization(
        config_path="ml/configs/presets/with_optimization.yaml",
        data_path=DATA_PATH,
        test_split=0.2,
        val_split=0.1,
    )
