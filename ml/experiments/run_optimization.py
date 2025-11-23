"""
Запуск с оптимизацией гиперпараметров.
"""

import sys
from pathlib import Path

# Добавляем корень проекта
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.configs.experiment import ExperimentConfig
from ml.utils.data_loader import DataLoader
from ml.experiments.runner import ExperimentRunner


def run_with_optimization():
    """
    Запуск эксперимента с оптимизацией гиперпараметров.
    """
    # 1. Загрузка конфига
    config_path = "ml/configs/presets/with_optimization.yaml"
    config = ExperimentConfig.from_yaml(config_path)
    
    # Проверка включения оптимизации
    if not config.optimization or not config.optimization.get('enabled'):
        raise ValueError("Optimization must be enabled in config")
    
    # 2. Загрузка данных
    loader = DataLoader.from_config(config)
    data = loader.load()
    
    # 3. Запуск эксперимента
    print("\n" + "="*80)
    print("ЗАПУСК ОПТИМИЗАЦИИ")
    print("="*80)
    
    runner = ExperimentRunner(config)
    results = runner.run(data)
    
    # 4. Вывод лучших параметров
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


if __name__ == "__main__":
    run_with_optimization()
