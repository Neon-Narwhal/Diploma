"""
Сравнение нескольких моделей.
"""

import numpy as np
from ml.configs.experiment import ExperimentConfig
from ml.experiments.runner import ExperimentRunner


def run_comparison(
    config_path: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
):
    """
    Запуск сравнения моделей из конфига.
    
    Args:
        config_path: путь к YAML конфигу
        X_train: признаки для обучения
        y_train: таргет для обучения
        X_test: признаки для теста
        y_test: таргет для теста
        
    Returns:
        Результаты сравнения
    """
    # Загрузка конфига
    config = ExperimentConfig.from_yaml(config_path)
    
    # Проверка что есть несколько моделей
    if len(config.models) < 2:
        raise ValueError("Для сравнения нужно минимум 2 модели в конфиге")
    
    # Запуск
    runner = ExperimentRunner(config)
    results = runner.run(X_train, y_train, X_test, y_test)
    
    return results


# Пример использования
if __name__ == "__main__":
    # Пример с dummy данными
    X_train = np.random.randn(100, 20)
    y_train = np.random.randint(0, 3, 100)
    X_test = np.random.randn(30, 20)
    y_test = np.random.randint(0, 3, 30)
    
    results = run_comparison(
        config_path="ml/configs/presets/compare_all.yaml",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
