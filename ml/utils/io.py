"""
Утилиты для сохранения/загрузки моделей и данных.
"""

import joblib
import pickle
from pathlib import Path
from typing import Any


def save_model(model: Any, path: str):
    """
    Сохранение модели.
    
    Args:
        model: модель для сохранения
        path: путь для сохранения
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    """
    Загрузка модели.
    
    Args:
        path: путь к модели
        
    Returns:
        Загруженная модель
    """
    return joblib.load(path)


def save_pickle(obj: Any, path: str):
    """Сохранение объекта в pickle"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    """Загрузка объекта из pickle"""
    with open(path, 'rb') as f:
        return pickle.load(f)
