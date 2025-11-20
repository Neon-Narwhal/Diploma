"""
Загрузка конфигов из YAML/dict.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Union


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Загрузка YAML конфига.
    
    Args:
        path: путь к YAML файлу
        
    Returns:
        Словарь с конфигом
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_yaml(config: Dict[str, Any], path: str):
    """
    Сохранение конфига в YAML.
    
    Args:
        config: словарь с конфигом
        path: путь для сохранения
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    Валидация конфига.
    
    Args:
        config: словарь с конфигом
        required_keys: список обязательных ключей
        
    Returns:
        True если конфиг валиден
        
    Raises:
        ValueError: если отсутствуют обязательные ключи
    """
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        raise ValueError(f"Config missing required keys: {missing_keys}")
    
    return True
