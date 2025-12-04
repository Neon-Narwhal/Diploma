import sys
import os
import yaml
from pathlib import Path

# Добавляем корень проекта в путь для импортов
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cfg_analysis.experiments.runner import CFGExperimentRunner
from ast_analysis.configs.experiment import ASTExperimentConfig
# Важно: импортируем сам анализатор, чтобы сработал декоратор @register
import cfg_analysis.core.analyzer 

def main():
    # Жёсткая привязка к конфигу
    config_path = PROJECT_ROOT / "cfg_analysis/configs/presets/basic.yaml"
    
    print(f"Running experiment with config: {config_path}")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Загрузка конфига
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
        
    config = ASTExperimentConfig(**config_dict)
    
    # Запуск
    runner = CFGExperimentRunner(config)
    runner.run()

if __name__ == "__main__":
    main()
