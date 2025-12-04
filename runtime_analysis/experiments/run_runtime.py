import sys
from pathlib import Path
import yaml

# Добавляем корень проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from runtime_analysis.experiments.runner import RuntimeExperimentRunner
from ast_analysis.configs.experiment import ASTExperimentConfig
# Импорт для регистрации анализатора
import runtime_analysis.core.analyzer 

def main():
    # Жёсткая привязка к конфигу
    config_path = PROJECT_ROOT / "runtime_analysis/configs/presets/runtime_test.yaml"
    
    print(f"Running runtime analysis with config: {config_path}")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
        
    config = ASTExperimentConfig(**config_dict)
    runner = RuntimeExperimentRunner(config)
    runner.run()

if __name__ == "__main__":
    main()
