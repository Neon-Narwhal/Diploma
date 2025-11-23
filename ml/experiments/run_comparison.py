import sys
from pathlib import Path

# Добавляем корень проекта
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.configs.experiment import ExperimentConfig
from ml.utils.data_loader import DataLoader  # Импорт из utils
from ml.experiments.runner import ExperimentRunner

def run_comparison():
    # Загрузка конфига
    config_path = "ml/configs/presets/compare_all.yaml"
    config = ExperimentConfig.from_yaml(config_path)
    
    # Загрузка данных
    loader = DataLoader.from_config(config)
    data = loader.load()
    
    # Запуск
    runner = ExperimentRunner(config)
    runner.run(data)

if __name__ == "__main__":
    run_comparison()
