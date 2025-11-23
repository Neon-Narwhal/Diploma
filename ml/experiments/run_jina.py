import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.configs.experiment import ExperimentConfig
from ml.utils.data_loader import DataLoader
from ml.experiments.runner import ExperimentRunner

def run_jina_experiment():
    print("="*80)
    print("ЗАПУСК JINA V2 + CATBOOST")
    print("="*80)
    
    # 1. Загрузка конфига
    config_path = "ml/configs/presets/optimize_catboost_hybrid_pca.yaml"
    config = ExperimentConfig.from_yaml(config_path)
    
    # 2. Загрузка данных (здесь скачается и запустится Jina)
    loader = DataLoader.from_config(config)
    data = loader.load()
    
    # 3. Запуск обучения и оптимизации
    runner = ExperimentRunner(config)
    results = runner.run(data)
    
    # 4. Вывод результатов
    if 'catboost_jina' in results and 'optimization' in results['catboost_jina']:
        opt = results['catboost_jina']['optimization']
        print(f"\nЛучший результат F1 Macro: {opt['best_value']:.4f}")
        print(f"Параметры: {opt['best_params']}")

if __name__ == "__main__":
    run_jina_experiment()
