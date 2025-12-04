"""
Запуск бенчмарка всех AST анализаторов.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ast_analysis.configs.experiment import ASTExperimentConfig
from ast_analysis.experiments.runner import ASTExperimentRunner


def run_benchmark():
    """Запуск бенчмарка на нескольких конфигурациях"""
    
    configs = [
        'ast_analysis/configs/presets/basic.yaml',
        'ast_analysis/configs/presets/full.yaml',
    ]
    
    all_results = {}
    
    for config_path in configs:
        print("\n" + "=" * 80)
        print(f"Running: {config_path}")
        print("=" * 80)
        
        try:
            config = ASTExperimentConfig.from_yaml(config_path)
            runner = ASTExperimentRunner(config)
            results = runner.run()
            all_results[config.name] = results
        
        except Exception as e:
            print(f"Failed to run {config_path}: {e}")
            continue
    
    # Сводная таблица
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    for exp_name, results in all_results.items():
        print(f"\n{exp_name}:")
        metrics = results.get('metrics', {})
        for split_name, split_metrics in metrics.items():
            success_rate = split_metrics.get('success_rate', 0)
            avg_time = split_metrics.get('avg_processing_time', 0)
            print(f"  {split_name}: success={success_rate:.2%}, time={avg_time:.3f}s")


if __name__ == "__main__":
    run_benchmark()
