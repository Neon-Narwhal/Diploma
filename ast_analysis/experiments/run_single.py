"""
Запуск одного AST эксперимента.
"""

import sys
import argparse
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ast_analysis.configs.experiment import ASTExperimentConfig
from ast_analysis.experiments.runner import ASTExperimentRunner


def main():
    parser = argparse.ArgumentParser(description='Run AST analysis experiment')
    parser.add_argument(
        '--config',
        type=str,
        default='ast_analysis/configs/presets/advanced.yaml',
        help='Path to experiment config'
    )
    args = parser.parse_args()
    
    # Загрузка конфига
    print(f"Loading config from: {args.config}")
    config = ASTExperimentConfig.from_yaml(args.config)
    
    # Запуск эксперимента
    runner = ASTExperimentRunner(config)
    results = runner.run()
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
