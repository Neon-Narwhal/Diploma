import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.configs.experiment import ExperimentConfig
from ml.utils.data_loader import DataLoader
from ml.experiments.runner import ExperimentRunner
import pandas as pd

def run_all_optimizations():
    # Список конфигов для оптимизации
    config_paths = [
        "ml/configs/presets/optimize_catboost.yaml",
        "ml/configs/presets/optimize_xgboost.yaml",
        "ml/configs/presets/optimize_lightgbm.yaml"
    ]
    
    print("="*80)
    print("ЗАПУСК ПОСЛЕДОВАТЕЛЬНОЙ ОПТИМИЗАЦИИ 3 МОДЕЛЕЙ")
    print("="*80)

    # 1. Загрузка данных ОДИН РАЗ (используем настройки из первого конфига)
    # Предполагается, что данные и препроцессинг у всех одинаковые
    base_config = ExperimentConfig.from_yaml(config_paths[0])
    loader = DataLoader.from_config(base_config)
    data = loader.load()
    
    final_results = []

    # 2. Цикл по моделям
    for config_path in config_paths:
        print(f"\n\n>>> Обработка конфига: {config_path}")
        
        try:
            config = ExperimentConfig.from_yaml(config_path)
            
            # Используем уже загруженные данные, но создаем новый runner для текущего конфига
            runner = ExperimentRunner(config)
            
            # Запуск (run вернет результаты обучения/оптимизации)
            run_results = runner.run(data)
            
            # Извлекаем результаты оптимизации
            for model_name, model_res in run_results.items():
                if 'optimization' in model_res:
                    opt = model_res['optimization']
                    final_results.append({
                        'model': model_name,
                        'best_score': opt['best_value'],
                        'n_trials': opt['n_trials'],
                        'best_params': opt['best_params']
                    })
                else:
                     # Если оптимизация упала или не запустилась, но есть метрики
                     if 'test_metrics' in model_res:
                         final_results.append({
                             'model': model_name,
                             'best_score': model_res['test_metrics'].get('test_f1_macro', 0),
                             'n_trials': 0,
                             'best_params': "Optimization failed or skipped"
                         })

        except Exception as e:
            print(f"❌ Ошибка при обработке {config_path}: {e}")

    # 3. Итоговое сравнение
    print("\n" + "="*80)
    print("ИТОГОВАЯ ТАБЛИЦА ЛУЧШИХ МОДЕЛЕЙ")
    print("="*80)
    
    if final_results:
        df = pd.DataFrame(final_results)
        # Сортируем по скору (по убыванию)
        df = df.sort_values(by='best_score', ascending=False)
        
        print(df[['model', 'best_score', 'n_trials']].to_string(index=False))
        
        print("\nЛучшие параметры победителя:")
        best_model = df.iloc[0]
        print(f"Модель: {best_model['model']}")
        print(f"Params: {best_model['best_params']}")
    else:
        print("Нет результатов для сравнения.")

if __name__ == "__main__":
    run_all_optimizations()
