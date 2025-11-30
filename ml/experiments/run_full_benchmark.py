import sys
import os
import yaml
import logging
from pathlib import Path
import pandas as pd

# Добавляем корень проекта в путь (на 3 уровня вверх от этого файла)
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.configs.experiment import ExperimentConfig
from ml.utils.data_loader import DataLoader
from ml.experiments.runner import ExperimentRunner

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_benchmark():
    # Список файлов конфигурации для бенчмарка
    configs_to_run = [
        "ml/configs/presets/benchmark_1_ast.yaml",
        "ml/configs/presets/benchmark_2_nlp.yaml",
        "ml/configs/presets/benchmark_3_hybrid.yaml",
        "ml/configs/presets/benchmark_4_ovr_hybrid.yaml",
        "ml/configs/presets/benchmark_5_stacking_hybrid.yaml"
    ]
    
    final_summary = []
    
    print("\n" + "="*80)
    print("ЗАПУСК ПОЛНОГО БЕНЧМАРКА (3 ТИПА ПРИЗНАКОВ x 3 МОДЕЛИ)")
    print("="*80 + "\n")
    
    for config_path in configs_to_run:
        full_path = project_root / config_path
        
        if not full_path.exists():
            logger.error(f"Config not found: {config_path}")
            continue
            
        logger.info(f"\n{'='*60}\nSTARTING BENCHMARK GROUP: {config_path}\n{'='*60}")
        
        try:
            # 1. Загружаем конфиг через метод from_yaml
            config = ExperimentConfig.from_yaml(str(full_path))
            
            # 2. Загружаем данные (препроцессинг зависит от конфига features!)
            logger.info("Loading and processing data...")
            loader = DataLoader.from_config(config)
            data = loader.load()
            
            # 3. Запускаем Runner
            # Runner сам создаст MLPipeline и переберет все модели из списка 'models' в конфиге
            runner = ExperimentRunner(config)
            run_results = runner.run(data)
            
            # 4. Собираем результаты для итоговой таблицы
            for model_name, res in run_results.items():
                score = 0
                
                # Пытаемся найти метрику F1 (сначала на тесте, потом на валидации)
                if 'test_metrics' in res and res['test_metrics']:
                    score = res['test_metrics'].get('f1_macro', 0)
                elif 'val_metrics' in res and res['val_metrics']:
                    score = res['val_metrics'].get('f1_macro', 0)
                
                # --- БЕЗОПАСНОЕ ОПРЕДЕЛЕНИЕ ТИПА ПРИЗНАКОВ ---
                feat_type = "unknown"
                
                # Проверяем config.features (это словарь или None)
                if config.features and isinstance(config.features, dict):
                    feat_type = config.features.get('type', 'unknown')
                
                # Если не нашли, ищем в config.data['features']
                elif hasattr(config, 'data'):
                    # config.data может быть словарем
                    if isinstance(config.data, dict):
                        feats = config.data.get('features', {})
                        if isinstance(feats, dict):
                            feat_type = feats.get('type', feat_type)
                    # Или объектом (если вдруг структура поменялась)
                    elif hasattr(config.data, 'features'):
                         f = config.data.features
                         if isinstance(f, dict):
                             feat_type = f.get('type', feat_type)
                # ---------------------------------------------
                
                final_summary.append({
                    'Feature Type': feat_type,
                    'Model': model_name,
                    'F1 Score': score,
                    'Config': config_path
                })
                
                logger.info(f"  >>> {feat_type} | {model_name}: F1 = {score:.4f}")
                
            logger.info(f"✓ Group {config_path} finished successfully.")
                
        except Exception as e:
            logger.error(f"❌ Failed benchmark group {config_path}: {e}", exc_info=True)
            # Не падаем, идем к следующему конфигу
            continue

    # 5. Выводим итоговую сводную таблицу
    logger.info("\n" + "="*60)
    logger.info("FINAL BENCHMARK RESULTS")
    logger.info("="*60)
    
    if final_summary:
        df = pd.DataFrame(final_summary)
        # Сортируем: сначала по типу фичей, потом по скору
        df = df.sort_values(by=['Feature Type', 'F1 Score'], ascending=[True, False])
        
        table_str = df.to_string(index=False)
        print("\n" + table_str)
        logger.info("\n" + table_str)
        
        # Сохраняем в CSV на всякий случай
        output_csv = project_root / "ml" / "outputs" / "reports" / "full_benchmark_results.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"Results saved to {output_csv}")
    else:
        logger.warning("No results gathered.")

if __name__ == "__main__":
    run_benchmark()
