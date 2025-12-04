import sys
import copy
from pathlib import Path

# === НАСТРОЙКИ ЗАПУСКА ===

# ПОСТАВЬ True, ЧТОБЫ ПРОВЕРИТЬ КОД ЗА 5 МИНУТ (на ноутбуке)
# ПОСТАВЬ False ПЕРЕД ОТПРАВКОЙ ДРУГУ НА A100
DEBUG_MODE = False 

# =========================

# Настройка путей (поднимаемся из ml/experiments/ в корень проекта)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.configs.experiment import ExperimentConfig
from ml.experiments.runner import ExperimentRunner
from ml.utils.data_loader import DataLoader


# === КОНФИГУРАЦИЯ ПАРАМЕТРОВ ===

if DEBUG_MODE:
    print(f"\n{'!'*40}")
    print("!!! ВНИМАНИЕ: ЗАПУЩЕН DEBUG РЕЖИМ !!!")
    print("Обучение пройдет на микро-датасете для проверки кода.")
    print(f"{'!'*40}\n")
    
    # Легкие параметры для теста
    GLOBAL_PARAMS_CB = {"iterations": 10, "depth": 2, "learning_rate": 0.1, "task_type": "CPU", "verbose": True}
    GLOBAL_PARAMS_XGB = {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.1, "device": "cpu"}
    GLOBAL_PARAMS_LGBM = {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.1, "device": "cpu", "verbose": -1}
    
    SPLIT_LIMITS = {"train": 200, "val": 50, "test": 50} # Микро-лимиты
    STACKING_CV = 2
    OVR_ITERATIONS = 5

else:
    # ТЯЖЕЛЫЕ ПАРАМЕТРЫ ДЛЯ A100
    # Используем 0:1, предполагая 2 GPU. Если одна — поменяй на "0"
    GLOBAL_PARAMS_CB = {
        "task_type": "GPU", "devices": "0:1", 
        "iterations": 5000, "depth": 8, "learning_rate": 0.03, 
        "early_stopping_rounds": 300, "verbose": 500
    }
    GLOBAL_PARAMS_XGB = {
        "tree_method": "hist", "device": "cuda", 
        "n_estimators": 5000, "max_depth": 8, "learning_rate": 0.03,
        "early_stopping_rounds": 300
    }
    GLOBAL_PARAMS_LGBM = {
        "device": "gpu", 
        "n_estimators": 5000, "max_depth": 8, "learning_rate": 0.03,
        "early_stopping_rounds": 300, "verbose": -1
    }
    
    SPLIT_LIMITS = None # Грузим всё (100k+)
    STACKING_CV = 5
    OVR_ITERATIONS = 3000


# Базовая структура конфига
BASE_CONFIG = {
    "data": {
        "train_path": "data/bigobench_mapped/train.jsonl",
        "val_path": "data/bigobench_mapped/val.jsonl",
        "test_path": "data/bigobench_mapped/test.jsonl",
        "preprocessing": {
            "min_code_length": 10,
            "split_limits": SPLIT_LIMITS
        }
    },
    "optimization": {"enabled": False}, # Optuna отключена
    "evaluation_metrics": ["f1_macro", "accuracy", "precision_macro", "recall_macro"],
    "save_models": True,
    "generate_report": True
}

def run_experiment(name, feature_type, models_list, description):
    print(f"\n\n{'='*80}")
    print(f"ЗАПУСК: {name}")
    print(f"{description}")
    print(f"{'='*80}\n")
    
    config_dict = copy.deepcopy(BASE_CONFIG)
    config_dict["name"] = name
    config_dict["description"] = description
    config_dict["features"] = {"type": feature_type}
    
    # Кэширование
    cache_key = f"debug_{feature_type}" if DEBUG_MODE else f"full_a100_{feature_type}"
    config_dict["data"]["feature_cache"] = {
        "enabled": True,
        "cache_dir": "data/feature_cache",
        "cache_key": cache_key
    }
    
    config_dict["models"] = models_list
    
    # Создаем конфиг
    config = ExperimentConfig(**config_dict)
    
    # === ВАЖНО: ЗАГРУЖАЕМ ДАННЫЕ ===
    from ml.utils.data_loader import DataLoader
    print("Загрузка данных...")
    loader = DataLoader.from_config(config)
    data = loader.load()
    print(f"  Train: {data.X_train.shape}, Val: {data.X_val.shape}, Test: {data.X_test.shape}")
    # ================================
    
    # Запускаем эксперименты
    runner = ExperimentRunner(config)
    runner.run(data)  # <--- Передаем данные сюда


def main():
    # 1. AST
    run_experiment(
        "1_full_ast", "ast",
        [   # models_list идет ТРЕТЬИМ
            {"name": "cb_ast", "type": "catboost", "params": GLOBAL_PARAMS_CB},
            {"name": "xgb_ast", "type": "xgboost", "params": GLOBAL_PARAMS_XGB}
        ],
        "Сравнение на AST признаках" # description идет ЧЕТВЕРТЫМ
    )

    # 2. NLP (Jina)
    run_experiment(
        "2_full_nlp", "jina",
        [
            {"name": "cb_nlp", "type": "catboost", "params": GLOBAL_PARAMS_CB},
            {"name": "xgb_nlp", "type": "xgboost", "params": GLOBAL_PARAMS_XGB}
        ],
        "Сравнение на Embeddings"
    )

    # 3. Hybrid (Base for Stacking)
    run_experiment(
        "3_full_hybrid", "hybrid",
        [
            {"name": "cb_hybrid", "type": "catboost", "params": GLOBAL_PARAMS_CB},
            {"name": "xgb_hybrid", "type": "xgboost", "params": GLOBAL_PARAMS_XGB},
            {"name": "lgbm_hybrid", "type": "lightgbm", "params": GLOBAL_PARAMS_LGBM}
        ],
        "Гибридные признаки (база для стекинга)"
    )

    # 4. OvR (Hybrid)
    ovr_params = copy.deepcopy(GLOBAL_PARAMS_CB)
    if not DEBUG_MODE:
        ovr_params["iterations"] = OVR_ITERATIONS
        ovr_params["depth"] = 6 
    
    run_experiment(
        "4_full_ovr_hybrid", "hybrid",
        [
            {"name": "ovr_cb_hybrid", "type": "ovr_catboost", "params": ovr_params}
        ],
        "One-vs-Rest подход"
    )

    # 5. Stacking
    pretrained_paths = [
        "ml/outputs/models/cb_hybrid.pkl",
        "ml/outputs/models/xgb_hybrid.pkl",
        "ml/outputs/models/lgbm_hybrid.pkl",
        "ml/outputs/models/ovr_cb_hybrid.pkl"
    ]
    
    stacking_params = {
        "meta_model": "ridge",
        "cv": STACKING_CV,
        "pretrained_models": pretrained_paths
    }
    
    run_experiment(
        "5_full_stacking", "hybrid",
        [
            {"name": "mega_stacking_full", "type": "stacking", "params": stacking_params}
        ],
        "Финальный ансамбль (Stacking)"
    )

    
    print(f"\n\n{'='*80}")
    print("ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ УСПЕШНО")
    if DEBUG_MODE:
        print("Это был ТЕСТОВЫЙ прогон. Теперь поставь DEBUG_MODE = False и отправляй другу!")
    else:
        print("Результаты ищи в ml/outputs/reports/ и ml/outputs/plots/")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
