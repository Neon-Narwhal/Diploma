import sys
from pathlib import Path
import yaml
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.configs.experiment import ExperimentConfig
from ml.utils.data_loader import DataLoader
from ml.experiments.runner import ExperimentRunner
from ml.core.model_factory import ModelFactory
import joblib

def run_evaluation_only():
    # 1. Конфиг (берем тот же, но включаем отчет)
    config_path = "ml/configs/presets/benchmark_5_stacking_hybrid.yaml"
    config = ExperimentConfig.from_yaml(config_path)
    
    # Включаем генерацию отчета
    config.generate_report = True
    
    # 2. Загружаем данные (быстро из кэша)
    print("Загрузка данных...")
    loader = DataLoader.from_config(config)
    data = loader.load()
    
    # 3. Загружаем УЖЕ ОБУЧЕННУЮ модель
    model_path = "ml/outputs/models/stacking_mega_hybrid.pkl"
    print(f"Загрузка модели из {model_path}...")
    
    if not Path(model_path).exists():
        print(f"Ошибка: Файл {model_path} не найден!")
        return

    # Загружаем pickle
    model = joblib.load(model_path)
    
    # 4. Создаем runner, но подменяем логику run
    runner = ExperimentRunner(config)
    
    # Хак: вручную запускаем evaluate и логирование
    print("Оценка на тестовой выборке...")
    
    # Нам нужен метод _evaluate из pipeline, но проще вызвать predict напрямую
    y_pred = model.predict(data.X_test)
    
    # Считаем метрики
    from sklearn.metrics import f1_score, accuracy_score
    
    metrics = {
        'test_accuracy': accuracy_score(data.y_test, y_pred),
        'test_f1_macro': f1_score(data.y_test, y_pred, average='macro'),
        'test_f1_micro': f1_score(data.y_test, y_pred, average='micro')
    }
    
    print("\nМетрики:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
        
    # 5. Сохраняем в CSV (как это делает ExperimentRunner)
    report_dir = Path("ml/outputs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = report_dir / "benchmark_5_results.csv"
    
    result_row = {
        'experiment_name': config.name,
        'model_type': 'stacking',
        **metrics
    }
    
    if report_path.exists():
        df = pd.read_csv(report_path)
        # Удаляем старую запись с таким именем, если есть
        df = df[df['experiment_name'] != config.name]
        df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)
    else:
        df = pd.DataFrame([result_row])
        
    df.to_csv(report_path, index=False)
    print(f"\nОтчет сохранен в {report_path}")

if __name__ == "__main__":
    run_evaluation_only()
