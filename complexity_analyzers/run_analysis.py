# complexity_analysis/run_analysis.py

import sys
import logging
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    INPUT_PATH, OUTPUT_DIR, ANALYZERS_TO_RUN, ANALYSIS_MODE,
    MAX_SAMPLES, LANGUAGE, LOG_LEVEL, LOG_TO_FILE, DEFAULT_PYTHON_DATASET
)
from processors import ComplexityProcessor

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_analysis():
    print("=" * 80)
    print("🚀 ЗАПУСК АНАЛИЗА СЛОЖНОСТИ")
    print("=" * 80)
    
    # Определяем входной путь
    input_path = INPUT_PATH or DEFAULT_PYTHON_DATASET
    if not input_path.exists():
        logger.error(f"Входной путь не найден: {input_path}")
        return

    # Определяем выходную директорию
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR or (PROJECT_ROOT / 'complexity_analyzers' / 'results' / f"{timestamp}_{ANALYSIS_MODE}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Конфигурация:")
    print(f"  Вход: {input_path}")
    print(f"  Выход: {output_dir}")
    print(f"  Анализаторы: {ANALYZERS_TO_RUN or 'все включенные'}")
    print(f"  Макс. образцов: {MAX_SAMPLES or 'все'}")
    print("=" * 80)

    try:
        from config import ANALYZERS_REGISTRY, MAX_WORKERS, USE_MULTIPROCESSING
        
        analyzers_to_run = ANALYZERS_TO_RUN or [name for name, cfg in ANALYZERS_REGISTRY.items() if cfg.enabled]
        
        processor = ComplexityProcessor(
            analyzers_to_use=analyzers_to_run,
            max_workers=MAX_WORKERS if USE_MULTIPROCESSING else 1
        )
        
        max_items = 10000 # MAX_SAMPLES

        results = processor.process_path(
            input_path=input_path,
            output_dir=output_dir,
            max_items=max_items
        )
        
        print("\n✅ Анализ успешно завершён!")
        
    except Exception as e:
        logger.error(f"❌ Произошла критическая ошибка: {e}", exc_info=True)

if __name__ == '__main__':
    run_analysis()
