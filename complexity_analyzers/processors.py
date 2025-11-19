# complexity_analysis/processors.py

import logging
import time
from pathlib import Path
import json
from typing import List, Dict, Any, Optional
import statistics
import traceback

logger = logging.getLogger(__name__)

class ComplexityProcessor:
    """Процессор для анализа сложности, поддерживающий файлы и датасеты."""
    
    def __init__(self, analyzers_to_use: List[str], max_workers: int = 1):
        self.analyzers_to_use = analyzers_to_use
        self.max_workers = max_workers
        logger.info(f"🔧 Инициализация ComplexityProcessor с анализаторами: {analyzers_to_use}")
        self._validate_analyzers()

    def _validate_analyzers(self):
        """Проверяет доступность анализаторов с детальным логированием."""
        logger.info("🔍 Валидация анализаторов...")
        try:
            from complexity_analyzers import create_analyzer
            self.create_analyzer = create_analyzer
            logger.info("✅ Импорт create_analyzer успешен")
        except ImportError as e:
            logger.error(f"❌ Не удалось импортировать `create_analyzer`: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

        # Тестируем создание каждого анализатора
        for analyzer_name in self.analyzers_to_use:
            try:
                logger.info(f"🧪 Тестирование создания анализатора: {analyzer_name}")
                analyzer = self.create_analyzer(analyzer_name)
                logger.info(f"✅ Анализатор {analyzer_name} создан успешно: {type(analyzer).__name__}")
                
                # Проверяем доступность
                if hasattr(analyzer, 'is_available'):
                    is_available = analyzer.is_available()
                    logger.info(f"📊 Анализатор {analyzer_name} доступен: {is_available}")
                else:
                    logger.warning(f"⚠️ Анализатор {analyzer_name} не имеет метода is_available()")
                    
            except Exception as e:
                logger.error(f"❌ Ошибка создания анализатора {analyzer_name}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")

    def load_jsonl(self, filepath: Path) -> List[Dict[str, Any]]:
        """Загружает данные из JSONL файла с пропагацией асимптотики."""
        data, complexity_cache = [], {}
        logger.info(f"📖 Чтение датасета из {filepath}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip(): 
                    continue
                try:
                    sample = json.loads(line)
                    task_id = sample.get('problem_id') or sample.get('problem', 'unknown')
                    
                    if sample.get('complexity'):
                        complexity_cache[task_id] = sample['complexity']
                    elif task_id in complexity_cache:
                        sample['complexity'] = complexity_cache[task_id]
                    
                    data.append(sample)
                    
                    # Логируем первые несколько образцов
                    if line_num <= 3:
                        logger.info(f"📝 Образец {line_num}: problem_id={sample.get('problem_id')}, "
                                  f"complexity={sample.get('complexity')}, src_length={len(sample.get('src', ''))}")
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ Не удалось прочитать строку {line_num} в {filepath}: {e}")
                    
        logger.info(f"✅ Загружено {len(data)} образцов из {len(complexity_cache)} уникальных задач")
        return data

    def analyze_single_item(self, item: Dict, analyzer_name: str, item_index: int = None) -> Dict:
        """Анализирует один элемент с детальным логированием."""
        start_time = time.time()
        
        # Подробное логирование входных данных (только для первых 3)
        if item_index is not None and item_index < 3:
            logger.info(f"🔬 Анализ образца #{item_index} анализатором {analyzer_name}")
            logger.info(f"   Problem ID: {item.get('problem_id', 'unknown')}")
            logger.info(f"   True complexity: {item.get('complexity', 'unknown')}")
            logger.info(f"   Source code length: {len(item.get('src', ''))}")
            if item.get('src'):
                logger.info(f"   Source preview: {item.get('src')[:100]}...")
        
        source_code = item.get('src') or item.get('code')
        
        if not source_code:
            logger.error(f"❌ Нет исходного кода в образце #{item_index}")
            return {
                'success': False, 
                'errors': ['Нет исходного кода'],
                'analysis_time': time.time() - start_time,
                'analyzer_name': analyzer_name
            }

        try:
            # Создание анализатора
            logger.debug(f"🏭 Создание анализатора {analyzer_name}...")
            analyzer = self.create_analyzer(analyzer_name)
            logger.debug(f"✅ Анализатор создан: {type(analyzer).__name__}")
            
            # Создание контекста
            logger.debug(f"📋 Создание контекста анализа...")
            from complexity_analyzers.core.base import AnalysisContext
            
            context = AnalysisContext(
                source_code=source_code,
                language='python',
                timeout=60,
                debug_mode=False
            )
            logger.debug(f"✅ Контекст создан")
            
            # Собственно анализ
            logger.debug(f"🚀 Запуск анализа...")
            result = analyzer.analyze(context)
            logger.debug(f"✅ Анализ завершён")
            
            analysis_time = time.time() - start_time
            
            # ИСПРАВЛЕНО: Правильное извлечение complexity класса
            complexity_class = result.complexity_class
            
            # Получаем строковое представление сложности
            if hasattr(complexity_class, 'notation'):
                # Если есть атрибут notation (из твоего enums.py)
                predicted_complexity = complexity_class.notation
            elif hasattr(complexity_class, 'value'):
                # Если value - это кортеж, берём первый элемент (notation)
                if isinstance(complexity_class.value, (tuple, list)):
                    predicted_complexity = complexity_class.value[0]  # "O(1)"
                else:
                    predicted_complexity = str(complexity_class.value)
            elif hasattr(complexity_class, 'class_name'):
                # Используем class_name
                predicted_complexity = complexity_class.class_name
            else:
                # Fallback - преобразуем в строку
                predicted_complexity = str(complexity_class)
            
            # Детальное логирование результата (только для первых 3)
            if item_index is not None and item_index < 3:
                logger.info(f"📊 Результат анализа #{item_index}:")
                logger.info(f"   Predicted complexity: {predicted_complexity}")
                logger.info(f"   Confidence: {result.confidence}")
                logger.info(f"   Errors: {result.errors}")
                logger.info(f"   Analysis time: {analysis_time:.4f}s")

            # Проверка на успешность
            from complexity_analyzers.core.enums import ComplexityClass
            success = (
                result.confidence >= 0.3 and
                result.complexity_class != ComplexityClass.UNKNOWN and
                not result.errors
            )
            
            if not success:
                logger.warning(f"⚠️ Анализ #{item_index} неуспешен: confidence={result.confidence}, "
                            f"class={result.complexity_class}, errors={result.errors}")
            
            return {
                'file_path': item.get('path', f"problem_{item.get('problem_id', 'unknown')}"),
                'analyzer_name': analyzer_name,
                'true_complexity': item.get('complexity'),
                'predicted_complexity': predicted_complexity,  # Теперь строка
                'confidence': result.confidence,
                'analysis_time': analysis_time,
                'success': success,
                'errors': result.errors,
                'warnings': getattr(result, 'warnings', [])
            }
            
        except Exception as e:
            logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА в анализе #{item_index} анализатором {analyzer_name}: {e}")
            logger.error(f"Полный traceback: {traceback.format_exc()}")
            
            return {
                'file_path': item.get('path', 'unknown'), 
                'analyzer_name': analyzer_name,
                'success': False, 
                'errors': [str(e)], 
                'analysis_time': time.time() - start_time,
                'true_complexity': item.get('complexity'),
                'predicted_complexity': 'ERROR',
                'confidence': 0.0
            }



    def process_path(self, input_path: Path, output_dir: Path, max_items: Optional[int]) -> Dict:
        """Обрабатывает путь с детальным логированием."""
        logger.info(f"🗂️ Обработка пути: {input_path}")
        
        if input_path.suffix == '.jsonl':
            logger.info("📄 Обнаружен JSONL файл, запускаем режим датасета")
            items = self.load_jsonl(input_path)
            for item in items: 
                item['path'] = 'from_dataset'
        else:
            logger.info("📁 Обнаружена директория/файл, запускаем файловый режим")
            files = self._find_files(input_path)
            items = [{'path': str(p), 'code': p.read_text(encoding='utf-8')} for p in files]

        if max_items:
            logger.info(f"✂️ Ограничиваем до {max_items} элементов (было {len(items)})")
            items = items[:max_items]

        logger.info(f"📊 Найдено {len(items)} элементов для анализа")
        if not items: 
            logger.warning("⚠️ Нет элементов для анализа!")
            return {}

        all_results = {}
        for analyzer_name in self.analyzers_to_use:
            logger.info(f"🔍 Запуск анализатора: {analyzer_name}")
            
            analyzer_results = []
            for i, item in enumerate(items):
                if i % 100 == 0:  # Логируем каждый 100-й элемент
                    logger.info(f"   Обработано {i}/{len(items)} элементов...")
                    
                result = self.analyze_single_item(item, analyzer_name, i)
                analyzer_results.append(result)
            
            logger.info(f"✅ Анализатор {analyzer_name} завершён")
            all_results[analyzer_name] = self._aggregate_results(analyzer_results, analyzer_name)
            self._save_individual_results(analyzer_name, all_results[analyzer_name], output_dir)

        self._save_combined_summary(all_results, output_dir)
        return all_results

    def _find_files(self, path: Path) -> List[Path]:
        """Находит Python файлы для анализа."""
        logger.info(f"🔍 Поиск файлов в {path}")
        if path.is_file():
            files = [path] if path.suffix == '.py' else []
        elif path.is_dir():
            from config import RECURSIVE_SEARCH, EXCLUDED_PATTERNS
            pattern = "**/*.py" if RECURSIVE_SEARCH else "*.py"
            all_files = list(path.glob(pattern))
            files = [f for f in all_files if not any(p in str(f) for p in EXCLUDED_PATTERNS)]
        else:
            files = []
        
        logger.info(f"📁 Найдено {len(files)} Python файлов")
        return files

    def _aggregate_results(self, results: List[Dict], analyzer_name: str) -> Dict:
        """Агрегирует результаты для одного анализатора."""
        logger.info(f"📈 Агрегация результатов для {analyzer_name}")
        
        successful = [r for r in results if r.get('success')]
        failed = [r for r in results if not r.get('success')]
        
        logger.info(f"   Успешных: {len(successful)}/{len(results)}")
        logger.info(f"   Неудачных: {len(failed)}/{len(results)}")
        
        # Логируем причины неудач
        if failed and len(failed) <= 10:
            logger.info("   Примеры ошибок:")
            for i, fail in enumerate(failed[:5]):
                logger.info(f"     {i+1}. {fail.get('errors', ['Неизвестная ошибка'])}")
        
        times = [r['analysis_time'] for r in results if 'analysis_time' in r]
        confidences = [r['confidence'] for r in successful if 'confidence' in r]

        # ИСПРАВЛЕНО: Правильная обработка данных для метрик
        y_true = []
        y_pred = []
        
        for r in successful:
            true_val = r.get('true_complexity')
            pred_val = r.get('predicted_complexity')
            
            if true_val and pred_val:
                # Нормализуем true_complexity через маппинг
                from config import COMPLEXITY_MAPPING
                normalized_true = COMPLEXITY_MAPPING.get(true_val, true_val)
                
                y_true.append(normalized_true)
                y_pred.append(str(pred_val))

        logger.info(f"   Данных для метрик: true={len(y_true)}, pred={len(y_pred)}")
        
        # Логируем уникальные значения для диагностики
        if y_true and y_pred:
            unique_true = set(y_true)
            unique_pred = set(y_pred)
            logger.info(f"   Уникальные true: {unique_true}")
            logger.info(f"   Уникальные pred: {unique_pred}")

        metrics = {}
        if y_true and y_pred and len(y_true) == len(y_pred):
            try:
                from sklearn.metrics import accuracy_score, f1_score
                
                # Убеждаемся, что данные в правильном формате
                accuracy = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
                metrics['accuracy'] = float(accuracy)
                metrics['f1_weighted'] = float(f1)
                
                logger.info(f"   Метрики: accuracy={accuracy:.3f}, f1={f1:.3f}")
                
            except Exception as e:
                logger.error(f"❌ Ошибка вычисления метрик: {e}")
                logger.error(f"   Пример true: {y_true[:5] if y_true else 'пусто'}")
                logger.error(f"   Пример pred: {y_pred[:5] if y_pred else 'пусто'}")
        else:
            logger.warning("⚠️ Недостаточно данных для вычисления метрик")
        
        return {
            'total_items': len(results),
            'successful_analyses': len(successful),
            'metrics': metrics,
            'time_statistics': {
                'total_time': sum(times),
                'avg_time_per_item': statistics.mean(times) if times else 0,
                'items_per_second': len(results) / sum(times) if sum(times) > 0 else 0
            },
            'confidence_statistics': {
                'avg_confidence': statistics.mean(confidences) if confidences else 0
            },
            'individual_results': results
        }


    def _save_individual_results(self, analyzer_name: str, results: Dict, output_dir: Path):
        """Сохраняет детальные результаты для одного анализатора."""
        path = output_dir / "individual_results"
        path.mkdir(exist_ok=True)
        filepath = path / f"{analyzer_name}_results.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"💾 Результаты {analyzer_name} сохранены: {filepath}")

    def _save_combined_summary(self, all_results: Dict, output_dir: Path):
        """Сохраняет сводку по всем анализаторам."""
        summary_data = {
            analyzer: {
                'total_items': res.get('total_items', 0),
                'successful_analyses': res.get('successful_analyses', 0),
                'accuracy': res.get('metrics', {}).get('accuracy', 0),
                'f1_weighted': res.get('metrics', {}).get('f1_weighted', 0),
                'avg_time_per_item': res.get('time_statistics', {}).get('avg_time_per_item', 0)
            } for analyzer, res in all_results.items()
        }
        
        path = output_dir / "comparison_results"
        path.mkdir(exist_ok=True)
        filepath = path / "summary.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        logger.info(f"📋 Сводка сохранена: {filepath}")
