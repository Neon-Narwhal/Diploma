# complexity_analyzers/processors.py

import logging
import time
from pathlib import Path
import json
from typing import List, Dict, Any, Optional
import statistics
import traceback
import re
from multiprocessing import Pool, TimeoutError
from functools import partial

logger = logging.getLogger(__name__)


def normalize_complexity(value: str) -> Optional[str]:
    """
    Нормализует синтаксис выражений сложности для точного сравнения.
    
    Преобразования:
    - Убирает пробелы: "O(n + m)" -> "O(n+m)"
    - Унифицирует степени: "O(n**2)" -> "O(n^2)"
    - Приводит к нижнему регистру переменные
    - Сортирует переменные в произведениях: "O(m*n)" -> "O(n*m)"
    
    Примеры:
        "O(n**2)" -> "O(n^2)"
        "O(n * m)" -> "O(n*m)"
        "O(m*n)" -> "O(n*m)"
    """
    if not value:
        return None
    
    # Очистка и базовая нормализация
    value = value.strip()
    
    # Приводим внутреннюю часть к нижнему регистру
    if value.startswith('O(') and value.endswith(')'):
        inner = value[2:-1]
        inner = inner.lower()
        value = f'O({inner})'
    
    # Убираем все пробелы
    value = value.replace(' ', '')
    
    # Унификация степеней: ** -> ^
    value = value.replace('**', '^')
    
    # Сортировка переменных в произведениях
    value = _sort_product_terms(value)
    
    return value


def _sort_product_terms(complexity: str) -> str:
    """
    Сортирует переменные в произведениях для канонической формы.
    
    Примеры:
        "O(m*n)" -> "O(n*m)"
        "O(k*m*n)" -> "O(k*m*n)"
    """
    if '*' not in complexity:
        return complexity
    
    def sort_product(term: str) -> str:
        """Сортирует один терм-произведение"""
        factors = term.split('*')
        sorted_factors = sorted(factors)
        return '*'.join(sorted_factors)
    
    # Извлекаем внутреннюю часть O(...)
    if complexity.startswith('O(') and complexity.endswith(')'):
        inner = complexity[2:-1]
        
        # Разбиваем по + (простой случай без вложенных скобок)
        if '(' not in inner:
            terms = inner.split('+')
            sorted_terms = [sort_product(term) for term in terms]
            return f'O({"+".join(sorted_terms)})'
    
    return complexity


def group_complexity(value: str) -> str:
    """
    Маппинг точных классов на группы для честного сравнения анализаторов.
    
    47 классов → 10 групп:
    - constant: O(1)
    - logarithmic: O(logn), O(logn*logm)
    - linear: O(n), O(m), O(n+m)
    - linearithmic: O(nlogn), O(nlogn+m)
    - quadratic: O(n^2), O(n*m)
    - cubic: O(n^3), O(n^2*m)
    - polynomial: O(n^k) где k > 3
    - exponential: O(2^n)
    - factorial: O(n!)
    - unknown: неопределенные
    
    Примеры:
        "O(n*m)" -> "quadratic"
        "O(n+m)" -> "linear"
        "O(nlogn+m)" -> "linearithmic"
    """
    value = normalize_complexity(value)
    
    if not value:
        return 'unknown'
    
    # Constant: O(1)
    if value == 'O(1)':
        return 'constant'
    
    # Exponential: O(2^n), O(3^n), любое число в степени n
    if re.search(r'\d+\^n', value) or re.search(r'\d+\^m', value):
        return 'exponential'
    
    # Factorial: O(n!)
    if '!' in value:
        return 'factorial'
    
    # Cubic: O(n^3), O(n^2*m), O(n*m*k) (3+ переменных)
    if 'n^3' in value or 'm^3' in value or 'k^3' in value:
        return 'cubic'
    
    # Произведение 3+ переменных → cubic
    if value.count('*') >= 2:
        return 'cubic'
    
    # O(n^2*m) и подобные → cubic
    if 'n^2' in value and '*' in value:
        return 'cubic'
    
    # Quadratic: O(n^2), O(n*m), O(m*n), O(n^2+m)
    if 'n^2' in value or 'm^2' in value or 'k^2' in value:
        return 'quadratic'
    
    # Произведение ровно двух переменных → quadratic
    if '*' in value and value.count('*') == 1 and '+' not in value:
        return 'quadratic'
    
    # Linearithmic: O(nlogn), O(nlogn+m), O(mlogm), O((n+m)log(n+m))
    # Проверяем паттерн: переменная сразу перед log
    if re.search(r'(n|m|k)log', value.replace(' ', '')):
        return 'linearithmic'
    
    # Linear: O(n), O(m), O(n+m), O(n+m+k)
    # Суммы без произведений ИЛИ одна переменная
    if ('+' in value and '*' not in value) or value in ['O(n)', 'O(m)', 'O(k)', 'O(l)']:
        return 'linear'
    
    # Logarithmic: O(logn), O(logn*logm), O(logn+logm)
    if 'log' in value and not re.search(r'(n|m|k)log', value):
        return 'logarithmic'
    
    # Polynomial: O(n^k) где k > 3, или сложные степени
    if '^' in value:
        # Пытаемся найти степень больше 3
        match = re.search(r'n\^(\d+)', value)
        if match:
            power = int(match.group(1))
            if power > 3:
                return 'polynomial'
        return 'polynomial'
    
    return 'unknown'


def _worker_analyze(item: Dict, item_index: int, analyzer_name: str) -> Dict:
    """Рабочая функция для параллелизации (глобальная)."""
    start_time = time.time()
    
    true_complexity = item.get('complexity')
    source_code = item.get('src') or item.get('code')
    
    if not source_code:
        return {
            'success': False,
            'errors': ['Нет кода'],
            'analysis_time': time.time() - start_time,
            'analyzer_name': analyzer_name,
            'true_complexity': true_complexity,
            'predicted_complexity': None,
            'confidence': 0.0,
        }

    try:
        from complexity_analyzers import create_analyzer
        from complexity_analyzers.core.base import AnalysisContext
        from complexity_analyzers.core.enums import ComplexityClass
        
        analyzer = create_analyzer(analyzer_name)
        
        context = AnalysisContext(
            source_code=source_code,
            language='python',
            timeout=5,  # Внутренний таймаут
            debug_mode=False
        )
        
        result = analyzer.analyze(context)
        analysis_time = time.time() - start_time
        
        # Извлечение предсказания
        complexity_class = result.complexity_class
        
        if hasattr(complexity_class, 'to_notation'):
            raw_prediction = complexity_class.to_notation()
        elif hasattr(complexity_class, 'notation'):
            raw_prediction = complexity_class.notation
        else:
            raw_prediction = str(complexity_class)
        
        predicted_complexity = normalize_complexity(raw_prediction)
        
        success = (
            result.confidence >= 0.3 and
            result.complexity_class != ComplexityClass.UNKNOWN and
            not result.errors
        )
        
        return {
            'file_path': item.get('path', f"problem_{item.get('problem_id', 'unknown')}"),
            'analyzer_name': analyzer_name,
            'true_complexity': true_complexity,
            'predicted_complexity': predicted_complexity,
            'confidence': result.confidence,
            'analysis_time': analysis_time,
            'success': success,
            'errors': result.errors,
            'warnings': getattr(result, 'warnings', [])
        }
        
    except Exception as e:
        return {
            'file_path': item.get('path', 'unknown'),
            'analyzer_name': analyzer_name,
            'success': False,
            'errors': [str(e)],
            'analysis_time': time.time() - start_time,
            'true_complexity': true_complexity,
            'predicted_complexity': None,
            'confidence': 0.0
        }

class ComplexityProcessor:
    """Процессор для анализа сложности с поддержкой многовариантных классов."""
    
    def __init__(self, analyzers_to_use: List[str], max_workers: int = 1):
        self.analyzers_to_use = analyzers_to_use
        self.max_workers = max_workers
        logger.info(f"🔧 Инициализация ComplexityProcessor")
        logger.info(f"   Анализаторы: {analyzers_to_use}")
        self._validate_analyzers()

    def _validate_analyzers(self):
        """Проверяет доступность анализаторов."""
        logger.info("🔍 Валидация анализаторов...")
        try:
            from complexity_analyzers import create_analyzer
            self.create_analyzer = create_analyzer
            logger.info("✅ Импорт create_analyzer успешен")
        except ImportError as e:
            logger.error(f"❌ Не удалось импортировать create_analyzer: {e}")
            logger.error(traceback.format_exc())
            raise

        for analyzer_name in self.analyzers_to_use:
            try:
                logger.info(f"🧪 Тест: {analyzer_name}")
                analyzer = self.create_analyzer(analyzer_name)
                logger.info(f"✅ {analyzer_name}: {type(analyzer).__name__}")
            except Exception as e:
                logger.error(f"❌ Ошибка {analyzer_name}: {e}")
                logger.error(traceback.format_exc())

    def load_jsonl(self, filepath: Path) -> List[Dict[str, Any]]:
        """
        Загружает данные из JSONL с нормализацией сложности.
        
        Поддерживает все форматы:
        - time_complexity: "O(n*m)", "O(n**2)", и т.д.
        - time_complexity_original: fallback
        """
        data = []
        complexity_cache = {}
        
        logger.info(f"📖 Чтение: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    sample = json.loads(line)
                    
                    # ПРИОРИТЕТ: оригинальное поле с точным выражением
                    raw_complexity = (
                        sample.get('time_complexity') or
                        sample.get('time_complexity_original') or
                        sample.get('complexity')
                    )
                    
                    # Нормализуем синтаксис
                    normalized = normalize_complexity(raw_complexity)
                    
                    # Сохраняем
                    sample['complexity'] = normalized
                    
                    # Кэш
                    task_id = sample.get('problem_id') or sample.get('problem', 'unknown')
                    if normalized:
                        complexity_cache[task_id] = normalized
                    
                    # Код
                    if not sample.get('src'):
                        sample['src'] = sample.get('code', '')
                    
                    data.append(sample)
                    
                    # Лог первых 3
                    if line_num <= 3:
                        logger.info(
                            f"📝 #{line_num}: "
                            f"id={sample.get('problem_id')}, "
                            f"raw='{raw_complexity}', "
                            f"norm='{normalized}'"
                        )
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ Строка {line_num}: {e}")
        
        # Статистика
        labeled = sum(1 for s in data if s.get('complexity'))
        logger.info(f"✅ Загружено {len(data)} образцов")
        logger.info(f"   С метками: {labeled}/{len(data)}")
        logger.info(f"   Уникальных задач: {len(complexity_cache)}")
        
        # Распределение
        complexity_counts = {}
        for s in data:
            c = s.get('complexity')
            if c:
                complexity_counts[c] = complexity_counts.get(c, 0) + 1
        
        # Топ-10
        top_classes = sorted(complexity_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("   Топ-10 классов:")
        for cls, count in top_classes:
            logger.info(f"     {cls}: {count}")
        
        logger.info(f"   Всего уникальных: {len(complexity_counts)}")
        
        return data

    def analyze_single_item(self, item: Dict, analyzer_name: str, item_index: int = None) -> Dict:
        """Анализирует один элемент."""
        start_time = time.time()
        
        true_complexity = item.get('complexity')
        
        # Лог для первых 3
        if item_index is not None and item_index < 3:
            logger.info(f"🔬 Анализ #{item_index}: {analyzer_name}")
            logger.info(f"   ID: {item.get('problem_id', 'unknown')}")
            logger.info(f"   True: {true_complexity}")
            logger.info(f"   Src: {len(item.get('src', ''))} chars")
        
        source_code = item.get('src') or item.get('code')
        
        if not source_code:
            return {
                'success': False,
                'errors': ['Нет кода'],
                'analysis_time': time.time() - start_time,
                'analyzer_name': analyzer_name,
                'true_complexity': true_complexity,
                'predicted_complexity': None,
                'confidence': 0.0,
            }

        try:
            analyzer = self.create_analyzer(analyzer_name)

            
            from complexity_analyzers.core.base import AnalysisContext
            context = AnalysisContext(
                source_code=source_code,
                language='python',
                timeout=60,
                debug_mode=False
            )
            
            result = analyzer.analyze(context)
            analysis_time = time.time() - start_time
            
            # Извлекаем предсказание из ComplexityClass
            complexity_class = result.complexity_class
            
            # СТАНДАРТИЗАЦИЯ: всегда через .to_notation()
            if hasattr(complexity_class, 'to_notation'):
                raw_prediction = complexity_class.to_notation()
            elif hasattr(complexity_class, 'notation'):
                raw_prediction = complexity_class.notation
            elif hasattr(complexity_class, 'value'):
                if isinstance(complexity_class.value, (tuple, list)):
                    raw_prediction = complexity_class.value[0] if len(complexity_class.value) > 0 else str(complexity_class)
                else:
                    raw_prediction = str(complexity_class.value)
            else:
                raw_prediction = str(complexity_class)
            
            # Нормализуем предсказание
            predicted_complexity = normalize_complexity(raw_prediction)
            
            # Лог для первых 3
            if item_index is not None and item_index < 3:
                logger.info(f"📊 Результат #{item_index}:")
                logger.info(f"   True: {true_complexity}")
                logger.info(f"   Pred (raw): {raw_prediction}")
                logger.info(f"   Pred (norm): {predicted_complexity}")
                logger.info(f"   Match: {true_complexity == predicted_complexity}")
                logger.info(f"   Confidence: {result.confidence:.2f}")
                logger.info(f"   Time: {analysis_time:.4f}s")

            # Успешность
            from complexity_analyzers.core.enums import ComplexityClass
            success = (
                result.confidence >= 0.3 and
                result.complexity_class != ComplexityClass.UNKNOWN and
                not result.errors
            )
            
            return {
                'file_path': item.get('path', f"problem_{item.get('problem_id', 'unknown')}"),
                'analyzer_name': analyzer_name,
                'true_complexity': true_complexity,
                'predicted_complexity': predicted_complexity,
                'confidence': result.confidence,
                'analysis_time': analysis_time,
                'success': success,
                'errors': result.errors,
                'warnings': getattr(result, 'warnings', [])
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка #{item_index}: {e}")
            logger.error(traceback.format_exc())
            
            return {
                'file_path': item.get('path', 'unknown'),
                'analyzer_name': analyzer_name,
                'success': False,
                'errors': [str(e)],
                'analysis_time': time.time() - start_time,
                'true_complexity': true_complexity,
                'predicted_complexity': None,
                'confidence': 0.0
            }

    def process_path(self, input_path: Path, output_dir: Path, max_items: Optional[int]) -> Dict:
        """Обрабатывает путь с параллелизацией."""
        logger.info(f"🗂️ Обработка: {input_path}")
        
        if input_path.suffix == '.jsonl':
            logger.info("📄 Режим: JSONL датасет")
            items = self.load_jsonl(input_path)
            for item in items:
                item['path'] = 'from_dataset'
        else:
            logger.info("📁 Режим: файловый")
            files = self._find_files(input_path)
            items = [{'path': str(p), 'code': p.read_text(encoding='utf-8')} for p in files]

        if max_items:
            logger.info(f"✂️ Ограничение: {max_items} из {len(items)}")
            items = items[:max_items]

        logger.info(f"📊 Элементов для анализа: {len(items)}")
        
        if not items:
            logger.warning("⚠️ Нет элементов")
            return {}

        all_results = {}
        for analyzer_name in self.analyzers_to_use:
            logger.info(f"\n🔍 Анализатор: {analyzer_name}")
            logger.info("=" * 60)
            
            # Параллельная обработка
            analyzer_results = self._analyze_parallel(items, analyzer_name)
            
            logger.info(f"✅ {analyzer_name} завершён")
            all_results[analyzer_name] = self._aggregate_results(analyzer_results, analyzer_name)
            self._save_individual_results(analyzer_name, all_results[analyzer_name], output_dir)

        self._save_combined_summary(all_results, output_dir)
        return all_results
    
    def _analyze_parallel(self, items: List[Dict], analyzer_name: str) -> List[Dict]:
        """Параллельный анализ."""
        num_workers = self.max_workers if self.max_workers > 1 else 4
        timeout_per_item = 10  # секунд
        
        logger.info(f"🚀 Запуск {num_workers} процессов с таймаутом {timeout_per_item}s")
        
        results = []
        
        with Pool(processes=num_workers) as pool:
            async_results = []
            
            for i, item in enumerate(items):
                async_result = pool.apply_async(
                    _worker_analyze,
                    (item, i, analyzer_name)
                )
                async_results.append((i, async_result))
            
            for i, async_result in async_results:
                try:
                    result = async_result.get(timeout=timeout_per_item)
                    results.append(result)
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"   Прогресс: {i + 1}/{len(items)}...")
                        
                except TimeoutError:
                    logger.warning(f"⏱️ Таймаут на образце #{i}")
                    results.append({
                        'success': False,
                        'errors': ['Timeout exceeded'],
                        'analysis_time': timeout_per_item,
                        'analyzer_name': analyzer_name,
                        'true_complexity': items[i].get('complexity'),
                        'predicted_complexity': None,
                        'confidence': 0.0,
                    })
                except Exception as e:
                    logger.error(f"❌ Ошибка #{i}: {e}")
                    results.append({
                        'success': False,
                        'errors': [str(e)],
                        'analysis_time': 0.0,
                        'analyzer_name': analyzer_name,
                        'true_complexity': items[i].get('complexity'),
                        'predicted_complexity': None,
                        'confidence': 0.0,
                    })
        
        return results

    def _analyze_single_wrapper(self, item: Dict, item_index: int, analyzer_name: str) -> Dict:
        """Обёртка для параллельного вызова (без self)."""
        return self.analyze_single_item(item, analyzer_name, item_index)

    def _find_files(self, path: Path) -> List[Path]:
        """Находит Python файлы."""
        logger.info(f"🔍 Поиск файлов: {path}")
        
        if path.is_file():
            files = [path] if path.suffix == '.py' else []
        elif path.is_dir():
            try:
                from config import RECURSIVE_SEARCH, EXCLUDED_PATTERNS
            except ImportError:
                RECURSIVE_SEARCH = True
                EXCLUDED_PATTERNS = ['test_', '__pycache__']
            
            pattern = "**/*.py" if RECURSIVE_SEARCH else "*.py"
            all_files = list(path.glob(pattern))
            files = [f for f in all_files if not any(p in str(f) for p in EXCLUDED_PATTERNS)]
        else:
            files = []
        
        logger.info(f"📁 Найдено: {len(files)} файлов")
        return files

    def _aggregate_results(self, results: List[Dict], analyzer_name: str) -> Dict:
        """
        Агрегация результатов с двумя уровнями метрик:
        - Fine-grained: 47 точных классов (для ML/Hybrid)
        - Coarse-grained: 10 групп (для честного сравнения всех анализаторов)
        """
        logger.info(f"\n📈 Агрегация для {analyzer_name}")
        logger.info("=" * 60)
        
        successful = [r for r in results if r.get('success')]
        failed = [r for r in results if not r.get('success')]
        
        logger.info(f"Успешных: {len(successful)}/{len(results)}")
        logger.info(f"Неудачных: {len(failed)}/{len(results)}")
        
        # === FINE-GRAINED МЕТРИКИ (47 классов) ===
        y_true_fine = []
        y_pred_fine = []
        
        for r in successful:
            true_val = r.get('true_complexity')
            pred_val = r.get('predicted_complexity')
            
            if true_val and pred_val:
                y_true_fine.append(true_val)
                y_pred_fine.append(pred_val)
        
        logger.info(f"Валидных пар (fine): {len(y_true_fine)}")
        
        # === COARSE-GRAINED МЕТРИКИ (10 групп) ===
        y_true_coarse = [group_complexity(t) for t in y_true_fine]
        y_pred_coarse = [group_complexity(p) for p in y_pred_fine]
        
        logger.info(f"\nПример группировки (первые 10):")
        for true_f, true_c, pred_f, pred_c in zip(y_true_fine[:10], y_true_coarse[:10], y_pred_fine[:10], y_pred_coarse[:10]):
            logger.info(f"  {true_f} -> {true_c} | {pred_f} -> {pred_c}")

        # Распределение grouped классов
        from collections import Counter
        logger.info(f"\nРаспределение true (grouped): {Counter(y_true_coarse)}")
        logger.info(f"Распределение pred (grouped): {Counter(y_pred_coarse)}")
        # Вычисляем метрики
        metrics = {}
        
        if len(y_true_fine) > 0:
            try:
                from sklearn.metrics import accuracy_score, f1_score, classification_report
                
                # Fine-grained метрики (47 классов)
                fine_accuracy = accuracy_score(y_true_fine, y_pred_fine)
                fine_f1 = f1_score(y_true_fine, y_pred_fine, average='weighted', zero_division=0)
                
                # Coarse-grained метрики (10 групп)
                coarse_accuracy = accuracy_score(y_true_coarse, y_pred_coarse)
                coarse_f1 = f1_score(y_true_coarse, y_pred_coarse, average='weighted', zero_division=0)
                
                metrics = {
                    # Fine-grained (47 классов)
                    'fine_accuracy': float(fine_accuracy),
                    'fine_f1': float(fine_f1),
                    'fine_unique_true': len(set(y_true_fine)),
                    'fine_unique_pred': len(set(y_pred_fine)),
                    
                    # Coarse-grained (10 групп) - ГЛАВНЫЕ МЕТРИКИ
                    'coarse_accuracy': float(coarse_accuracy),
                    'coarse_f1': float(coarse_f1),
                    'coarse_unique_true': len(set(y_true_coarse)),
                    'coarse_unique_pred': len(set(y_pred_coarse)),
                }
                
                logger.info(f"\n📊 FINE-GRAINED МЕТРИКИ (47 классов):")
                logger.info(f"   Accuracy: {fine_accuracy:.3f}")
                logger.info(f"   F1: {fine_f1:.3f}")
                logger.info(f"   Уникальных true: {metrics['fine_unique_true']}")
                logger.info(f"   Уникальных pred: {metrics['fine_unique_pred']}")
                
                logger.info(f"\n📊 COARSE-GRAINED МЕТРИКИ (10 групп) — ОСНОВНЫЕ:")
                logger.info(f"   Accuracy: {coarse_accuracy:.3f}")
                logger.info(f"   F1: {coarse_f1:.3f}")
                logger.info(f"   Уникальных true: {metrics['coarse_unique_true']}")
                logger.info(f"   Уникальных pred: {metrics['coarse_unique_pred']}")
                
                # Classification report для coarse-grained
                if len(set(y_true_coarse)) <= 20:
                    report = classification_report(
                        y_true_coarse,
                        y_pred_coarse,
                        zero_division=0
                    )
                    logger.info(f"\n📋 Classification Report (Coarse):\n{report}")
                
                # Анализ ошибок (fine-grained)
                from collections import Counter
                
                mismatches_fine = []
                for true_val, pred_val in zip(y_true_fine, y_pred_fine):
                    if true_val != pred_val:
                        mismatches_fine.append((true_val, pred_val))
                
                # Топ-10 ошибок (fine)
                mismatch_counter_fine = Counter(mismatches_fine)
                top_errors_fine = mismatch_counter_fine.most_common(10)
                
                logger.info("\n🔴 Топ-10 ошибок (fine-grained, true -> pred):")
                for (true_val, pred_val), count in top_errors_fine:
                    logger.info(f"   {true_val} -> {pred_val}: {count} раз")
                
                # Анализ ошибок (coarse-grained)
                mismatches_coarse = []
                for true_g, pred_g in zip(y_true_coarse, y_pred_coarse):
                    if true_g != pred_g:
                        mismatches_coarse.append((true_g, pred_g))
                
                # Топ-5 групповых ошибок
                mismatch_counter_coarse = Counter(mismatches_coarse)
                top_errors_coarse = mismatch_counter_coarse.most_common(5)
                
                logger.info("\n🔴 Топ-5 ошибок (coarse-grained):")
                for (true_g, pred_g), count in top_errors_coarse:
                    logger.info(f"   {true_g} -> {pred_g}: {count} раз")
                
            except Exception as e:
                logger.error(f"❌ Ошибка вычисления метрик: {e}")
                logger.error(traceback.format_exc())
        else:
            logger.warning("⚠️ Недостаточно данных для метрик")
        
        # Статистика времени
        times = [r['analysis_time'] for r in results if 'analysis_time' in r]
        confidences = [r['confidence'] for r in successful if 'confidence' in r]
        
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
                'avg_confidence': statistics.mean(confidences) if confidences else 0,
                'min_confidence': min(confidences) if confidences else 0,
                'max_confidence': max(confidences) if confidences else 0
            },
            'individual_results': results
        }


    def _save_individual_results(self, analyzer_name: str, results: Dict, output_dir: Path):
        """Сохраняет детальные результаты."""
        path = output_dir / "individual_results"
        path.mkdir(exist_ok=True, parents=True)
        filepath = path / f"{analyzer_name}_results.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"💾 Сохранено: {filepath}")

    def _save_combined_summary(self, all_results: Dict, output_dir: Path):
        """Сохраняет сводку со всеми метриками"""
        summary_data = {
            analyzer: {
                'total_items': res.get('total_items', 0),
                'successful_analyses': res.get('successful_analyses', 0),
                
                # Fine-grained метрики (47 классов)
                'fine_accuracy': res.get('metrics', {}).get('fine_accuracy', 0),
                'fine_f1': res.get('metrics', {}).get('fine_f1', 0),
                'fine_unique_true': res.get('metrics', {}).get('fine_unique_true', 0),
                'fine_unique_pred': res.get('metrics', {}).get('fine_unique_pred', 0),
                
                # Coarse-grained метрики (10 групп) - ГЛАВНЫЕ
                'coarse_accuracy': res.get('metrics', {}).get('coarse_accuracy', 0),
                'coarse_f1': res.get('metrics', {}).get('coarse_f1', 0),
                'coarse_unique_true': res.get('metrics', {}).get('coarse_unique_true', 0),
                'coarse_unique_pred': res.get('metrics', {}).get('coarse_unique_pred', 0),
                
                # Производительность
                'avg_time_per_item': res.get('time_statistics', {}).get('avg_time_per_item', 0),
                'avg_confidence': res.get('confidence_statistics', {}).get('avg_confidence', 0)
            } for analyzer, res in all_results.items()
        }
        
        path = output_dir / "comparison_results"
        path.mkdir(exist_ok=True, parents=True)
        filepath = path / "summary.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 Сводка: {filepath}")

