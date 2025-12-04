"""
Универсальный батчевый процессор с поддержкой параллелизации.
"""

import time
import multiprocessing as mp
from typing import List, Dict, Any, Callable, Optional
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class ProcessingResult:
    """Результат обработки одного образца"""
    success: bool
    data: Any
    error: Optional[str] = None
    processing_time: float = 0.0
    timeout: bool = False


class BatchProcessor:
    """
    Универсальный процессор для батчевой обработки с поддержкой:
    - Параллелизации (multiprocessing)
    - Таймаутов
    - Progress bar
    - Обработки ошибок
    """
    
    def __init__(self,
                 batch_size: int = 32,
                 n_workers: int = 4,
                 timeout_per_sample: Optional[float] = None,
                 show_progress: bool = True,
                 skip_on_error: bool = True):
        """
        Args:
            batch_size: Размер батча
            n_workers: Количество параллельных процессов
            timeout_per_sample: Таймаут на обработку одного образца (секунды)
            show_progress: Показывать progress bar
            skip_on_error: Пропускать ошибки и продолжать
        """
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.timeout_per_sample = timeout_per_sample
        self.show_progress = show_progress
        self.skip_on_error = skip_on_error
    
    def process(self,
                items: List[Any],
                process_fn: Callable[[Any], Any],
                parallel: bool = True) -> List[ProcessingResult]:
        """
        Обработка списка элементов.
        
        Args:
            items: Список элементов для обработки
            process_fn: Функция обработки одного элемента
            parallel: Использовать параллелизацию
        
        Returns:
            Список результатов обработки
        """
        if not items:
            return []
        
        print(f"\nProcessing {len(items)} items...")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Workers: {self.n_workers}")
        print(f"  Parallel: {parallel}")
        print(f"  Timeout: {self.timeout_per_sample}s per sample")
        
        if parallel and self.n_workers > 1:
            results = self._process_parallel(items, process_fn)
        else:
            results = self._process_sequential(items, process_fn)
        
        # Статистика
        self._print_stats(results)
        
        return results
    
    def _process_sequential(self,
                           items: List[Any],
                           process_fn: Callable) -> List[ProcessingResult]:
        """Последовательная обработка"""
        results = []
        
        iterator = tqdm(items, desc="Processing") if self.show_progress else items
        
        for item in iterator:
            result = self._process_single(item, process_fn)
            results.append(result)
            
            if not result.success and not self.skip_on_error:
                break
        
        return results
    
    def _process_parallel(self,
                         items: List[Any],
                         process_fn: Callable) -> List[ProcessingResult]:
        """Параллельная обработка"""
        results = [None] * len(items)
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Отправляем задачи
            future_to_idx = {
                executor.submit(self._process_single_safe, item, process_fn): idx
                for idx, item in enumerate(items)
            }
            
            # Собираем результаты с progress bar
            if self.show_progress:
                futures = tqdm(as_completed(future_to_idx), 
                              total=len(items), 
                              desc="Processing")
            else:
                futures = as_completed(future_to_idx)
            
            for future in futures:
                idx = future_to_idx[future]
                try:
                    result = future.result(timeout=self.timeout_per_sample)
                    results[idx] = result
                except TimeoutError:
                    results[idx] = ProcessingResult(
                        success=False,
                        data=None,
                        error="Timeout",
                        timeout=True
                    )
                except Exception as e:
                    results[idx] = ProcessingResult(
                        success=False,
                        data=None,
                        error=str(e)
                    )
        
        return results
    
    def _process_single(self, item: Any, process_fn: Callable) -> ProcessingResult:
        """Обработка одного элемента с замером времени"""
        start_time = time.time()
        
        try:
            # Применяем таймаут если указан
            if self.timeout_per_sample:
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Processing timeout")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(self.timeout_per_sample))
            
            # Обработка
            data = process_fn(item)
            
            if self.timeout_per_sample:
                signal.alarm(0)  # Отключаем таймаут
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                data=data,
                processing_time=processing_time
            )
        
        except TimeoutError:
            return ProcessingResult(
                success=False,
                data=None,
                error="Timeout",
                timeout=True,
                processing_time=time.time() - start_time
            )
        
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=None,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    def _process_single_safe(self, item: Any, process_fn: Callable) -> ProcessingResult:
        """Обёртка для безопасной обработки в subprocess"""
        try:
            return self._process_single(item, process_fn)
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=None,
                error=f"Subprocess error: {str(e)}"
            )
    
    def _print_stats(self, results: List[ProcessingResult]) -> None:
        """Вывод статистики обработки"""
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        timeouts = sum(1 for r in results if r.timeout)
        
        processing_times = [r.processing_time for r in results if r.processing_time > 0]
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        print("\n" + "=" * 60)
        print("PROCESSING STATISTICS")
        print("=" * 60)
        print(f"Total samples:    {total}")
        print(f"Successful:       {successful} ({successful/total*100:.1f}%)")
        print(f"Failed:           {failed} ({failed/total*100:.1f}%)")
        print(f"Timeouts:         {timeouts} ({timeouts/total*100:.1f}%)")
        print(f"Avg time/sample:  {avg_time:.3f}s")
        print("=" * 60)


class BatchGenerator:
    """Генератор батчей для ленивой обработки"""
    
    def __init__(self, items: List[Any], batch_size: int):
        self.items = items
        self.batch_size = batch_size
    
    def __iter__(self):
        for i in range(0, len(self.items), self.batch_size):
            yield self.items[i:i + self.batch_size]
    
    def __len__(self):
        return (len(self.items) + self.batch_size - 1) // self.batch_size
