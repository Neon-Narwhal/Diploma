"""
Модуль безопасного выполнения кода и замера времени.
"""

import time
import multiprocessing
import traceback
from typing import Any

class CodeExecutor:
    """
    Изолированный исполнитель кода.
    """
    
    def __init__(self, timeout: float = 2.0):
        self.timeout = timeout
        
    def measure_time(self, code: str, input_data: Any) -> float:
        """
        Запуск кода в отдельном процессе с замером времени.
        Returns: время в секундах или -1.0 при ошибке/таймауте.
        """
        # Используем fork метод для Linux (быстрее и надежнее для нашей задачи)
        ctx = multiprocessing.get_context('fork') if hasattr(multiprocessing, 'get_context') else multiprocessing
        queue = ctx.Queue()
        
        p = ctx.Process(target=self._worker, args=(code, input_data, queue))
        p.start()
        p.join(self.timeout)
        
        if p.is_alive():
            p.terminate()
            p.join()
            return -1.0  # Timeout
            
        if not queue.empty():
            result = queue.get()
            if isinstance(result, dict):
                if result.get('success'):
                    return result['time']
                else:
                    # Можно логировать ошибку
                    # print(f"Execution failed: {result.get('error')}")
                    pass
        
        return -1.0
        
    def _worker(self, code: str, input_data: Any, queue: multiprocessing.Queue):
        """Воркер, который выполняется в отдельном процессе"""
        try:
            # Компилируем и выполняем код пользователя в изолированном скоупе
            local_scope = {}
            global_scope = {
                '__builtins__': __builtins__,
                # Можно добавить разрешенные импорты
            }
            
            exec(code, global_scope, local_scope)
            
            # Ищем целевую функцию (берем последнюю определенную функцию)
            target_func = None
            for name, obj in local_scope.items():
                if callable(obj) and not name.startswith('_'):
                    target_func = obj
            
            if not target_func:
                queue.put({'success': False, 'error': 'No callable function found'})
                return

            # Замер времени
            # Запускаем несколько раз и берем минимум для стабильности
            times = []
            for _ in range(3):
                start = time.perf_counter()
                result = target_func(input_data)
                end = time.perf_counter()
                times.append(end - start)
            
            queue.put({'success': True, 'time': min(times)})
            
        except TimeoutError:
            queue.put({'success': False, 'error': 'Timeout in function execution'})
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            queue.put({'success': False, 'error': error_msg})
