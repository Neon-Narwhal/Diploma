"""Бенчмаркинг алгоритмов для определения сложности"""
import time
import gc
import statistics
import subprocess
import sys
import tempfile
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

@dataclass
class BenchmarkResult:
    """Результат бенчмарка"""
    input_size: int
    execution_times: List[float]
    avg_time: float
    median_time: float
    std_deviation: float
    min_time: float
    max_time: float
    memory_usage: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class BenchmarkConfig:
    """Конфигурация бенчмарка"""
    input_sizes: List[int]
    iterations_per_size: int
    timeout_per_iteration: float
    warmup_iterations: int
    gc_between_runs: bool
    measure_memory: bool
    use_subprocess: bool
    max_workers: int

class TestDataGenerator:
    """Генератор тестовых данных различных типов"""
    
    @staticmethod
    def generate_list(size: int, data_type: str = 'sequential') -> List:
        """Генерация списка данных"""
        if data_type == 'sequential':
            return list(range(size))
        elif data_type == 'random':
            import random
            return [random.randint(0, size * 10) for _ in range(size)]
        elif data_type == 'reverse':
            return list(range(size, 0, -1))
        elif data_type == 'partially_sorted':
            import random
            data = list(range(size))
            # Перемешиваем только 10% элементов
            shuffle_count = max(1, size // 10)
            for _ in range(shuffle_count):
                i, j = random.randint(0, size-1), random.randint(0, size-1)
                data[i], data[j] = data[j], data[i]
            return data
        else:
            return list(range(size))
    
    @staticmethod
    def generate_matrix(size: int) -> List[List[int]]:
        """Генерация квадратной матрицы"""
        return [[j + i * size for j in range(size)] for i in range(size)]
    
    @staticmethod
    def generate_graph(size: int, density: float = 0.1) -> Dict[int, List[int]]:
        """Генерация графа с заданной плотностью"""
        import random
        graph = {i: [] for i in range(size)}
        
        edge_count = int(size * (size - 1) * density / 2)
        edges_added = 0
        
        while edges_added < edge_count:
            u, v = random.randint(0, size-1), random.randint(0, size-1)
            if u != v and v not in graph[u]:
                graph[u].append(v)
                graph[v].append(u)  # Неориентированный граф
                edges_added += 1
        
        return graph
    
    @staticmethod
    def generate_string(size: int, alphabet: str = 'abcdefghijklmnopqrstuvwxyz') -> str:
        """Генерация строки"""
        import random
        return ''.join(random.choice(alphabet) for _ in range(size))
    
    @staticmethod
    def detect_data_type_from_code(source_code: str) -> str:
        """Автоматическое определение типа данных из кода"""
        code_lower = source_code.lower()
        
        if any(keyword in code_lower for keyword in ['matrix', 'grid', '2d', 'board']):
            return 'matrix'
        elif any(keyword in code_lower for keyword in ['graph', 'node', 'edge', 'vertex']):
            return 'graph'
        elif any(keyword in code_lower for keyword in ['string', 'str', 'char', 'text']):
            return 'string'
        elif any(keyword in code_lower for keyword in ['sort', 'sorted']):
            return 'list_random'  # Для сортировки лучше случайные данные
        else:
            return 'list'

class FunctionExtractor:
    """Извлекатель функций из исходного кода"""
    
    @staticmethod
    def extract_main_function(source_code: str) -> Optional[str]:
        """Извлечение главной функции"""
        import ast
        
        try:
            tree = ast.parse(source_code)
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
            
            if not functions:
                return None
            
            # Приоритет функций
            priority_names = ['main', 'solve', 'algorithm', 'run', 'benchmark', 'test']
            for name in priority_names:
                if name in functions:
                    return name
            
            # Возвращаем первую функцию
            return functions[0]
            
        except SyntaxError:
            return None
    
    @staticmethod
    def analyze_function_signature(source_code: str, function_name: str) -> Dict[str, Any]:
        """Анализ сигнатуры функции"""
        import ast
        
        try:
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    return {
                        'name': function_name,
                        'args_count': len(node.args.args),
                        'args_names': [arg.arg for arg in node.args.args],
                        'has_defaults': len(node.args.defaults) > 0,
                        'has_varargs': node.args.vararg is not None,
                        'has_kwargs': node.args.kwarg is not None,
                        'line_number': node.lineno
                    }
            
            return {}
            
        except SyntaxError:
            return {}

class MemoryProfiler:
    """Профайлер памяти"""
    
    def __init__(self):
        self.available = self._check_memory_profiler()
    
    def _check_memory_profiler(self) -> bool:
        """Проверка доступности memory_profiler"""
        try:
            import psutil
            return True
        except ImportError:
            try:
                import memory_profiler
                return True
            except ImportError:
                return False
    
    def measure_memory_usage(self, func: Callable, *args, **kwargs) -> Tuple[Any, float]:
        """Измерение использования памяти"""
        if not self.available:
            result = func(*args, **kwargs)
            return result, 0.0
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # Измерение до выполнения
            memory_before = process.memory_info().rss / 1024 / 1024  # МБ
            
            # Выполнение функции
            result = func(*args, **kwargs)
            
            # Измерение после выполнения
            memory_after = process.memory_info().rss / 1024 / 1024  # МБ
            
            memory_used = memory_after - memory_before
            
            return result, max(0, memory_used)
            
        except Exception:
            result = func(*args, **kwargs)
            return result, 0.0

class AlgorithmBenchmarker:
    """Основной класс для бенчмаркинга алгоритмов"""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or self._default_config()
        self.data_generator = TestDataGenerator()
        self.function_extractor = FunctionExtractor()
        self.memory_profiler = MemoryProfiler()
        
    def _default_config(self) -> BenchmarkConfig:
        """Конфигурация по умолчанию"""
        return BenchmarkConfig(
            input_sizes=[10, 50, 100, 200, 500, 1000, 2000],
            iterations_per_size=5,
            timeout_per_iteration=10.0,
            warmup_iterations=2,
            gc_between_runs=True,
            measure_memory=True,
            use_subprocess=True,
            max_workers=1
        )
    
    def benchmark_function(self, source_code: str, function_name: Optional[str] = None) -> List[BenchmarkResult]:
        """Бенчмарк функции"""
        if function_name is None:
            function_name = self.function_extractor.extract_main_function(source_code)
        
        if not function_name:
            return []
        
        # Анализ сигнатуры функции
        signature = self.function_extractor.analyze_function_signature(source_code, function_name)
        
        # Определение типа данных
        data_type = self.data_generator.detect_data_type_from_code(source_code)
        
        results = []
        
        for size in self.config.input_sizes:
            if self.config.use_subprocess:
                result = self._benchmark_size_subprocess(source_code, function_name, size, data_type)
            else:
                result = self._benchmark_size_direct(source_code, function_name, size, data_type)
            
            results.append(result)
            
            # Если время выполнения слишком большое, прекращаем
            if result.avg_time > self.config.timeout_per_iteration * 2:
                break
        
        return results
    
    def _benchmark_size_direct(self, source_code: str, function_name: str, 
                              size: int, data_type: str) -> BenchmarkResult:
        """Прямой бенчмарк в текущем процессе"""
        try:
            # Компиляция кода
            compiled_code = compile(source_code, '<string>', 'exec')
            namespace = {}
            exec(compiled_code, namespace)
            
            if function_name not in namespace:
                return BenchmarkResult(
                    input_size=size,
                    execution_times=[],
                    avg_time=0,
                    median_time=0,
                    std_deviation=0,
                    min_time=0,
                    max_time=0,
                    success=False,
                    error_message=f"Function {function_name} not found"
                )
            
            func = namespace[function_name]
            
            # Генерация тестовых данных
            test_data = self._generate_test_data(size, data_type)
            
            execution_times = []
            memory_usages = []
            
            # Прогревочные итерации
            for _ in range(self.config.warmup_iterations):
                try:
                    func(test_data)
                except:
                    pass
            
            # Основные измерения
            for _ in range(self.config.iterations_per_size):
                if self.config.gc_between_runs:
                    gc.collect()
                
                try:
                    start_time = time.perf_counter()
                    
                    if self.config.measure_memory:
                        result, memory_used = self.memory_profiler.measure_memory_usage(func, test_data)
                        memory_usages.append(memory_used)
                    else:
                        result = func(test_data)
                    
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    execution_times.append(execution_time)
                    
                except Exception as e:
                    return BenchmarkResult(
                        input_size=size,
                        execution_times=[],
                        avg_time=0,
                        median_time=0,
                        std_deviation=0,
                        min_time=0,
                        max_time=0,
                        success=False,
                        error_message=str(e)
                    )
            
            # Статистика
            if execution_times:
                avg_memory = statistics.mean(memory_usages) if memory_usages else None
                
                return BenchmarkResult(
                    input_size=size,
                    execution_times=execution_times.copy(),
                    avg_time=statistics.mean(execution_times),
                    median_time=statistics.median(execution_times),
                    std_deviation=statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                    min_time=min(execution_times),
                    max_time=max(execution_times),
                    memory_usage=avg_memory,
                    success=True
                )
            else:
                return BenchmarkResult(
                    input_size=size,
                    execution_times=[],
                    avg_time=0,
                    median_time=0,
                    std_deviation=0,
                    min_time=0,
                    max_time=0,
                    success=False,
                    error_message="No successful executions"
                )
                
        except Exception as e:
            return BenchmarkResult(
                input_size=size,
                execution_times=[],
                avg_time=0,
                median_time=0,
                std_deviation=0,
                min_time=0,
                max_time=0,
                success=False,
                error_message=str(e)
            )
    
    def _benchmark_size_subprocess(self, source_code: str, function_name: str,
                                  size: int, data_type: str) -> BenchmarkResult:
        """Бенчмарк в отдельном процессе"""
        try:
            # Создание скрипта для выполнения
            script_content = self._create_benchmark_script(source_code, function_name, size, data_type)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script_content)
                f.flush()
                script_path = f.name
            
            try:
                # Выполнение в subprocess
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_per_iteration * self.config.iterations_per_size * 2
                )
                
                if result.returncode == 0:
                    # Парсинг результата
                    return self._parse_benchmark_result(result.stdout, size)
                else:
                    return BenchmarkResult(
                        input_size=size,
                        execution_times=[],
                        avg_time=0,
                        median_time=0,
                        std_deviation=0,
                        min_time=0,
                        max_time=0,
                        success=False,
                        error_message=result.stderr
                    )
            
            finally:
                import os
                try:
                    os.unlink(script_path)
                except:
                    pass
                    
        except subprocess.TimeoutExpired:
            return BenchmarkResult(
                input_size=size,
                execution_times=[],
                avg_time=0,
                median_time=0,
                std_deviation=0,
                min_time=0,
                max_time=0,
                success=False,
                error_message="Timeout"
            )
        except Exception as e:
            return BenchmarkResult(
                input_size=size,
                execution_times=[],
                avg_time=0,
                median_time=0,
                std_deviation=0,
                min_time=0,
                max_time=0,
                success=False,
                error_message=str(e)
            )
    
    def _generate_test_data(self, size: int, data_type: str):
        """Генерация тестовых данных"""
        if data_type == 'matrix':
            return self.data_generator.generate_matrix(size)
        elif data_type == 'graph':
            return self.data_generator.generate_graph(size)
        elif data_type == 'string':
            return self.data_generator.generate_string(size)
        elif data_type == 'list_random':
            return self.data_generator.generate_list(size, 'random')
        else:
            return self.data_generator.generate_list(size, 'sequential')
    
    def _create_benchmark_script(self, source_code: str, function_name: str,
                                size: int, data_type: str) -> str:
        """Создание скрипта для бенчмарка"""
        test_data_repr = repr(self._generate_test_data(size, data_type))
        
        return f"""
import time
import gc
import statistics
import json

# Исходный код
{source_code}

# Конфигурация
iterations = {self.config.iterations_per_size}
warmup_iterations = {self.config.warmup_iterations}
gc_between_runs = {self.config.gc_between_runs}

# Тестовые данные
test_data = {test_data_repr}

# Прогрев
for _ in range(warmup_iterations):
    try:
        {function_name}(test_data)
    except:
        pass

# Измерения
execution_times = []

for _ in range(iterations):
    if gc_between_runs:
        gc.collect()
    
    try:
        start_time = time.perf_counter()
        result = {function_name}(test_data)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        execution_times.append(execution_time)
        
    except Exception as e:
        print(json.dumps({{"error": str(e)}}))
        exit(1)

# Результат
if execution_times:
    result_data = {{
        "execution_times": execution_times,
        "avg_time": statistics.mean(execution_times),
        "median_time": statistics.median(execution_times),
        "std_deviation": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
        "min_time": min(execution_times),
        "max_time": max(execution_times),
        "success": True
    }}
else:
    result_data = {{
        "execution_times": [],
        "success": False,
        "error": "No successful executions"
    }}

print(json.dumps(result_data))
"""
    
    def _parse_benchmark_result(self, output: str, size: int) -> BenchmarkResult:
        """Парсинг результата бенчмарка"""
        try:
            import json
            data = json.loads(output.strip())
            
            if data.get("success", False):
                return BenchmarkResult(
                    input_size=size,
                    execution_times=data["execution_times"],
                    avg_time=data["avg_time"],
                    median_time=data["median_time"],
                    std_deviation=data["std_deviation"],
                    min_time=data["min_time"],
                    max_time=data["max_time"],
                    success=True
                )
            else:
                return BenchmarkResult(
                    input_size=size,
                    execution_times=[],
                    avg_time=0,
                    median_time=0,
                    std_deviation=0,
                    min_time=0,
                    max_time=0,
                    success=False,
                    error_message=data.get("error", "Unknown error")
                )
                
        except json.JSONDecodeError:
            return BenchmarkResult(
                input_size=size,
                execution_times=[],
                avg_time=0,
                median_time=0,
                std_deviation=0,
                min_time=0,
                max_time=0,
                success=False,
                error_message="Failed to parse benchmark output"
            )
    
    def parallel_benchmark(self, algorithms: Dict[str, str]) -> Dict[str, List[BenchmarkResult]]:
        """Параллельный бенчмарк нескольких алгоритмов"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_name = {
                executor.submit(self.benchmark_function, code): name
                for name, code in algorithms.items()
            }
            
            for future in future_to_name:
                name = future_to_name[future]
                try:
                    results[name] = future.result(timeout=self.config.timeout_per_iteration * 10)
                except Exception as e:
                    results[name] = [BenchmarkResult(
                        input_size=0,
                        execution_times=[],
                        avg_time=0,
                        median_time=0,
                        std_deviation=0,
                        min_time=0,
                        max_time=0,
                        success=False,
                        error_message=str(e)
                    )]
        
        return results
