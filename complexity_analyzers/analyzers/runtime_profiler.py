"""Анализатор времени выполнения"""
import time
import timeit
import gc
import sys
import subprocess
import tempfile
from typing import Dict, Any, List, Tuple, Optional, Callable
import numpy as np
from scipy.optimize import curve_fit
from complexity_analyzers.core.base import BaseComplexityAnalyzer, AnalyzerType
from complexity_analyzers.core.result import ComplexityResult, ComplexityClass

class TestDataGenerator:
    """Генератор тестовых данных"""
    
    @staticmethod
    def generate_list(size: int, data_type: str = 'int') -> List[Any]:
        """Генерация списка"""
        if data_type == 'int':
            return list(range(size))
        elif data_type == 'random_int':
            import random
            return [random.randint(0, size) for _ in range(size)]
        elif data_type == 'str':
            return [f"item_{i}" for i in range(size)]
        else:
            return list(range(size))
    
    @staticmethod
    def generate_matrix(size: int) -> List[List[int]]:
        """Генерация матрицы"""
        return [[j for j in range(size)] for i in range(size)]
    
    @staticmethod
    def generate_graph(size: int) -> Dict[int, List[int]]:
        """Генерация графа"""
        graph = {}
        for i in range(size):
            # Каждая вершина соединена с несколькими случайными
            import random
            neighbors = random.sample(range(size), min(3, size-1))
            graph[i] = [n for n in neighbors if n != i]
        return graph

class ComplexityFitter:
    """Подгонщик функций сложности"""
    
    def __init__(self):
        self.complexity_functions = {
            ComplexityClass.CONSTANT: lambda x, a: np.full_like(x, a, dtype=float),
            ComplexityClass.LOGARITHMIC: lambda x, a, b: a * np.log(x + 1) + b,
            ComplexityClass.LINEAR: lambda x, a, b: a * x + b,
            ComplexityClass.LINEARITHMIC: lambda x, a, b: a * x * np.log(x + 1) + b,
            ComplexityClass.QUADRATIC: lambda x, a, b: a * x**2 + b,
            ComplexityClass.CUBIC: lambda x, a, b: a * x**3 + b,
            ComplexityClass.EXPONENTIAL: lambda x, a, b: a * np.exp(x/1000) + b  # Масштабированная
        }
    
    def fit_complexity(self, sizes: List[int], times: List[float]) -> Tuple[ComplexityClass, float]:
        """Подгонка кривой сложности"""
        sizes_array = np.array(sizes, dtype=float)
        times_array = np.array(times, dtype=float)
        
        best_complexity = ComplexityClass.LINEAR
        best_score = -float('inf')
        
        for complexity, func in self.complexity_functions.items():
            try:
                # Попытка подгонки
                popt, _ = curve_fit(func, sizes_array, times_array, maxfev=1000)
                
                # Вычисление R²
                y_pred = func(sizes_array, *popt)
                ss_res = np.sum((times_array - y_pred) ** 2)
                ss_tot = np.sum((times_array - np.mean(times_array)) ** 2)
                r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                if r2_score > best_score:
                    best_score = r2_score
                    best_complexity = complexity
                    
            except Exception:
                continue
        
        return best_complexity, max(0.0, best_score)

class RuntimeProfiler(BaseComplexityAnalyzer):
    """Профайлер времени выполнения"""
    
    def __init__(self):
        super().__init__("runtime_profiler", AnalyzerType.RUNTIME_PROFILER)
        self.test_sizes: List[int] = [10, 50, 100, 200, 500, 1000]
        self.iterations: int = 5
        self.timeout: int = 30
        self.data_generator = TestDataGenerator()
        self.complexity_fitter = ComplexityFitter()
    
    def is_available(self) -> bool:
        """Проверка доступности"""
        return True
    
    def analyze(self, context) -> ComplexityResult:
        """Анализ времени выполнения"""
        try:
            # Извлекаем главную функцию
            func_name, func_code = self._extract_main_function(context.source_code)
            if not func_name:
                return ComplexityResult(
                    complexity_class=ComplexityClass.UNKNOWN,
                    confidence=0.0,
                    analyzer_name=self.name,
                    errors=["No main function found"]
                )
            
            # Измеряем время выполнения
            measurements = self._measure_execution_times(func_code, func_name)
            
            if not measurements:
                return ComplexityResult(
                    complexity_class=ComplexityClass.UNKNOWN,
                    confidence=0.0,
                    analyzer_name=self.name,
                    errors=["No measurements obtained"]
                )
            
            # Подгоняем кривую сложности
            sizes = [m['size'] for m in measurements]
            times = [m['avg_time'] for m in measurements]
            
            complexity_class, confidence = self.complexity_fitter.fit_complexity(sizes, times)
            
            return ComplexityResult(
                complexity_class=complexity_class,
                confidence=confidence,
                analyzer_name=self.name,
                runtime_data={
                    'measurements': measurements,
                    'sizes': sizes,
                    'times': times,
                    'function_name': func_name
                }
            )
            
        except Exception as e:
            return ComplexityResult(
                complexity_class=ComplexityClass.UNKNOWN,
                confidence=0.0,
                analyzer_name=self.name,
                errors=[f"Runtime analysis error: {e}"]
            )
    
    def _extract_main_function(self, source_code: str) -> Tuple[Optional[str], str]:
        """Извлечение главной функции для тестирования"""
        import ast
        import re
        
        try:
            tree = ast.parse(source_code)
            
            # Ищем функции
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
            
            if not functions:
                return None, source_code
            
            # Выбираем главную функцию (эвристики)
            main_function = None
            
            # 1. Ищем функцию с именем main, solve, algorithm
            for priority_name in ['main', 'solve', 'algorithm', 'run']:
                if priority_name in functions:
                    main_function = priority_name
                    break
            
            # 2. Если не найдена, берем первую функцию
            if not main_function:
                main_function = functions[0]
            
            return main_function, source_code
            
        except Exception:
            return None, source_code
    
    def _measure_execution_times(self, code: str, func_name: str) -> List[Dict[str, Any]]:
        """Измерение времени выполнения для разных размеров входных данных"""
        measurements = []
        
        for size in self.test_sizes:
            try:
                # Генерируем тестовые данные
                test_data = self._generate_test_data_for_function(code, size)
                
                # Измеряем время
                times = []
                for _ in range(self.iterations):
                    start_time = time.perf_counter()
                    
                    try:
                        # Выполняем функцию в изолированном окружении
                        result = self._execute_function_safely(code, func_name, test_data)
                        end_time = time.perf_counter()
                        
                        if result is not None:  # Функция выполнилась успешно
                            times.append(end_time - start_time)
                    except Exception:
                        continue
                
                if times:
                    avg_time = np.mean(times)
                    std_time = np.std(times)
                    
                    measurements.append({
                        'size': size,
                        'avg_time': avg_time,
                        'std_time': std_time,
                        'measurements_count': len(times),
                        'raw_times': times
                    })
                
            except Exception:
                continue
        
        return measurements
    
    def _generate_test_data_for_function(self, code: str, size: int) -> Any:
        """Генерация тестовых данных на основе анализа кода"""
        # Простая эвристика: если в коде есть упоминания матриц, генерируем матрицу
        if any(keyword in code.lower() for keyword in ['matrix', 'grid', '2d', 'board']):
            return self.data_generator.generate_matrix(size)
        
        # Если упоминается граф
        elif any(keyword in code.lower() for keyword in ['graph', 'node', 'edge', 'vertex']):
            return self.data_generator.generate_graph(size)
        
        # По умолчанию - список
        else:
            return self.data_generator.generate_list(size)
    
    def _execute_function_safely(self, code: str, func_name: str, test_data: Any) -> Any:
        """Безопасное выполнение функции в изолированном окружении"""
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Записываем код и вызов функции
            execution_code = f"""
{code}

import sys
import json

try:
    test_data = {repr(test_data)}
    result = {func_name}(test_data)
    print(json.dumps({{"success": True, "result": str(result)}}))
except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}))
"""
            f.write(execution_code)
            f.flush()
            
            try:
                # Выполняем в отдельном процессе с таймаутом
                result = subprocess.run(
                    [sys.executable, f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                
                if result.returncode == 0:
                    import json
                    output = json.loads(result.stdout.strip())
                    if output.get('success'):
                        return output.get('result')
                return None
                
            except subprocess.TimeoutExpired:
                return None
            except Exception:
                return None
            finally:
                import os
                try:
                    os.unlink(f.name)
                except:
                    pass

class BenchmarkRunner:
    """Запускатель бенчмарков"""
    
    def __init__(self):
        self.profiler = RuntimeProfiler()
    
    def run_benchmark_suite(self, algorithms: Dict[str, str], 
                          test_sizes: List[int] = None) -> Dict[str, ComplexityResult]:
        """Запуск набора бенчмарков"""
        if test_sizes:
            self.profiler.test_sizes = test_sizes
        
        results = {}
        
        for name, code in algorithms.items():
            from complexity_analyzers.base.analyzer import AnalysisContext
            context = AnalysisContext(source_code=code)
            result = self.profiler.analyze(context)
            results[name] = result
        
        return results
    
    def compare_algorithms(self, algorithms: Dict[str, str]) -> Dict[str, Any]:
        """Сравнение алгоритмов"""
        results = self.run_benchmark_suite(algorithms)
        
        comparison = {
            'algorithms': {},
            'ranking': [],
            'complexity_distribution': {}
        }
        
        for name, result in results.items():
            comparison['algorithms'][name] = {
                'complexity': result.complexity_class.notation,
                'confidence': result.confidence,
                'avg_time': np.mean([m['avg_time'] for m in result.runtime_data.get('measurements', [])]) if result.runtime_data else 0
            }
        
        # Ранжирование по эффективности
        ranking = sorted(
            comparison['algorithms'].items(),
            key=lambda x: (x[1]['complexity'], x[1]['avg_time'])
        )
        comparison['ranking'] = [name for name, _ in ranking]
        
        # Распределение классов сложности
        complexity_counts = {}
        for result in results.values():
            complexity = result.complexity_class.notation
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        comparison['complexity_distribution'] = complexity_counts
        
        return comparison
