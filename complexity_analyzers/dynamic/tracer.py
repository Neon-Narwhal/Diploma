"""Динамический анализатор с трассировкой выполнения"""
import sys
import time
import inspect
import functools
import subprocess
import tempfile
from typing import Dict, Any, List, Optional, Callable, Set, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from complexity_analyzers.base.analyzer import BaseComplexityAnalyzer, AnalyzerType
from complexity_analyzers.base.result import ComplexityResult, ComplexityClass

@dataclass
class CallTrace:
    """Трасса вызова функции"""
    function_name: str
    args_signature: str
    input_size: int
    execution_time: float
    call_depth: int
    line_number: int
    call_count: int = 1

@dataclass 
class RecurrencePattern:
    """Паттерн рекуррентного соотношения"""
    function_name: str
    recurrence_type: str  # 'linear', 'divide_conquer', 'tree', 'exponential'
    base_cases: List[int]
    recursive_calls_per_input: Dict[int, int]
    time_complexity: ComplexityClass
    confidence: float

class ExecutionTracer:
    """Трассировщик выполнения кода"""
    
    def __init__(self):
        self.traces: List[CallTrace] = []
        self.call_stack: List[str] = []
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.start_time: float = 0
        self.current_input_size: int = 0
        self.max_recursion_depth: int = 0
        self.function_times: Dict[str, List[float]] = defaultdict(list)
        
    def reset(self):
        """Сброс состояния трассировщика"""
        self.traces.clear()
        self.call_stack.clear()
        self.call_counts.clear()
        self.start_time = 0
        self.current_input_size = 0
        self.max_recursion_depth = 0
        self.function_times.clear()
    
    def trace_calls(self, frame, event: str, arg):
        """Функция трассировки вызовов"""
        if event == 'call':
            return self._handle_call(frame)
        elif event == 'return':
            return self._handle_return(frame, arg)
        return self.trace_calls
    
    def _handle_call(self, frame):
        """Обработка вызова функции"""
        func_name = frame.f_code.co_name
        line_no = frame.f_lineno
        
        # Игнорируем встроенные и системные функции
        if func_name.startswith('_') or 'site-packages' in frame.f_code.co_filename:
            return
        
        # Определяем размер входных данных
        input_size = self._estimate_input_size(frame)
        
        # Обновляем стек вызовов
        self.call_stack.append(func_name)
        call_depth = len(self.call_stack)
        self.max_recursion_depth = max(self.max_recursion_depth, call_depth)
        
        # Увеличиваем счетчик вызовов
        self.call_counts[func_name] += 1
        
        # Создаем трассу
        args_sig = self._get_args_signature(frame)
        
        trace = CallTrace(
            function_name=func_name,
            args_signature=args_sig,
            input_size=input_size,
            execution_time=0.0,  # Будет обновлено при возврате
            call_depth=call_depth,
            line_number=line_no,
            call_count=self.call_counts[func_name]
        )
        
        self.traces.append(trace)
        return self.trace_calls
    
    def _handle_return(self, frame, return_value):
        """Обработка возврата из функции"""
        func_name = frame.f_code.co_name
        
        if func_name.startswith('_'):
            return
        
        # Обновляем стек вызовов
        if self.call_stack and self.call_stack[-1] == func_name:
            self.call_stack.pop()
        
        return self.trace_calls
    
    def _estimate_input_size(self, frame) -> int:
        """Оценка размера входных данных"""
        local_vars = frame.f_locals
        
        # Ищем переменные, которые могут быть входными данными
        for var_name, var_value in local_vars.items():
            if var_name in ['n', 'size', 'length', 'count']:
                if isinstance(var_value, int):
                    return var_value
            elif var_name in ['arr', 'array', 'list', 'data', 'items']:
                if hasattr(var_value, '__len__'):
                    return len(var_value)
        
        # Если не нашли явного размера, используем текущий
        return self.current_input_size
    
    def _get_args_signature(self, frame) -> str:
        """Получение сигнатуры аргументов"""
        try:
            local_vars = frame.f_locals
            args = []
            
            for name, value in local_vars.items():
                if not name.startswith('_'):
                    if isinstance(value, (int, float, str)):
                        args.append(f"{name}={value}")
                    elif hasattr(value, '__len__'):
                        args.append(f"{name}=<len={len(value)}>")
                    else:
                        args.append(f"{name}=<{type(value).__name__}>")
            
            return f"({', '.join(args[:3])})"  # Ограничиваем 3 аргументами
        except:
            return "()"

class RecurrenceAnalyzer:
    """Анализатор рекуррентных соотношений"""
    
    def __init__(self):
        self.patterns: List[RecurrencePattern] = []
    
    def analyze_traces(self, traces: List[CallTrace]) -> List[RecurrencePattern]:
        """Анализ трасс для выявления рекуррентных паттернов"""
        patterns = []
        
        # Группируем трассы по функциям
        function_traces = defaultdict(list)
        for trace in traces:
            function_traces[trace.function_name].append(trace)
        
        # Анализируем каждую функцию
        for func_name, func_traces in function_traces.items():
            pattern = self._analyze_function_recursion(func_name, func_traces)
            if pattern:
                patterns.append(pattern)
        
        self.patterns = patterns
        return patterns
    
    def _analyze_function_recursion(self, func_name: str, traces: List[CallTrace]) -> Optional[RecurrencePattern]:
        """Анализ рекурсии конкретной функции"""
        if len(traces) < 2:
            return None
        
        # Группируем по размеру входных данных
        size_to_calls = defaultdict(list)
        for trace in traces:
            size_to_calls[trace.input_size].append(trace)
        
        # Анализируем количество вызовов для каждого размера
        recursive_calls = {}
        for size, calls in size_to_calls.items():
            recursive_calls[size] = len(calls)
        
        # Определяем тип рекуррентного соотношения
        recurrence_type = self._classify_recurrence_type(recursive_calls)
        time_complexity = self._infer_complexity_from_recurrence(recurrence_type, recursive_calls)
        
        # Находим базовые случаи
        base_cases = [size for size, count in recursive_calls.items() if count == 1]
        
        confidence = self._calculate_pattern_confidence(recursive_calls, recurrence_type)
        
        return RecurrencePattern(
            function_name=func_name,
            recurrence_type=recurrence_type,
            base_cases=base_cases,
            recursive_calls_per_input=recursive_calls,
            time_complexity=time_complexity,
            confidence=confidence
        )
    
    def _classify_recurrence_type(self, recursive_calls: Dict[int, int]) -> str:
        """Классификация типа рекуррентного соотношения"""
        sizes = sorted(recursive_calls.keys())
        
        if len(sizes) < 3:
            return 'unknown'
        
        # Проверяем паттерны роста
        ratios = []
        for i in range(1, len(sizes)):
            if recursive_calls[sizes[i-1]] > 0:
                ratio = recursive_calls[sizes[i]] / recursive_calls[sizes[i-1]]
                ratios.append(ratio)
        
        if not ratios:
            return 'unknown'
        
        avg_ratio = sum(ratios) / len(ratios)
        
        if avg_ratio <= 1.5:
            return 'linear'
        elif avg_ratio <= 2.5:
            return 'divide_conquer'
        elif avg_ratio <= 4.0:
            return 'tree'
        else:
            return 'exponential'
    
    def _infer_complexity_from_recurrence(self, recurrence_type: str, 
                                        recursive_calls: Dict[int, int]) -> ComplexityClass:
        """Вывод сложности из типа рекуррентного соотношения"""
        if recurrence_type == 'linear':
            return ComplexityClass.LINEAR
        elif recurrence_type == 'divide_conquer':
            return ComplexityClass.LINEARITHMIC
        elif recurrence_type == 'tree':
            return ComplexityClass.QUADRATIC
        elif recurrence_type == 'exponential':
            return ComplexityClass.EXPONENTIAL
        else:
            return ComplexityClass.UNKNOWN
    
    def _calculate_pattern_confidence(self, recursive_calls: Dict[int, int], 
                                    recurrence_type: str) -> float:
        """Расчет уверенности в паттерне"""
        base_confidence = 0.7
        
        # Увеличиваем уверенность при большем количестве точек данных
        if len(recursive_calls) >= 5:
            base_confidence += 0.2
        
        # Увеличиваем при четком паттерне
        if recurrence_type in ['linear', 'divide_conquer', 'exponential']:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)

class DynamicComplexityTracer(BaseComplexityAnalyzer):
    """Динамический анализатор сложности с трассировкой"""
    
    def __init__(self):
        super().__init__("dynamic_tracer", AnalyzerType.DYNAMIC_TRACER)
        self.tracer = ExecutionTracer()
        self.recurrence_analyzer = RecurrenceAnalyzer()
        self.test_sizes = [5, 10, 20, 50, 100]
        self.timeout = 10  # секунд на каждый тест
    
    def is_available(self) -> bool:
        """Проверка доступности"""
        return True  # Трассировка всегда доступна
    
    def analyze(self, context) -> ComplexityResult:
        """Динамический анализ с трассировкой"""
        try:
            # Извлекаем главную функцию
            main_function = self._extract_main_function(context.source_code)
            if not main_function:
                return ComplexityResult(
                    complexity_class=ComplexityClass.UNKNOWN,
                    confidence=0.0,
                    analyzer_name=self.name,
                    errors=["No main function found for tracing"]
                )
            
            # Выполняем трассировку для разных размеров входных данных
            all_traces = []
            for size in self.test_sizes:
                traces = self._trace_execution(context.source_code, main_function, size)
                all_traces.extend(traces)
            
            if not all_traces:
                return ComplexityResult(
                    complexity_class=ComplexityClass.UNKNOWN,
                    confidence=0.0,
                    analyzer_name=self.name,
                    errors=["No traces collected"]
                )
            
            # Анализируем рекуррентные паттерны
            patterns = self.recurrence_analyzer.analyze_traces(all_traces)
            
            # Определяем итоговую сложность
            final_complexity, confidence = self._determine_final_complexity(patterns, all_traces)
            
            return ComplexityResult(
                complexity_class=final_complexity,
                confidence=confidence,
                analyzer_name=self.name,
                dynamic_traces={
                    'total_traces': len(all_traces),
                    'max_recursion_depth': self.tracer.max_recursion_depth,
                    'recurrence_patterns': [self._pattern_to_dict(p) for p in patterns],
                    'function_call_counts': dict(self.tracer.call_counts),
                    'test_sizes': self.test_sizes
                }
            )
            
        except Exception as e:
            return ComplexityResult(
                complexity_class=ComplexityClass.UNKNOWN,
                confidence=0.0,
                analyzer_name=self.name,
                errors=[f"Dynamic tracing error: {e}"]
            )
    
    def _extract_main_function(self, source_code: str) -> Optional[str]:
        """Извлечение главной функции для трассировки"""
        import ast
        
        try:
            tree = ast.parse(source_code)
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
            
            if not functions:
                return None
            
            # Приоритет: main, solve, algorithm, первая функция
            for priority_name in ['main', 'solve', 'algorithm', 'run']:
                if priority_name in functions:
                    return priority_name
            
            return functions[0]
            
        except:
            return None
    
    def _trace_execution(self, source_code: str, function_name: str, input_size: int) -> List[CallTrace]:
        """Трассировка выполнения для конкретного размера входных данных"""
        self.tracer.reset()
        self.tracer.current_input_size = input_size
        
        try:
            # Создаем исполняемый код
            test_data = self._generate_test_data(input_size)
            
            exec_code = f"""
{source_code}

# Тестовые данные
test_input = {repr(test_data)}

# Вызов функции
result = {function_name}(test_input)
"""
            
            # Устанавливаем трассировку
            sys.settrace(self.tracer.trace_calls)
            
            # Выполняем код
            exec_globals = {}
            exec(exec_code, exec_globals)
            
            # Отключаем трассировку
            sys.settrace(None)
            
            return self.tracer.traces.copy()
            
        except Exception as e:
            sys.settrace(None)  # Обязательно отключаем трассировку
            return []
    
    def _generate_test_data(self, size: int) -> Any:
        """Генерация тестовых данных"""
        # Простая эвристика - возвращаем список чисел
        return list(range(size))
    
    def _determine_final_complexity(self, patterns: List[RecurrencePattern], 
                                  all_traces: List[CallTrace]) -> Tuple[ComplexityClass, float]:
        """Определение итоговой сложности из паттернов и трасс"""
        if not patterns:
            # Если нет рекуррентных паттернов, анализируем общий рост вызовов
            return self._analyze_trace_growth(all_traces)
        
        # Если есть паттерны, выбираем наиболее уверенный
        best_pattern = max(patterns, key=lambda p: p.confidence)
        return best_pattern.time_complexity, best_pattern.confidence
    
    def _analyze_trace_growth(self, traces: List[CallTrace]) -> Tuple[ComplexityClass, float]:
        """Анализ роста количества вызовов"""
        # Группируем по размеру входных данных
        size_to_call_count = defaultdict(int)
        for trace in traces:
            size_to_call_count[trace.input_size] += 1
        
        if len(size_to_call_count) < 3:
            return ComplexityClass.UNKNOWN, 0.1
        
        sizes = sorted(size_to_call_count.keys())
        call_counts = [size_to_call_count[size] for size in sizes]
        
        # Простой анализ роста
        growth_ratios = []
        for i in range(1, len(call_counts)):
            if call_counts[i-1] > 0:
                ratio = call_counts[i] / call_counts[i-1]
                growth_ratios.append(ratio)
        
        if not growth_ratios:
            return ComplexityClass.UNKNOWN, 0.1
        
        avg_growth = sum(growth_ratios) / len(growth_ratios)
        
        if avg_growth <= 1.2:
            return ComplexityClass.CONSTANT, 0.6
        elif avg_growth <= 2.0:
            return ComplexityClass.LINEAR, 0.7
        elif avg_growth <= 4.0:
            return ComplexityClass.QUADRATIC, 0.6
        else:
            return ComplexityClass.EXPONENTIAL, 0.8
    
    def _pattern_to_dict(self, pattern: RecurrencePattern) -> Dict[str, Any]:
        """Преобразование паттерна в словарь"""
        return {
            'function_name': pattern.function_name,
            'recurrence_type': pattern.recurrence_type,
            'base_cases': pattern.base_cases,
            'recursive_calls_per_input': pattern.recursive_calls_per_input,
            'time_complexity': pattern.time_complexity.notation,
            'confidence': pattern.confidence
        }

class SafeExecutionTracer(DynamicComplexityTracer):
    """Безопасный трассировщик с изолированным выполнением"""
    
    def __init__(self):
        super().__init__()
        self.name = "safe_dynamic_tracer"
        self.use_subprocess = True
    
    def _trace_execution(self, source_code: str, function_name: str, input_size: int) -> List[CallTrace]:
        """Безопасная трассировка через subprocess"""
        if not self.use_subprocess:
            return super()._trace_execution(source_code, function_name, input_size)
        
        try:
            # Создаем временный файл с кодом трассировки
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                trace_code = self._generate_trace_code(source_code, function_name, input_size)
                f.write(trace_code)
                f.flush()
                
                # Выполняем в отдельном процессе
                result = subprocess.run(
                    [sys.executable, f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                
                if result.returncode == 0:
                    # Парсим результат трассировки
                    return self._parse_trace_output(result.stdout)
                else:
                    return []
                    
        except subprocess.TimeoutExpired:
            return []
        except Exception:
            return []
        finally:
            try:
                import os
                os.unlink(f.name)
            except:
                pass
    
    def _generate_trace_code(self, source_code: str, function_name: str, input_size: int) -> str:
        """Генерация кода для трассировки"""
        test_data = self._generate_test_data(input_size)
        
        return f"""
import sys
import json
from collections import defaultdict

# Трассировка
traces = []
call_stack = []
call_counts = defaultdict(int)

def trace_calls(frame, event, arg):
    if event == 'call':
        func_name = frame.f_code.co_name
        if not func_name.startswith('_') and 'site-packages' not in frame.f_code.co_filename:
            call_stack.append(func_name)
            call_counts[func_name] += 1
            
            traces.append({{
                'function_name': func_name,
                'input_size': {input_size},
                'call_depth': len(call_stack),
                'line_number': frame.f_lineno,
                'call_count': call_counts[func_name]
            }})
    
    elif event == 'return':
        func_name = frame.f_code.co_name
        if call_stack and call_stack[-1] == func_name:
            call_stack.pop()
    
    return trace_calls

# Исходный код
{source_code}

# Трассировка
sys.settrace(trace_calls)

try:
    test_input = {repr(test_data)}
    result = {function_name}(test_input)
    sys.settrace(None)
    
    # Вывод результатов
    print(json.dumps(traces))
    
except Exception as e:
    sys.settrace(None)
    print(json.dumps([]))
"""
    
    def _parse_trace_output(self, output: str) -> List[CallTrace]:
        """Парсинг вывода трассировки"""
        try:
            import json
            trace_data = json.loads(output.strip())
            
            traces = []
            for data in trace_data:
                trace = CallTrace(
                    function_name=data['function_name'],
                    args_signature="()",
                    input_size=data['input_size'],
                    execution_time=0.0,
                    call_depth=data['call_depth'],
                    line_number=data['line_number'],
                    call_count=data['call_count']
                )
                traces.append(trace)
            
            return traces
            
        except:
            return []
