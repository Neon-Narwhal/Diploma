"""Интеграция с инструментами профилирования"""
import subprocess
import tempfile
import json
import time
from typing import Dict, Any, List, Optional
from complexity_analyzers.core.base import BaseComplexityAnalyzer, AnalyzerType
from complexity_analyzers.core.result import ComplexityResult, ComplexityClass

class PySpyIntegration:
    """Интеграция с py-spy профайлером"""
    
    def __init__(self):
        self.is_available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Проверка доступности py-spy"""
        try:
            result = subprocess.run(['py-spy', '--version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def profile_code(self, source_code: str, duration: int = 5) -> Dict[str, Any]:
        """Профилирование кода с помощью py-spy"""
        if not self.is_available:
            return {'error': 'py-spy not available'}
        
        try:
            # Создаем временный файл с кодом
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(source_code)
                f.flush()
                
                # Запускаем профилирование
                cmd = [
                    'py-spy', 'record', '-o', '/tmp/profile.svg',
                    '-d', str(duration), 
                    '--', 'python', f.name
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 5)
                
                if result.returncode == 0:
                    return {'success': True, 'profile_file': '/tmp/profile.svg'}
                else:
                    return {'error': f'py-spy failed: {result.stderr}'}
                    
        except Exception as e:
            return {'error': f'Profiling error: {e}'}
        finally:
            try:
                import os
                os.unlink(f.name)
            except:
                pass

class LineProfilerIntegration:
    """Интеграция с line_profiler"""
    
    def __init__(self):
        self.is_available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Проверка доступности line_profiler"""
        try:
            import line_profiler
            return True
        except ImportError:
            return False
    
    def profile_function(self, source_code: str, function_name: str) -> Dict[str, Any]:
        """Профилирование функции построчно"""
        if not self.is_available:
            return {'error': 'line_profiler not available'}
        
        try:
            import line_profiler
            
            # Компилируем код
            code_obj = compile(source_code, '<string>', 'exec')
            namespace = {}
            exec(code_obj, namespace)
            
            if function_name not in namespace:
                return {'error': f'Function {function_name} not found'}
            
            func = namespace[function_name]
            
            # Создаем профайлер
            profiler = line_profiler.LineProfiler()
            profiler.add_function(func)
            
            # Профилируем
            profiler.enable_by_count()
            
            # Нужны тестовые данные для вызова функции
            test_data = list(range(100))
            func(test_data)
            
            profiler.disable_by_count()
            
            # Получаем статистику
            stats = profiler.get_stats()
            
            return {
                'success': True,
                'stats': self._format_line_stats(stats),
                'function_name': function_name
            }
            
        except Exception as e:
            return {'error': f'Line profiling error: {e}'}
    
    def _format_line_stats(self, stats) -> Dict[str, Any]:
        """Форматирование статистики line_profiler"""
        formatted = {}
        
        for (filename, start_lineno, func_name), timings in stats.timings.items():
            formatted[func_name] = {
                'filename': filename,
                'start_line': start_lineno,
                'line_timings': []
            }
            
            for lineno, nhits, time in timings:
                formatted[func_name]['line_timings'].append({
                    'line_number': lineno,
                    'hits': nhits,
                    'time': time,
                    'time_per_hit': time / nhits if nhits > 0 else 0
                })
        
        return formatted

class MemoryProfilerIntegration:
    """Интеграция с memory_profiler"""
    
    def __init__(self):
        self.is_available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Проверка доступности memory_profiler"""
        try:
            import memory_profiler
            return True
        except ImportError:
            return False
    
    def profile_memory(self, source_code: str, function_name: str) -> Dict[str, Any]:
        """Профилирование использования памяти"""
        if not self.is_available:
            return {'error': 'memory_profiler not available'}
        
        try:
            from memory_profiler import profile, memory_usage
            
            # Создаем декорированную функцию
            exec_code = f"""
import memory_profiler

{source_code}

@memory_profiler.profile
def profiled_{function_name}(*args, **kwargs):
    return {function_name}(*args, **kwargs)
"""
            
            namespace = {}
            exec(exec_code, namespace)
            
            profiled_func = namespace[f'profiled_{function_name}']
            
            # Тестовые данные
            test_data = list(range(1000))
            
            # Измеряем использование памяти
            mem_usage = memory_usage((profiled_func, (test_data,)), interval=0.1)
            
            return {
                'success': True,
                'memory_usage': mem_usage,
                'peak_memory': max(mem_usage) if mem_usage else 0,
                'memory_growth': max(mem_usage) - min(mem_usage) if mem_usage else 0
            }
            
        except Exception as e:
            return {'error': f'Memory profiling error: {e}'}

class ScaleneIntegration:
    """Интеграция с Scalene профайлером"""
    
    def __init__(self):
        self.is_available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Проверка доступности Scalene"""
        try:
            result = subprocess.run(['scalene', '--version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def profile_code(self, source_code: str, duration: int = 10) -> Dict[str, Any]:
        """Профилирование с помощью Scalene"""
        if not self.is_available:
            return {'error': 'Scalene not available'}
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(source_code)
                f.flush()
                
                # Запускаем Scalene
                cmd = [
                    'scalene', '--json', '--outfile', '/tmp/scalene_profile.json',
                    f.name
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration)
                
                if result.returncode == 0:
                    # Читаем результаты
                    try:
                        with open('/tmp/scalene_profile.json', 'r') as profile_file:
                            profile_data = json.load(profile_file)
                        return {'success': True, 'profile_data': profile_data}
                    except:
                        return {'error': 'Failed to read Scalene output'}
                else:
                    return {'error': f'Scalene failed: {result.stderr}'}
                    
        except Exception as e:
            return {'error': f'Scalene profiling error: {e}'}
        finally:
            try:
                import os
                os.unlink(f.name)
            except:
                pass

class ToolsIntegrationAnalyzer(BaseComplexityAnalyzer):
    """Анализатор интеграции с внешними инструментами"""
    
    def __init__(self):
        super().__init__("tools_integration", AnalyzerType.TOOLS_INTEGRATION)
        
        # Инициализация интеграций
        self.py_spy = PySpyIntegration()
        self.line_profiler = LineProfilerIntegration()
        self.memory_profiler = MemoryProfilerIntegration()
        self.scalene = ScaleneIntegration()
        
        # Доступные инструменты
        self.available_tools = self._check_available_tools()
    
    def _check_available_tools(self) -> List[str]:
        """Проверка доступных инструментов"""
        tools = []
        
        if self.py_spy.is_available:
            tools.append('py-spy')
        if self.line_profiler.is_available:
            tools.append('line_profiler')
        if self.memory_profiler.is_available:
            tools.append('memory_profiler')
        if self.scalene.is_available:
            tools.append('scalene')
        
        return tools
    
    def is_available(self) -> bool:
        """Проверка доступности"""
        return len(self.available_tools) > 0
    
    def analyze(self, context) -> ComplexityResult:
        """Анализ с помощью внешних инструментов"""
        if not self.available_tools:
            return ComplexityResult(
                complexity_class=ComplexityClass.UNKNOWN,
                confidence=0.0,
                analyzer_name=self.name,
                errors=["No profiling tools available"]
            )
        
        try:
            # Извлекаем главную функцию
            main_function = self._extract_main_function(context.source_code)
            
            # Результаты от всех инструментов
            tool_results = {}
            
            # Line profiler
            if 'line_profiler' in self.available_tools and main_function:
                result = self.line_profiler.profile_function(context.source_code, main_function)
                tool_results['line_profiler'] = result
            
            # Memory profiler
            if 'memory_profiler' in self.available_tools and main_function:
                result = self.memory_profiler.profile_memory(context.source_code, main_function)
                tool_results['memory_profiler'] = result
            
            # py-spy (если код может выполняться длительное время)
            if 'py-spy' in self.available_tools:
                # Только для длительных операций
                pass
            
            # Scalene
            if 'scalene' in self.available_tools:
                result = self.scalene.profile_code(context.source_code, duration=5)
                tool_results['scalene'] = result
            
            # Анализируем результаты
            complexity_class, confidence = self._analyze_tool_results(tool_results)
            
            return ComplexityResult(
                complexity_class=complexity_class,
                confidence=confidence,
                analyzer_name=self.name,
                tool_outputs=tool_results,
                debug_info={
                    'available_tools': self.available_tools,
                    'main_function': main_function,
                    'successful_tools': len([r for r in tool_results.values() 
                                           if r.get('success', False)])
                }
            )
            
        except Exception as e:
            return ComplexityResult(
                complexity_class=ComplexityClass.UNKNOWN,
                confidence=0.0,
                analyzer_name=self.name,
                errors=[f"Tools integration error: {e}"]
            )
    
    def _extract_main_function(self, source_code: str) -> Optional[str]:
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
            for priority_name in ['main', 'solve', 'algorithm', 'run']:
                if priority_name in functions:
                    return priority_name
            
            return functions[0]
            
        except:
            return None
    
    def _analyze_tool_results(self, tool_results: Dict[str, Dict[str, Any]]) -> tuple[ComplexityClass, float]:
        """Анализ результатов от инструментов"""
        complexity_indicators = []
        confidence_scores = []
        
        # Анализ line_profiler
        if 'line_profiler' in tool_results:
            result = tool_results['line_profiler']
            if result.get('success'):
                complexity, confidence = self._analyze_line_profiler_result(result)
                complexity_indicators.append(complexity)
                confidence_scores.append(confidence)
        
        # Анализ memory_profiler
        if 'memory_profiler' in tool_results:
            result = tool_results['memory_profiler']
            if result.get('success'):
                complexity, confidence = self._analyze_memory_profiler_result(result)
                complexity_indicators.append(complexity)
                confidence_scores.append(confidence)
        
        # Анализ Scalene
        if 'scalene' in tool_results:
            result = tool_results['scalene']
            if result.get('success'):
                complexity, confidence = self._analyze_scalene_result(result)
                complexity_indicators.append(complexity)
                confidence_scores.append(confidence)
        
        # Определяем итоговую сложность
        if not complexity_indicators:
            return ComplexityClass.UNKNOWN, 0.0
        
        # Простое голосование или среднее
        from collections import Counter
        complexity_votes = Counter(complexity_indicators)
        final_complexity = complexity_votes.most_common(1)[0][0]
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return final_complexity, avg_confidence
    
    def _analyze_line_profiler_result(self, result: Dict[str, Any]) -> tuple[ComplexityClass, float]:
        """Анализ результатов line_profiler"""
        stats = result.get('stats', {})
        
        if not stats:
            return ComplexityClass.UNKNOWN, 0.1
        
        # Ищем самые медленные строки
        max_time_per_hit = 0
        total_hits = 0
        
        for func_name, func_stats in stats.items():
            for line_timing in func_stats.get('line_timings', []):
                time_per_hit = line_timing.get('time_per_hit', 0)
                hits = line_timing.get('hits', 0)
                
                max_time_per_hit = max(max_time_per_hit, time_per_hit)
                total_hits += hits
        
        # Простая эвристика
        if total_hits > 1000:
            return ComplexityClass.QUADRATIC, 0.6
        elif total_hits > 100:
            return ComplexityClass.LINEAR, 0.7
        else:
            return ComplexityClass.CONSTANT, 0.8
    
    def _analyze_memory_profiler_result(self, result: Dict[str, Any]) -> tuple[ComplexityClass, float]:
        """Анализ результатов memory_profiler"""
        memory_growth = result.get('memory_growth', 0)
        peak_memory = result.get('peak_memory', 0)
        
        # Анализируем рост памяти
        if memory_growth > 100:  # МБ
            return ComplexityClass.QUADRATIC, 0.6
        elif memory_growth > 10:
            return ComplexityClass.LINEAR, 0.7
        else:
            return ComplexityClass.CONSTANT, 0.8
    
    def _analyze_scalene_result(self, result: Dict[str, Any]) -> tuple[ComplexityClass, float]:
        """Анализ результатов Scalene"""
        profile_data = result.get('profile_data', {})
        
        # Упрощенный анализ - возвращаем среднюю оценку
        return ComplexityClass.LINEAR, 0.5
