"""Расширенный AST-анализатор с детекцией сложных паттернов"""
import ast
import math
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict, deque

from complexity_analyzers.core.base import BaseComplexityAnalyzer, AnalyzerType, AnalysisContext
from core.result import ComplexityResult, ComplexityClass, ComplexityMetrics
from core.enums import PatternType, DataStructureUsage
from .ast_patterns import PatternDetectorRegistry
from .ast_features import ASTFeatureExtractor

class AdvancedLoopAnalyzer(ast.NodeVisitor):
    """Продвинутый анализатор циклов"""
    
    def __init__(self):
        self.loops: List[Dict[str, Any]] = []
        self.nesting_stack: List[Dict[str, Any]] = []
        self.current_function: Optional[str] = None
        self.max_nesting: int = 0
        self.loop_variables: Set[str] = set()
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Обработка определения функции"""
        prev_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = prev_function
    
    def visit_For(self, node: ast.For):
        """Анализ for-цикла"""
        loop_info = self._analyze_loop(node, 'for')
        self._enter_loop(loop_info)
        self.generic_visit(node)
        self._exit_loop()
    
    def visit_While(self, node: ast.While):
        """Анализ while-цикла"""
        loop_info = self._analyze_loop(node, 'while')
        self._enter_loop(loop_info)
        self.generic_visit(node)
        self._exit_loop()
    
    def visit_ListComp(self, node: ast.ListComp):
        """Анализ list comprehension"""
        loop_info = self._analyze_comprehension(node, 'list_comp')
        self._enter_loop(loop_info)
        self.generic_visit(node)
        self._exit_loop()
    
    def visit_DictComp(self, node: ast.DictComp):
        """Анализ dict comprehension"""
        loop_info = self._analyze_comprehension(node, 'dict_comp')
        self._enter_loop(loop_info)
        self.generic_visit(node)
        self._exit_loop()
    
    def _analyze_loop(self, node: ast.AST, loop_type: str) -> Dict[str, Any]:
        """Анализ характеристик цикла"""
        loop_info = {
            'type': loop_type,
            'line': getattr(node, 'lineno', 0),
            'nesting_level': len(self.nesting_stack),
            'function': self.current_function,
            'variables': set(),
            'complexity_indicators': []
        }
        
        if isinstance(node, ast.For):
            # Анализ переменной цикла и итерируемого объекта
            if isinstance(node.target, ast.Name):
                loop_info['variables'].add(node.target.id)
                self.loop_variables.add(node.target.id)
            
            # Анализ итерируемого объекта
            loop_info['iteration_complexity'] = self._analyze_iteration_complexity(node.iter)
            
        elif isinstance(node, ast.While):
            # Анализ условия while
            loop_info['condition_complexity'] = self._analyze_condition_complexity(node.test)
            
            # Поиск переменных в условии
            for var_node in ast.walk(node.test):
                if isinstance(var_node, ast.Name):
                    loop_info['variables'].add(var_node.id)
        
        return loop_info
    
    def _analyze_comprehension(self, node: ast.AST, comp_type: str) -> Dict[str, Any]:
        """Анализ list/dict comprehension"""
        return {
            'type': comp_type,
            'line': getattr(node, 'lineno', 0),
            'nesting_level': len(self.nesting_stack),
            'function': self.current_function,
            'variables': set(),
            'complexity_indicators': ['comprehension']
        }
    
    def _analyze_iteration_complexity(self, iter_node: ast.AST) -> str:
        """Анализ сложности итерации"""
        if isinstance(iter_node, ast.Call):
            if isinstance(iter_node.func, ast.Name):
                func_name = iter_node.func.id
                if func_name == 'range':
                    return 'linear'
                elif func_name == 'enumerate':
                    return 'linear'
                elif func_name == 'zip':
                    return 'linear'
                elif func_name in ['sorted', 'reversed']:
                    return 'linearithmic'
        elif isinstance(iter_node, ast.Name):
            return 'depends_on_container'
        elif isinstance(iter_node, ast.List):
            return 'constant_list'
        
        return 'unknown'
    
    def _analyze_condition_complexity(self, condition: ast.AST) -> str:
        """Анализ сложности условия"""
        if isinstance(condition, ast.Compare):
            # Простое сравнение
            return 'constant'
        elif isinstance(condition, ast.BoolOp):
            # Логические операции
            return 'constant'
        elif isinstance(condition, ast.Call):
            # Вызов функции в условии
            return 'function_call'
        
        return 'unknown'
    
    def _enter_loop(self, loop_info: Dict[str, Any]):
        """Вход в цикл"""
        self.nesting_stack.append(loop_info)
        self.max_nesting = max(self.max_nesting, len(self.nesting_stack))
        self.loops.append(loop_info)
    
    def _exit_loop(self):
        """Выход из цикла"""
        if self.nesting_stack:
            self.nesting_stack.pop()

class RecursionComplexityAnalyzer(ast.NodeVisitor):
    """Анализатор рекурсивной сложности"""
    
    def __init__(self):
        self.functions: Dict[str, Dict[str, Any]] = {}
        self.call_graph: Dict[str, Set[str]] = defaultdict(set)
        self.current_function: Optional[str] = None
        self.recursion_patterns: List[Dict[str, Any]] = []
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Анализ определения функции"""
        func_name = node.name
        self.functions[func_name] = {
            'name': func_name,
            'line': node.lineno,
            'args_count': len(node.args.args),
            'calls': set(),
            'is_recursive': False,
            'recursion_type': 'none',
            'base_cases': [],
            'recursive_calls': []
        }
        
        prev_function = self.current_function
        self.current_function = func_name
        self.call_graph[func_name] = set()
        
        self.generic_visit(node)
        
        # Анализируем рекурсию после обхода тела функции
        self._analyze_recursion_pattern(func_name)
        
        self.current_function = prev_function
    
    def visit_Call(self, node: ast.Call):
        """Анализ вызовов функций"""
        if self.current_function and isinstance(node.func, ast.Name):
            called_func = node.func.id
            self.call_graph[self.current_function].add(called_func)
            self.functions[self.current_function]['calls'].add(called_func)
            
            # Если это рекурсивный вызов
            if called_func == self.current_function:
                self.functions[self.current_function]['is_recursive'] = True
                self.functions[self.current_function]['recursive_calls'].append({
                    'line': node.lineno,
                    'args_count': len(node.args),
                    'context': self._get_call_context(node)
                })
        
        self.generic_visit(node)
    
    def visit_Return(self, node: ast.Return):
        """Анализ return-ов (поиск базовых случаев)"""
        if self.current_function:
            # Простая эвристика для определения базового случая
            if node.value and not self._contains_recursive_call(node.value):
                self.functions[self.current_function]['base_cases'].append({
                    'line': node.lineno,
                    'is_constant': self._is_constant_return(node.value)
                })
        
        self.generic_visit(node)
    
    def _analyze_recursion_pattern(self, func_name: str):
        """Анализ паттерна рекурсии"""
        func_info = self.functions[func_name]
        
        if not func_info['is_recursive']:
            return
        
        recursive_calls = func_info['recursive_calls']
        base_cases = func_info['base_cases']
        
        # Определяем тип рекурсии
        if len(recursive_calls) == 1:
            func_info['recursion_type'] = 'linear'
        elif len(recursive_calls) == 2:
            func_info['recursion_type'] = 'binary'
        elif len(recursive_calls) > 2:
            func_info['recursion_type'] = 'tree'
        
        # Анализируем сложность
        complexity = self._infer_recursion_complexity(func_info)
        func_info['estimated_complexity'] = complexity
        
        self.recursion_patterns.append(func_info)
    
    def _get_call_context(self, call_node: ast.Call) -> str:
        """Получение контекста вызова"""
        # Упрощенная реализация
        return f"args_count:{len(call_node.args)}"
    
    def _contains_recursive_call(self, node: ast.AST) -> bool:
        """Проверка наличия рекурсивного вызова в узле"""
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                if child.func.id == self.current_function:
                    return True
        return False
    
    def _is_constant_return(self, node: ast.AST) -> bool:
        """Проверка константного возвращаемого значения"""
        return isinstance(node, (ast.Constant, ast.Num, ast.Str))
    
    def _infer_recursion_complexity(self, func_info: Dict[str, Any]) -> ComplexityClass:
        """Вывод сложности рекурсии"""
        recursion_type = func_info['recursion_type']
        recursive_calls_count = len(func_info['recursive_calls'])
        
        if recursion_type == 'linear':
            return ComplexityClass.LINEAR
        elif recursion_type == 'binary':
            return ComplexityClass.EXPONENTIAL
        elif recursion_type == 'tree':
            if recursive_calls_count <= 2:
                return ComplexityClass.EXPONENTIAL
            else:
                return ComplexityClass.FACTORIAL
        
        return ComplexityClass.UNKNOWN

class DataStructureAnalyzer(ast.NodeVisitor):
    """Анализатор использования структур данных"""
    
    def __init__(self):
        self.data_structures: Dict[DataStructureUsage, int] = defaultdict(int)
        self.operations: Dict[str, int] = defaultdict(int)
        self.complexity_operations: List[Dict[str, Any]] = []
    
    def visit_Call(self, node: ast.Call):
        """Анализ вызовов методов и функций"""
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            
            # Операции со списками
            if method_name in ['append', 'insert', 'pop', 'remove']:
                self.operations[f'list_{method_name}'] += 1
                self._record_operation('list', method_name, node.lineno)
            
            # Операции со словарями
            elif method_name in ['get', 'keys', 'values', 'items', 'pop']:
                self.operations[f'dict_{method_name}'] += 1
                self._record_operation('dict', method_name, node.lineno)
            
            # Операции с множествами
            elif method_name in ['add', 'remove', 'union', 'intersection']:
                self.operations[f'set_{method_name}'] += 1
                self._record_operation('set', method_name, node.lineno)
        
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            # Конструкторы структур данных
            if func_name in ['list', 'dict', 'set', 'tuple']:
                self.data_structures[DataStructureUsage.ARRAY_LIST if func_name == 'list' 
                                  else DataStructureUsage.DICTIONARY_HASH if func_name == 'dict'
                                  else DataStructureUsage.SET] += 1
            
            # Функции с известной сложностью
            elif func_name in ['sorted', 'max', 'min', 'sum']:
                self._record_operation('builtin', func_name, node.lineno)
        
        self.generic_visit(node)
    
    def visit_Subscript(self, node: ast.Subscript):
        """Анализ обращений по индексу"""
        if isinstance(node.ctx, ast.Load):
            self.operations['subscript_access'] += 1
        elif isinstance(node.ctx, ast.Store):
            self.operations['subscript_assignment'] += 1
        
        self.generic_visit(node)
    
    def _record_operation(self, structure_type: str, operation: str, line: int):
        """Запись операции с известной сложностью"""
        complexity_map = {
            ('list', 'append'): ComplexityClass.CONSTANT,
            ('list', 'insert'): ComplexityClass.LINEAR,
            ('list', 'pop'): ComplexityClass.CONSTANT,
            ('list', 'remove'): ComplexityClass.LINEAR,
            ('dict', 'get'): ComplexityClass.CONSTANT,
            ('dict', 'keys'): ComplexityClass.LINEAR,
            ('set', 'add'): ComplexityClass.CONSTANT,
            ('set', 'union'): ComplexityClass.LINEAR,
            ('builtin', 'sorted'): ComplexityClass.LINEARITHMIC,
            ('builtin', 'max'): ComplexityClass.LINEAR,
            ('builtin', 'min'): ComplexityClass.LINEAR,
            ('builtin', 'sum'): ComplexityClass.LINEAR,
        }
        
        key = (structure_type, operation)
        if key in complexity_map:
            self.complexity_operations.append({
                'structure': structure_type,
                'operation': operation,
                'line': line,
                'complexity': complexity_map[key]
            })

class AdvancedASTAnalyzer(BaseComplexityAnalyzer):
    """Расширенный AST-анализатор"""
    
    def __init__(self):
        super().__init__("ast_advanced", AnalyzerType.STATIC_AST)
        
        # Специализированные анализаторы
        self.loop_analyzer = AdvancedLoopAnalyzer()
        self.recursion_analyzer = RecursionComplexityAnalyzer()
        self.data_structure_analyzer = DataStructureAnalyzer()
        
        # Реестры детекторов и экстракторов
        self.pattern_detector_registry = PatternDetectorRegistry()
        self.feature_extractor = ASTFeatureExtractor()
    
    def is_available(self) -> bool:
        """Проверка доступности"""
        return True
    
    def analyze(self, context: AnalysisContext) -> ComplexityResult:
        """Расширенный анализ AST"""
        try:
            tree = ast.parse(context.source_code)
            
            # Сброс состояния анализаторов
            self._reset_analyzers()
            
            # Запуск всех анализаторов
            self.loop_analyzer.visit(tree)
            self.recursion_analyzer.visit(tree)
            self.data_structure_analyzer.visit(tree)
            
            # Детекция паттернов
            detected_patterns = self.pattern_detector_registry.detect_all(tree)
            
            # Извлечение признаков
            features = self.feature_extractor.extract_all_features(tree, context.source_code)
            
            # Объединение результатов анализа
            analysis_results = self._combine_analysis_results(
                detected_patterns, features
            )
            
            # Определение итоговой сложности
            complexity_class = self._determine_complexity(analysis_results)
            confidence = self._calculate_confidence(analysis_results)
            
            # Создание метрик
            metrics = self._create_metrics(analysis_results, complexity_class)
            
            return ComplexityResult(
                complexity_class=complexity_class,
                confidence=confidence,
                analyzer_name=self.name,
                metrics=metrics,
                ast_features=features,
                debug_info={
                    'detected_patterns': detected_patterns,
                    'loop_analysis': self._serialize_loop_analysis(),
                    'recursion_analysis': self._serialize_recursion_analysis(),
                    'data_structure_analysis': self._serialize_data_structure_analysis()
                }
            )
            
        except SyntaxError as e:
            return ComplexityResult(
                complexity_class=ComplexityClass.UNKNOWN,
                confidence=0.0,
                analyzer_name=self.name,
                errors=[f"Syntax error: {e}"]
            )
        except Exception as e:
            return ComplexityResult(
                complexity_class=ComplexityClass.UNKNOWN,
                confidence=0.0,
                analyzer_name=self.name,
                errors=[f"Advanced AST analysis error: {e}"]
            )
    
    def _reset_analyzers(self):
        """Сброс состояния всех анализаторов"""
        self.loop_analyzer = AdvancedLoopAnalyzer()
        self.recursion_analyzer = RecursionComplexityAnalyzer()
        self.data_structure_analyzer = DataStructureAnalyzer()
    
    def _combine_analysis_results(self, patterns: Dict[str, Any], 
                                features: Dict[str, Any]) -> Dict[str, Any]:
        """Объединение результатов всех анализов"""
        return {
            'patterns': patterns,
            'features': features,
            'loop_analysis': {
                'total_loops': len(self.loop_analyzer.loops),
                'max_nesting': self.loop_analyzer.max_nesting,
                'loops': self.loop_analyzer.loops
            },
            'recursion_analysis': {
                'recursive_functions': len([f for f in self.recursion_analyzer.functions.values() 
                                          if f['is_recursive']]),
                'patterns': self.recursion_analyzer.recursion_patterns
            },
            'data_structure_analysis': {
                'structures_used': dict(self.data_structure_analyzer.data_structures),
                'operations': dict(self.data_structure_analyzer.operations),
                'complexity_operations': self.data_structure_analyzer.complexity_operations
            }
        }
    
    def _determine_complexity(self, analysis_results: Dict[str, Any]) -> ComplexityClass:
        """Определение итоговой сложности на основе всех анализов"""
        complexity_indicators = []
        
        # Анализ циклов
        loop_analysis = analysis_results['loop_analysis']
        max_nesting = loop_analysis['max_nesting']
        
        if max_nesting >= 3:
            complexity_indicators.append(ComplexityClass.CUBIC)
        elif max_nesting == 2:
            complexity_indicators.append(ComplexityClass.QUADRATIC)
        elif max_nesting == 1:
            complexity_indicators.append(ComplexityClass.LINEAR)
        
        # Анализ рекурсии
        recursion_analysis = analysis_results['recursion_analysis']
        for pattern in recursion_analysis['patterns']:
            if 'estimated_complexity' in pattern:
                complexity_indicators.append(pattern['estimated_complexity'])
        
        # Анализ операций со структурами данных
        data_analysis = analysis_results['data_structure_analysis']
        for op in data_analysis['complexity_operations']:
            complexity_indicators.append(op['complexity'])
        
        # Анализ паттернов
        patterns = analysis_results['patterns']
        for pattern_type, pattern_data in patterns.items():
            if pattern_data.get('detected_patterns'):
                # Примерная оценка сложности на основе паттернов
                if 'sorting' in pattern_type:
                    complexity_indicators.append(ComplexityClass.LINEARITHMIC)
                elif 'search' in pattern_type and 'binary' in str(pattern_data):
                    complexity_indicators.append(ComplexityClass.LOGARITHMIC)
        
        # Выбор максимальной сложности
        if complexity_indicators:
            return max(complexity_indicators)
        else:
            return ComplexityClass.CONSTANT
    
    def _calculate_confidence(self, analysis_results: Dict[str, Any]) -> float:
        """Расчет уверенности в результате"""
        base_confidence = 0.8
        
        # Увеличиваем уверенность при наличии четких индикаторов
        loop_count = analysis_results['loop_analysis']['total_loops']
        recursive_count = analysis_results['recursion_analysis']['recursive_functions']
        pattern_count = sum(1 for p in analysis_results['patterns'].values() 
                          if p.get('detected_patterns'))
        
        if loop_count > 0:
            base_confidence += 0.1
        if recursive_count > 0:
            base_confidence += 0.1
        if pattern_count > 0:
            base_confidence += 0.05
        
        return min(base_confidence, 1.0)
    
    def _create_metrics(self, analysis_results: Dict[str, Any], 
                       complexity_class: ComplexityClass) -> ComplexityMetrics:
        """Создание объекта метрик"""
        loop_analysis = analysis_results['loop_analysis']
        recursion_analysis = analysis_results['recursion_analysis']
        
        return ComplexityMetrics(
            time_complexity=complexity_class,
            nested_depth=loop_analysis['max_nesting'],
            loop_count=loop_analysis['total_loops'],
            recursive_calls=recursion_analysis['recursive_functions']
        )
    
    def _serialize_loop_analysis(self) -> Dict[str, Any]:
        """Сериализация результатов анализа циклов"""
        return {
            'total_loops': len(self.loop_analyzer.loops),
            'max_nesting': self.loop_analyzer.max_nesting,
            'loop_details': [
                {
                    'type': loop['type'],
                    'line': loop['line'],
                    'nesting_level': loop['nesting_level'],
                    'function': loop['function']
                }
                for loop in self.loop_analyzer.loops
            ]
        }
    
    def _serialize_recursion_analysis(self) -> Dict[str, Any]:
        """Сериализация результатов анализа рекурсии"""
        return {
            'total_functions': len(self.recursion_analyzer.functions),
            'recursive_functions': len([f for f in self.recursion_analyzer.functions.values() 
                                      if f['is_recursive']]),
            'recursion_patterns': [
                {
                    'function_name': pattern['name'],
                    'recursion_type': pattern['recursion_type'],
                    'estimated_complexity': pattern.get('estimated_complexity', ComplexityClass.UNKNOWN).notation
                }
                for pattern in self.recursion_analyzer.recursion_patterns
            ]
        }
    
    def _serialize_data_structure_analysis(self) -> Dict[str, Any]:
        """Сериализация результатов анализа структур данных"""
        return {
            'structures_used': {k.value: v for k, v in self.data_structure_analyzer.data_structures.items()},
            'total_operations': sum(self.data_structure_analyzer.operations.values()),
            'high_complexity_operations': len([op for op in self.data_structure_analyzer.complexity_operations
                                             if op['complexity'].complexity_order >= 3])
        }
