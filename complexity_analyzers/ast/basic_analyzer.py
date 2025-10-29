"""Базовый AST-анализатор"""
import ast
from typing import Dict, Any, List, Optional, Set
from complexity_analyzers.base.analyzer import BaseComplexityAnalyzer, AnalyzerType
from complexity_analyzers.base.result import ComplexityResult, ComplexityClass, ComplexityMetrics

class ASTNodeCounter(ast.NodeVisitor):
    """Счетчик AST-узлов"""
    
    def __init__(self):
        self.node_counts: Dict[str, int] = {}
        self.total_nodes: int = 0
    
    def visit(self, node: ast.AST) -> None:
        """Посещение узла"""
        node_type = type(node).__name__
        self.node_counts[node_type] = self.node_counts.get(node_type, 0) + 1
        self.total_nodes += 1
        self.generic_visit(node)
    
    def get_counts(self) -> Dict[str, int]:
        """Получение счетчиков"""
        return self.node_counts.copy()

class LoopAnalyzer(ast.NodeVisitor):
    """Анализатор циклов"""
    
    def __init__(self):
        self.loops: List[Dict[str, Any]] = []
        self.nesting_depth: int = 0
        self.max_nesting: int = 0
        self.current_function: Optional[str] = None
    
    def visit_For(self, node: ast.For) -> None:
        """Обработка for-цикла"""
        self.nesting_depth += 1
        self.max_nesting = max(self.max_nesting, self.nesting_depth)
        
        loop_info = {
            'type': 'for',
            'line': node.lineno,
            'nesting_level': self.nesting_depth,
            'function': self.current_function,
            'has_else': bool(node.orelse)
        }
        self.loops.append(loop_info)
        
        self.generic_visit(node)
        self.nesting_depth -= 1
    
    def visit_While(self, node: ast.While) -> None:
        """Обработка while-цикла"""
        self.nesting_depth += 1
        self.max_nesting = max(self.max_nesting, self.nesting_depth)
        
        loop_info = {
            'type': 'while',
            'line': node.lineno,
            'nesting_level': self.nesting_depth,
            'function': self.current_function,
            'has_else': bool(node.orelse)
        }
        self.loops.append(loop_info)
        
        self.generic_visit(node)
        self.nesting_depth -= 1
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Обработка определения функции"""
        prev_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = prev_function

class RecursionAnalyzer(ast.NodeVisitor):
    """Анализатор рекурсии"""
    
    def __init__(self):
        self.functions: Dict[str, Dict[str, Any]] = {}
        self.current_function: Optional[str] = None
        self.call_graph: Dict[str, Set[str]] = {}
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Обработка определения функции"""
        func_name = node.name
        self.functions[func_name] = {
            'name': func_name,
            'line': node.lineno,
            'args_count': len(node.args.args),
            'calls': set(),
            'is_recursive': False
        }
        
        prev_function = self.current_function
        self.current_function = func_name
        self.call_graph[func_name] = set()
        
        self.generic_visit(node)
        
        # Проверяем рекурсию
        if func_name in self.call_graph[func_name]:
            self.functions[func_name]['is_recursive'] = True
        
        self.current_function = prev_function
    
    def visit_Call(self, node: ast.Call) -> None:
        """Обработка вызова функции"""
        if self.current_function and isinstance(node.func, ast.Name):
            called_func = node.func.id
            self.call_graph[self.current_function].add(called_func)
            self.functions[self.current_function]['calls'].add(called_func)
        
        self.generic_visit(node)
    
    def detect_mutual_recursion(self) -> List[List[str]]:
        """Обнаружение взаимной рекурсии"""
        cycles = []
        visited = set()
        
        def dfs(node: str, path: List[str]) -> None:
            if node in path:
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                if len(cycle) > 2:  # Взаимная рекурсия
                    cycles.append(cycle)
                return
            
            if node in visited or node not in self.call_graph:
                return
            
            visited.add(node)
            for neighbor in self.call_graph[node]:
                dfs(neighbor, path + [node])
        
        for func in self.functions:
            if func not in visited:
                dfs(func, [])
        
        return cycles

class BasicASTAnalyzer(BaseComplexityAnalyzer):
    """Базовый анализатор AST"""
    
    def __init__(self):
        super().__init__("ast_basic", AnalyzerType.STATIC_AST)
        self.node_counter: Optional[ASTNodeCounter] = None
        self.loop_analyzer: Optional[LoopAnalyzer] = None
        self.recursion_analyzer: Optional[RecursionAnalyzer] = None
    
    def is_available(self) -> bool:
        """Проверка доступности"""
        return True  # AST всегда доступен в Python
    
    def analyze(self, context) -> ComplexityResult:
        """Анализ кода через AST"""
        try:
            tree = ast.parse(context.source_code)
            
            # Инициализируем анализаторы
            self.node_counter = ASTNodeCounter()
            self.loop_analyzer = LoopAnalyzer()
            self.recursion_analyzer = RecursionAnalyzer()
            
            # Проходим по дереву
            self.node_counter.visit(tree)
            self.loop_analyzer.visit(tree)
            self.recursion_analyzer.visit(tree)
            
            # Извлекаем признаки
            features = self._extract_features()
            
            # Определяем сложность
            complexity_class = self._infer_complexity(features)
            confidence = self._calculate_confidence(features)
            
            return ComplexityResult(
                complexity_class=complexity_class,
                confidence=confidence,
                analyzer_name=self.name,
                metrics=ComplexityMetrics(
                    time_complexity=complexity_class,
                    nested_depth=features.get('max_nesting', 0),
                    loop_count=features.get('total_loops', 0),
                    recursive_calls=features.get('recursive_functions', 0)
                ),
                ast_features=features
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
                errors=[f"Analysis error: {e}"]
            )
    
    def _extract_features(self) -> Dict[str, Any]:
        """Извлечение признаков из AST"""
        features = {
            # Узлы
            'total_nodes': self.node_counter.total_nodes,
            'node_types': self.node_counter.get_counts(),
            
            # Циклы
            'total_loops': len(self.loop_analyzer.loops),
            'max_nesting': self.loop_analyzer.max_nesting,
            'for_loops': sum(1 for loop in self.loop_analyzer.loops if loop['type'] == 'for'),
            'while_loops': sum(1 for loop in self.loop_analyzer.loops if loop['type'] == 'while'),
            
            # Рекурсия
            'total_functions': len(self.recursion_analyzer.functions),
            'recursive_functions': sum(1 for f in self.recursion_analyzer.functions.values() 
                                     if f['is_recursive']),
            'mutual_recursion_cycles': self.recursion_analyzer.detect_mutual_recursion(),
            
            # Сложные конструкции
            'list_comprehensions': self.node_counter.node_counts.get('ListComp', 0),
            'dict_comprehensions': self.node_counter.node_counts.get('DictComp', 0),
            'generator_expressions': self.node_counter.node_counts.get('GeneratorExp', 0),
            'lambda_functions': self.node_counter.node_counts.get('Lambda', 0),
        }
        
        return features
    
    def _infer_complexity(self, features: Dict[str, Any]) -> ComplexityClass:
        """Вывод класса сложности из признаков"""
        max_nesting = features.get('max_nesting', 0)
        total_loops = features.get('total_loops', 0)
        recursive_functions = features.get('recursive_functions', 0)
        
        # Рекурсивные функции
        if recursive_functions > 0:
            mutual_cycles = features.get('mutual_recursion_cycles', [])
            if mutual_cycles:
                return ComplexityClass.EXPONENTIAL
            return ComplexityClass.EXPONENTIAL  # Консервативная оценка
        
        # Вложенные циклы
        if max_nesting >= 3:
            return ComplexityClass.CUBIC
        elif max_nesting == 2:
            return ComplexityClass.QUADRATIC
        elif max_nesting == 1 or total_loops > 0:
            return ComplexityClass.LINEAR
        else:
            return ComplexityClass.CONSTANT
    
    def _calculate_confidence(self, features: Dict[str, Any]) -> float:
        """Расчет уверенности в результате"""
        base_confidence = 0.7
        
        # Увеличиваем уверенность при наличии четких паттернов
        if features.get('max_nesting', 0) > 0:
            base_confidence += 0.2
        
        if features.get('recursive_functions', 0) > 0:
            base_confidence += 0.1
        
        # Уменьшаем при сложных конструкциях
        complex_constructs = (
            features.get('list_comprehensions', 0) +
            features.get('lambda_functions', 0) +
            features.get('generator_expressions', 0)
        )
        
        if complex_constructs > 0:
            base_confidence -= 0.1 * min(complex_constructs, 3)
        
        return max(0.1, min(base_confidence, 1.0))
