"""
Анализатор рекурсии в AST.
"""

import ast
from typing import Dict, Any, Set
from collections import defaultdict
from ast_analysis.core.enums import ComplexityClass


class RecursionAnalyzer(ast.NodeVisitor):
    """
    Анализатор рекурсивных функций.
    Определяет тип рекурсии (linear, binary, tree) и оценивает сложность.
    """
    
    def __init__(self):
        self.functions: Dict[str, Dict[str, Any]] = {}
        self.call_graph: Dict[str, Set[str]] = defaultdict(set)
        self.current_function: str = None
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
            'recursive_calls': [],
            'base_cases': []
        }
        
        prev_function = self.current_function
        self.current_function = func_name
        self.call_graph[func_name] = set()
        
        self.generic_visit(node)
        
        # Анализ после обхода
        self._analyze_recursion_pattern(func_name)
        
        self.current_function = prev_function
    
    def visit_Call(self, node: ast.Call):
        """Анализ вызовов функций"""
        if self.current_function and isinstance(node.func, ast.Name):
            called_func = node.func.id
            self.call_graph[self.current_function].add(called_func)
            self.functions[self.current_function]['calls'].add(called_func)
            
            # Рекурсивный вызов
            if called_func == self.current_function:
                self.functions[self.current_function]['is_recursive'] = True
                self.functions[self.current_function]['recursive_calls'].append({
                    'line': node.lineno,
                    'args_count': len(node.args)
                })
        
        self.generic_visit(node)
    
    def visit_Return(self, node: ast.Return):
        """Анализ return (поиск базовых случаев)"""
        if self.current_function:
            if node.value and not self._contains_recursive_call(node.value):
                self.functions[self.current_function]['base_cases'].append({
                    'line': node.lineno,
                    'is_constant': isinstance(node.value, (ast.Constant, ast.Num, ast.Str))
                })
        
        self.generic_visit(node)
    
    def _analyze_recursion_pattern(self, func_name: str):
        """Определение типа рекурсии и оценка сложности"""
        func_info = self.functions[func_name]
        
        if not func_info['is_recursive']:
            return
        
        recursive_calls_count = len(func_info['recursive_calls'])
        
        # Тип рекурсии
        if recursive_calls_count == 1:
            func_info['recursion_type'] = 'linear'
            func_info['estimated_complexity'] = ComplexityClass.LINEAR
        elif recursive_calls_count == 2:
            func_info['recursion_type'] = 'binary'
            func_info['estimated_complexity'] = ComplexityClass.EXPONENTIAL
        elif recursive_calls_count > 2:
            func_info['recursion_type'] = 'tree'
            func_info['estimated_complexity'] = ComplexityClass.FACTORIAL
        
        self.recursion_patterns.append(func_info)
    
    def _contains_recursive_call(self, node: ast.AST) -> bool:
        """Проверка рекурсивного вызова в узле"""
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                if child.func.id == self.current_function:
                    return True
        return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Сводка анализа рекурсии"""
        return {
            'total_functions': len(self.functions),
            'recursive_functions': len([f for f in self.functions.values() if f['is_recursive']]),
            'recursion_patterns': self.recursion_patterns
        }
