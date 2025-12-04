"""
Анализатор циклов в AST.
"""

import ast
from typing import Dict, Any, List, Set


class LoopAnalyzer(ast.NodeVisitor):
    """
    Анализатор циклов с детекцией:
    - Уровня вложенности
    - Логарифмических шагов (i *= 2, i //= 2)
    - Зависимых вложенных циклов (внутренний цикл зависит от переменной внешнего)
    """
    
    def __init__(self):
        self.loops: List[Dict[str, Any]] = []
        self.nesting_stack: List[Dict[str, Any]] = []
        self.current_function: str = None
        self.max_nesting: int = 0
        self.loop_variables: Set[str] = set()
        
        # Индикаторы сложности
        self.has_logarithmic_step: bool = False
        self.has_dependent_inner_loop: bool = False
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Отслеживание текущей функции"""
        prev_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = prev_function
    
    def visit_For(self, node: ast.For):
        """Анализ for-цикла"""
        loop_info = self._analyze_for_loop(node)
        self._enter_loop(loop_info)
        self.generic_visit(node)
        self._exit_loop()
    
    def visit_While(self, node: ast.While):
        """Анализ while-цикла"""
        loop_info = self._analyze_while_loop(node)
        self._enter_loop(loop_info)
        self.generic_visit(node)
        self._exit_loop()
    
    def visit_ListComp(self, node: ast.ListComp):
        """Анализ list comprehension"""
        loop_info = {
            'type': 'list_comp',
            'line': node.lineno,
            'nesting_level': len(self.nesting_stack),
            'function': self.current_function
        }
        self._enter_loop(loop_info)
        self.generic_visit(node)
        self._exit_loop()
    
    def visit_DictComp(self, node: ast.DictComp):
        """Анализ dict comprehension"""
        loop_info = {
            'type': 'dict_comp',
            'line': node.lineno,
            'nesting_level': len(self.nesting_stack),
            'function': self.current_function
        }
        self._enter_loop(loop_info)
        self.generic_visit(node)
        self._exit_loop()
    
    def visit_AugAssign(self, node: ast.AugAssign):
        """Детекция логарифмических шагов (i *= 2, i //= 2)"""
        if isinstance(node.op, (ast.Mult, ast.Div, ast.FloorDiv)):
            if self.nesting_stack:
                self.has_logarithmic_step = True
                self.nesting_stack[-1].setdefault('complexity_indicators', []).append('logarithmic_step')
        self.generic_visit(node)
    
    def _analyze_for_loop(self, node: ast.For) -> Dict[str, Any]:
        """Анализ for-цикла"""
        loop_info = {
            'type': 'for',
            'line': node.lineno,
            'nesting_level': len(self.nesting_stack),
            'function': self.current_function,
            'variables': set(),
            'complexity_indicators': []
        }
        
        # Переменная цикла
        if isinstance(node.target, ast.Name):
            loop_info['variables'].add(node.target.id)
            self.loop_variables.add(node.target.id)
        
        # Проверка зависимости от внешнего цикла
        if isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                for arg in node.iter.args:
                    if isinstance(arg, ast.Name) and arg.id in self.loop_variables:
                        self.has_dependent_inner_loop = True
                        loop_info['complexity_indicators'].append('dependent_range')
        
        return loop_info
    
    def _analyze_while_loop(self, node: ast.While) -> Dict[str, Any]:
        """Анализ while-цикла"""
        loop_info = {
            'type': 'while',
            'line': node.lineno,
            'nesting_level': len(self.nesting_stack),
            'function': self.current_function,
            'variables': set(),
            'complexity_indicators': []
        }
        
        # Переменные в условии
        for var_node in ast.walk(node.test):
            if isinstance(var_node, ast.Name):
                loop_info['variables'].add(var_node.id)
        
        return loop_info
    
    def _enter_loop(self, loop_info: Dict[str, Any]):
        """Вход в цикл"""
        self.nesting_stack.append(loop_info)
        self.max_nesting = max(self.max_nesting, len(self.nesting_stack))
        self.loops.append(loop_info)
    
    def _exit_loop(self):
        """Выход из цикла"""
        if self.nesting_stack:
            self.nesting_stack.pop()
    
    def get_summary(self) -> Dict[str, Any]:
        """Сводка анализа циклов"""
        return {
            'total_loops': len(self.loops),
            'max_nesting': self.max_nesting,
            'has_logarithmic_step': self.has_logarithmic_step,
            'has_dependent_inner_loop': self.has_dependent_inner_loop,
            'loops': self.loops
        }
