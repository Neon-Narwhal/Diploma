"""
Анализатор диапазонов итераторов для CFG.
Определяет границы циклов и паттерны роста переменных.
"""

import ast
import networkx as nx
from typing import Dict, Any, List, Optional
from collections import defaultdict


class IteratorRangeAnalyzer:
    """Анализатор диапазонов итераторов"""
    
    def __init__(self, cfg: nx.DiGraph, ast_tree: ast.AST, dfa_results: Dict):
        """
        Инициализация анализатора итераторов.
        
        Args:
            cfg: Граф потока управления
            ast_tree: AST дерево исходного кода
            dfa_results: Результаты анализа потока данных
        """
        self.cfg = cfg
        self.ast_tree = ast_tree
        self.dfa_results = dfa_results
        
        self.loop_bounds: Dict[int, Dict[str, Any]] = {}
        self.growth_patterns: Dict[str, str] = {}
    
    def analyze(self) -> Dict[str, Any]:
        """
        Основной метод анализа.
        
        Returns:
            Словарь с результатами анализа итераторов
        """
        self._analyze_for_loops()
        self._analyze_while_loops()
        self._detect_growth_patterns()
        
        return {
            'loop_bounds': self.loop_bounds,
            'growth_patterns': self.growth_patterns
        }
    
    def _analyze_for_loops(self):
        """Анализ for-циклов"""
        for node_id in self.cfg.nodes():
            node_data = self.cfg.nodes[node_id]
            
            if node_data.get('type') == 'loop':
                ast_node = node_data.get('ast_node')
                
                if isinstance(ast_node, ast.For):
                    bounds = self._extract_for_loop_bounds(ast_node)
                    self.loop_bounds[node_id] = bounds
    
    def _extract_for_loop_bounds(self, node: ast.For) -> Dict[str, Any]:
        """
        Извлечение границ for-цикла.
        
        Args:
            node: AST узел for-цикла
            
        Returns:
            Словарь с информацией о границах
        """
        if isinstance(node.iter, ast.Call):
            func = node.iter.func
            
            # range(...)
            if isinstance(func, ast.Name) and func.id == 'range':
                return self._analyze_range_call(node.iter)
            
            # enumerate(...)
            elif isinstance(func, ast.Name) and func.id == 'enumerate':
                return {'type': 'enumerate', 'iterations': 'n', 'param': 'n'}
            
            # zip(...)
            elif isinstance(func, ast.Name) and func.id == 'zip':
                return {'type': 'zip', 'iterations': 'min(n,m)', 'param': 'n'}
        
        # for x in arr
        elif isinstance(node.iter, ast.Name):
            var_name = node.iter.id
            return {'type': 'iterable', 'iterations': f'len({var_name})', 'param': var_name}
        
        # for x in [...]
        elif isinstance(node.iter, ast.List):
            length = len(node.iter.elts)
            return {'type': 'list_literal', 'iterations': str(length), 'param': str(length)}
        
        return {'type': 'unknown', 'iterations': 'unknown', 'param': 'n'}
    
    def _analyze_range_call(self, call_node: ast.Call) -> Dict[str, Any]:
        """
        Анализ вызова range().
        
        Args:
            call_node: AST узел вызова range()
            
        Returns:
            Словарь с информацией о границах range
        """
        args = call_node.args
        
        if len(args) == 1:
            # range(n)
            param = self._extract_parameter_name(args[0])
            return {
                'type': 'range',
                'param': param,
                'iterations': param,
                'start': '0',
                'end': param,
                'step': '1'
            }
        
        elif len(args) == 2:
            # range(start, end)
            start = self._extract_parameter_name(args[0])
            end = self._extract_parameter_name(args[1])
            return {
                'type': 'range',
                'param': end,
                'iterations': f'{end}-{start}',
                'start': start,
                'end': end,
                'step': '1'
            }
        
        elif len(args) == 3:
            # range(start, end, step)
            start = self._extract_parameter_name(args[0])
            end = self._extract_parameter_name(args[1])
            step = self._extract_parameter_name(args[2])
            return {
                'type': 'range',
                'param': end,
                'iterations': f'({end}-{start})/{step}',
                'start': start,
                'end': end,
                'step': step
            }
        
        return {'type': 'range', 'iterations': 'unknown', 'param': 'n'}
    
    def _extract_parameter_name(self, node: ast.AST) -> str:
        """
        Извлечение имени параметра из AST узла.
        
        Args:
            node: AST узел
            
        Returns:
            Строковое представление параметра
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, (ast.Num, ast.Str)):  # Python 3.7 compatibility
            return str(node.n if isinstance(node, ast.Num) else node.s)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id == 'len' and node.args:
                    if isinstance(node.args[0], ast.Name):
                        return f'len({node.args[0].id})'
        elif isinstance(node, ast.BinOp):
            left = self._extract_parameter_name(node.left)
            right = self._extract_parameter_name(node.right)
            op = self._get_operator_symbol(node.op)
            return f'{left}{op}{right}'
        
        return 'expr'
    
    def _get_operator_symbol(self, op: ast.operator) -> str:
        """Получение символа оператора"""
        op_map = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.Mod: '%',
            ast.Pow: '**'
        }
        return op_map.get(type(op), '?')
    
    def _analyze_while_loops(self):
        """Анализ while-циклов"""
        for node_id in self.cfg.nodes():
            node_data = self.cfg.nodes[node_id]
            
            if node_data.get('type') == 'loop':
                ast_node = node_data.get('ast_node')
                
                if isinstance(ast_node, ast.While):
                    loop_var = self._find_loop_variable(ast_node)
                    if loop_var:
                        pattern = self._detect_variable_growth_in_loop(ast_node, loop_var)
                        self.growth_patterns[loop_var] = pattern
                        
                        self.loop_bounds[node_id] = {
                            'type': 'while',
                            'loop_var': loop_var,
                            'growth_pattern': pattern,
                            'iterations': self._estimate_iterations_by_pattern(pattern),
                            'param': loop_var
                        }
    
    def _find_loop_variable(self, while_node: ast.While) -> Optional[str]:
        """
        Находит переменную цикла в while.
        
        Args:
            while_node: AST узел while-цикла
            
        Returns:
            Имя переменной цикла или None
        """
        # Ищем переменную в условии
        for node in ast.walk(while_node.test):
            if isinstance(node, ast.Name):
                return node.id
        return None
    
    def _detect_variable_growth_in_loop(self, while_node: ast.While, var_name: str) -> str:
        """
        Определяет паттерн роста переменной в цикле.
        
        Args:
            while_node: AST узел while-цикла
            var_name: Имя переменной
            
        Returns:
            Тип паттерна роста
        """
        for node in ast.walk(while_node):
            # i += 1, i += k
            if isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name) and node.target.id == var_name:
                    if isinstance(node.op, ast.Add):
                        return 'linear'
                    elif isinstance(node.op, ast.Mult):
                        return 'logarithmic'
                    elif isinstance(node.op, ast.Pow):
                        return 'polynomial'
            
            # i = i * 2, i = i + 1
            elif isinstance(node, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id == var_name for t in node.targets):
                    if isinstance(node.value, ast.BinOp):
                        if isinstance(node.value.op, ast.Mult):
                            return 'logarithmic'
                        elif isinstance(node.value.op, ast.Add):
                            return 'linear'
                        elif isinstance(node.value.op, ast.Pow):
                            return 'polynomial'
        
        return 'unknown'
    
    def _estimate_iterations_by_pattern(self, pattern: str) -> str:
        """Оценка количества итераций по паттерну роста"""
        if pattern == 'linear':
            return 'n'
        elif pattern == 'logarithmic':
            return 'logn'
        elif pattern == 'polynomial':
            return 'sqrt(n)'
        return 'unknown'
    
    def _detect_growth_patterns(self):
        """Детекция паттернов роста для всех переменных"""
        # Уже обработано в _analyze_while_loops
        pass
    
    def estimate_iterations(self, loop_node_id: int) -> str:
        """
        Оценивает количество итераций цикла.
        
        Args:
            loop_node_id: ID узла цикла в CFG
            
        Returns:
            Строковое представление количества итераций
        """
        if loop_node_id not in self.loop_bounds:
            return 'unknown'
        
        bounds = self.loop_bounds[loop_node_id]
        return bounds.get('iterations', 'unknown')
