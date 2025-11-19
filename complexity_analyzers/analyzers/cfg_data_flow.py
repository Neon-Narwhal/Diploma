"""
Анализатор потока данных для CFG.
Отслеживает переменные через граф для определения зависимостей.
"""

import ast
import networkx as nx
from typing import Dict, Any, List, Set, Tuple, Optional
from collections import defaultdict, deque


class DataFlowAnalyzer:
    """Анализатор потока данных для CFG"""
    
    def __init__(self, cfg: nx.DiGraph, ast_tree: ast.AST):
        """
        Инициализация анализатора потока данных.
        
        Args:
            cfg: Граф потока управления
            ast_tree: AST дерево исходного кода
        """
        self.cfg = cfg
        self.ast_tree = ast_tree
        
        # Основные структуры данных
        self.variable_definitions: Dict[int, Dict[str, Any]] = {}
        self.variable_uses: Dict[int, Set[str]] = {}
        self.def_use_chains: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.live_variables: Dict[int, Set[str]] = {}
        
        # Дополнительная информация о переменных
        self.variable_scopes: Dict[str, str] = {}  # var -> scope_type
        self.function_parameters: Set[str] = set()
        self.loop_variables: Set[str] = set()
    
    def analyze(self) -> Dict[str, Any]:
        """
        Основной метод анализа.
        
        Returns:
            Словарь с результатами анализа потока данных
        """
        self._extract_function_parameters()
        self._build_def_use_chains()
        self._compute_live_variables()
        self._analyze_variable_scopes()
        
        return {
            'variable_definitions': self.variable_definitions,
            'variable_uses': self.variable_uses,
            'def_use_chains': {k: list(v) for k, v in self.def_use_chains.items()},
            'live_variables': {k: list(v) for k, v in self.live_variables.items()},
            'variable_scopes': self.variable_scopes,
            'function_parameters': list(self.function_parameters),
            'loop_variables': list(self.loop_variables)
        }
    
    def _extract_function_parameters(self):
        """Извлечение параметров функций из AST"""
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.FunctionDef):
                for arg in node.args.args:
                    self.function_parameters.add(arg.arg)
                    self.variable_scopes[arg.arg] = 'parameter'
    
    def _build_def_use_chains(self):
        """Построение цепочек определение-использование"""
        for node_id in self.cfg.nodes():
            node_data = self.cfg.nodes[node_id]
            ast_node = node_data.get('ast_node')
            
            if not ast_node:
                continue
            
            # Извлекаем определения и использования из узла
            definitions = self._extract_definitions(ast_node, node_id)
            uses = self._extract_uses(ast_node, node_id)
            
            self.variable_definitions[node_id] = definitions
            self.variable_uses[node_id] = uses
            
            # Строим цепочки def-use
            for var_name in definitions:
                # Ищем все использования этой переменной в последующих узлах
                for use_node_id in self.variable_uses:
                    if var_name in self.variable_uses[use_node_id]:
                        if self._has_path(node_id, use_node_id):
                            self.def_use_chains[var_name].append((node_id, use_node_id))
    
    def _extract_definitions(self, node: ast.AST, node_id: int) -> Dict[str, Any]:
        """Извлечение определений переменных из AST узла"""
        definitions = {}
        
        for child in ast.walk(node):
            # Обычное присваивание: x = value
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        definitions[target.id] = {
                            'type': 'assign',
                            'line': child.lineno,
                            'node_id': node_id
                        }
            
            # Расширенное присваивание: x += value
            elif isinstance(child, ast.AugAssign):
                if isinstance(child.target, ast.Name):
                    definitions[child.target.id] = {
                        'type': 'augassign',
                        'line': child.lineno,
                        'node_id': node_id
                    }
            
            # For цикл: for i in ...
            elif isinstance(child, ast.For):
                if isinstance(child.target, ast.Name):
                    var_name = child.target.id
                    definitions[var_name] = {
                        'type': 'loop_var',
                        'line': child.lineno,
                        'node_id': node_id
                    }
                    self.loop_variables.add(var_name)
                    self.variable_scopes[var_name] = 'loop_var'
        
        return definitions
    
    def _extract_uses(self, node: ast.AST, node_id: int) -> Set[str]:
        """Извлечение использований переменных из AST узла"""
        uses = set()
        
        for child in ast.walk(node):
            # Использование переменной в выражении
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                uses.add(child.id)
        
        return uses
    
    def _has_path(self, from_node: int, to_node: int) -> bool:
        """Проверка существования пути между узлами в CFG"""
        try:
            return nx.has_path(self.cfg, from_node, to_node)
        except:
            return False
    
    def _compute_live_variables(self):
        """Вычисление живых переменных (backward analysis)"""
        # Инициализация: все переменные мертвы
        for node_id in self.cfg.nodes():
            self.live_variables[node_id] = set()
        
        # Итеративный алгоритм обратного распространения
        changed = True
        max_iterations = 100
        iteration = 0
        
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            
            # Обратный обход графа
            for node_id in reversed(list(self.cfg.nodes())):
                old_live = self.live_variables[node_id].copy()
                
                # live_out = union of live_in of successors
                live_out = set()
                for successor in self.cfg.successors(node_id):
                    live_out |= self.live_variables[successor]
                
                # live_in = (live_out - def) | use
                definitions = set(self.variable_definitions.get(node_id, {}).keys())
                uses = self.variable_uses.get(node_id, set())
                
                live_in = (live_out - definitions) | uses
                
                self.live_variables[node_id] = live_in
                
                if live_in != old_live:
                    changed = True
    
    def _analyze_variable_scopes(self):
        """Анализ областей видимости переменных"""
        # Уже частично заполнено в других методах
        # Дополняем для переменных, чей scope не определён
        
        all_variables = set()
        for defs in self.variable_definitions.values():
            all_variables.update(defs.keys())
        
        for var in all_variables:
            if var not in self.variable_scopes:
                # Если не параметр и не loop_var, то локальная
                self.variable_scopes[var] = 'local'
    
    def get_variable_scope(self, var_name: str) -> str:
        """
        Возвращает scope переменной.
        
        Args:
            var_name: Имя переменной
            
        Returns:
            Тип scope: 'parameter', 'local', 'loop_var'
        """
        return self.variable_scopes.get(var_name, 'unknown')
    
    def are_variables_independent(self, var1: str, var2: str) -> bool:
        """
        Проверяет независимость переменных.
        
        Args:
            var1: Первая переменная
            var2: Вторая переменная
            
        Returns:
            True если переменные независимы
        """
        # Переменные независимы, если:
        # 1. Обе являются параметрами функции
        if var1 in self.function_parameters and var2 in self.function_parameters:
            return True
        
        # 2. Одна не зависит от другой в def-use цепочках
        deps1 = self.get_variable_dependencies(var1)
        deps2 = self.get_variable_dependencies(var2)
        
        if var2 in deps1 or var1 in deps2:
            return False
        
        return True
    
    def get_variable_dependencies(self, var_name: str) -> Set[str]:
        """
        Возвращает множество переменных, от которых зависит данная.
        
        Args:
            var_name: Имя переменной
            
        Returns:
            Множество зависимых переменных
        """
        dependencies = set()
        
        # Ищем все узлы, где переменная определяется
        for node_id, defs in self.variable_definitions.items():
            if var_name in defs:
                # Смотрим, какие переменные используются в этом узле
                uses = self.variable_uses.get(node_id, set())
                dependencies.update(uses)
        
        return dependencies
