"""
Трекер множественных переменных для CFG.
Различает независимые переменные (O(m*n) vs O(n^2)).
"""

import networkx as nx
from typing import Dict, Any, List, Set
from collections import defaultdict, deque


class MultiVariableTracker:
    """Трекер множественных переменных"""
    
    def __init__(self, cfg: nx.DiGraph, dfa_results: Dict, ira_results: Dict):
        """
        Инициализация трекера.
        
        Args:
            cfg: Граф потока управления
            dfa_results: Результаты анализа потока данных
            ira_results: Результаты анализа итераторов
        """
        self.cfg = cfg
        self.dfa_results = dfa_results
        self.ira_results = ira_results
        
        self.loop_parameters: Dict[int, str] = {}
        self.parameter_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.function_parameters: Set[str] = set(dfa_results.get('function_parameters', []))
    
    def analyze(self) -> Dict[str, Any]:
        """
        Основной метод анализа.
        
        Returns:
            Словарь с результатами анализа переменных
        """
        self._extract_loop_parameters()
        self._build_parameter_dependencies()
        
        return {
            'loop_parameters': self.loop_parameters,
            'parameter_dependencies': {k: list(v) for k, v in self.parameter_dependencies.items()},
            'function_parameters': list(self.function_parameters)
        }
    
    def _extract_loop_parameters(self):
        """Извлечение параметров для каждого цикла"""
        loop_bounds = self.ira_results.get('loop_bounds', {})
        
        for loop_id, bounds in loop_bounds.items():
            param = bounds.get('param', 'n')
            
            # Очищаем параметр от выражений типа len(...), оставляем имя
            clean_param = self._clean_parameter_name(param)
            self.loop_parameters[loop_id] = clean_param
    
    def _clean_parameter_name(self, param: str) -> str:
        """
        Очистка имени параметра от дополнительных выражений.
        
        Args:
            param: Параметр (может быть 'n', 'len(arr)', 'n-1', etc.)
            
        Returns:
            Очищенное имя параметра
        """
        # len(arr) -> arr
        if param.startswith('len(') and param.endswith(')'):
            return param[4:-1]
        
        # Если содержит операторы, берём первую переменную
        for op in ['+', '-', '*', '/', '%']:
            if op in param:
                parts = param.split(op)
                for part in parts:
                    part = part.strip()
                    if part.isidentifier():
                        return part
        
        # Если число, возвращаем как есть
        if param.isdigit():
            return param
        
        # Если идентификатор, возвращаем как есть
        if param.isidentifier():
            return param
        
        # По умолчанию 'n'
        return 'n'
    
    def _build_parameter_dependencies(self):
        """Построение графа зависимостей параметров"""
        dfa_chains = self.dfa_results.get('def_use_chains', {})
        
        for var, chains in dfa_chains.items():
            # Для каждой переменной находим, от каких она зависит
            dependencies = self.dfa_results.get('variable_dependencies', {})
            if var in dependencies:
                for dep_var in dependencies[var]:
                    self.parameter_dependencies[var].add(dep_var)
    
    def are_parameters_independent(self, param1: str, param2: str) -> bool:
        """
        Проверка независимости параметров.
        
        Args:
            param1: Первый параметр
            param2: Второй параметр
            
        Returns:
            True если параметры независимы
        """
        # Одинаковые параметры зависимы
        if param1 == param2:
            return False
        
        # Оба являются входными параметрами функции → независимы
        if param1 in self.function_parameters and param2 in self.function_parameters:
            return True
        
        # Проверка циклических зависимостей
        if param2 in self._get_all_dependencies(param1):
            return False
        if param1 in self._get_all_dependencies(param2):
            return False
        
        # Если один параметр функции, а другой нет
        if param1 in self.function_parameters or param2 in self.function_parameters:
            return True
        
        return True
    
    def _get_all_dependencies(self, param: str) -> Set[str]:
        """
        Получение всех зависимостей параметра (транзитивное замыкание).
        
        Args:
            param: Имя параметра
            
        Returns:
            Множество зависимых параметров
        """
        visited = set()
        queue = deque([param])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            
            visited.add(current)
            deps = self.parameter_dependencies.get(current, set())
            queue.extend(deps)
        
        visited.discard(param)
        return visited
    
    def generate_precise_notation(self, nested_loops: List[int]) -> str:
        """
        Генерация точной нотации O(m*n) vs O(n^2).
        
        Args:
            nested_loops: Список ID вложенных циклов (от внешнего к внутреннему)
            
        Returns:
            Нотация сложности
        """
        if len(nested_loops) == 0:
            return 'O(1)'
        
        if len(nested_loops) == 1:
            loop_id = nested_loops[0]
            param = self.loop_parameters.get(loop_id, 'n')
            return f'O({param})'
        
        if len(nested_loops) == 2:
            param1 = self.loop_parameters.get(nested_loops[0], 'n')
            param2 = self.loop_parameters.get(nested_loops[1], 'm')
            
            # Проверяем независимость
            if self.are_parameters_independent(param1, param2):
                # Независимые параметры → O(m*n)
                # Сортируем для канонической формы
                params = sorted([param1, param2])
                return f'O({params[0]}*{params[1]})'
            else:
                # Зависимые параметры → O(n^2)
                return f'O({param1}^2)'
        
        if len(nested_loops) >= 3:
            # Три и более вложенных цикла
            param1 = self.loop_parameters.get(nested_loops[0], 'n')
            return f'O({param1}^{len(nested_loops)})'
        
        return 'O(?)'
