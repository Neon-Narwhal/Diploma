"""
Композитор сложностей для CFG.
Объединяет сложности циклов и библиотечных вызовов.
"""

import networkx as nx
from typing import Dict, Any, List, Set
from collections import deque

from complexity_analyzers.core.enums import ComplexityClass


class ComplexityComposer:
    """Композитор сложностей"""
    
    # Правила композиции сложностей
    COMPOSITION_RULES = {
        # (outer, inner) -> result
        ('O(1)', 'O(1)'): 'O(1)',
        ('O(1)', 'O(logn)'): 'O(logn)',
        ('O(1)', 'O(n)'): 'O(n)',
        
        ('O(logn)', 'O(1)'): 'O(logn)',
        ('O(logn)', 'O(logn)'): 'O(log^2n)',
        ('O(logn)', 'O(n)'): 'O(nlogn)',
        
        ('O(n)', 'O(1)'): 'O(n)',
        ('O(n)', 'O(logn)'): 'O(nlogn)',
        ('O(n)', 'O(n)'): 'O(n^2)',
        ('O(n)', 'O(nlogn)'): 'O(n^2logn)',
        ('O(n)', 'O(n^2)'): 'O(n^3)',
        
        ('O(nlogn)', 'O(1)'): 'O(nlogn)',
        ('O(nlogn)', 'O(logn)'): 'O(nlog^2n)',
        ('O(nlogn)', 'O(n)'): 'O(n^2logn)',
        
        ('O(n^2)', 'O(1)'): 'O(n^2)',
        ('O(n^2)', 'O(logn)'): 'O(n^2logn)',
        ('O(n^2)', 'O(n)'): 'O(n^3)',
        ('O(n^2)', 'O(nlogn)'): 'O(n^3logn)',
        
        ('O(n^3)', 'O(1)'): 'O(n^3)',
        ('O(n^3)', 'O(logn)'): 'O(n^3logn)',
        ('O(n^3)', 'O(n)'): 'O(n^4)',
    }
    
    def __init__(self, cfg: nx.DiGraph, ira_results: Dict, lcr_results: Dict):
        """
        Инициализация композитора.
        
        Args:
            cfg: Граф потока управления
            ira_results: Результаты анализа итераторов
            lcr_results: Результаты распознавания библиотечных вызовов
        """
        self.cfg = cfg
        self.ira_results = ira_results
        self.lcr_results = lcr_results
        
        self.node_complexities: Dict[int, str] = {}
    
    def analyze(self) -> Dict[str, Any]:
        """
        Основной метод анализа.
        
        Returns:
            Словарь с композированными сложностями
        """
        self._compute_node_complexities()
        overall_complexity = self._compute_overall_complexity()
        
        return {
            'node_complexities': self.node_complexities,
            'overall_complexity': overall_complexity
        }
    
    def _compute_node_complexities(self):
        """Вычисление сложности для каждого узла"""
        # Сначала вычисляем сложность statement-узлов (листья)
        for node_id in self.cfg.nodes():
            node_data = self.cfg.nodes[node_id]
            node_type = node_data.get('type')
            
            if node_type == 'statement':
                # Проверяем на библиотечные вызовы
                calls = self._get_calls_in_node(node_id)
                if calls:
                    # Берём максимальную сложность среди вызовов
                    max_call = max(calls, key=lambda c: c['complexity'].complexity_order)
                    self.node_complexities[node_id] = max_call['complexity'].to_notation()
                else:
                    self.node_complexities[node_id] = 'O(1)'
            elif node_type in ['condition', 'function_entry', 'function_exit', 'merge']:
                self.node_complexities[node_id] = 'O(1)'
        
        # Затем вычисляем сложность циклов (снизу вверх)
        loop_nodes = [n for n in self.cfg.nodes() 
                     if self.cfg.nodes[n].get('type') == 'loop']
        
        # Сортируем циклы по глубине вложенности (внутренние сначала)
        loop_nodes_sorted = self._sort_loops_by_depth(loop_nodes)
        
        for loop_id in loop_nodes_sorted:
            iterations = self._get_loop_iterations(loop_id)
            body_complexity = self._compute_loop_body_complexity(loop_id)
            
            composed = self.compose(f'O({iterations})', body_complexity)
            self.node_complexities[loop_id] = composed
    
    def _sort_loops_by_depth(self, loop_nodes: List[int]) -> List[int]:
        """Сортировка циклов по глубине вложенности (внутренние сначала)"""
        # Простая эвристика: считаем количество предшественников-циклов
        loop_depths = {}
        
        for loop_id in loop_nodes:
            depth = 0
            visited = set()
            queue = deque([loop_id])
            
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                
                for predecessor in self.cfg.predecessors(current):
                    pred_type = self.cfg.nodes[predecessor].get('type')
                    if pred_type == 'loop' and predecessor != loop_id:
                        depth += 1
                    queue.append(predecessor)
            
            loop_depths[loop_id] = depth
        
        # Сортируем по глубине (меньшая глубина = внутренний цикл)
        return sorted(loop_nodes, key=lambda x: loop_depths.get(x, 0), reverse=True)
    
    def _get_loop_iterations(self, loop_id: int) -> str:
        """Получение количества итераций цикла"""
        loop_bounds = self.ira_results.get('loop_bounds', {})
        
        if loop_id in loop_bounds:
            iterations = loop_bounds[loop_id].get('iterations', 'n')
            
            # Упрощаем выражения
            if iterations in ['n', 'm', 'k', 'logn', 'log(n)']:
                return iterations
            elif iterations.startswith('len('):
                return 'n'
            elif iterations.isdigit():
                return '1'  # Константное число итераций
            else:
                return 'n'
        
        return 'n'
    
    def _compute_loop_body_complexity(self, loop_node_id: int) -> str:
        """Вычисление сложности тела цикла"""
        body_nodes = self._get_loop_body_nodes(loop_node_id)
        
        if not body_nodes:
            return 'O(1)'
        
        # Находим максимальную сложность среди узлов тела
        max_complexity = 'O(1)'
        max_order = 0
        
        for node_id in body_nodes:
            if node_id in self.node_complexities:
                complexity = self.node_complexities[node_id]
                order = self._complexity_order(complexity)
                if order > max_order:
                    max_order = order
                    max_complexity = complexity
        
        return max_complexity
    
    def _get_loop_body_nodes(self, loop_node_id: int) -> Set[int]:
        """Получение узлов тела цикла"""
        body_nodes = set()
        visited = set()
        queue = deque([loop_node_id])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            
            visited.add(current)
            
            for successor in self.cfg.successors(current):
                edge_data = self.cfg.edges[current, successor]
                edge_type = edge_data.get('type')
                
                # Не идём по обратному ребру и не выходим из цикла
                if edge_type not in ['loop_back', 'false_branch']:
                    if successor != loop_node_id:
                        body_nodes.add(successor)
                        queue.append(successor)
        
        return body_nodes
    
    def _get_calls_in_node(self, node_id: int) -> List[Dict]:
        """Получение библиотечных вызовов в узле"""
        detected_calls = self.lcr_results.get('detected_calls', [])
        return [call for call in detected_calls 
                if call.get('cfg_node_id') == node_id]
    
    def compose(self, outer: str, inner: str) -> str:
        """
        Композиция двух сложностей.
        
        Args:
            outer: Внешняя сложность (цикл)
            inner: Внутренняя сложность (тело)
            
        Returns:
            Композированная сложность
        """
        key = (outer, inner)
        
        if key in self.COMPOSITION_RULES:
            return self.COMPOSITION_RULES[key]
        
        # Эвристическая композиция
        outer_order = self._complexity_order(outer)
        inner_order = self._complexity_order(inner)
        
        total_order = outer_order + inner_order - 1  # -1 потому что O(1)*O(n) = O(n), не O(n^2)
        
        if total_order <= 1:
            return 'O(1)'
        elif total_order == 2:
            return 'O(n)'
        elif total_order == 3:
            return 'O(n^2)'
        elif total_order == 4:
            return 'O(n^3)'
        else:
            return f'O(n^{total_order})'
    
    def _complexity_order(self, complexity: str) -> int:
        """
        Приближённый порядок сложности.
        
        Args:
            complexity: Нотация сложности
            
        Returns:
            Числовой порядок
        """
        complexity_lower = complexity.lower()
        
        if 'o(1)' in complexity_lower:
            return 1
        elif 'o(logn)' in complexity_lower or 'o(log(n))' in complexity_lower:
            return 2
        elif 'o(n)' in complexity_lower and '^' not in complexity_lower:
            return 3
        elif 'o(nlogn)' in complexity_lower or 'o(n*logn)' in complexity_lower:
            return 3
        elif 'o(n^2)' in complexity_lower or 'o(n*n)' in complexity_lower:
            return 4
        elif 'o(n^3)' in complexity_lower:
            return 5
        elif 'o(n^4)' in complexity_lower:
            return 6
        elif 'o(2^n)' in complexity_lower:
            return 10
        
        return 5
    
    def _compute_overall_complexity(self) -> str:
        """Вычисление общей сложности функции"""
        if not self.node_complexities:
            return 'O(1)'
        
        # Берём максимальную сложность среди всех узлов
        max_complexity = 'O(1)'
        max_order = 0
        
        for complexity in self.node_complexities.values():
            order = self._complexity_order(complexity)
            if order > max_order:
                max_order = order
                max_complexity = complexity
        
        return max_complexity
