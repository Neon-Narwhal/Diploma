"""
Распознаватель библиотечных вызовов для CFG.
Определяет сложность вызовов стандартных функций и методов.
"""

import ast
import networkx as nx
from typing import Dict, Any, List, Optional
from collections import defaultdict

from complexity_analyzers.core.enums import ComplexityClass


class LibraryCallRecognizer:
    """Распознаватель библиотечных вызовов"""
    
    # База данных сложностей стандартных функций
    BUILTIN_COMPLEXITY = {
        # Сортировки
        'sorted': ComplexityClass.LINEARITHMIC,
        'list.sort': ComplexityClass.LINEARITHMIC,
        
        # Поиск
        'bisect.bisect': ComplexityClass.LOGARITHMIC,
        'bisect.bisect_left': ComplexityClass.LOGARITHMIC,
        'bisect.bisect_right': ComplexityClass.LOGARITHMIC,
        'bisect.insort': ComplexityClass.LINEAR,
        
        # Heap операции
        'heapq.heappush': ComplexityClass.LOGARITHMIC,
        'heapq.heappop': ComplexityClass.LOGARITHMIC,
        'heapq.heapify': ComplexityClass.LINEAR,
        'heapq.nlargest': ComplexityClass.LINEARITHMIC,
        'heapq.nsmallest': ComplexityClass.LINEARITHMIC,
        
        # Агрегации
        'max': ComplexityClass.LINEAR,
        'min': ComplexityClass.LINEAR,
        'sum': ComplexityClass.LINEAR,
        'len': ComplexityClass.CONSTANT,
        'any': ComplexityClass.LINEAR,
        'all': ComplexityClass.LINEAR,
        
        # Методы списков
        'list.append': ComplexityClass.CONSTANT,
        'list.extend': ComplexityClass.LINEAR,
        'list.insert': ComplexityClass.LINEAR,
        'list.pop': ComplexityClass.CONSTANT,
        'list.remove': ComplexityClass.LINEAR,
        'list.index': ComplexityClass.LINEAR,
        'list.count': ComplexityClass.LINEAR,
        'list.reverse': ComplexityClass.LINEAR,
        'list.copy': ComplexityClass.LINEAR,
        'list.clear': ComplexityClass.LINEAR,
        
        # Методы словарей
        'dict.get': ComplexityClass.CONSTANT,
        'dict.keys': ComplexityClass.LINEAR,
        'dict.values': ComplexityClass.LINEAR,
        'dict.items': ComplexityClass.LINEAR,
        'dict.pop': ComplexityClass.CONSTANT,
        'dict.popitem': ComplexityClass.CONSTANT,
        'dict.clear': ComplexityClass.LINEAR,
        'dict.update': ComplexityClass.LINEAR,
        'dict.copy': ComplexityClass.LINEAR,
        
        # Методы множеств
        'set.add': ComplexityClass.CONSTANT,
        'set.remove': ComplexityClass.CONSTANT,
        'set.discard': ComplexityClass.CONSTANT,
        'set.pop': ComplexityClass.CONSTANT,
        'set.clear': ComplexityClass.LINEAR,
        'set.union': ComplexityClass.LINEAR,
        'set.intersection': ComplexityClass.LINEAR,
        'set.difference': ComplexityClass.LINEAR,
        'set.symmetric_difference': ComplexityClass.LINEAR,
        'set.issubset': ComplexityClass.LINEAR,
        'set.issuperset': ComplexityClass.LINEAR,
        
        # Строковые методы
        'str.split': ComplexityClass.LINEAR,
        'str.join': ComplexityClass.LINEAR,
        'str.replace': ComplexityClass.LINEAR,
        'str.find': ComplexityClass.LINEAR,
        'str.count': ComplexityClass.LINEAR,
    }
    
    def __init__(self, cfg: nx.DiGraph, ast_tree: ast.AST):
        """
        Инициализация распознавателя.
        
        Args:
            cfg: Граф потока управления
            ast_tree: AST дерево исходного кода
        """
        self.cfg = cfg
        self.ast_tree = ast_tree
        self.detected_calls: List[Dict[str, Any]] = []
    
    def analyze(self) -> Dict[str, Any]:
        """
        Основной метод анализа.
        
        Returns:
            Словарь с обнаруженными вызовами
        """
        self._detect_calls_in_cfg()
        
        return {
            'detected_calls': self.detected_calls,
            'calls_by_complexity': self._group_by_complexity(),
            'total_calls': len(self.detected_calls)
        }
    
    def _detect_calls_in_cfg(self):
        """Обнаружение вызовов в CFG"""
        for node_id in self.cfg.nodes():
            node_data = self.cfg.nodes[node_id]
            ast_node = node_data.get('ast_node')
            
            if ast_node:
                for call_node in ast.walk(ast_node):
                    if isinstance(call_node, ast.Call):
                        call_info = self._recognize_call(call_node)
                        if call_info:
                            call_info['cfg_node_id'] = node_id
                            self.detected_calls.append(call_info)
    
    def _recognize_call(self, call_node: ast.Call) -> Optional[Dict[str, Any]]:
        """
        Распознавание конкретного вызова.
        
        Args:
            call_node: AST узел вызова функции
            
        Returns:
            Словарь с информацией о вызове или None
        """
        # Простой вызов функции: sorted(arr)
        if isinstance(call_node.func, ast.Name):
            func_name = call_node.func.id
            if func_name in self.BUILTIN_COMPLEXITY:
                return {
                    'type': 'builtin',
                    'name': func_name,
                    'complexity': self.BUILTIN_COMPLEXITY[func_name],
                    'line': call_node.lineno
                }
        
        # Вызов метода: arr.sort()
        elif isinstance(call_node.func, ast.Attribute):
            method_name = call_node.func.attr
            obj_type = self._infer_object_type(call_node.func.value)
            
            full_name = f'{obj_type}.{method_name}'
            if full_name in self.BUILTIN_COMPLEXITY:
                return {
                    'type': 'method',
                    'name': full_name,
                    'complexity': self.BUILTIN_COMPLEXITY[full_name],
                    'object_type': obj_type,
                    'method_name': method_name,
                    'line': call_node.lineno
                }
        
        return None
    
    def _infer_object_type(self, node: ast.AST) -> str:
        """
        Вывод типа объекта (упрощённая эвристика).
        
        Args:
            node: AST узел объекта
            
        Returns:
            Предполагаемый тип объекта
        """
        if isinstance(node, ast.Name):
            name = node.id
            
            # Эвристики по имени переменной
            name_lower = name.lower()
            
            if any(x in name_lower for x in ['list', 'arr', 'array', 'items']):
                return 'list'
            elif any(x in name_lower for x in ['dict', 'map', 'mapping']):
                return 'dict'
            elif any(x in name_lower for x in ['set', 'unique']):
                return 'set'
            elif any(x in name_lower for x in ['str', 'text', 'string']):
                return 'str'
            
            # Эвристика по окончанию (множественное число часто список)
            if name.endswith('s') and len(name) > 1:
                return 'list'
        
        # Литералы
        elif isinstance(node, ast.List):
            return 'list'
        elif isinstance(node, ast.Dict):
            return 'dict'
        elif isinstance(node, ast.Set):
            return 'set'
        elif isinstance(node, (ast.Str, ast.Constant)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return 'str'
            elif isinstance(node, ast.Str):
                return 'str'
        
        return 'unknown'
    
    def _group_by_complexity(self) -> Dict[str, List[Dict]]:
        """
        Группировка вызовов по сложности.
        
        Returns:
            Словарь {нотация_сложности: [вызовы]}
        """
        grouped = defaultdict(list)
        
        for call in self.detected_calls:
            complexity = call['complexity']
            grouped[complexity.to_notation()].append(call)
        
        return dict(grouped)
    
    def get_calls_in_node(self, node_id: int) -> List[Dict[str, Any]]:
        """
        Получение всех вызовов в конкретном узле CFG.
        
        Args:
            node_id: ID узла CFG
            
        Returns:
            Список вызовов в узле
        """
        return [call for call in self.detected_calls 
                if call.get('cfg_node_id') == node_id]
