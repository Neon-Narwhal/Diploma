"""Детекторы алгоритмических паттернов"""
import ast
from typing import Dict, Any, List, Optional, Set
from abc import ABC, abstractmethod

class PatternDetector(ABC):
    """Базовый класс для детекторов паттернов"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def detect(self, tree: ast.AST) -> Dict[str, Any]:
        """Обнаружение паттерна"""
        pass

class SortingPatternDetector(PatternDetector):
    """Детектор алгоритмов сортировки"""
    
    def __init__(self):
        super().__init__("sorting_patterns")
        self.sorting_patterns = {
            'bubble_sort': self._detect_bubble_sort,
            'selection_sort': self._detect_selection_sort,
            'insertion_sort': self._detect_insertion_sort,
            'merge_sort': self._detect_merge_sort,
            'quick_sort': self._detect_quick_sort
        }
    
    def detect(self, tree: ast.AST) -> Dict[str, Any]:
        """Обнаружение паттернов сортировки"""
        results = {}
        
        for pattern_name, detector in self.sorting_patterns.items():
            results[pattern_name] = detector(tree)
        
        return {
            'detected_patterns': [k for k, v in results.items() if v],
            'pattern_details': results
        }
    
    def _detect_bubble_sort(self, tree: ast.AST) -> bool:
        """Обнаружение пузырьковой сортировки"""
        class BubbleSortVisitor(ast.NodeVisitor):
            def __init__(self):
                self.nested_loops = 0
                self.has_swap = False
                self.has_comparison = False
            
            def visit_For(self, node):
                self.nested_loops += 1
                self.generic_visit(node)
                self.nested_loops -= 1
            
            def visit_Compare(self, node):
                if self.nested_loops >= 2:
                    self.has_comparison = True
                self.generic_visit(node)
            
            def visit_Assign(self, node):
                # Простое обнаружение swap паттерна
                if (isinstance(node.value, ast.Subscript) and 
                    isinstance(node.targets[0], ast.Subscript)):
                    self.has_swap = True
                self.generic_visit(node)
        
        visitor = BubbleSortVisitor()
        visitor.visit(tree)
        
        return (visitor.nested_loops >= 2 and 
                visitor.has_swap and 
                visitor.has_comparison)
    
    def _detect_selection_sort(self, tree: ast.AST) -> bool:
        """Обнаружение сортировки выбором"""
        # Упрощенная реализация
        return False
    
    def _detect_insertion_sort(self, tree: ast.AST) -> bool:
        """Обнаружение сортировки вставками"""
        # Упрощенная реализация
        return False
    
    def _detect_merge_sort(self, tree: ast.AST) -> bool:
        """Обнаружение сортировки слиянием"""
        class MergeSortVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_recursion = False
                self.has_merge_logic = False
                self.current_function = None
            
            def visit_FunctionDef(self, node):
                prev_func = self.current_function
                self.current_function = node.name
                self.generic_visit(node)
                self.current_function = prev_func
            
            def visit_Call(self, node):
                if (self.current_function and 
                    isinstance(node.func, ast.Name) and
                    node.func.id == self.current_function):
                    self.has_recursion = True
                self.generic_visit(node)
        
        visitor = MergeSortVisitor()
        visitor.visit(tree)
        
        return visitor.has_recursion
    
    def _detect_quick_sort(self, tree: ast.AST) -> bool:
        """Обнаружение быстрой сортировки"""
        # Аналогично merge_sort, но с дополнительными проверками
        return False

class SearchPatternDetector(PatternDetector):
    """Детектор алгоритмов поиска"""
    
    def __init__(self):
        super().__init__("search_patterns")
        self.search_patterns = {
            'linear_search': self._detect_linear_search,
            'binary_search': self._detect_binary_search,
            'hash_search': self._detect_hash_search
        }
    
    def detect(self, tree: ast.AST) -> Dict[str, Any]:
        """Обнаружение паттернов поиска"""
        results = {}
        
        for pattern_name, detector in self.search_patterns.items():
            results[pattern_name] = detector(tree)
        
        return {
            'detected_patterns': [k for k, v in results.items() if v],
            'pattern_details': results
        }
    
    def _detect_linear_search(self, tree: ast.AST) -> bool:
        """Обнаружение линейного поиска"""
        class LinearSearchVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_loop = False
                self.has_comparison = False
                self.has_return_in_loop = False
                self.in_loop = False
            
            def visit_For(self, node):
                self.has_loop = True
                self.in_loop = True
                self.generic_visit(node)
                self.in_loop = False
            
            def visit_Compare(self, node):
                if self.in_loop:
                    self.has_comparison = True
                self.generic_visit(node)
            
            def visit_Return(self, node):
                if self.in_loop:
                    self.has_return_in_loop = True
                self.generic_visit(node)
        
        visitor = LinearSearchVisitor()
        visitor.visit(tree)
        
        return (visitor.has_loop and 
                visitor.has_comparison and 
                visitor.has_return_in_loop)
    
    def _detect_binary_search(self, tree: ast.AST) -> bool:
        """Обнаружение бинарного поиска"""
        class BinarySearchVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_while_loop = False
                self.has_midpoint = False
                self.has_bounds_update = False
            
            def visit_While(self, node):
                self.has_while_loop = True
                self.generic_visit(node)
            
            def visit_BinOp(self, node):
                # Ищем вычисление средней точки
                if isinstance(node.op, ast.Div) or isinstance(node.op, ast.FloorDiv):
                    self.has_midpoint = True
                self.generic_visit(node)
            
            def visit_Assign(self, node):
                # Упрощенное обнаружение обновления границ
                if isinstance(node.targets[0], ast.Name):
                    if node.targets[0].id in ['left', 'right', 'low', 'high', 'start', 'end']:
                        self.has_bounds_update = True
                self.generic_visit(node)
        
        visitor = BinarySearchVisitor()
        visitor.visit(tree)
        
        return (visitor.has_while_loop and 
                visitor.has_midpoint and 
                visitor.has_bounds_update)
    
    def _detect_hash_search(self, tree: ast.AST) -> bool:
        """Обнаружение поиска по хешу"""
        # Поиск использования словарей для поиска
        class HashSearchVisitor(ast.NodeVisitor):
            def __init__(self):
                self.uses_dict = False
                self.has_dict_access = False
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == 'dict':
                    self.uses_dict = True
                self.generic_visit(node)
            
            def visit_Subscript(self, node):
                if isinstance(node.value, ast.Name):
                    self.has_dict_access = True
                self.generic_visit(node)
        
        visitor = HashSearchVisitor()
        visitor.visit(tree)
        
        return visitor.uses_dict and visitor.has_dict_access

class DynamicProgrammingDetector(PatternDetector):
    """Детектор паттернов динамического программирования"""
    
    def __init__(self):
        super().__init__("dp_patterns")
    
    def detect(self, tree: ast.AST) -> Dict[str, Any]:
        """Обнаружение DP паттернов"""
        class DPVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_memoization = False
                self.has_tabulation = False
                self.has_recursive_calls = False
                self.uses_cache = False
                self.current_function = None
            
            def visit_FunctionDef(self, node):
                prev_func = self.current_function
                self.current_function = node.name
                
                # Поиск декораторов кеширования
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        if decorator.id in ['lru_cache', 'cache', 'memoize']:
                            self.has_memoization = True
                
                self.generic_visit(node)
                self.current_function = prev_func
            
            def visit_Call(self, node):
                # Рекурсивные вызовы
                if (self.current_function and 
                    isinstance(node.func, ast.Name) and
                    node.func.id == self.current_function):
                    self.has_recursive_calls = True
                
                self.generic_visit(node)
            
            def visit_Subscript(self, node):
                # Обращения к массиву/словарю (табуляция)
                if isinstance(node.value, ast.Name):
                    var_name = node.value.id
                    if any(name in var_name.lower() for name in ['dp', 'memo', 'cache', 'table']):
                        self.has_tabulation = True
                        self.uses_cache = True
                
                self.generic_visit(node)
        
        visitor = DPVisitor()
        visitor.visit(tree)
        
        return {
            'has_memoization': visitor.has_memoization,
            'has_tabulation': visitor.has_tabulation,
            'has_recursive_calls': visitor.has_recursive_calls,
            'uses_cache': visitor.uses_cache,
            'likely_dp': (visitor.has_memoization or visitor.has_tabulation or 
                         (visitor.has_recursive_calls and visitor.uses_cache))
        }

class DataStructurePatternDetector(PatternDetector):
    """Детектор паттернов использования структур данных"""
    
    def __init__(self):
        super().__init__("data_structure_patterns")
    
    def detect(self, tree: ast.AST) -> Dict[str, Any]:
        """Анализ использования структур данных"""
        class DataStructureVisitor(ast.NodeVisitor):
            def __init__(self):
                self.data_structures = {
                    'list': 0,
                    'dict': 0,
                    'set': 0,
                    'tuple': 0,
                    'deque': 0,
                    'heapq': 0,
                    'defaultdict': 0
                }
                self.operations = {
                    'list_append': 0,
                    'list_pop': 0,
                    'list_insert': 0,
                    'dict_get': 0,
                    'dict_keys': 0,
                    'set_add': 0,
                    'set_union': 0
                }
            
            def visit_Call(self, node):
                # Конструкторы структур данных
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in self.data_structures:
                        self.data_structures[func_name] += 1
                
                # Методы структур данных
                elif isinstance(node.func, ast.Attribute):
                    method_name = node.func.attr
                    if method_name == 'append':
                        self.operations['list_append'] += 1
                    elif method_name == 'pop':
                        self.operations['list_pop'] += 1
                    elif method_name == 'insert':
                        self.operations['list_insert'] += 1
                    elif method_name == 'get':
                        self.operations['dict_get'] += 1
                    elif method_name == 'keys':
                        self.operations['dict_keys'] += 1
                    elif method_name == 'add':
                        self.operations['set_add'] += 1
                    elif method_name == 'union':
                        self.operations['set_union'] += 1
                
                self.generic_visit(node)
        
        visitor = DataStructureVisitor()
        visitor.visit(tree)
        
        return {
            'data_structures_used': visitor.data_structures,
            'operations_used': visitor.operations,
            'primary_structure': max(visitor.data_structures, 
                                   key=visitor.data_structures.get) if any(visitor.data_structures.values()) else None
        }

class PatternDetectorRegistry:
    """Реестр детекторов паттернов"""
    
    def __init__(self):
        self.detectors: Dict[str, PatternDetector] = {}
        self._register_default_detectors()
    
    def _register_default_detectors(self):
        """Регистрация стандартных детекторов"""
        self.register(SortingPatternDetector())
        self.register(SearchPatternDetector())
        self.register(DynamicProgrammingDetector())
        self.register(DataStructurePatternDetector())
    
    def register(self, detector: PatternDetector):
        """Регистрация детектора"""
        self.detectors[detector.name] = detector
    
    def detect_all(self, tree: ast.AST) -> Dict[str, Any]:
        """Запуск всех детекторов"""
        results = {}
        
        for name, detector in self.detectors.items():
            try:
                results[name] = detector.detect(tree)
            except Exception as e:
                results[name] = {'error': str(e)}
        
        return results
    
    def get_detector(self, name: str) -> Optional[PatternDetector]:
        """Получение детектора по имени"""
        return self.detectors.get(name)
