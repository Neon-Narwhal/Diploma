import ast
from typing import Dict, Any
from .base import PatternDetector

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
        results = {}
        for pattern_name, detector in self.sorting_patterns.items():
            results[pattern_name] = detector(tree)
        return {
            'detected_patterns': [k for k, v in results.items() if v],
            'pattern_details': results
        }
    
    def _detect_bubble_sort(self, tree: ast.AST) -> bool:
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
                if (isinstance(node.value, ast.Subscript) and 
                    len(node.targets) > 0 and 
                    isinstance(node.targets[0], ast.Subscript)):
                    self.has_swap = True
                elif (len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple) and 
                      isinstance(node.value, ast.Tuple)):
                      # Python style swap: a, b = b, a
                      self.has_swap = True
                self.generic_visit(node)
        
        visitor = BubbleSortVisitor()
        visitor.visit(tree)
        return (visitor.nested_loops >= 2 and visitor.has_swap and visitor.has_comparison)
    
    def _detect_selection_sort(self, tree: ast.AST) -> bool:
        return False
    
    def _detect_insertion_sort(self, tree: ast.AST) -> bool:
        return False
    
    def _detect_merge_sort(self, tree: ast.AST) -> bool:
        class MergeSortVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_recursion = False
                self.current_function = None
            
            def visit_FunctionDef(self, node):
                prev_func = self.current_function
                self.current_function = node.name
                self.generic_visit(node)
                self.current_function = prev_func
            
            def visit_Call(self, node):
                if (self.current_function and isinstance(node.func, ast.Name) and
                    node.func.id == self.current_function):
                    self.has_recursion = True
                self.generic_visit(node)
        
        visitor = MergeSortVisitor()
        visitor.visit(tree)
        return visitor.has_recursion
    
    def _detect_quick_sort(self, tree: ast.AST) -> bool:
        return False
