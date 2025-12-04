import ast
from typing import Dict, Any
from .base import PatternDetector

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
        results = {}
        for pattern_name, detector in self.search_patterns.items():
            results[pattern_name] = detector(tree)
        return {
            'detected_patterns': [k for k, v in results.items() if v],
            'pattern_details': results
        }
    
    def _detect_linear_search(self, tree: ast.AST) -> bool:
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
        return (visitor.has_loop and visitor.has_comparison and visitor.has_return_in_loop)
    
    def _detect_binary_search(self, tree: ast.AST) -> bool:
        class BinarySearchVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_while_loop = False
                self.has_midpoint = False
                self.has_bounds_update = False
            
            def visit_While(self, node):
                self.has_while_loop = True
                self.generic_visit(node)
            
            def visit_BinOp(self, node):
                if isinstance(node.op, (ast.Div, ast.FloorDiv)):
                    self.has_midpoint = True
                self.generic_visit(node)
            
            def visit_Assign(self, node):
                if len(node.targets) > 0 and isinstance(node.targets[0], ast.Name):
                    if node.targets[0].id in ['left', 'right', 'low', 'high', 'start', 'end', 'l', 'r']:
                        self.has_bounds_update = True
                self.generic_visit(node)
        
        visitor = BinarySearchVisitor()
        visitor.visit(tree)
        return (visitor.has_while_loop and visitor.has_midpoint and visitor.has_bounds_update)
    
    def _detect_hash_search(self, tree: ast.AST) -> bool:
        class HashSearchVisitor(ast.NodeVisitor):
            def __init__(self):
                self.uses_dict = False
                self.has_dict_access = False
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == 'dict':
                    self.uses_dict = True
                self.generic_visit(node)
            
            def visit_Dict(self, node):
                self.uses_dict = True
                self.generic_visit(node)
            
            def visit_Subscript(self, node):
                if isinstance(node.value, ast.Name):
                    self.has_dict_access = True
                self.generic_visit(node)
        
        visitor = HashSearchVisitor()
        visitor.visit(tree)
        return visitor.uses_dict and visitor.has_dict_access
