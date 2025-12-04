import ast
from typing import Dict, Any
from .base import PatternDetector

class DynamicProgrammingDetector(PatternDetector):
    """Детектор паттернов динамического программирования"""
    
    def __init__(self):
        super().__init__("dp_patterns")
    
    def detect(self, tree: ast.AST) -> Dict[str, Any]:
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
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        if decorator.id in ['lru_cache', 'cache', 'memoize']:
                            self.has_memoization = True
                    elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                        if decorator.func.id in ['lru_cache', 'cache']:
                            self.has_memoization = True
                            
                self.generic_visit(node)
                self.current_function = prev_func
            
            def visit_Call(self, node):
                if (self.current_function and isinstance(node.func, ast.Name) and
                    node.func.id == self.current_function):
                    self.has_recursive_calls = True
                self.generic_visit(node)
            
            def visit_Subscript(self, node):
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
