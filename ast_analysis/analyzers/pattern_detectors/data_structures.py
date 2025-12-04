import ast
from typing import Dict, Any
from .base import PatternDetector

class DataStructurePatternDetector(PatternDetector):
    """Детектор паттернов использования структур данных"""
    
    def __init__(self):
        super().__init__("data_structure_patterns")
    
    def detect(self, tree: ast.AST) -> Dict[str, Any]:
        class DataStructureVisitor(ast.NodeVisitor):
            def __init__(self):
                self.data_structures = {
                    'list': 0, 'dict': 0, 'set': 0, 'tuple': 0,
                    'deque': 0, 'heapq': 0, 'defaultdict': 0
                }
                self.operations = {
                    'list_append': 0, 'list_pop': 0, 'list_insert': 0,
                    'dict_get': 0, 'dict_keys': 0,
                    'set_add': 0, 'set_union': 0
                }
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in self.data_structures:
                        self.data_structures[func_name] += 1
                elif isinstance(node.func, ast.Attribute):
                    method_name = node.func.attr
                    if method_name == 'append': self.operations['list_append'] += 1
                    elif method_name == 'pop': self.operations['list_pop'] += 1
                    elif method_name == 'insert': self.operations['list_insert'] += 1
                    elif method_name == 'get': self.operations['dict_get'] += 1
                    elif method_name == 'keys': self.operations['dict_keys'] += 1
                    elif method_name == 'add': self.operations['set_add'] += 1
                    elif method_name == 'union': self.operations['set_union'] += 1
                self.generic_visit(node)
            
            def visit_List(self, node):
                self.data_structures['list'] += 1
                self.generic_visit(node)
            
            def visit_Dict(self, node):
                self.data_structures['dict'] += 1
                self.generic_visit(node)
                
        visitor = DataStructureVisitor()
        visitor.visit(tree)
        
        primary_structure = None
        if any(visitor.data_structures.values()):
            primary_structure = max(visitor.data_structures, key=visitor.data_structures.get)

        return {
            'data_structures_used': visitor.data_structures,
            'operations_used': visitor.operations,
            'primary_structure': primary_structure
        }
