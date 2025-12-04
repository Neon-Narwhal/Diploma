"""
Расширенный AST-анализатор (AdvancedASTAnalyzer).
"""

import ast
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict

from ast_analysis.core.base_analyzer import BaseASTAnalyzer
from ast_analysis.core.result import ASTAnalysisResult
from ast_analysis.core.registry import register_analyzer
from ast_analysis.core.enums import ComplexityClass

from ast_analysis.analyzers.feature_extractor import ASTFeatureExtractor
from ast_analysis.analyzers.pattern_detectors.registry import PatternDetectorRegistry

class AdvancedLoopAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.loops: List[Dict[str, Any]] = []
        self.nesting_stack: List[Dict[str, Any]] = []
        self.current_function: Optional[str] = None
        self.max_nesting: int = 0
        self.loop_variables: Set[str] = set()
        self.has_logarithmic_step = False 
        self.has_dependent_inner_loop = False
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        prev_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = prev_function
    
    def visit_For(self, node: ast.For):
        loop_info = self._analyze_loop(node, 'for')
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
            for arg in node.iter.args:
                if isinstance(arg, ast.Name) and arg.id in self.loop_variables:
                    self.has_dependent_inner_loop = True
                    loop_info['complexity_indicators'].append('dependent_range')
        self._enter_loop(loop_info)
        self.generic_visit(node)
        self._exit_loop()
    
    def visit_While(self, node: ast.While):
        loop_info = self._analyze_loop(node, 'while')
        self._enter_loop(loop_info)
        self.generic_visit(node)
        self._exit_loop()
    
    def visit_AugAssign(self, node: ast.AugAssign):
        if isinstance(node.op, (ast.Mult, ast.Div, ast.FloorDiv)):
            if self.nesting_stack:
                self.has_logarithmic_step = True
                self.nesting_stack[-1]['complexity_indicators'].append('logarithmic_step')
        self.generic_visit(node)
    
    def _analyze_loop(self, node: ast.AST, loop_type: str) -> Dict[str, Any]:
        loop_info = {
            'type': loop_type,
            'line': getattr(node, 'lineno', 0),
            'nesting_level': len(self.nesting_stack),
            'function': self.current_function,
            'variables': set(),
            'complexity_indicators': []
        }
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            loop_info['variables'].add(node.target.id)
            self.loop_variables.add(node.target.id)
        elif isinstance(node, ast.While):
            for var_node in ast.walk(node.test):
                if isinstance(var_node, ast.Name):
                    loop_info['variables'].add(var_node.id)
        return loop_info
    
    def _enter_loop(self, loop_info: Dict[str, Any]):
        self.nesting_stack.append(loop_info)
        self.max_nesting = max(self.max_nesting, len(self.nesting_stack))
        self.loops.append(loop_info)
    
    def _exit_loop(self):
        if self.nesting_stack:
            self.nesting_stack.pop()

class RecursionComplexityAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.functions: Dict[str, Dict[str, Any]] = {}
        self.current_function: Optional[str] = None
        self.recursion_patterns: List[Dict[str, Any]] = []
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        func_name = node.name
        self.functions[func_name] = {
            'name': func_name,
            'line': node.lineno,
            'is_recursive': False,
            'recursion_type': 'none',
            'recursive_calls': []
        }
        prev_function = self.current_function
        self.current_function = func_name
        self.generic_visit(node)
        self._analyze_recursion_pattern(func_name)
        self.current_function = prev_function
    
    def visit_Call(self, node: ast.Call):
        if self.current_function and isinstance(node.func, ast.Name):
            called_func = node.func.id
            if called_func == self.current_function:
                self.functions[self.current_function]['is_recursive'] = True
                self.functions[self.current_function]['recursive_calls'].append({})
        self.generic_visit(node)
    
    def _analyze_recursion_pattern(self, func_name: str):
        func_info = self.functions[func_name]
        if not func_info['is_recursive']: return
        
        count = len(func_info['recursive_calls'])
        if count == 1:
            func_info['recursion_type'] = 'linear'
            func_info['estimated_complexity'] = ComplexityClass.LINEAR
        elif count == 2:
            func_info['recursion_type'] = 'binary'
            func_info['estimated_complexity'] = ComplexityClass.EXPONENTIAL
        elif count > 2:
            func_info['recursion_type'] = 'tree'
            func_info['estimated_complexity'] = ComplexityClass.FACTORIAL
            
        self.recursion_patterns.append(func_info)

class DataStructureAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.data_structures: Dict[str, int] = defaultdict(int)
        self.operations: Dict[str, int] = defaultdict(int)
        self.complexity_operations: List[Dict[str, Any]] = []
    
    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            if method_name in ['append', 'pop']:
                self._record_operation('list', method_name, node.lineno, ComplexityClass.CONSTANT)
            elif method_name in ['insert', 'remove']:
                self._record_operation('list', method_name, node.lineno, ComplexityClass.LINEAR)
            elif method_name in ['sort']:
                 self._record_operation('list', method_name, node.lineno, ComplexityClass.LINEARITHMIC)
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name == 'sorted':
                self._record_operation('builtin', func_name, node.lineno, ComplexityClass.LINEARITHMIC)
        self.generic_visit(node)
        
    def _record_operation(self, structure: str, operation: str, line: int, complexity: ComplexityClass):
        self.complexity_operations.append({
            'structure': structure, 'operation': operation, 
            'line': line, 'complexity': complexity
        })

@register_analyzer('advanced')
class AdvancedASTAnalyzer(BaseASTAnalyzer):
    """Продвинутый AST анализатор"""
    
    def __init__(self, name: str = "ast_advanced", **config):
        super().__init__(name, **config)
        self.loop_analyzer = AdvancedLoopAnalyzer()
        self.recursion_analyzer = RecursionComplexityAnalyzer()
        self.data_structure_analyzer = DataStructureAnalyzer()
        self.pattern_detector_registry = PatternDetectorRegistry()
        self.feature_extractor = ASTFeatureExtractor()
        
    def analyze(self, code: str) -> ASTAnalysisResult:
        try:
            tree = ast.parse(code)
            self._reset_analyzers()
            
            self.loop_analyzer.visit(tree)
            self.recursion_analyzer.visit(tree)
            self.data_structure_analyzer.visit(tree)
            
            detected_patterns = self.pattern_detector_registry.detect_all(tree)
            features = self.feature_extractor.extract_all_features(tree, code)
            
            analysis_results = self._combine_analysis_results(detected_patterns, features)
            
            complexity_enum = self._determine_complexity(analysis_results)
            prediction = complexity_enum.value
            
            metadata = {
                'max_nesting': self.loop_analyzer.max_nesting,
                'patterns': detected_patterns
            }
            
            return ASTAnalysisResult.from_success(
                features=features,
                analyzer_name=self.name,
                code_length=len(code),
                prediction=prediction,
                confidence=0.9,
                prediction_metadata=metadata
            )
        except Exception as e:
            return ASTAnalysisResult.from_error(f"Error: {e}", self.name)

    def _reset_analyzers(self):
        self.loop_analyzer = AdvancedLoopAnalyzer()
        self.recursion_analyzer = RecursionComplexityAnalyzer()
        self.data_structure_analyzer = DataStructureAnalyzer()

    def _combine_analysis_results(self, patterns, features):
        return {
            'patterns': patterns,
            'features': features,
            'loop_analysis': {
                'max_nesting': self.loop_analyzer.max_nesting,
                'has_log_step': self.loop_analyzer.has_logarithmic_step
            },
            'recursion_analysis': {
                'patterns': self.recursion_analyzer.recursion_patterns
            },
            'data_analysis': {
                'ops': self.data_structure_analyzer.complexity_operations
            }
        }

    def _determine_complexity(self, results: Dict[str, Any]) -> ComplexityClass:
        # 1. Recursion
        rec_patterns = results['recursion_analysis']['patterns']
        if rec_patterns:
            return rec_patterns[0].get('estimated_complexity', ComplexityClass.EXPONENTIAL)
            
        # 2. Loops
        nesting = results['loop_analysis']['max_nesting']
        has_log = results['loop_analysis']['has_log_step']
        
        # Check for sorting
        has_sort = False
        for op in results['data_analysis']['ops']:
            if op['complexity'] == ComplexityClass.LINEARITHMIC:
                has_sort = True
        
        if nesting == 0: return ComplexityClass.CONSTANT
        if nesting == 1:
            if has_log: return ComplexityClass.LOGARITHMIC
            if has_sort: return ComplexityClass.LINEARITHMIC
            return ComplexityClass.LINEAR
        if nesting == 2:
            if has_log: return ComplexityClass.LINEARITHMIC
            return ComplexityClass.QUADRATIC
        
        return ComplexityClass.CUBIC
