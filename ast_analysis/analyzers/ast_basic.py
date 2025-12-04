"""
Базовый AST анализатор с предсказанием сложности.
"""

import ast as python_ast
from typing import Dict, Any
from ast_analysis.core.base_analyzer import BaseASTAnalyzer
from ast_analysis.core.result import ASTAnalysisResult
from ast_analysis.core.registry import register_analyzer
from ast_analysis.analyzers.loop_analyzer import LoopAnalyzer
from ast_analysis.analyzers.recursion_analyzer import RecursionAnalyzer
from ast_analysis.analyzers.complexity_predictor import ComplexityPredictor


@register_analyzer('basic')
class ASTBasicAnalyzer(BaseASTAnalyzer):
    """
    Базовый AST анализатор с предсказанием сложности.
    Извлекает признаки + предсказывает класс сложности.
    """
    
    def __init__(self, name: str = "ast_basic", **config):
        super().__init__(name, **config)
        self.max_depth = config.get('max_depth', 100)
        self.track_all_nodes = config.get('track_all_nodes', True)
        
        # Специализированные анализаторы
        self.predictor = ComplexityPredictor()
    
    def analyze(self, code: str) -> ASTAnalysisResult:
        """Анализ кода с предсказанием"""
        try:
            # Парсинг AST
            tree = python_ast.parse(code)
            
            # 1. Извлечение базовых признаков
            features = self._extract_features(tree)
            
            # 2. Анализ циклов
            loop_analyzer = LoopAnalyzer()
            loop_analyzer.visit(tree)
            loop_summary = loop_analyzer.get_summary()
            
            # 3. Анализ рекурсии
            recursion_analyzer = RecursionAnalyzer()
            recursion_analyzer.visit(tree)
            recursion_summary = recursion_analyzer.get_summary()
            
            # 4. Предсказание сложности
            analysis_data = {
                'features': features,
                'loop_summary': loop_summary,
                'recursion_summary': recursion_summary
            }
            prediction, confidence = self.predictor.predict(analysis_data)
            
            # 5. Метаданные предсказания
            prediction_metadata = {
                'max_nesting': loop_summary.get('max_nesting', 0),
                'has_logarithmic_step': loop_summary.get('has_logarithmic_step', False),
                'has_dependent_inner_loop': loop_summary.get('has_dependent_inner_loop', False),
                'recursive_functions': recursion_summary.get('recursive_functions', 0),
                'total_loops': loop_summary.get('total_loops', 0)
            }
            
            return ASTAnalysisResult.from_success(
                features=features,
                analyzer_name=self.name,
                code_length=len(code),
                prediction=prediction,
                confidence=confidence,
                prediction_metadata=prediction_metadata
            )
        
        except SyntaxError as e:
            return ASTAnalysisResult.from_error(
                error=f"SyntaxError: {str(e)}",
                analyzer_name=self.name
            )
        
        except Exception as e:
            return ASTAnalysisResult.from_error(
                error=f"Error: {str(e)}",
                analyzer_name=self.name
            )
    
    def _extract_features(self, tree: python_ast.AST) -> Dict[str, Any]:
        """Извлечение базовых признаков из AST"""
        features = {}
        
        # Подсчёт узлов
        node_counts = self._count_nodes(tree)
        features.update(node_counts)
        
        # Метрики дерева
        features['max_depth'] = self._compute_depth(tree)
        features['max_width'] = self._compute_width(tree)
        features['total_nodes'] = sum(node_counts.values())
        features['unique_node_types'] = len(node_counts)
        
        # Базовые структурные метрики
        features['num_functions'] = node_counts.get('FunctionDef', 0) + node_counts.get('AsyncFunctionDef', 0)
        features['num_classes'] = node_counts.get('ClassDef', 0)
        features['num_imports'] = node_counts.get('Import', 0) + node_counts.get('ImportFrom', 0)
        features['num_assignments'] = node_counts.get('Assign', 0) + node_counts.get('AugAssign', 0)
        
        # Сложность управления потоком
        features['num_loops'] = (
            node_counts.get('For', 0) + 
            node_counts.get('While', 0) + 
            node_counts.get('AsyncFor', 0)
        )
        features['num_conditionals'] = node_counts.get('If', 0)
        features['num_try_except'] = node_counts.get('Try', 0)
        
        return features
    
    def _count_nodes(self, tree: python_ast.AST) -> Dict[str, int]:
        """Подсчёт узлов каждого типа"""
        counts = {}
        
        for node in python_ast.walk(tree):
            node_type = node.__class__.__name__
            counts[node_type] = counts.get(node_type, 0) + 1
        
        return counts
    
    def _compute_depth(self, node: python_ast.AST, current_depth: int = 0) -> int:
        """Вычисление максимальной глубины дерева"""
        if current_depth > self.max_depth:
            return current_depth
        
        max_child_depth = current_depth
        
        for child in python_ast.iter_child_nodes(node):
            child_depth = self._compute_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def _compute_width(self, tree: python_ast.AST) -> int:
        """Вычисление максимальной ширины дерева"""
        level_widths = {}
        
        def traverse(node, depth):
            level_widths[depth] = level_widths.get(depth, 0) + 1
            for child in python_ast.iter_child_nodes(node):
                traverse(child, depth + 1)
        
        traverse(tree, 0)
        
        return max(level_widths.values()) if level_widths else 0
    
    def get_feature_names(self) -> list[str]:
        """Список извлекаемых признаков"""
        base_features = [
            'total_nodes',
            'unique_node_types',
            'max_depth',
            'max_width',
            'num_functions',
            'num_classes',
            'num_imports',
            'num_assignments',
            'num_loops',
            'num_conditionals',
            'num_try_except'
        ]
        
        return base_features
