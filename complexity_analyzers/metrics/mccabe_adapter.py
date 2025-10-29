"""Адаптер для библиотеки McCabe"""
from typing import Dict, Any, List, Optional
import ast
import logging

from complexity_analyzers.metrics.calculator import BaseMetricsCalculator

logger = logging.getLogger(__name__)

class McCabeAdapter(BaseMetricsCalculator):
    """Адаптер для интеграции с библиотекой McCabe"""
    
    def __init__(self):
        super().__init__('mccabe')
        self._mccabe_available = self._check_mccabe_availability()
    
    def _check_mccabe_availability(self) -> bool:
        """Проверка доступности McCabe"""
        try:
            import mccabe
            return True
        except ImportError:
            logger.warning("McCabe library not available. Install with: pip install mccabe")
            return False
    
    def is_available(self) -> bool:
        """Проверка доступности адаптера"""
        return self._mccabe_available
    
    def calculate(self, source_code: str) -> Dict[str, Any]:
        """Вычисление метрик через McCabe"""
        if not self._mccabe_available:
            return {'error': 'McCabe not available'}
        
        try:
            from mccabe import PathGraphingAstVisitor
            
            # Парсинг кода
            try:
                tree = ast.parse(source_code)
            except SyntaxError as e:
                return {'error': f'Syntax error: {e}'}
            
            # Создание visitor
            visitor = PathGraphingAstVisitor()
            
            # Обход AST
            visitor.preorder(tree, visitor)
            
            metrics = {}
            
            if visitor.graphs:
                complexities = []
                graph_details = []
                
                for graph in visitor.graphs.values():
                    complexity = graph.complexity()
                    complexities.append(complexity)
                    
                    graph_info = {
                        'name': graph.name,
                        'complexity': complexity,
                        'line': graph.lineno,
                        'nodes': len(graph.nodes),
                        'edges': len(graph.edges)
                    }
                    
                    # Дополнительная информация о графе
                    graph_info.update(self._analyze_graph_structure(graph))
                    graph_details.append(graph_info)
                
                # Основные метрики
                metrics['cyclomatic_complexity'] = max(complexities)
                metrics['average_cyclomatic_complexity'] = sum(complexities) / len(complexities)
                metrics['total_functions'] = len(complexities)
                metrics['complexity_sum'] = sum(complexities)
                
                # Детальная информация
                metrics['function_complexities'] = graph_details
                
                # Статистика распределения
                metrics['complexity_distribution'] = self._calculate_complexity_distribution(complexities)
                
                # Метрики графов
                total_nodes = sum(len(g.nodes) for g in visitor.graphs.values())
                total_edges = sum(len(g.edges) for g in visitor.graphs.values())
                
                metrics['total_nodes'] = total_nodes
                metrics['total_edges'] = total_edges
                metrics['avg_nodes_per_function'] = total_nodes / len(visitor.graphs) if visitor.graphs else 0
                metrics['avg_edges_per_function'] = total_edges / len(visitor.graphs) if visitor.graphs else 0
                
            else:
                # Нет функций в коде
                metrics['cyclomatic_complexity'] = 1
                metrics['average_cyclomatic_complexity'] = 1
                metrics['total_functions'] = 0
                metrics['complexity_sum'] = 1
                metrics['function_complexities'] = []
                metrics['complexity_distribution'] = {'low': 1, 'moderate': 0, 'high': 0, 'very_high': 0}
                metrics['total_nodes'] = 0
                metrics['total_edges'] = 0
                metrics['avg_nodes_per_function'] = 0
                metrics['avg_edges_per_function'] = 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"McCabe calculation failed: {e}")
            return {'error': str(e)}
    
    def _analyze_graph_structure(self, graph) -> Dict[str, Any]:
        """Анализ структуры графа потока управления"""
        structure_info = {}
        
        try:
            # Количество узлов и рёбер
            nodes_count = len(graph.nodes)
            edges_count = len(graph.edges)
            
            structure_info['nodes_count'] = nodes_count
            structure_info['edges_count'] = edges_count
            
            # Плотность графа
            if nodes_count > 1:
                max_edges = nodes_count * (nodes_count - 1)
                structure_info['density'] = edges_count / max_edges if max_edges > 0 else 0
            else:
                structure_info['density'] = 0
            
            # Анализ узлов
            if hasattr(graph, 'nodes') and graph.nodes:
                # Подсчёт различных типов узлов
                node_types = {}
                for node in graph.nodes:
                    node_type = getattr(node, 'type', 'unknown')
                    node_types[node_type] = node_types.get(node_type, 0) + 1
                
                structure_info['node_types'] = node_types
                
                # Узлы с высокой степенью (возможные точки ветвления)
                high_degree_nodes = 0
                for node in graph.nodes:
                    # Примерная оценка степени узла
                    if hasattr(node, 'successors') and len(node.successors) > 1:
                        high_degree_nodes += 1
                
                structure_info['branching_nodes'] = high_degree_nodes
            
            # Оценка структурной сложности
            if nodes_count > 0 and edges_count > 0:
                # Простая метрика структурной сложности
                structure_complexity = (edges_count - nodes_count + 2) / nodes_count
                structure_info['structural_complexity'] = structure_complexity
            else:
                structure_info['structural_complexity'] = 1.0
            
        except Exception as e:
            logger.warning(f"Failed to analyze graph structure: {e}")
            structure_info['analysis_error'] = str(e)
        
        return structure_info
    
    def _calculate_complexity_distribution(self, complexities: List[int]) -> Dict[str, int]:
        """Распределение функций по уровням сложности"""
        distribution = {
            'low': 0,       # 1-5
            'moderate': 0,  # 6-10
            'high': 0,      # 11-20
            'very_high': 0  # 21+
        }
        
        for complexity in complexities:
            if complexity <= 5:
                distribution['low'] += 1
            elif complexity <= 10:
                distribution['moderate'] += 1
            elif complexity <= 20:
                distribution['high'] += 1
            else:
                distribution['very_high'] += 1
        
        return distribution
    
    def calculate_detailed_metrics(self, source_code: str) -> Dict[str, Any]:
        """Расширенные метрики McCabe с дополнительным анализом"""
        base_metrics = self.calculate(source_code)
        
        if 'error' in base_metrics:
            return base_metrics
        
        try:
            # Дополнительный анализ на основе AST
            tree = ast.parse(source_code)
            
            detailed_metrics = base_metrics.copy()
            
            # Анализ типов конструкций
            constructs_analysis = self._analyze_control_flow_constructs(tree)
            detailed_metrics['control_flow_analysis'] = constructs_analysis
            
            # Анализ вложенности
            nesting_analysis = self._analyze_nesting_levels(tree)
            detailed_metrics['nesting_analysis'] = nesting_analysis
            
            # Метрики качества кода на основе сложности
            quality_metrics = self._calculate_quality_metrics(base_metrics)
            detailed_metrics['quality_metrics'] = quality_metrics
            
            return detailed_metrics
            
        except Exception as e:
            logger.warning(f"Failed to calculate detailed metrics: {e}")
            base_metrics['detailed_analysis_error'] = str(e)
            return base_metrics
    
    def _analyze_control_flow_constructs(self, tree: ast.AST) -> Dict[str, int]:
        """Анализ конструкций управления потоком"""
        class ControlFlowAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.constructs = {
                    'if_statements': 0,
                    'for_loops': 0,
                    'while_loops': 0,
                    'try_blocks': 0,
                    'with_statements': 0,
                    'function_calls': 0,
                    'return_statements': 0,
                    'break_statements': 0,
                    'continue_statements': 0,
                    'raise_statements': 0
                }
            
            def visit_If(self, node):
                self.constructs['if_statements'] += 1
                self.generic_visit(node)
            
            def visit_For(self, node):
                self.constructs['for_loops'] += 1
                self.generic_visit(node)
            
            def visit_While(self, node):
                self.constructs['while_loops'] += 1
                self.generic_visit(node)
            
            def visit_Try(self, node):
                self.constructs['try_blocks'] += 1
                self.generic_visit(node)
            
            def visit_With(self, node):
                self.constructs['with_statements'] += 1
                self.generic_visit(node)
            
            def visit_Call(self, node):
                self.constructs['function_calls'] += 1
                self.generic_visit(node)
            
            def visit_Return(self, node):
                self.constructs['return_statements'] += 1
                self.generic_visit(node)
            
            def visit_Break(self, node):
                self.constructs['break_statements'] += 1
                self.generic_visit(node)
            
            def visit_Continue(self, node):
                self.constructs['continue_statements'] += 1
                self.generic_visit(node)
            
            def visit_Raise(self, node):
                self.constructs['raise_statements'] += 1
                self.generic_visit(node)
        
        analyzer = ControlFlowAnalyzer()
        analyzer.visit(tree)
        
        return analyzer.constructs
    
    def _analyze_nesting_levels(self, tree: ast.AST) -> Dict[str, int]:
        """Анализ уровней вложенности"""
        class NestingAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.current_depth = 0
                self.max_depth = 0
                self.depth_distribution = {}
                self.nesting_contexts = []
            
            def _enter_context(self, context_type: str):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                
                if self.current_depth not in self.depth_distribution:
                    self.depth_distribution[self.current_depth] = 0
                self.depth_distribution[self.current_depth] += 1
                
                self.nesting_contexts.append(context_type)
            
            def _exit_context(self):
                if self.nesting_contexts:
                    self.nesting_contexts.pop()
                self.current_depth = max(0, self.current_depth - 1)
            
            def visit_If(self, node):
                self._enter_context('if')
                self.generic_visit(node)
                self._exit_context()
            
            def visit_For(self, node):
                self._enter_context('for')
                self.generic_visit(node)
                self._exit_context()
            
            def visit_While(self, node):
                self._enter_context('while')
                self.generic_visit(node)
                self._exit_context()
            
            def visit_Try(self, node):
                self._enter_context('try')
                self.generic_visit(node)
                self._exit_context()
            
            def visit_With(self, node):
                self._enter_context('with')
                self.generic_visit(node)
                self._exit_context()
            
            def visit_FunctionDef(self, node):
                self._enter_context('function')
                self.generic_visit(node)
                self._exit_context()
        
        analyzer = NestingAnalyzer()
        analyzer.visit(tree)
        
        return {
            'max_nesting_depth': analyzer.max_depth,
            'depth_distribution': analyzer.depth_distribution,
            'total_nested_blocks': sum(analyzer.depth_distribution.values())
        }
    
    def _calculate_quality_metrics(self, base_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Метрики качества кода на основе сложности"""
        quality_metrics = {}
        
        # Средняя сложность на функцию
        avg_complexity = base_metrics.get('average_cyclomatic_complexity', 1)
        max_complexity = base_metrics.get('cyclomatic_complexity', 1)
        total_functions = base_metrics.get('total_functions', 1)
        
        # Индекс качества (обратно пропорционален сложности)
        quality_metrics['quality_index'] = 100 / (1 + avg_complexity)
        
        # Индекс рефакторинга (процент функций с высокой сложностью)
        distribution = base_metrics.get('complexity_distribution', {})
        high_complexity_functions = distribution.get('high', 0) + distribution.get('very_high', 0)
        
        if total_functions > 0:
            quality_metrics['refactoring_index'] = (high_complexity_functions / total_functions) * 100
        else:
            quality_metrics['refactoring_index'] = 0
        
        # Индекс консистентности (насколько равномерно распределена сложность)
        if total_functions > 1:
            # Коэффициент вариации сложности
            complexities = [info['complexity'] for info in base_metrics.get('function_complexities', [])]
            if complexities:
                import statistics
                try:
                    std_dev = statistics.stdev(complexities)
                    mean_complexity = statistics.mean(complexities)
                    if mean_complexity > 0:
                        cv = std_dev / mean_complexity
                        quality_metrics['consistency_index'] = max(0, 100 - (cv * 100))
                    else:
                        quality_metrics['consistency_index'] = 100
                except statistics.StatisticsError:
                    quality_metrics['consistency_index'] = 100
            else:
                quality_metrics['consistency_index'] = 100
        else:
            quality_metrics['consistency_index'] = 100
        
        # Рекомендации
        recommendations = []
        
        if avg_complexity > 10:
            recommendations.append("Consider breaking down complex functions")
        
        if max_complexity > 20:
            recommendations.append("Refactor functions with very high complexity")
        
        if quality_metrics['refactoring_index'] > 25:
            recommendations.append("High percentage of complex functions needs attention")
        
        quality_metrics['recommendations'] = recommendations
        
        return quality_metrics
