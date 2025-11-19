"""Анализатор графа потока управления (CFG v2.0)"""
import ast
import networkx as nx
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass

from complexity_analyzers.core.base import BaseComplexityAnalyzer, AnalyzerType
from complexity_analyzers.core.result import ComplexityResult, ComplexityClass, ComplexityMetrics
import numpy as np

# Импорты новых компонентов CFG v2.0
from complexity_analyzers.analyzers.cfg_builder import PythonCFGBuilder
from complexity_analyzers.analyzers.cfg_data_flow import DataFlowAnalyzer
from complexity_analyzers.analyzers.cfg_iterator_analysis import IteratorRangeAnalyzer
from complexity_analyzers.analyzers.cfg_library_calls import LibraryCallRecognizer
#from complexity_analyzers.analyzers.cfg_multi_variable import MultiVariableTracker
from complexity_analyzers.analyzers.cfg_complexity_composer import ComplexityComposer


class CFGAnalyzer:
    """Анализатор метрик CFG"""
    
    def __init__(self):
        self.graph: Optional[nx.DiGraph] = None
    
    def analyze_cfg(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Анализ метрик CFG"""
        self.graph = graph
        
        metrics = {
            # Базовые метрики
            'nodes_count': graph.number_of_nodes(),
            'edges_count': graph.number_of_edges(),
            
            # Цикломатическая сложность
            'cyclomatic_complexity': self._calculate_cyclomatic_complexity(),
            
            # Метрики связности
            'strongly_connected_components': len(list(nx.strongly_connected_components(graph))),
            'weakly_connected_components': len(list(nx.weakly_connected_components(graph))),
            
            # Метрики путей
            'longest_path_length': self._calculate_longest_path(),
            'average_path_length': self._calculate_average_path_length(),
            
            # Метрики узлов
            'decision_nodes': self._count_decision_nodes(),
            'loop_nodes': self._count_loop_nodes(),
            'max_indegree': max([graph.in_degree(n) for n in graph.nodes()] or [0]),
            'max_outdegree': max([graph.out_degree(n) for n in graph.nodes()] or [0]),
            
            # Метрики сложности
            'nesting_depth': self._calculate_nesting_depth(),
            'nested_loop_depth': self._count_nested_loops(),
            'fan_in_out': self._calculate_fan_metrics(),
            
            # Структурные метрики
            'back_edges': self._count_back_edges(),
            'forward_edges': self._count_forward_edges(),
            'cross_edges': self._count_cross_edges()
        }
        
        return metrics
    
    def _calculate_cyclomatic_complexity(self) -> int:
        """Вычисление цикломатической сложности"""
        if not self.graph:
            return 1
        
        edges = self.graph.number_of_edges()
        nodes = self.graph.number_of_nodes()
        components = len(list(nx.weakly_connected_components(self.graph)))
        
        return max(1, edges - nodes + 2 * components)
    
    def _calculate_longest_path(self) -> int:
        """Вычисление длины самого длинного пути"""
        if not self.graph or self.graph.number_of_nodes() == 0:
            return 0
        
        try:
            if nx.is_directed_acyclic_graph(self.graph):
                return nx.dag_longest_path_length(self.graph)
            else:
                max_length = 0
                for node in self.graph.nodes():
                    try:
                        lengths = nx.single_source_shortest_path_length(
                            self.graph, node, cutoff=50
                        )
                        if lengths:
                            max_length = max(max_length, max(lengths.values()))
                    except:
                        continue
                return max_length
        except:
            return 0
    
    def _calculate_average_path_length(self) -> float:
        """Вычисление средней длины пути"""
        if not self.graph or self.graph.number_of_nodes() <= 1:
            return 0.0
        
        try:
            if nx.is_strongly_connected(self.graph):
                return nx.average_shortest_path_length(self.graph)
            else:
                total_length = 0
                total_pairs = 0
                
                for component in nx.strongly_connected_components(self.graph):
                    if len(component) > 1:
                        subgraph = self.graph.subgraph(component)
                        if nx.is_strongly_connected(subgraph):
                            avg_length = nx.average_shortest_path_length(subgraph)
                            pairs = len(component) * (len(component) - 1)
                            total_length += avg_length * pairs
                            total_pairs += pairs
                
                return total_length / total_pairs if total_pairs > 0 else 0.0
        except:
            return 0.0
    
    def _count_decision_nodes(self) -> int:
        """Подсчет узлов принятия решений"""
        count = 0
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            if (node_data.get('type') == 'condition' or 
                self.graph.out_degree(node_id) > 1):
                count += 1
        return count
    
    def _count_loop_nodes(self) -> int:
        """Подсчет узлов циклов"""
        count = 0
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            if node_data.get('type') == 'loop':
                count += 1
        return count
    
    def _calculate_nesting_depth(self) -> int:
        """Вычисление глубины вложенности (циклы + условия)"""
        max_depth = 0
        
        def dfs_depth(node_id: int, current_depth: int, visited: Set[int]):
            nonlocal max_depth
            
            if node_id in visited:
                return
            
            visited.add(node_id)
            max_depth = max(max_depth, current_depth)
            
            node_data = self.graph.nodes[node_id]
            new_depth = current_depth
            
            if node_data.get('type') in ['loop', 'condition']:
                new_depth += 1
            
            for successor in self.graph.successors(node_id):
                dfs_depth(successor, new_depth, visited.copy())
        
        entry_nodes = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        
        for entry_node in entry_nodes:
            dfs_depth(entry_node, 0, set())
        
        return max_depth
    
    def _count_nested_loops(self) -> int:
        """Подсчёт максимальной вложенности ТОЛЬКО циклов"""
        max_loop_depth = 0
        
        def dfs_loop_depth(node_id: int, current_loop_depth: int, visited: Set[int]):
            nonlocal max_loop_depth
            
            if node_id in visited:
                return
            
            visited.add(node_id)
            node_data = self.graph.nodes[node_id]
            
            new_depth = current_loop_depth
            if node_data.get('type') == 'loop':
                new_depth += 1
                max_loop_depth = max(max_loop_depth, new_depth)
            
            for successor in self.graph.successors(node_id):
                dfs_loop_depth(successor, new_depth, visited.copy())
        
        entry_nodes = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        
        for entry_node in entry_nodes:
            dfs_loop_depth(entry_node, 0, set())
        
        return max_loop_depth
    
    def _calculate_fan_metrics(self) -> Dict[str, float]:
        """Вычисление метрик fan-in/fan-out"""
        if not self.graph.nodes():
            return {'avg_fan_in': 0.0, 'avg_fan_out': 0.0, 
                    'max_fan_in': 0, 'max_fan_out': 0}
        
        fan_ins = [self.graph.in_degree(n) for n in self.graph.nodes()]
        fan_outs = [self.graph.out_degree(n) for n in self.graph.nodes()]
        
        return {
            'avg_fan_in': np.mean(fan_ins) if fan_ins else 0.0,
            'avg_fan_out': np.mean(fan_outs) if fan_outs else 0.0,
            'max_fan_in': max(fan_ins) if fan_ins else 0,
            'max_fan_out': max(fan_outs) if fan_outs else 0
        }
    
    def _count_back_edges(self) -> int:
        """Подсчет обратных рёбер"""
        back_edges = 0
        
        try:
            for edge in self.graph.edges():
                source, target = edge
                
                try:
                    if nx.has_path(self.graph, target, source):
                        back_edges += 1
                except:
                    continue
                    
        except:
            pass
        
        return back_edges
    
    def _count_forward_edges(self) -> int:
        """Подсчет прямых рёбер"""
        return max(0, self.graph.number_of_edges() - self._count_back_edges())
    
    def _count_cross_edges(self) -> int:
        """Подсчет перекрестных рёбер"""
        return 0


class CFGComplexityAnalyzer(BaseComplexityAnalyzer):
    """Анализатор сложности на основе CFG v2.0"""
    
    def __init__(self):
        super().__init__("cfg_analyzer", AnalyzerType.CFG)
        self.cfg_builder = PythonCFGBuilder()
        self.cfg_analyzer = CFGAnalyzer()
    
    def is_available(self) -> bool:
        """Проверка доступности"""
        try:
            import networkx
            return True
        except ImportError:
            return False
    
    def analyze(self, context) -> ComplexityResult:
        """Анализ сложности через CFG v2.0"""
        try:
            # 1. Парсинг и построение CFG
            tree = ast.parse(context.source_code)
            cfg = self.cfg_builder.build_cfg(context.source_code)
            
            # 2. Базовые метрики CFG
            cfg_metrics = self.cfg_analyzer.analyze_cfg(cfg)
            
            # 3. НОВОЕ: Анализ потока данных
            dfa = DataFlowAnalyzer(cfg, tree)
            dfa_results = dfa.analyze()
            cfg_metrics['data_flow'] = dfa_results
            
            # 4. НОВОЕ: Анализ итераторов
            ira = IteratorRangeAnalyzer(cfg, tree, dfa_results)
            ira_results = ira.analyze()
            cfg_metrics['iterator_analysis'] = ira_results
            
            # 5. НОВОЕ: Распознавание библиотечных вызовов
            lcr = LibraryCallRecognizer(cfg, tree)
            lcr_results = lcr.analyze()
            cfg_metrics['library_calls'] = lcr_results
            
            # 6. НОВОЕ: Отслеживание множественных переменных
            mvt = MultiVariableTracker(cfg, dfa_results, ira_results)
            mvt_results = mvt.analyze()
            cfg_metrics['multi_variable'] = mvt_results
            
            # 7. НОВОЕ: Композиция сложностей
            composer = ComplexityComposer(cfg, ira_results, lcr_results)
            composer_results = composer.analyze()
            cfg_metrics['complexity_composition'] = composer_results
            
            # 8. Определение класса сложности
            complexity_class = self._infer_complexity_from_cfg(cfg_metrics)
            confidence = self._calculate_confidence(cfg_metrics)
            
            return ComplexityResult(
                complexity_class=complexity_class,
                confidence=confidence,
                analyzer_name=self.name,
                metrics=ComplexityMetrics(
                    time_complexity=complexity_class,
                    cyclomatic_complexity=cfg_metrics.get('cyclomatic_complexity'),
                    nested_depth=cfg_metrics.get('nesting_depth')
                ),
                cfg_metrics=cfg_metrics
            )
            
        except Exception as e:
            return ComplexityResult(
                complexity_class=ComplexityClass.UNKNOWN,
                confidence=0.0,
                analyzer_name=self.name,
                errors=[f"CFG v2.0 analysis error: {e}"]
            )
    
    def _infer_complexity_from_cfg(self, metrics: Dict[str, Any]) -> ComplexityClass:
        """Определение класса сложности с использованием CFG v2.0"""
        
        # Получаем результаты композитора
        composer_results = metrics.get('complexity_composition', {})
        overall_notation = composer_results.get('overall_complexity', 'O(1)')
        
        # Маппинг нотации на ComplexityClass
        notation_lower = overall_notation.lower()
        
        if 'o(1)' in notation_lower:
            return ComplexityClass.CONSTANT
        elif 'o(logn)' in notation_lower or 'o(log(n))' in notation_lower:
            return ComplexityClass.LOGARITHMIC
        elif 'o(nlogn)' in notation_lower or 'o(n*logn)' in notation_lower:
            return ComplexityClass.LINEARITHMIC
        elif 'o(n)' in notation_lower and '^' not in notation_lower and '*' not in notation_lower:
            return ComplexityClass.LINEAR
        elif 'o(n^2)' in notation_lower or 'o(m*n)' in notation_lower or 'o(n*m)' in notation_lower:
            return ComplexityClass.QUADRATIC
        elif 'o(n^3)' in notation_lower:
            return ComplexityClass.CUBIC
        elif 'o(n^' in notation_lower:
            return ComplexityClass.POLYNOMIAL
        elif 'o(2^n)' in notation_lower:
            return ComplexityClass.EXPONENTIAL
        elif 'o(n!)' in notation_lower:
            return ComplexityClass.FACTORIAL
        
        return ComplexityClass.UNKNOWN
    
    def _calculate_confidence(self, metrics: Dict[str, Any]) -> float:
        """Расчет уверенности в результате"""
        base_confidence = 0.8
        
        # Увеличиваем уверенность при наличии четких индикаторов
        nodes_count = metrics.get('nodes_count', 0)
        if nodes_count > 5:
            base_confidence += 0.1
        
        # Уменьшаем при сложных графах
        avg_path_length = metrics.get('average_path_length', 0)
        if avg_path_length > 10:
            base_confidence -= 0.2
        
        # Увеличиваем при распознанных библиотечных вызовах
        library_calls = metrics.get('library_calls', {})
        if library_calls.get('total_calls', 0) > 0:
            base_confidence += 0.05
        
        return max(0.1, min(base_confidence, 1.0))
