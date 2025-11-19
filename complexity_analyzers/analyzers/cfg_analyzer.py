"""Анализатор графа потока управления"""
import ast
import networkx as nx
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from complexity_analyzers.core.base import BaseComplexityAnalyzer, AnalyzerType
from complexity_analyzers.core.result import ComplexityResult, ComplexityClass, ComplexityMetrics
import numpy as np

@dataclass
class CFGNode:
    """Узел графа потока управления"""
    id: int
    type: str  # 'statement', 'condition', 'loop', 'function_entry', 'function_exit'
    line_number: int
    code: str = ""
    ast_node: Optional[ast.AST] = None
    
    def __hash__(self):
        return hash(self.id)

@dataclass
class CFGEdge:
    """Ребро графа потока управления"""
    source: CFGNode
    target: CFGNode
    type: str  # 'sequential', 'true_branch', 'false_branch', 'loop_back', 'call', 'return'
    condition: Optional[str] = None

class CFGBuilder:
    """Построитель графа потока управления"""
    
    def __init__(self):
        self.nodes: List[CFGNode] = []
        self.edges: List[CFGEdge] = []
        self.node_counter = 0
        self.current_function = None
    
    def build_cfg(self, source_code: str) -> nx.DiGraph:
        """Построение CFG из исходного кода"""
        try:
            tree = ast.parse(source_code)
            self._reset()
            
            # Создаем граф
            graph = nx.DiGraph()
            
            # Обходим AST и строим CFG
            self._visit_node(tree, graph)
            
            return graph
            
        except Exception as e:
            raise ValueError(f"Error building CFG: {e}")
    
    def _reset(self):
        """Сброс состояния билдера"""
        self.nodes.clear()
        self.edges.clear()
        self.node_counter = 0
        self.current_function = None
    
    def _create_node(self, node_type: str, line_number: int, 
                    code: str = "", ast_node: ast.AST = None) -> CFGNode:
        """Создание нового узла CFG"""
        cfg_node = CFGNode(
            id=self.node_counter,
            type=node_type,
            line_number=line_number,
            code=code,
            ast_node=ast_node
        )
        self.nodes.append(cfg_node)
        self.node_counter += 1
        return cfg_node
    
    def _add_edge(self, graph: nx.DiGraph, source: CFGNode, target: CFGNode, 
                 edge_type: str = 'sequential', condition: str = None):
        """Добавление ребра в граф"""
        edge = CFGEdge(source, target, edge_type, condition)
        self.edges.append(edge)
        
        graph.add_node(source.id, **{
            'type': source.type,
            'line': source.line_number,
            'code': source.code
        })
        graph.add_node(target.id, **{
            'type': target.type,
            'line': target.line_number,
            'code': target.code
        })
        
        graph.add_edge(source.id, target.id, **{
            'type': edge_type,
            'condition': condition
        })
    
    def _visit_node(self, node: ast.AST, graph: nx.DiGraph, 
                   entry_node: CFGNode = None) -> Tuple[CFGNode, CFGNode]:
        """Обход узла AST и построение CFG"""
        if isinstance(node, ast.Module):
            return self._visit_module(node, graph)
        elif isinstance(node, ast.FunctionDef):
            return self._visit_function(node, graph)
        elif isinstance(node, ast.If):
            return self._visit_if(node, graph, entry_node)
        elif isinstance(node, ast.For):
            return self._visit_for(node, graph, entry_node)
        elif isinstance(node, ast.While):
            return self._visit_while(node, graph, entry_node)
        elif isinstance(node, ast.Return):
            return self._visit_return(node, graph)
        elif isinstance(node, (ast.Break, ast.Continue)):
            return self._visit_break_continue(node, graph)
        else:
            return self._visit_statement(node, graph)
    
    def _visit_module(self, node: ast.Module, graph: nx.DiGraph) -> Tuple[CFGNode, CFGNode]:
        """Обработка модуля"""
        if not node.body:
            entry = self._create_node('module_entry', 1, 'module start')
            exit = self._create_node('module_exit', 1, 'module end')
            self._add_edge(graph, entry, exit)
            return entry, exit
        
        entry = self._create_node('module_entry', 1, 'module start')
        current = entry
        
        for stmt in node.body:
            stmt_entry, stmt_exit = self._visit_node(stmt, graph)
            if current != entry:
                self._add_edge(graph, current, stmt_entry)
            else:
                self._add_edge(graph, entry, stmt_entry)
            current = stmt_exit
        
        exit = self._create_node('module_exit', getattr(node.body[-1], 'lineno', 1), 'module end')
        self._add_edge(graph, current, exit)
        
        return entry, exit
    
    def _visit_function(self, node: ast.FunctionDef, graph: nx.DiGraph) -> Tuple[CFGNode, CFGNode]:
        """Обработка функции"""
        prev_function = self.current_function
        self.current_function = node.name
        
        entry = self._create_node(
            'function_entry', 
            node.lineno, 
            f'def {node.name}(...):',
            node
        )
        
        if not node.body:
            exit = self._create_node('function_exit', node.lineno, 'return', node)
            self._add_edge(graph, entry, exit)
        else:
            current = entry
            
            for stmt in node.body:
                stmt_entry, stmt_exit = self._visit_node(stmt, graph)
                self._add_edge(graph, current, stmt_entry)
                current = stmt_exit
            
            exit = self._create_node(
                'function_exit', 
                getattr(node.body[-1], 'lineno', node.lineno), 
                'return',
                node
            )
            self._add_edge(graph, current, exit)
        
        self.current_function = prev_function
        return entry, exit
    
    def _visit_if(self, node: ast.If, graph: nx.DiGraph, 
                 entry_node: CFGNode = None) -> Tuple[CFGNode, CFGNode]:
        """Обработка условного оператора"""
        condition_node = self._create_node(
            'condition', 
            node.lineno, 
            f'if {ast.unparse(node.test) if hasattr(ast, "unparse") else "condition"}:',
            node
        )
        
        # True ветка
        true_entry = None
        true_exit = None
        
        if node.body:
            current = None
            for stmt in node.body:
                stmt_entry, stmt_exit = self._visit_node(stmt, graph)
                if current is None:
                    true_entry = stmt_entry
                    self._add_edge(graph, condition_node, stmt_entry, 'true_branch', 'True')
                else:
                    self._add_edge(graph, current, stmt_entry)
                current = stmt_exit
            true_exit = current
        
        # False ветка (else)
        false_entry = None
        false_exit = None
        
        if node.orelse:
            current = None
            for stmt in node.orelse:
                stmt_entry, stmt_exit = self._visit_node(stmt, graph)
                if current is None:
                    false_entry = stmt_entry
                    self._add_edge(graph, condition_node, stmt_entry, 'false_branch', 'False')
                else:
                    self._add_edge(graph, current, stmt_entry)
                current = stmt_exit
            false_exit = current
        
        # Создаем узел схождения
        merge_node = self._create_node('merge', node.lineno, 'endif')
        
        if true_exit:
            self._add_edge(graph, true_exit, merge_node)
        else:
            self._add_edge(graph, condition_node, merge_node, 'true_branch', 'True')
        
        if false_exit:
            self._add_edge(graph, false_exit, merge_node)
        else:
            self._add_edge(graph, condition_node, merge_node, 'false_branch', 'False')
        
        return condition_node, merge_node
    
    def _visit_for(self, node: ast.For, graph: nx.DiGraph, 
                  entry_node: CFGNode = None) -> Tuple[CFGNode, CFGNode]:
        """Обработка цикла for"""
        loop_header = self._create_node(
            'loop', 
            node.lineno, 
            f'for {ast.unparse(node.target) if hasattr(ast, "unparse") else "var"} in ...:',
            node
        )
        
        # Тело цикла
        if node.body:
            current = None
            body_entry = None
            
            for stmt in node.body:
                stmt_entry, stmt_exit = self._visit_node(stmt, graph)
                if current is None:
                    body_entry = stmt_entry
                    self._add_edge(graph, loop_header, stmt_entry, 'true_branch', 'continue')
                else:
                    self._add_edge(graph, current, stmt_entry)
                current = stmt_exit
            
            # Обратная связь к заголовку цикла
            self._add_edge(graph, current, loop_header, 'loop_back')
        
        # Выход из цикла
        exit_node = self._create_node('statement', node.lineno, 'end for')
        self._add_edge(graph, loop_header, exit_node, 'false_branch', 'break')
        
        # Обработка else ветки
        if node.orelse:
            else_current = None
            for stmt in node.orelse:
                stmt_entry, stmt_exit = self._visit_node(stmt, graph)
                if else_current is None:
                    self._add_edge(graph, loop_header, stmt_entry, 'else_branch')
                else:
                    self._add_edge(graph, else_current, stmt_entry)
                else_current = stmt_exit
            
            if else_current:
                self._add_edge(graph, else_current, exit_node)
        
        return loop_header, exit_node
    
    def _visit_while(self, node: ast.While, graph: nx.DiGraph, 
                    entry_node: CFGNode = None) -> Tuple[CFGNode, CFGNode]:
        """Обработка цикла while"""
        loop_header = self._create_node(
            'loop', 
            node.lineno, 
            f'while {ast.unparse(node.test) if hasattr(ast, "unparse") else "condition"}:',
            node
        )
        
        # Тело цикла
        if node.body:
            current = None
            
            for stmt in node.body:
                stmt_entry, stmt_exit = self._visit_node(stmt, graph)
                if current is None:
                    self._add_edge(graph, loop_header, stmt_entry, 'true_branch', 'True')
                else:
                    self._add_edge(graph, current, stmt_entry)
                current = stmt_exit
            
            # Обратная связь к заголовку цикла
            self._add_edge(graph, current, loop_header, 'loop_back')
        
        # Выход из цикла
        exit_node = self._create_node('statement', node.lineno, 'end while')
        self._add_edge(graph, loop_header, exit_node, 'false_branch', 'False')
        
        return loop_header, exit_node
    
    def _visit_return(self, node: ast.Return, graph: nx.DiGraph) -> Tuple[CFGNode, CFGNode]:
        """Обработка return"""
        return_node = self._create_node(
            'return', 
            node.lineno, 
            f'return {ast.unparse(node.value) if node.value and hasattr(ast, "unparse") else ""}',
            node
        )
        return return_node, return_node
    
    def _visit_break_continue(self, node: ast.AST, graph: nx.DiGraph) -> Tuple[CFGNode, CFGNode]:
        """Обработка break/continue"""
        stmt_type = 'break' if isinstance(node, ast.Break) else 'continue'
        stmt_node = self._create_node(stmt_type, node.lineno, stmt_type, node)
        return stmt_node, stmt_node
    
    def _visit_statement(self, node: ast.AST, graph: nx.DiGraph) -> Tuple[CFGNode, CFGNode]:
        """Обработка обычного оператора"""
        code = ast.unparse(node) if hasattr(ast, 'unparse') else str(type(node).__name__)
        stmt_node = self._create_node('statement', getattr(node, 'lineno', 1), code, node)
        return stmt_node, stmt_node

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
        
        # M = E - N + 2P
        # где E - количество ребер, N - количество узлов, P - количество связных компонентов
        edges = self.graph.number_of_edges()
        nodes = self.graph.number_of_nodes()
        components = len(list(nx.weakly_connected_components(self.graph)))
        
        return max(1, edges - nodes + 2 * components)
    
    def _calculate_longest_path(self) -> int:
        """Вычисление длины самого длинного пути"""
        if not self.graph or self.graph.number_of_nodes() == 0:
            return 0
        
        try:
            # Для DAG можем использовать топологическую сортировку
            if nx.is_directed_acyclic_graph(self.graph):
                return nx.dag_longest_path_length(self.graph)
            else:
                # Для графа с циклами приближенная оценка
                max_length = 0
                for node in self.graph.nodes():
                    try:
                        lengths = nx.single_source_shortest_path_length(self.graph, node, cutoff=50)
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
                # Для несвязного графа вычисляем среднее по компонентам
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
        """Вычисление глубины вложенности"""
        max_depth = 0
        
        def dfs_depth(node_id: int, current_depth: int, visited: Set[int]):
            nonlocal max_depth
            
            if node_id in visited:
                return
            
            visited.add(node_id)
            max_depth = max(max_depth, current_depth)
            
            node_data = self.graph.nodes[node_id]
            new_depth = current_depth
            
            # Увеличиваем глубину для циклов и условий
            if node_data.get('type') in ['loop', 'condition']:
                new_depth += 1
            
            for successor in self.graph.successors(node_id):
                dfs_depth(successor, new_depth, visited.copy())
        
        # Начинаем с узлов без предшественников
        entry_nodes = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        
        for entry_node in entry_nodes:
            dfs_depth(entry_node, 0, set())
        
        return max_depth
    
    def _calculate_fan_metrics(self) -> Dict[str, float]:
        """Вычисление метрик fan-in/fan-out"""
        if not self.graph.nodes():
            return {'avg_fan_in': 0.0, 'avg_fan_out': 0.0, 'max_fan_in': 0, 'max_fan_out': 0}
        
        fan_ins = [self.graph.in_degree(n) for n in self.graph.nodes()]
        fan_outs = [self.graph.out_degree(n) for n in self.graph.nodes()]
        
        return {
            'avg_fan_in': np.mean(fan_ins) if fan_ins else 0.0,
            'avg_fan_out': np.mean(fan_outs) if fan_outs else 0.0,
            'max_fan_in': max(fan_ins) if fan_ins else 0,
            'max_fan_out': max(fan_outs) if fan_outs else 0
        }
    
    def _count_back_edges(self) -> int:
        """Подсчет обратных ребер (индикатор циклов)"""
        back_edges = 0
        
        try:
            # Находим обратные ребра через DFS
            for edge in self.graph.edges():
                source, target = edge
                
                # Если есть путь от target к source, это может быть обратным ребром
                try:
                    if nx.has_path(self.graph, target, source):
                        back_edges += 1
                except:
                    continue
                    
        except:
            pass
        
        return back_edges
    
    def _count_forward_edges(self) -> int:
        """Подсчет прямых ребер"""
        # Упрощенная реализация
        return max(0, self.graph.number_of_edges() - self._count_back_edges())
    
    def _count_cross_edges(self) -> int:
        """Подсчет перекрестных ребер"""
        # Упрощенная реализация - возвращаем 0
        return 0

class CFGComplexityAnalyzer(BaseComplexityAnalyzer):
    """Анализатор сложности на основе CFG"""
    
    def __init__(self):
        super().__init__("cfg_analyzer", AnalyzerType.CFG)
        self.cfg_builder = CFGBuilder()
        self.cfg_analyzer = CFGAnalyzer()
    
    def is_available(self) -> bool:
        """Проверка доступности"""
        try:
            import networkx
            return True
        except ImportError:
            return False
    
    def analyze(self, context) -> ComplexityResult:
        """Анализ сложности через CFG"""
        try:
            # Построение CFG
            cfg = self.cfg_builder.build_cfg(context.source_code)
            
            # Анализ метрик CFG
            cfg_metrics = self.cfg_analyzer.analyze_cfg(cfg)
            
            # Определение класса сложности
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
                errors=[f"CFG analysis error: {e}"]
            )
    
    def _infer_complexity_from_cfg(self, metrics: Dict[str, Any]) -> ComplexityClass:
        """Определяет класс сложности с возможностью точной нотации."""
        loop_nodes = metrics.get('loop_nodes', 0)
        nested_loop_depth = metrics.get('nested_loop_depth', 0)
        decision_nodes = metrics.get('decision_nodes', 0)
        nodes_count = metrics.get('nodes_count', 0)
        
        # Нет циклов
        if loop_nodes == 0:
            if decision_nodes > nodes_count * 0.3:
                return ComplexityClass.LOGARITHMIC
            return ComplexityClass.CONSTANT
        
        # Один цикл
        if nested_loop_depth == 1:
            if decision_nodes > 3:
                return ComplexityClass.LINEARITHMIC  # O(nlogn)
            return ComplexityClass.LINEAR  # O(n)
        
        # Два вложенных цикла
        if nested_loop_depth == 2:
            return ComplexityClass.QUADRATIC  # O(n^2)
        
        # Три вложенных цикла
        if nested_loop_depth == 3:
            return ComplexityClass.CUBIC  # O(n^3)
        
        # Больше трёх
        if nested_loop_depth >= 4:
            return ComplexityClass.POLYNOMIAL  # O(n^k)
        
        # Несколько последовательных циклов
        if loop_nodes > 1:
            return ComplexityClass.LINEAR  # O(n+m)
        
        return ComplexityClass.CONSTANT


    
    def _calculate_confidence(self, metrics: Dict[str, Any]) -> float:
        """Расчет уверенности в результате"""
        base_confidence = 0.8
        
        # Увеличиваем уверенность при наличии четких структур
        nodes_count = metrics.get('nodes_count', 0)
        if nodes_count > 5:  # Достаточно узлов для анализа
            base_confidence += 0.1
        
        # Уменьшаем при сложных графах
        avg_path_length = metrics.get('average_path_length', 0)
        if avg_path_length > 10:
            base_confidence -= 0.2
        
        return max(0.1, min(base_confidence, 1.0))
