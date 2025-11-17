"""Построитель графа потока управления (CFG) из Python кода"""
import ast
import networkx as nx
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

class NodeType(Enum):
    """Типы узлов CFG"""
    ENTRY = "entry"
    EXIT = "exit"
    STATEMENT = "statement"
    CONDITION = "condition"
    LOOP_HEADER = "loop_header"
    LOOP_BODY = "loop_body"
    EXCEPTION_HANDLER = "exception_handler"
    FUNCTION_CALL = "function_call"
    RETURN = "return"
    BREAK = "break"
    CONTINUE = "continue"
    MERGE = "merge"

class EdgeType(Enum):
    """Типы рёбер CFG"""
    SEQUENTIAL = "sequential"
    TRUE_BRANCH = "true_branch"
    FALSE_BRANCH = "false_branch"
    LOOP_BACK = "loop_back"
    EXCEPTION = "exception"
    CALL = "call"
    RETURN_EDGE = "return"
    BREAK_EDGE = "break"
    CONTINUE_EDGE = "continue"

@dataclass
class CFGNode:
    """Узел графа потока управления"""
    id: int
    node_type: NodeType
    line_number: int
    code: str = ""
    ast_node: Optional[ast.AST] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, CFGNode) and self.id == other.id

@dataclass
class CFGEdge:
    """Ребро графа потока управления"""
    source: CFGNode
    target: CFGNode
    edge_type: EdgeType
    condition: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

class CFGContext:
    """Контекст построения CFG"""
    
    def __init__(self):
        self.break_targets: List[CFGNode] = []
        self.continue_targets: List[CFGNode] = []
        self.exception_handlers: List[CFGNode] = []
        self.function_exits: List[CFGNode] = []
        self.current_function: Optional[str] = None

class PythonCFGBuilder:
    """Построитель CFG для Python кода"""
    
    def __init__(self):
        self.nodes: List[CFGNode] = []
        self.edges: List[CFGEdge] = []
        self.node_counter = 0
        self.graph: Optional[nx.DiGraph] = None
        self.context = CFGContext()
        
    def build_from_source(self, source_code: str) -> nx.DiGraph:
        """Построение CFG из исходного кода"""
        try:
            tree = ast.parse(source_code)
            return self.build_from_ast(tree)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in source code: {e}")
    
    def build_from_ast(self, tree: ast.AST) -> nx.DiGraph:
        """Построение CFG из AST"""
        self._reset()
        self.graph = nx.DiGraph()
        
        # Построение CFG
        entry_node, exit_node = self._visit_node(tree)
        
        # Добавление узлов и рёбер в NetworkX граф
        self._build_networkx_graph()
        
        return self.graph
    
    def _reset(self):
        """Сброс состояния построителя"""
        self.nodes.clear()
        self.edges.clear()
        self.node_counter = 0
        self.graph = None
        self.context = CFGContext()
    
    def _create_node(self, node_type: NodeType, line_number: int, 
                    code: str = "", ast_node: ast.AST = None, 
                    **properties) -> CFGNode:
        """Создание нового узла CFG"""
        node = CFGNode(
            id=self.node_counter,
            node_type=node_type,
            line_number=line_number,
            code=code,
            ast_node=ast_node,
            properties=properties
        )
        self.nodes.append(node)
        self.node_counter += 1
        return node
    
    def _add_edge(self, source: CFGNode, target: CFGNode, 
                 edge_type: EdgeType = EdgeType.SEQUENTIAL, 
                 condition: str = None, **properties):
        """Добавление ребра в CFG"""
        edge = CFGEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            condition=condition,
            properties=properties
        )
        self.edges.append(edge)
    
    def _visit_node(self, node: ast.AST) -> Tuple[CFGNode, CFGNode]:
        """Обход узла AST и построение соответствующей части CFG"""
        if isinstance(node, ast.Module):
            return self._visit_module(node)
        elif isinstance(node, ast.FunctionDef):
            return self._visit_function_def(node)
        elif isinstance(node, ast.AsyncFunctionDef):
            return self._visit_async_function_def(node)
        elif isinstance(node, ast.ClassDef):
            return self._visit_class_def(node)
        elif isinstance(node, ast.If):
            return self._visit_if(node)
        elif isinstance(node, ast.For):
            return self._visit_for(node)
        elif isinstance(node, ast.While):
            return self._visit_while(node)
        elif isinstance(node, ast.Try):
            return self._visit_try(node)
        elif isinstance(node, ast.With):
            return self._visit_with(node)
        elif isinstance(node, ast.Return):
            return self._visit_return(node)
        elif isinstance(node, ast.Break):
            return self._visit_break(node)
        elif isinstance(node, ast.Continue):
            return self._visit_continue(node)
        elif isinstance(node, ast.Raise):
            return self._visit_raise(node)
        elif isinstance(node, ast.Assert):
            return self._visit_assert(node)
        else:
            return self._visit_statement(node)
    
    def _visit_module(self, node: ast.Module) -> Tuple[CFGNode, CFGNode]:
        """Обработка модуля"""
        entry = self._create_node(NodeType.ENTRY, 1, "module entry")
        
        if not node.body:
            exit_node = self._create_node(NodeType.EXIT, 1, "module exit")
            self._add_edge(entry, exit_node)
            return entry, exit_node
        
        # Обработка тела модуля
        current = entry
        for stmt in node.body:
            stmt_entry, stmt_exit = self._visit_node(stmt)
            self._add_edge(current, stmt_entry)
            current = stmt_exit
        
        exit_node = self._create_node(NodeType.EXIT, 
                                     getattr(node.body[-1], 'lineno', 1), 
                                     "module exit")
        self._add_edge(current, exit_node)
        
        return entry, exit_node
    
    def _visit_function_def(self, node: ast.FunctionDef) -> Tuple[CFGNode, CFGNode]:
        """Обработка определения функции"""
        prev_function = self.context.current_function
        self.context.current_function = node.name
        
        # Сохраняем текущие цели для break/continue
        prev_break_targets = self.context.break_targets.copy()
        prev_continue_targets = self.context.continue_targets.copy()
        prev_function_exits = self.context.function_exits.copy()
        
        self.context.break_targets.clear()
        self.context.continue_targets.clear()
        self.context.function_exits.clear()
        
        # Создаём узлы входа и выхода функции
        entry = self._create_node(
            NodeType.ENTRY, 
            node.lineno, 
            f"def {node.name}({self._format_args(node.args)}):",
            node,
            function_name=node.name,
            decorators=[self._ast_to_string(d) for d in node.decorator_list]
        )
        
        exit_node = self._create_node(
            NodeType.EXIT,
            node.lineno,
            f"end def {node.name}",
            node,
            function_name=node.name
        )
        
        self.context.function_exits.append(exit_node)
        
        if not node.body:
            self._add_edge(entry, exit_node)
        else:
            # Обработка тела функции
            current = entry
            for stmt in node.body:
                stmt_entry, stmt_exit = self._visit_node(stmt)
                self._add_edge(current, stmt_entry)
                current = stmt_exit
            
            # Если последний узел не return, добавляем неявный return
            if current.node_type != NodeType.RETURN:
                implicit_return = self._create_node(
                    NodeType.RETURN,
                    getattr(node.body[-1], 'lineno', node.lineno),
                    "return None",
                    properties={'implicit': True}
                )
                self._add_edge(current, implicit_return)
                self._add_edge(implicit_return, exit_node, EdgeType.RETURN_EDGE)
            else:
                self._add_edge(current, exit_node, EdgeType.RETURN_EDGE)
        
        # Восстанавливаем контекст
        self.context.current_function = prev_function
        self.context.break_targets = prev_break_targets
        self.context.continue_targets = prev_continue_targets
        self.context.function_exits = prev_function_exits
        
        return entry, exit_node
    
    def _visit_async_function_def(self, node: ast.AsyncFunctionDef) -> Tuple[CFGNode, CFGNode]:
        """Обработка асинхронной функции"""
        # Аналогично обычной функции, но с пометкой async
        prev_function = self.context.current_function
        self.context.current_function = node.name
        
        entry = self._create_node(
            NodeType.ENTRY,
            node.lineno,
            f"async def {node.name}({self._format_args(node.args)}):",
            node,
            function_name=node.name,
            is_async=True
        )
        
        exit_node = self._create_node(
            NodeType.EXIT,
            node.lineno,
            f"end async def {node.name}",
            node,
            function_name=node.name,
            is_async=True
        )
        
        if node.body:
            current = entry
            for stmt in node.body:
                stmt_entry, stmt_exit = self._visit_node(stmt)
                self._add_edge(current, stmt_entry)
                current = stmt_exit
            self._add_edge(current, exit_node)
        else:
            self._add_edge(entry, exit_node)
        
        self.context.current_function = prev_function
        return entry, exit_node
    
    def _visit_class_def(self, node: ast.ClassDef) -> Tuple[CFGNode, CFGNode]:
        """Обработка определения класса"""
        entry = self._create_node(
            NodeType.STATEMENT,
            node.lineno,
            f"class {node.name}({', '.join(self._ast_to_string(base) for base in node.bases)}):",
            node,
            class_name=node.name
        )
        
        if not node.body:
            return entry, entry
        
        current = entry
        for stmt in node.body:
            stmt_entry, stmt_exit = self._visit_node(stmt)
            self._add_edge(current, stmt_entry)
            current = stmt_exit
        
        return entry, current
    
    def _visit_if(self, node: ast.If) -> Tuple[CFGNode, CFGNode]:
        """Обработка условного оператора"""
        # Создаём узел условия
        condition_node = self._create_node(
            NodeType.CONDITION,
            node.lineno,
            f"if {self._ast_to_string(node.test)}:",
            node,
            condition=self._ast_to_string(node.test)
        )
        
        # Узел слияния
        merge_node = self._create_node(
            NodeType.MERGE,
            node.lineno,
            "end if",
            properties={'merge_type': 'if'}
        )
        
        # Обработка true-ветки
        if node.body:
            true_current = None
            for stmt in node.body:
                stmt_entry, stmt_exit = self._visit_node(stmt)
                if true_current is None:
                    # Первый узел в true-ветке
                    self._add_edge(condition_node, stmt_entry, EdgeType.TRUE_BRANCH, "True")
                    true_current = stmt_exit
                else:
                    self._add_edge(true_current, stmt_entry)
                    true_current = stmt_exit
            
            if true_current:
                self._add_edge(true_current, merge_node)
        else:
            # Пустая true-ветка
            self._add_edge(condition_node, merge_node, EdgeType.TRUE_BRANCH, "True")
        
        # Обработка false-ветки (else/elif)
        if node.orelse:
            false_current = None
            for stmt in node.orelse:
                stmt_entry, stmt_exit = self._visit_node(stmt)
                if false_current is None:
                    # Первый узел в false-ветке
                    self._add_edge(condition_node, stmt_entry, EdgeType.FALSE_BRANCH, "False")
                    false_current = stmt_exit
                else:
                    self._add_edge(false_current, stmt_entry)
                    false_current = stmt_exit
            
            if false_current:
                self._add_edge(false_current, merge_node)
        else:
            # Нет else-ветки
            self._add_edge(condition_node, merge_node, EdgeType.FALSE_BRANCH, "False")
        
        return condition_node, merge_node
    
    def _visit_for(self, node: ast.For) -> Tuple[CFGNode, CFGNode]:
        """Обработка цикла for"""
        # Заголовок цикла
        loop_header = self._create_node(
            NodeType.LOOP_HEADER,
            node.lineno,
            f"for {self._ast_to_string(node.target)} in {self._ast_to_string(node.iter)}:",
            node,
            loop_type='for',
            target=self._ast_to_string(node.target),
            iterable=self._ast_to_string(node.iter)
        )
        
        # Узел выхода из цикла
        loop_exit = self._create_node(
            NodeType.MERGE,
            node.lineno,
            "end for",
            properties={'merge_type': 'for_exit'}
        )
        
        # Сохраняем текущие цели для break/continue
        prev_break_targets = self.context.break_targets.copy()
        prev_continue_targets = self.context.continue_targets.copy()
        
        self.context.break_targets.append(loop_exit)
        self.context.continue_targets.append(loop_header)
        
        # Обработка тела цикла
        if node.body:
            body_current = None
            for stmt in node.body:
                stmt_entry, stmt_exit = self._visit_node(stmt)
                if body_current is None:
                    # Первый узел в теле цикла
                    self._add_edge(loop_header, stmt_entry, EdgeType.TRUE_BRANCH, "continue")
                    body_current = stmt_exit
                else:
                    self._add_edge(body_current, stmt_entry)
                    body_current = stmt_exit
            
            # Обратная связь к заголовку цикла
            if body_current:
                self._add_edge(body_current, loop_header, EdgeType.LOOP_BACK)
        
        # Выход из цикла (когда итерация завершена)
        self._add_edge(loop_header, loop_exit, EdgeType.FALSE_BRANCH, "break")
        
        # Обработка else-ветки
        if node.orelse:
            else_current = None
            for stmt in node.orelse:
                stmt_entry, stmt_exit = self._visit_node(stmt)
                if else_current is None:
                    # else выполняется только при нормальном завершении цикла
                    self._add_edge(loop_exit, stmt_entry)
                    else_current = stmt_exit
                else:
                    self._add_edge(else_current, stmt_entry)
                    else_current = stmt_exit
            
            if else_current:
                final_exit = self._create_node(NodeType.MERGE, node.lineno, "end for-else")
                self._add_edge(else_current, final_exit)
                loop_exit = final_exit
        
        # Восстанавливаем контекст
        self.context.break_targets = prev_break_targets
        self.context.continue_targets = prev_continue_targets
        
        return loop_header, loop_exit
    
    def _visit_while(self, node: ast.While) -> Tuple[CFGNode, CFGNode]:
        """Обработка цикла while"""
        # Заголовок цикла с условием
        loop_header = self._create_node(
            NodeType.LOOP_HEADER,
            node.lineno,
            f"while {self._ast_to_string(node.test)}:",
            node,
            loop_type='while',
            condition=self._ast_to_string(node.test)
        )
        
        # Узел выхода из цикла
        loop_exit = self._create_node(
            NodeType.MERGE,
            node.lineno,
            "end while",
            properties={'merge_type': 'while_exit'}
        )
        
        # Сохраняем контекст
        prev_break_targets = self.context.break_targets.copy()
        prev_continue_targets = self.context.continue_targets.copy()
        
        self.context.break_targets.append(loop_exit)
        self.context.continue_targets.append(loop_header)
        
        # Обработка тела цикла
        if node.body:
            body_current = None
            for stmt in node.body:
                stmt_entry, stmt_exit = self._visit_node(stmt)
                if body_current is None:
                    self._add_edge(loop_header, stmt_entry, EdgeType.TRUE_BRANCH, "True")
                    body_current = stmt_exit
                else:
                    self._add_edge(body_current, stmt_entry)
                    body_current = stmt_exit
            
            # Обратная связь
            if body_current:
                self._add_edge(body_current, loop_header, EdgeType.LOOP_BACK)
        
        # Выход из цикла
        self._add_edge(loop_header, loop_exit, EdgeType.FALSE_BRANCH, "False")
        
        # Обработка else-ветки
        if node.orelse:
            else_current = None
            for stmt in node.orelse:
                stmt_entry, stmt_exit = self._visit_node(stmt)
                if else_current is None:
                    self._add_edge(loop_exit, stmt_entry)
                    else_current = stmt_exit
                else:
                    self._add_edge(else_current, stmt_entry)
                    else_current = stmt_exit
            
            if else_current:
                final_exit = self._create_node(NodeType.MERGE, node.lineno, "end while-else")
                self._add_edge(else_current, final_exit)
                loop_exit = final_exit
        
        # Восстанавливаем контекст
        self.context.break_targets = prev_break_targets
        self.context.continue_targets = prev_continue_targets
        
        return loop_header, loop_exit
    
    def _visit_try(self, node: ast.Try) -> Tuple[CFGNode, CFGNode]:
        """Обработка try-except блока"""
        # Начало try-блока
        try_entry = self._create_node(
            NodeType.STATEMENT,
            node.lineno,
            "try:",
            node
        )
        
        # Узел слияния для всех веток
        merge_node = self._create_node(
            NodeType.MERGE,
            node.lineno,
            "end try",
            properties={'merge_type': 'try_except'}
        )
        
        # Обработка тела try
        try_current = try_entry
        for stmt in node.body:
            stmt_entry, stmt_exit = self._visit_node(stmt)
            self._add_edge(try_current, stmt_entry)
            try_current = stmt_exit
        
        # Обработка except-блоков
        for handler in node.handlers:
            handler_entry = self._create_node(
                NodeType.EXCEPTION_HANDLER,
                handler.lineno,
                f"except {self._format_exception_handler(handler)}:",
                handler,
                exception_type=self._ast_to_string(handler.type) if handler.type else None,
                exception_name=handler.name
            )
            
            # Связываем каждый узел в try с обработчиком исключений
            # (упрощённая модель - в реальности исключение может возникнуть в любом месте)
            self._add_edge(try_entry, handler_entry, EdgeType.EXCEPTION)
            
            handler_current = handler_entry
            for stmt in handler.body:
                stmt_entry, stmt_exit = self._visit_node(stmt)
                self._add_edge(handler_current, stmt_entry)
                handler_current = stmt_exit
            
            self._add_edge(handler_current, merge_node)
        
        # Нормальное завершение try-блока
        self._add_edge(try_current, merge_node)
        
        # Обработка else-блока
        if node.orelse:
            else_current = merge_node
            for stmt in node.orelse:
                stmt_entry, stmt_exit = self._visit_node(stmt)
                self._add_edge(else_current, stmt_entry)
                else_current = stmt_exit
            merge_node = else_current
        
        # Обработка finally-блока
        if node.finalbody:
            finally_entry = self._create_node(
                NodeType.STATEMENT,
                node.finalbody[0].lineno,
                "finally:",
                properties={'block_type': 'finally'}
            )
            
            self._add_edge(merge_node, finally_entry)
            
            finally_current = finally_entry
            for stmt in node.finalbody:
                stmt_entry, stmt_exit = self._visit_node(stmt)
                self._add_edge(finally_current, stmt_entry)
                finally_current = stmt_exit
            
            merge_node = finally_current
        
        return try_entry, merge_node
    
    def _visit_with(self, node: ast.With) -> Tuple[CFGNode, CFGNode]:
        """Обработка with-блока"""
        with_entry = self._create_node(
            NodeType.STATEMENT,
            node.lineno,
            f"with {', '.join(self._format_with_item(item) for item in node.items)}:",
            node,
            context_managers=[self._ast_to_string(item.context_expr) for item in node.items]
        )
        
        if not node.body:
            return with_entry, with_entry
        
        current = with_entry
        for stmt in node.body:
            stmt_entry, stmt_exit = self._visit_node(stmt)
            self._add_edge(current, stmt_entry)
            current = stmt_exit
        
        # Создаём узел выхода из with
        with_exit = self._create_node(
            NodeType.STATEMENT,
            current.line_number,
            "end with",
            properties={'block_type': 'with_exit'}
        )
        self._add_edge(current, with_exit)
        
        return with_entry, with_exit
    
    def _visit_return(self, node: ast.Return) -> Tuple[CFGNode, CFGNode]:
        """Обработка return"""
        return_node = self._create_node(
            NodeType.RETURN,
            node.lineno,
            f"return {self._ast_to_string(node.value) if node.value else ''}",
            node,
            return_value=self._ast_to_string(node.value) if node.value else None
        )
        
        # Связываем с выходом функции
        for exit_node in self.context.function_exits:
            self._add_edge(return_node, exit_node, EdgeType.RETURN_EDGE)
        
        return return_node, return_node
    
    def _visit_break(self, node: ast.Break) -> Tuple[CFGNode, CFGNode]:
        """Обработка break"""
        break_node = self._create_node(
            NodeType.BREAK,
            node.lineno,
            "break",
            node
        )
        
        # Связываем с ближайшим выходом из цикла
        if self.context.break_targets:
            target = self.context.break_targets[-1]
            self._add_edge(break_node, target, EdgeType.BREAK_EDGE)
        
        return break_node, break_node
    
    def _visit_continue(self, node: ast.Continue) -> Tuple[CFGNode, CFGNode]:
        """Обработка continue"""
        continue_node = self._create_node(
            NodeType.CONTINUE,
            node.lineno,
            "continue",
            node
        )
        
        # Связываем с ближайшим заголовком цикла
        if self.context.continue_targets:
            target = self.context.continue_targets[-1]
            self._add_edge(continue_node, target, EdgeType.CONTINUE_EDGE)
        
        return continue_node, continue_node
    
    def _visit_raise(self, node: ast.Raise) -> Tuple[CFGNode, CFGNode]:
        """Обработка raise"""
        raise_node = self._create_node(
            NodeType.STATEMENT,
            node.lineno,
            f"raise {self._ast_to_string(node.exc) if node.exc else ''}",
            node,
            statement_type='raise',
            exception=self._ast_to_string(node.exc) if node.exc else None
        )
        
        return raise_node, raise_node
    
    def _visit_assert(self, node: ast.Assert) -> Tuple[CFGNode, CFGNode]:
        """Обработка assert"""
        assert_node = self._create_node(
            NodeType.CONDITION,
            node.lineno,
            f"assert {self._ast_to_string(node.test)}",
            node,
            condition=self._ast_to_string(node.test),
            statement_type='assert'
        )
        
        return assert_node, assert_node
    
    def _visit_statement(self, node: ast.AST) -> Tuple[CFGNode, CFGNode]:
        """Обработка обычного оператора"""
        stmt_node = self._create_node(
            NodeType.STATEMENT,
            getattr(node, 'lineno', 1),
            self._ast_to_string(node),
            node,
            statement_type=type(node).__name__
        )
        
        return stmt_node, stmt_node
    
    def _build_networkx_graph(self):
        """Построение NetworkX графа из узлов и рёбер"""
        # Добавление узлов
        for node in self.nodes:
            self.graph.add_node(node.id, **{
                'type': node.node_type.value,
                'line': node.line_number,
                'code': node.code,
                'ast_type': type(node.ast_node).__name__ if node.ast_node else None,
                **node.properties
            })
        
        # Добавление рёбер
        for edge in self.edges:
            self.graph.add_edge(edge.source.id, edge.target.id, **{
                'type': edge.edge_type.value,
                'condition': edge.condition,
                **edge.properties
            })
    
    def _ast_to_string(self, node: Optional[ast.AST]) -> str:
        """Преобразование AST узла в строку"""
        if node is None:
            return ""
        
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(node)
            else:
                # Fallback для старых версий Python
                return self._simple_ast_to_string(node)
        except:
            return type(node).__name__
    
    def _simple_ast_to_string(self, node: ast.AST) -> str:
        """Упрощённое преобразование AST в строку"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, (ast.Num, ast.Str)):  # Для совместимости со старыми версиями
            return repr(node.n if hasattr(node, 'n') else node.s)
        elif isinstance(node, ast.BinOp):
            left = self._simple_ast_to_string(node.left)
            right = self._simple_ast_to_string(node.right)
            op = self._op_to_string(node.op)
            return f"{left} {op} {right}"
        elif isinstance(node, ast.Compare):
            left = self._simple_ast_to_string(node.left)
            parts = [left]
            for op, comp in zip(node.ops, node.comparators):
                op_str = self._op_to_string(op)
                comp_str = self._simple_ast_to_string(comp)
                parts.extend([op_str, comp_str])
            return " ".join(parts)
        else:
            return type(node).__name__
    
    def _op_to_string(self, op: ast.AST) -> str:
        """Преобразование оператора в строку"""
        op_map = {
            ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/',
            ast.Mod: '%', ast.Pow: '**', ast.LShift: '<<', ast.RShift: '>>',
            ast.BitOr: '|', ast.BitXor: '^', ast.BitAnd: '&',
            ast.FloorDiv: '//', ast.Eq: '==', ast.NotEq: '!=',
            ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=',
            ast.Is: 'is', ast.IsNot: 'is not', ast.In: 'in', ast.NotIn: 'not in',
            ast.And: 'and', ast.Or: 'or', ast.Not: 'not'
        }
        return op_map.get(type(op), str(type(op).__name__))
    
    def _format_args(self, args: ast.arguments) -> str:
        """Форматирование аргументов функции"""
        arg_parts = []
        
        # Обычные аргументы
        for arg in args.args:
            arg_parts.append(arg.arg)
        
        # Аргументы с defaults
        defaults_start = len(args.args) - len(args.defaults)
        for i, default in enumerate(args.defaults):
            idx = defaults_start + i
            if idx < len(arg_parts):
                arg_parts[idx] += f"={self._ast_to_string(default)}"
        
        # *args
        if args.vararg:
            arg_parts.append(f"*{args.vararg.arg}")
        
        # **kwargs
        if args.kwarg:
            arg_parts.append(f"**{args.kwarg.arg}")
        
        return ", ".join(arg_parts)
    
    def _format_exception_handler(self, handler: ast.ExceptHandler) -> str:
        """Форматирование обработчика исключений"""
        parts = []
        if handler.type:
            parts.append(self._ast_to_string(handler.type))
        if handler.name:
            parts.append(f"as {handler.name}")
        return " ".join(parts) if parts else ""
    
    def _format_with_item(self, item: ast.withitem) -> str:
        """Форматирование элемента with"""
        result = self._ast_to_string(item.context_expr)
        if item.optional_vars:
            result += f" as {self._ast_to_string(item.optional_vars)}"
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики построенного CFG"""
        if not self.graph:
            return {}
        
        node_types = {}
        edge_types = {}
        
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            node_type = node_data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        for source, target in self.graph.edges():
            edge_data = self.graph.edges[source, target]
            edge_type = edge_data.get('type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': node_types,
            'edge_types': edge_types,
            'strongly_connected_components': len(list(nx.strongly_connected_components(self.graph))),
            'weakly_connected_components': len(list(nx.weakly_connected_components(self.graph))),
            'is_dag': nx.is_directed_acyclic_graph(self.graph)
        }
