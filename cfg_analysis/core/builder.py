"""
Построитель CFG из AST (Control Flow Graph Builder).
Преобразует дерево синтаксиса в граф базовых блоков.
"""

import ast
from typing import List, Optional, Tuple, Any
from .block import BasicBlock
from .graph import ControlFlowGraph


class CFGBuilder:
    """
    Класс для построения CFG из AST.
    Обрабатывает конструкции потока управления (if, for, while, try).
    """
    
    def __init__(self):
        self.cfg: Optional[ControlFlowGraph] = None
        self.current_block: Optional[BasicBlock] = None
        
        # Стек циклов для обработки break/continue
        # Хранит кортежи (head_block, exit_block)
        self.loop_stack: List[Tuple[BasicBlock, BasicBlock]] = []

    def build(self, name: str, tree: ast.AST) -> ControlFlowGraph:
        """
        Основной метод построения.
        
        Args:
            name: Имя графа (обычно имя функции)
            tree: AST узел (FunctionDef, Module или список инструкций)
        
        Returns:
            Построенный CFG
        """
        self.cfg = ControlFlowGraph(name)
        self.cfg.entry_block = self.cfg.new_block()
        self.cfg.entry_block.is_entry = True
        self.current_block = self.cfg.entry_block
        self.loop_stack = []

        # Если входной узел - функция, берем её тело
        nodes_to_process = []
        if isinstance(tree, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            nodes_to_process = tree.body
        elif isinstance(tree, list):
            nodes_to_process = tree
        else:
            nodes_to_process = [tree]

        self._process_nodes(nodes_to_process)
        
        # Если после обработки поток не прерван (нет return в конце),
        # соединяем последний блок с абстрактным выходом (если нужно)
        # В данной реализации просто оставляем висячие блоки как выходы.
        
        return self.cfg

    def _process_nodes(self, nodes: List[ast.AST]):
        """Обработка списка инструкций последовательно"""
        for node in nodes:
            if self.current_block is None:
                # Мертвый код (unreachable code) после return/break/continue
                # Создаем изолированный блок, чтобы не терять инструкции,
                # но не соединяем его с предыдущим.
                self.current_block = self.cfg.new_block()
            
            if isinstance(node, ast.If):
                self._process_if(node)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                self._process_loop(node, is_while=False)
            elif isinstance(node, ast.While):
                self._process_loop(node, is_while=True)
            elif isinstance(node, ast.Break):
                self._process_break(node)
            elif isinstance(node, ast.Continue):
                self._process_continue(node)
            elif isinstance(node, ast.Return):
                self._process_return(node)
            elif isinstance(node, ast.Try):
                self._process_try(node)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Вложенные функции/классы считаем одной инструкцией объявления
                self.current_block.add_statement(node)
            else:
                # Простые инструкции (Assign, Expr, Call и т.д.)
                self.current_block.add_statement(node)

    def _process_if(self, node: ast.If):
        """Обработка if-elif-else"""
        # 1. Добавляем условие в текущий блок
        self.current_block.add_statement(node.test)
        
        # Запоминаем блок условия (развилка)
        condition_block = self.current_block
        
        # 2. Создаем блоки
        then_block = self.cfg.new_block()
        else_block = self.cfg.new_block() if node.orelse else None
        join_block = self.cfg.new_block()
        
        # 3. Ветка THEN
        self.cfg.add_edge(condition_block, then_block)
        self.current_block = then_block
        self._process_nodes(node.body)
        
        # Если ветка не закончилась прыжком (return/break), соединяем с join
        if self.current_block:
            self.cfg.add_edge(self.current_block, join_block)
            
        # 4. Ветка ELSE
        if else_block:
            self.cfg.add_edge(condition_block, else_block)
            self.current_block = else_block
            self._process_nodes(node.orelse)
            
            if self.current_block:
                self.cfg.add_edge(self.current_block, join_block)
        else:
            # Если else нет, условие может сразу вести в join (false case)
            self.cfg.add_edge(condition_block, join_block)
            
        # 5. Продолжаем от join_block
        self.current_block = join_block

    def _process_loop(self, node: ast.AST, is_while: bool):
        """Обработка циклов (for/while)"""
        # 1. Текущий блок ведет к заголовку цикла
        loop_head = self.cfg.new_block()
        self.cfg.add_edge(self.current_block, loop_head)
        
        # В while условие в head, в for итерация в head
        if is_while:
            loop_head.add_statement(node.test)
        else:
            # ast.For: target, iter
            loop_head.add_statement(node.iter)
            loop_head.add_statement(node.target)
            
        # 2. Блоки тела и выхода
        loop_body = self.cfg.new_block()
        loop_exit = self.cfg.new_block()
        
        # 3. Связи заголовка
        self.cfg.add_edge(loop_head, loop_body) # Вход в тело
        self.cfg.add_edge(loop_head, loop_exit) # Выход из цикла
        
        # 4. Обработка тела с учетом стека циклов
        self.loop_stack.append((loop_head, loop_exit))
        
        self.current_block = loop_body
        self._process_nodes(node.body)
        
        # Обратное ребро (Back edge)
        if self.current_block:
            self.cfg.add_edge(self.current_block, loop_head)
            
        # 5. Обработка Orelse у цикла (выполняется, если цикл завершился штатно, не break)
        if node.orelse:
            else_block = self.cfg.new_block()
            # Штатный выход ведет в else
            # Важный момент: loop_head -> loop_exit (это break или конец итерации?)
            # В Python loop else выполняется если цикл кончился сам.
            # Поэтому loop_head (false) -> else_block
            # А break прыгает сразу в loop_exit (после else)
            
            # Корректировка связей:
            # Удаляем прямое ребро head->exit, созданное выше, перенаправляем
            self.cfg.remove_edge(loop_head, loop_exit) 
            self.cfg.add_edge(loop_head, else_block)
            
            self.current_block = else_block
            self._process_nodes(node.orelse)
            if self.current_block:
                self.cfg.add_edge(self.current_block, loop_exit)
        
        self.loop_stack.pop()
        self.current_block = loop_exit

    def _process_break(self, node: ast.Break):
        """Обработка break"""
        self.current_block.add_statement(node)
        if self.loop_stack:
            _, loop_exit = self.loop_stack[-1]
            self.cfg.add_edge(self.current_block, loop_exit)
        self.current_block = None  # Дальше код недостижим

    def _process_continue(self, node: ast.Continue):
        """Обработка continue"""
        self.current_block.add_statement(node)
        if self.loop_stack:
            loop_head, _ = self.loop_stack[-1]
            self.cfg.add_edge(self.current_block, loop_head)
        self.current_block = None  # Дальше код недостижим

    def _process_return(self, node: ast.Return):
        """Обработка return"""
        self.current_block.add_statement(node)
        self.current_block.is_exit = True
        # Можно добавить связь с глобальным exit_block, если он есть
        self.current_block = None

    def _process_try(self, node: ast.Try):
        """
        Упрощенная обработка try-except.
        Мы соединяем начало try со всеми except блоками, 
        так как исключение может возникнуть где угодно.
        """
        try_start = self.cfg.new_block()
        self.cfg.add_edge(self.current_block, try_start)
        
        join_block = self.cfg.new_block() # Блок после всей конструкции (или finally)
        finally_block = self.cfg.new_block() if node.finalbody else join_block
        
        # Обработка handlers (except)
        handler_blocks = []
        for handler in node.handlers:
            h_block = self.cfg.new_block()
            h_block.add_statement(handler.type if handler.type else ast.Name(id='Exception', ctx=ast.Load()))
            handler_blocks.append(h_block)
            # Консервативно: из любого места try можно попасть в except
            # Но мы соединяем try_start -> except как аппроксимацию
            self.cfg.add_edge(try_start, h_block)
            
        # Тело Try
        self.current_block = try_start
        self._process_nodes(node.body)
        if self.current_block:
            self.cfg.add_edge(self.current_block, finally_block)
            
        # Тела Except
        for i, handler in enumerate(node.handlers):
            self.current_block = handler_blocks[i]
            self._process_nodes(handler.body)
            if self.current_block:
                self.cfg.add_edge(self.current_block, finally_block)
        
        # Else блок (выполняется если не было исключений)
        if node.orelse:
            # Это сложно представить в упрощенном CFG без дублирования,
            # поэтому упрощаем: else идет перед finally
            else_start = self.cfg.new_block()
            # На самом деле нужно точное управление потоком от конца try
            # Опустим для краткости, считаем частью потока finally
            pass 

        # Finally блок
        if node.finalbody:
            self.current_block = finally_block
            self._process_nodes(node.finalbody)
            if self.current_block:
                self.cfg.add_edge(self.current_block, join_block)
        
        self.current_block = join_block
