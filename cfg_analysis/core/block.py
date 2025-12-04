"""
Представление базового блока (Basic Block) в CFG.
"""

import ast
from typing import List, Optional, Set


class BasicBlock:
    """
    Базовый блок — это последовательность инструкций без ветвлений внутри.
    Вход в блок только в начале, выход только в конце.
    """
    
    def __init__(self, bid: int):
        self.bid: int = bid  # Уникальный ID блока
        self.statements: List[ast.AST] = []  # Список инструкций
        self.predecessors: Set[int] = set()  # ID блоков, из которых можно попасть сюда
        self.successors: Set[int] = set()    # ID блоков, в которые можно попасть отсюда
        
        # Метаданные
        self.is_entry: bool = False
        self.is_exit: bool = False
        self.loop_depth: int = 0
    
    def add_statement(self, stmt: ast.AST):
        """Добавление инструкции в блок"""
        self.statements.append(stmt)
    
    def is_empty(self) -> bool:
        """Проверка на пустоту"""
        return len(self.statements) == 0
    
    def last_statement(self) -> Optional[ast.AST]:
        """Возвращает последнюю инструкцию блока (обычно условие перехода)"""
        return self.statements[-1] if self.statements else None
    
    def get_source_lines(self) -> List[int]:
        """Получение номеров строк кода в блоке"""
        lines = []
        for stmt in self.statements:
            if hasattr(stmt, 'lineno'):
                lines.append(stmt.lineno)
        return sorted(list(set(lines)))

    def __repr__(self):
        return f"Block({self.bid}, stmts={len(self.statements)})"
