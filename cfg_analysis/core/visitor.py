"""
Визитор для обхода CFG.
"""

from typing import Set, List, Any
from .graph import ControlFlowGraph
from .block import BasicBlock


class CFGVisitor:
    """
    Базовый класс для обхода графа.
    Реализует DFS (обход в глубину).
    """
    
    def __init__(self):
        self.visited: Set[int] = set()

    def visit(self, cfg: ControlFlowGraph):
        """Запуск обхода графа"""
        self.visited.clear()
        if cfg.entry_block:
            self._dfs(cfg, cfg.entry_block)
            
    def _dfs(self, cfg: ControlFlowGraph, block: BasicBlock):
        """Рекурсивный DFS"""
        if block.bid in self.visited:
            return
        
        self.visited.add(block.bid)
        self.visit_block(block)
        
        for succ_id in block.successors:
            succ_block = cfg.get_block(succ_id)
            if succ_block:
                self._dfs(cfg, succ_block)
    
    def visit_block(self, block: BasicBlock):
        """Метод обработки блока (переопределяется наследниками)"""
        pass


class GraphPrinter(CFGVisitor):
    """Простой принтер графа для отладки"""
    
    def __init__(self):
        super().__init__()
        self.output: List[str] = []
        
    def visit_block(self, block: BasicBlock):
        stmts_summary = [type(s).__name__ for s in block.statements]
        line = f"Block {block.bid}: {stmts_summary} -> {list(block.successors)}"
        if block.is_entry:
            line += " [ENTRY]"
        if block.is_exit:
            line += " [EXIT]"
        self.output.append(line)
        
    def get_output(self) -> str:
        return "\n".join(self.output)
