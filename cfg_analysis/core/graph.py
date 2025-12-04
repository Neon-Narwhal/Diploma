"""
Представление графа потока управления (CFG).
"""

from typing import Dict, List, Optional, Iterator, Tuple
from .block import BasicBlock


class ControlFlowGraph:
    """
    Граф потока управления.
    Хранит блоки и управляет связями между ними.
    """
    
    def __init__(self, name: str = "cfg"):
        self.name = name
        self.blocks: Dict[int, BasicBlock] = {}
        self.entry_block: Optional[BasicBlock] = None
        self.exit_blocks: List[BasicBlock] = []
        self._next_bid: int = 0
        
    def new_block(self) -> BasicBlock:
        """Создание нового пустого блока"""
        block = BasicBlock(self._next_bid)
        self.blocks[self._next_bid] = block
        self._next_bid += 1
        return block
    
    def add_edge(self, source: BasicBlock, target: BasicBlock):
        """Добавление направленного ребра между блоками"""
        if source.bid not in self.blocks or target.bid not in self.blocks:
            raise ValueError("Blocks must belong to the graph")
            
        source.successors.add(target.bid)
        target.predecessors.add(source.bid)
        
    def remove_edge(self, source: BasicBlock, target: BasicBlock):
        """Удаление ребра"""
        if target.bid in source.successors:
            source.successors.remove(target.bid)
        if source.bid in target.predecessors:
            target.predecessors.remove(source.bid)
            
    def get_block(self, bid: int) -> Optional[BasicBlock]:
        """Получение блока по ID"""
        return self.blocks.get(bid)
        
    def get_edges(self) -> Iterator[Tuple[int, int]]:
        """Итератор по всем ребрам (source_id, target_id)"""
        for source_id, block in self.blocks.items():
            for target_id in block.successors:
                yield (source_id, target_id)
                
    def get_cyclomatic_complexity(self) -> int:
        """
        Вычисление цикломатической сложности МакКейба.
        M = E - N + 2P
        Где:
        E = количество ребер
        N = количество узлов
        P = количество компонент связности (обычно 1 для функции)
        """
        num_edges = sum(len(b.successors) for b in self.blocks.values())
        num_nodes = len(self.blocks)
        # Для одной функции P=1. Формула M = E - N + 2
        if num_nodes == 0:
            return 0
        return max(1, num_edges - num_nodes + 2)
    
    def to_adjacency_matrix(self) -> List[List[int]]:
        """Экспорт в матрицу смежности (для отладки)"""
        size = len(self.blocks)
        matrix = [[0] * size for _ in range(size)]
        
        # Маппинг bid -> индекс массива (так как bid могут быть не последовательны при удалении)
        ids = sorted(self.blocks.keys())
        id_map = {bid: i for i, bid in enumerate(ids)}
        
        for u, v in self.get_edges():
            i, j = id_map[u], id_map[v]
            matrix[i][j] = 1
            
        return matrix

    def __str__(self):
        return f"CFG(nodes={len(self.blocks)}, edges={sum(len(b.successors) for b in self.blocks.values())})"
