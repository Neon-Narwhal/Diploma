"""
Анализ циклов в CFG на основе доминаторов.
"""

from typing import Dict, Set, List, Tuple, Optional, Any
from cfg_analysis.core.graph import ControlFlowGraph
from cfg_analysis.core.block import BasicBlock


class LoopAnalyzer:
    """
    Анализатор циклов с использованием Dominator Tree.
    """
    
    def __init__(self, cfg: ControlFlowGraph):
        self.cfg = cfg
        self.dominators: Dict[int, Set[int]] = {}
        self.back_edges: List[Tuple[int, int]] = []
        self.loops: List[Dict[str, Any]] = []
        
    def analyze(self) -> List[Dict[str, Any]]:
        """
        Полный анализ циклов.
        Returns: Список найденных циклов с метаданными.
        """
        if not self.cfg.entry_block:
            return []
            
        # 1. Строим дерево доминаторов
        self._compute_dominators()
        
        # 2. Ищем обратные ребра (Back edges)
        self._find_back_edges()
        
        # 3. Собираем натуральные циклы
        self.loops = []
        for source, target in self.back_edges:
            loop_nodes = self._get_natural_loop(source, target)
            loop_info = {
                'header': target,
                'back_edge_source': source,
                'nodes': list(loop_nodes),
                'size': len(loop_nodes),
                'depth': 0  # Будет вычислено позже
            }
            self.loops.append(loop_info)
            
        # 4. Вычисляем глубину вложенности
        self._compute_nesting_depth()
        
        return self.loops
    
    def _compute_dominators(self):
        """Алгоритм вычисления доминаторов (Iterative Data Flow Analysis)"""
        all_nodes = set(self.cfg.blocks.keys())
        
        # Init: Dom(n0) = {n0}, Dom(n) = All Nodes
        entry_id = self.cfg.entry_block.bid
        self.dominators = {n: all_nodes.copy() for n in all_nodes}
        self.dominators[entry_id] = {entry_id}
        
        changed = True
        while changed:
            changed = False
            for bid, block in self.cfg.blocks.items():
                if bid == entry_id:
                    continue
                
                # Dom(n) = {n} U (Intersection of Dom(p) for all predecessors p)
                preds = list(block.predecessors)
                if not preds:
                    continue
                    
                new_dom = self.dominators[preds[0]].copy()
                for p in preds[1:]:
                    if p in self.dominators:
                        new_dom &= self.dominators[p]
                
                new_dom.add(bid)
                
                if new_dom != self.dominators[bid]:
                    self.dominators[bid] = new_dom
                    changed = True

    def _find_back_edges(self):
        """
        Обратное ребро N -> D существует, если D доминирует над N.
        """
        self.back_edges = []
        for source_id, block in self.cfg.blocks.items():
            for target_id in block.successors:
                # Если target доминирует над source -> это обратное ребро (цикл)
                if target_id in self.dominators.get(source_id, set()):
                    self.back_edges.append((source_id, target_id))

    def _get_natural_loop(self, source: int, header: int) -> Set[int]:
        """
        Сбор всех узлов натурального цикла для обратного ребра source -> header.
        """
        loop_nodes = {header, source}
        if source == header:
            return loop_nodes
            
        stack = [source]
        while stack:
            node = stack.pop()
            block = self.cfg.get_block(node)
            if not block: continue
            
            for pred in block.predecessors:
                if pred not in loop_nodes:
                    loop_nodes.add(pred)
                    stack.append(pred)
        return loop_nodes

    def _compute_nesting_depth(self):
        """
        Определение глубины вложенности циклов.
        Цикл A вложен в B, если все узлы A содержатся в B.
        """
        for loop in self.loops:
            depth = 1
            nodes_a = set(loop['nodes'])
            
            for other_loop in self.loops:
                if loop == other_loop:
                    continue
                
                nodes_b = set(other_loop['nodes'])
                # Если A подмножество B, значит A внутри B
                if nodes_a.issubset(nodes_b):
                    depth += 1
            
            loop['depth'] = depth
            
            # Обновляем глубину в самих блоках для удобства
            for nid in nodes_a:
                blk = self.cfg.get_block(nid)
                if blk:
                    blk.loop_depth = max(blk.loop_depth, depth)
