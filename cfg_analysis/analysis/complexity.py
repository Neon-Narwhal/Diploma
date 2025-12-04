"""
Вычисление метрик сложности CFG.
"""

from typing import Dict, Any
from cfg_analysis.core.graph import ControlFlowGraph
from .loops import LoopAnalyzer


class CFGComplexityMetrics:
    """Калькулятор метрик сложности"""
    
    def __init__(self, cfg: ControlFlowGraph):
        self.cfg = cfg
        self.loop_analyzer = LoopAnalyzer(cfg)
        
    def compute(self) -> Dict[str, Any]:
        """Вычисление всех метрик"""
        
        # 1. Базовые метрики графа
        num_blocks = len(self.cfg.blocks)
        num_edges = sum(len(b.successors) for b in self.cfg.blocks.values())
        
        # 2. McCabe Cyclomatic Complexity
        cyclomatic = self.cfg.get_cyclomatic_complexity()
        
        # 3. Анализ циклов
        loops = self.loop_analyzer.analyze()
        num_loops = len(loops)
        max_loop_depth = max((l['depth'] for l in loops), default=0)
        
        # 4. Дополнительные структурные метрики
        # Средняя степень ветвления (average branching factor)
        avg_degree = (num_edges / num_blocks) if num_blocks > 0 else 0
        
        # Количество выходов (return statements)
        num_exits = len([b for b in self.cfg.blocks.values() if b.is_exit or not b.successors])
        
        return {
            'cfg_blocks': num_blocks,
            'cfg_edges': num_edges,
            'cyclomatic_complexity': cyclomatic,
            'num_loops': num_loops,
            'max_loop_depth': max_loop_depth,
            'avg_branching_factor': avg_degree,
            'num_exits': num_exits
        }
