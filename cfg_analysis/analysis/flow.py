"""
Упрощенный анализ потока данных (Data Flow Analysis).
Цель: найти паттерны изменения переменных цикла (i++, i*=2).
"""

import ast
from typing import Dict, Set, Any, List
from cfg_analysis.core.graph import ControlFlowGraph


class DataFlowAnalyzer:
    """
    Анализатор изменений переменных.
    """
    
    def __init__(self, cfg: ControlFlowGraph):
        self.cfg = cfg
        
    def analyze_loop_variables(self, loop_nodes: List[int]) -> Dict[str, str]:
        """
        Анализ того, как меняются переменные внутри цикла.
        Returns: {var_name: 'linear' | 'multiplicative' | 'unknown'}
        """
        changes = {}
        
        for bid in loop_nodes:
            block = self.cfg.get_block(bid)
            if not block: continue
            
            for stmt in block.statements:
                self._analyze_statement(stmt, changes)
                
        return changes
    
    def _analyze_statement(self, stmt: ast.AST, changes: Dict[str, str]):
        """Анализ одной инструкции"""
        # i += 1, i = i + 1
        if isinstance(stmt, (ast.AugAssign, ast.Assign)):
            targets = []
            if isinstance(stmt, ast.AugAssign):
                targets = [stmt.target]
                op = stmt.op
            else: # Assign
                targets = stmt.targets
                # Упрощение: считаем Assign сложным, если это не простое присваивание
                # Для простоты пока смотрим только AugAssign, так как это самый частый паттерн
                return 

            for target in targets:
                if isinstance(target, ast.Name):
                    var_name = target.id
                    current_type = changes.get(var_name, 'none')
                    
                    if isinstance(op, (ast.Add, ast.Sub)):
                        new_type = 'linear'
                    elif isinstance(op, (ast.Mult, ast.Div, ast.FloorDiv, ast.LShift, ast.RShift)):
                        new_type = 'multiplicative'
                    else:
                        new_type = 'unknown'
                        
                    # Агрегация: если уже было mult, то mult сильнее linear
                    if current_type == 'multiplicative' or new_type == 'multiplicative':
                        changes[var_name] = 'multiplicative'
                    elif new_type == 'linear':
                        changes[var_name] = 'linear'
