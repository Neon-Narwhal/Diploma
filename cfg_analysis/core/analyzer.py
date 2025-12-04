"""
Основной класс анализатора CFG.
"""

import ast
from typing import Dict, Any, Optional

from ast_analysis.core.base_analyzer import BaseASTAnalyzer
from ast_analysis.core.result import ASTAnalysisResult
from ast_analysis.core.registry import register_analyzer
from ast_analysis.core.enums import ComplexityClass

from cfg_analysis.core.builder import CFGBuilder
from cfg_analysis.analysis.complexity import CFGComplexityMetrics
from cfg_analysis.analysis.flow import DataFlowAnalyzer


@register_analyzer('cfg_basic')
class CFGAnalyzer(BaseASTAnalyzer):
    """
    Анализатор сложности на основе Control Flow Graph.
    """
    
    def __init__(self, name: str = "cfg_basic", **config):
        super().__init__(name, **config)
        self.builder = CFGBuilder()
        
    def analyze(self, code: str) -> ASTAnalysisResult:
        try:
            # 1. Парсинг AST (нужен как промежуточный шаг)
            tree = ast.parse(code)
            
            # 2. Построение CFG
            cfg = self.builder.build("main", tree)
            
            # 3. Расчет метрик
            metrics_calc = CFGComplexityMetrics(cfg)
            metrics = metrics_calc.compute()
            
            # 4. Анализ потока данных (для определения типа роста переменных)
            # Собираем данные по всем циклам
            flow_analyzer = DataFlowAnalyzer(cfg)
            has_multiplicative_step = False
            
            # LoopAnalyzer уже отработал внутри metrics_calc, 
            # но нам нужен доступ к циклам для data flow
            # Поэтому вызываем отдельно или берем из метрик если бы сохраняли
            # (В текущей реализации metrics_calc.loop_analyzer хранит состояние)
            
            for loop in metrics_calc.loop_analyzer.loops:
                changes = flow_analyzer.analyze_loop_variables(loop['nodes'])
                if 'multiplicative' in changes.values():
                    has_multiplicative_step = True
            
            # 5. Эвристическое предсказание сложности
            complexity = self._predict_complexity(metrics, has_multiplicative_step)
            
            # 6. Формирование результата
            # Обогащаем features CFG-метриками
            features = {f"cfg_{k}": v for k, v in metrics.items()}
            features['cfg_has_multiplicative'] = int(has_multiplicative_step)
            
            return ASTAnalysisResult.from_success(
                features=features,
                analyzer_name=self.name,
                code_length=len(code),
                prediction=complexity.value,
                confidence=0.85,
                prediction_metadata={
                    'cyclomatic': metrics['cyclomatic_complexity'],
                    'max_loop_depth': metrics['max_loop_depth']
                }
            )
            
        except SyntaxError as e:
            return ASTAnalysisResult.from_error(f"SyntaxError: {e}", self.name)
        except Exception as e:
            return ASTAnalysisResult.from_error(f"CFG Error: {e}", self.name)

    def _predict_complexity(self, metrics: Dict[str, Any], has_multiplicative: bool) -> ComplexityClass:
        """
        Предсказание на основе графовых метрик.
        """
        loops = metrics['num_loops']
        depth = metrics['max_loop_depth']
        
        # 1. Нет циклов -> O(1)
        if loops == 0:
            return ComplexityClass.CONSTANT
            
        # 2. Одинарный цикл
        if depth == 1:
            if has_multiplicative:
                return ComplexityClass.LOGARITHMIC # O(log N)
            return ComplexityClass.LINEAR # O(N)
            
        # 3. Вложенные циклы
        if depth == 2:
            if has_multiplicative:
                return ComplexityClass.LINEARITHMIC # O(N log N)
            return ComplexityClass.QUADRATIC # O(N^2)
            
        if depth >= 3:
            return ComplexityClass.CUBIC # O(N^3)
            
        return ComplexityClass.LINEAR
