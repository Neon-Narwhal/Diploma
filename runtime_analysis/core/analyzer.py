"""
Runtime Analyzer implementation.
"""

from typing import Any
from ast_analysis.core.base_analyzer import BaseASTAnalyzer
from ast_analysis.core.result import ASTAnalysisResult
from ast_analysis.core.registry import register_analyzer
from ast_analysis.core.enums import ComplexityClass

from runtime_analysis.core.generators import InputGenerator
from runtime_analysis.core.execution import CodeExecutor
from runtime_analysis.analysis.fitter import ComplexityFitter

@register_analyzer('runtime')
class RuntimeAnalyzer(BaseASTAnalyzer):
    
    def __init__(self, name: str = "runtime", **config):
        super().__init__(name, **config)
        self.generator = InputGenerator()
        self.executor = CodeExecutor(timeout=config.get('timeout', 1.0))
        self.fitter = ComplexityFitter()
        # Меньше размеров для отладки
        self.test_sizes = [10, 50, 100] 
        
    def analyze(self, code: str) -> ASTAnalysisResult:
        try:
            input_type = self.generator.infer_input_type(code)
            
            times = []
            valid_ns = []
            
            # Запускаем тесты на возрастающих N
            for n in self.test_sizes:
                try:
                    data = self.generator.generate(input_type, n)
                    t = self.executor.measure_time(code, data)
                    
                    if t > 0:  # Изменил с >= на >, так как 0 тоже ошибка
                        times.append(t)
                        valid_ns.append(n)
                    else:
                        # Если провал на малом N, может код битый
                        break
                except Exception as e:
                    # Логируем для отладки
                    break
            
            # Если удалось хотя бы 2 точки, пытаемся фиттить
            if len(valid_ns) >= 2:
                prediction, error = self.fitter.fit(valid_ns, times)
                pred_str = prediction.value
                conf = max(0.0, min(1.0, 1.0 - error * 5))
            else:
                # Провал
                pred_str = ComplexityClass.UNKNOWN.value
                conf = 0.0
                
            return ASTAnalysisResult.from_success(
                features={'runtime_points': len(valid_ns)},
                analyzer_name=self.name,
                code_length=len(code),
                prediction=pred_str,
                confidence=conf,
                prediction_metadata={'ns': valid_ns, 'times': times}
            )
            
        except Exception as e:
            return ASTAnalysisResult.from_error(f"Runtime Error: {str(e)}", self.name)
