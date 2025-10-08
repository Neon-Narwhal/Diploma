# complexity_analyzer/analyzers_baseline.py
"""
БАЗОВЫЕ реализации анализаторов - используют только встроенные возможности инструментов.
Без модификаций, только прямой маппинг метрик в классы сложности.
"""

import subprocess
import tempfile
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from pathlib import Path
import logging

# Импортируем библиотеки напрямую (Python API)
try:
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit, h_visit
    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False

try:
    import lizard
    LIZARD_AVAILABLE = True
except ImportError:
    LIZARD_AVAILABLE = False

try:
    from mccabe import PathGraphingAstVisitor  # ← ВОТ ТАК
    MCCABE_AVAILABLE = True
except ImportError:
    MCCABE_AVAILABLE = False
    PathGraphingAstVisitor = None


try:
    import subprocess
    # Проверяем наличие prospector
    result = subprocess.run(['prospector', '--version'], capture_output=True)
    PROSPECTOR_AVAILABLE = result.returncode == 0
except:
    PROSPECTOR_AVAILABLE = False

from config import (
    ComplexityClass,
    CYCLOMATIC_TO_TIME_COMPLEXITY,
    COGNITIVE_TO_TIME_COMPLEXITY
)

logger = logging.getLogger(__name__)


class BaselineAnalyzer(ABC):
    """Базовый класс для baseline анализаторов"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def analyze(self, source_code: str, language: str = 'python') -> Dict[str, Any]:
        """Анализирует код и возвращает метрики"""
        pass
    
    @abstractmethod
    def map_to_complexity(self, metrics: Dict[str, Any]) -> ComplexityClass:
        """Маппинг метрик в временную сложность"""
        pass
    
    def _map_by_threshold(self, value: float, threshold_rules: Dict) -> ComplexityClass:
        """Базовый маппинг по пороговым значениям"""
        for (min_val, max_val), complexity in threshold_rules.items():
            if min_val <= value < max_val:
                return complexity
        return ComplexityClass.EXPONENTIAL


class RadonBaselineAnalyzer(BaselineAnalyzer):
    """
    Baseline: Radon через Python API
    Использует только цикломатическую сложность
    """
    
    def __init__(self):
        super().__init__('radon_baseline')
        if not RADON_AVAILABLE:
            raise ImportError("radon is not installed. Install with: pip install radon")
    
    def analyze(self, source_code: str, language: str = 'python') -> Dict[str, Any]:
        """Анализ с помощью Radon API"""
        try:
            if not source_code or not source_code.strip():
                return {'cyclomatic_complexity': 1, 'error': 'Empty source code'}
            
            # Используем Radon API напрямую
            results = cc_visit(source_code)
            
            # Получаем максимальную сложность
            max_complexity = 1
            for item in results:
                max_complexity = max(max_complexity, item.complexity)
            
            return {
                'cyclomatic_complexity': max_complexity,
                'functions_count': len(results),
                'raw_output': [
                    {
                        'name': item.name,
                        'complexity': item.complexity,
                        'lineno': item.lineno,
                        'type': item.letter
                    } for item in results
                ]
            }
            
        except Exception as e:
            logger.error(f"Radon baseline analysis error: {e}")
            return {'cyclomatic_complexity': 1, 'error': str(e)}
    
    def map_to_complexity(self, metrics: Dict[str, Any]) -> ComplexityClass:
        """Простой маппинг CCN → временная сложность"""
        cc = metrics.get('cyclomatic_complexity', 1)
        return self._map_by_threshold(cc, CYCLOMATIC_TO_TIME_COMPLEXITY['radon'])


class LizardBaselineAnalyzer(BaselineAnalyzer):
    """
    Baseline: Lizard через Python API
    Использует только CCN из lizard
    """
    
    def __init__(self):
        super().__init__('lizard_baseline')
        if not LIZARD_AVAILABLE:
            raise ImportError("lizard is not installed. Install with: pip install lizard")
    
    def analyze(self, source_code: str, language: str = 'python') -> Dict[str, Any]:
        """Анализ с помощью Lizard API"""
        try:
            if not source_code or not source_code.strip():
                return {'cyclomatic_complexity': 1, 'error': 'Empty source code'}
            
            # Создаем временный файл (lizard требует файл)
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(source_code)
                temp_path = f.name
            
            # Используем Lizard API
            analysis = lizard.analyze_file(temp_path)
            
            # Удаляем временный файл
            Path(temp_path).unlink()
            
            # Получаем максимальную сложность
            max_complexity = 1
            for func in analysis.function_list:
                max_complexity = max(max_complexity, func.cyclomatic_complexity)
            
            return {
                'cyclomatic_complexity': max_complexity,
                'nloc': analysis.nloc,
                'functions_count': len(analysis.function_list),
                'raw_output': [
                    {
                        'name': func.name,
                        'complexity': func.cyclomatic_complexity,
                        'nloc': func.nloc,
                        'token_count': func.token_count,
                        'parameters': func.parameter_count
                    } for func in analysis.function_list
                ]
            }
            
        except Exception as e:
            logger.error(f"Lizard baseline analysis error: {e}")
            return {'cyclomatic_complexity': 1, 'error': str(e)}
    
    def map_to_complexity(self, metrics: Dict[str, Any]) -> ComplexityClass:
        """Простой маппинг CCN → временная сложность"""
        cc = metrics.get('cyclomatic_complexity', 1)
        return self._map_by_threshold(cc, CYCLOMATIC_TO_TIME_COMPLEXITY['lizard'])


class McCabeBaselineAnalyzer(BaselineAnalyzer):
    """Baseline: McCabe через Python API"""
    
    def __init__(self):
        super().__init__('mccabe_baseline')
        if not MCCABE_AVAILABLE:
            raise ImportError("mccabe is not installed. Install with: pip install mccabe")
    
    def analyze(self, source_code: str, language: str = 'python') -> Dict[str, Any]:
        """Анализ с помощью McCabe API"""
        try:
            if not source_code or not source_code.strip():
                return {'cyclomatic_complexity': 1, 'error': 'Empty source code'}
            
            import ast
            from mccabe import PathGraphingAstVisitor
            
            # Парсим код в AST
            tree = ast.parse(source_code)
            
            # Создаем visitor
            visitor = PathGraphingAstVisitor()
            
            # ПРАВИЛЬНО: передаем visitor как второй аргумент!
            visitor.preorder(tree, visitor)  # ← ВОТ ТАК!
            
            if not visitor.graphs:
                return {'cyclomatic_complexity': 1, 'functions_count': 0}
            
            # Получаем максимальную сложность
            max_complexity = 1
            results = []
            
            for graph in visitor.graphs.values():
                complexity = graph.complexity()
                max_complexity = max(max_complexity, complexity)
                results.append({
                    'name': graph.name,
                    'complexity': complexity,
                    'lineno': graph.lineno
                })
            
            return {
                'cyclomatic_complexity': max_complexity,
                'functions_count': len(visitor.graphs),
                'raw_output': results
            }
            
        except Exception as e:
            logger.error(f"McCabe baseline analysis error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'cyclomatic_complexity': 1, 'error': str(e)}
    
    def map_to_complexity(self, metrics: Dict[str, Any]) -> ComplexityClass:
        cc = metrics.get('cyclomatic_complexity', 1)
        return self._map_by_threshold(cc, CYCLOMATIC_TO_TIME_COMPLEXITY['mccabe'])


try:
    from complexipy import code_complexity
    COMPLEXIPY_AVAILABLE = True
except ImportError:
    COMPLEXIPY_AVAILABLE = False



class ComplexipyBaselineAnalyzer(BaselineAnalyzer):
    """Baseline: Complexipy через Python API"""
    
    def __init__(self):
        super().__init__('complexipy_baseline')
        if not COMPLEXIPY_AVAILABLE:
            raise ImportError("complexipy is not installed. Install with: pip install complexipy")
    
    def analyze(self, source_code: str, language: str = 'python') -> Dict[str, Any]:
        """Анализ с помощью Complexipy API"""
        try:
            if not source_code or not source_code.strip():
                return {'cognitive_complexity': 0, 'error': 'Empty source code'}
            
            result = code_complexity(source_code)
            max_cognitive = result.complexity
            
            return {
                'cognitive_complexity': max_cognitive,
                'functions_count': len(result.functions),
                'raw_output': {
                    'file_complexity': result.complexity,
                    'functions': [
                        {
                            'name': func.name,
                            'complexity': func.complexity,
                            # У Complexipy нет lineno, только line_start и line_end
                            'line_start': getattr(func, 'line_start', 0),
                            'line_end': getattr(func, 'line_end', 0)
                        } for func in result.functions
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Complexipy baseline analysis error: {e}")
            return {'cognitive_complexity': 0, 'error': str(e)}
    
    def map_to_complexity(self, metrics: Dict[str, Any]) -> ComplexityClass:
        """Маппинг когнитивной сложности → временная сложность"""
        cognitive = metrics.get('cognitive_complexity', 0)
        return self._map_by_threshold(cognitive, COGNITIVE_TO_TIME_COMPLEXITY['complexipy'])



class HalsteadAnalyzer(BaselineAnalyzer):
    """Baseline: Halstead Metrics через radon"""
    
    def __init__(self):
        super().__init__('halstead_baseline')
        if not RADON_AVAILABLE:
            raise ImportError("radon is not installed. Install with: pip install radon")
    
    def analyze(self, source_code: str, language: str = 'python') -> Dict[str, Any]:
        """Анализ Halstead метрик через radon"""
        try:
            if not source_code or not source_code.strip():
                return {'halstead_difficulty': 1, 'error': 'Empty source code'}
            
            from radon.metrics import h_visit
            
            # h_visit возвращает КОРТЕЖ (total, functions)
            # где total - HalsteadReport для всего файла, functions - список для функций
            result = h_visit(source_code)
            
            # result это кортеж: (total_halstead, [function_halsteads])
            if isinstance(result, tuple):
                total_h = result[0]  # Общие Halstead метрики для файла
                
                return {
                    'halstead_difficulty': total_h.difficulty if total_h.difficulty else 1,
                    'halstead_volume': total_h.volume if total_h.volume else 0,
                    'halstead_effort': total_h.effort if total_h.effort else 0,
                    'halstead_bugs': total_h.bugs if total_h.bugs else 0,
                    'halstead_time': total_h.time if total_h.time else 0,
                    'raw_output': {
                        'total': {
                            'difficulty': total_h.difficulty,
                            'volume': total_h.volume,
                            'effort': total_h.effort,
                            'bugs': total_h.bugs,
                            'time': total_h.time
                        }
                    }
                }
            else:
                # Fallback если формат другой
                return {
                    'halstead_difficulty': 1,
                    'halstead_volume': 0,
                    'error': 'Unexpected h_visit result format'
                }
            
        except Exception as e:
            logger.error(f"Halstead analysis error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'halstead_difficulty': 1, 'error': str(e)}
    
    def map_to_complexity(self, metrics: Dict[str, Any]) -> ComplexityClass:
        """Маппинг Halstead difficulty в временную сложность"""
        difficulty = metrics.get('halstead_difficulty', 1)
        
        if difficulty < 5:
            return ComplexityClass.CONSTANT
        elif difficulty < 10:
            return ComplexityClass.LINEAR
        elif difficulty < 20:
            return ComplexityClass.QUADRATIC
        elif difficulty < 40:
            return ComplexityClass.CUBIC
        else:
            return ComplexityClass.EXPONENTIAL



class MaintainabilityIndexAnalyzer(BaselineAnalyzer):
    """Baseline: Maintainability Index через Radon"""
    
    def __init__(self):
        super().__init__('mi_baseline')
        if not RADON_AVAILABLE:
            raise ImportError("radon is not installed. Install with: pip install radon")
    
    def analyze(self, source_code: str, language: str = 'python') -> Dict[str, Any]:
        """Анализ Maintainability Index"""
        try:
            if not source_code or not source_code.strip():
                return {'maintainability_index': 100, 'error': 'Empty source code'}
            
            from radon.metrics import mi_visit
            
            # mi_visit возвращает ОДНО ЧИСЛО (float), а не список!
            mi_score = mi_visit(source_code, multi=True)
            
            # Если mi_score это float (одно значение для файла)
            if isinstance(mi_score, (int, float)):
                avg_mi = float(mi_score)
            else:
                # Если это список/итерабельный объект
                try:
                    avg_mi = sum(mi_score) / len(mi_score) if mi_score else 100
                except TypeError:
                    avg_mi = 100
            
            return {
                'maintainability_index': avg_mi,
                'mi_rank': self._mi_to_rank(avg_mi),
                'raw_output': mi_score
            }
            
        except Exception as e:
            logger.error(f"MI analysis error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'maintainability_index': 100, 'error': str(e)}
    
    def _mi_to_rank(self, mi: float) -> str:
        """Преобразует MI в ранг (A-F)"""
        if mi >= 80:
            return 'A'
        elif mi >= 60:
            return 'B'
        elif mi >= 40:
            return 'C'
        elif mi >= 20:
            return 'D'
        else:
            return 'F'
    
    def map_to_complexity(self, metrics: Dict[str, Any]) -> ComplexityClass:
        """Маппинг MI в временную сложность"""
        mi = metrics.get('maintainability_index', 100)
        
        # Обратная корреляция: высокий MI = простой код
        if mi >= 80:
            return ComplexityClass.CONSTANT
        elif mi >= 60:
            return ComplexityClass.LINEAR
        elif mi >= 40:
            return ComplexityClass.QUADRATIC
        elif mi >= 20:
            return ComplexityClass.CUBIC
        else:
            return ComplexityClass.EXPONENTIAL


class NestedBlockDepthAnalyzer(BaselineAnalyzer):
    """
    Baseline: Nested Block Depth через AST
    Измеряет максимальную глубину вложенности блоков
    """
    
    def __init__(self):
        super().__init__('nbd_baseline')
    
    def analyze(self, source_code: str, language: str = 'python') -> Dict[str, Any]:
        """Анализ глубины вложенности"""
        try:
            if not source_code or not source_code.strip():
                return {'nested_block_depth': 0, 'error': 'Empty source code'}
            
            import ast
            
            class DepthVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.max_depth = 0
                    self.current_depth = 0
                
                def visit_If(self, node):
                    self.current_depth += 1
                    self.max_depth = max(self.max_depth, self.current_depth)
                    self.generic_visit(node)
                    self.current_depth -= 1
                
                def visit_For(self, node):
                    self.current_depth += 1
                    self.max_depth = max(self.max_depth, self.current_depth)
                    self.generic_visit(node)
                    self.current_depth -= 1
                
                def visit_While(self, node):
                    self.current_depth += 1
                    self.max_depth = max(self.max_depth, self.current_depth)
                    self.generic_visit(node)
                    self.current_depth -= 1
                
                def visit_With(self, node):
                    self.current_depth += 1
                    self.max_depth = max(self.max_depth, self.current_depth)
                    self.generic_visit(node)
                    self.current_depth -= 1
            
            tree = ast.parse(source_code)
            visitor = DepthVisitor()
            visitor.visit(tree)
            
            return {
                'nested_block_depth': visitor.max_depth,
                'raw_output': {'max_depth': visitor.max_depth}
            }
            
        except Exception as e:
            logger.error(f"NBD analysis error: {e}")
            return {'nested_block_depth': 0, 'error': str(e)}
    
    def map_to_complexity(self, metrics: Dict[str, Any]) -> ComplexityClass:
        """Маппинг NBD в временную сложность"""
        depth = metrics.get('nested_block_depth', 0)
        
        # Прямая корреляция с вложенностью циклов
        if depth == 0:
            return ComplexityClass.CONSTANT
        elif depth == 1:
            return ComplexityClass.LINEAR
        elif depth == 2:
            return ComplexityClass.QUADRATIC
        elif depth == 3:
            return ComplexityClass.CUBIC
        else:
            return ComplexityClass.EXPONENTIAL


# Обновляем фабрику
BASELINE_ANALYZER_FACTORY = {
    'radon_baseline': RadonBaselineAnalyzer,
    'lizard_baseline': LizardBaselineAnalyzer,
    'mccabe_baseline': McCabeBaselineAnalyzer,
    'complexipy_baseline': ComplexipyBaselineAnalyzer,
    'halstead_baseline': HalsteadAnalyzer,
    'mi_baseline': MaintainabilityIndexAnalyzer,
    'nbd_baseline': NestedBlockDepthAnalyzer,
}



def get_baseline_analyzer(tool_name: str) -> Optional[BaselineAnalyzer]:
    """Получить baseline анализатор по имени"""
    analyzer_class = BASELINE_ANALYZER_FACTORY.get(tool_name)
    if analyzer_class:
        return analyzer_class()
    return None
