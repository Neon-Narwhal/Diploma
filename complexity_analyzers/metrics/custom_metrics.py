"""Собственные метрики сложности"""
import ast
import re
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter
import math

from complexity_analyzers.metrics.calculator import BaseMetricsCalculator

class CustomMetricsCalculator(BaseMetricsCalculator):
    """Калькулятор собственных метрик сложности"""
    
    def __init__(self):
        super().__init__('custom')
    
    def is_available(self) -> bool:
        """Всегда доступен"""
        return True
    
    def calculate(self, source_code: str) -> Dict[str, Any]:
        """Вычисление собственных метрик"""
        try:
            # Базовые текстовые метрики
            text_metrics = self._calculate_text_metrics(source_code)
            
            # AST метрики
            try:
                tree = ast.parse(source_code)
                ast_metrics = self._calculate_ast_metrics(tree)
                structural_metrics = self._calculate_structural_metrics(tree)
                complexity_metrics = self._calculate_complexity_metrics(tree)
            except SyntaxError:
                ast_metrics = {}
                structural_metrics = {}
                complexity_metrics = {}
            
            # Объединяем все метрики
            all_metrics = {}
            all_metrics.update(text_metrics)
            all_metrics.update(ast_metrics)
            all_metrics.update(structural_metrics)
            all_metrics.update(complexity_metrics)
            
            return all_metrics
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_text_metrics(self, source_code: str) -> Dict[str, Any]:
        """Текстовые метрики кода"""
        lines = source_code.split('\n')
        
        # Базовые подсчёты
        total_lines = len(lines)
        blank_lines = sum(1 for line in lines if not line.strip())
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        code_lines = total_lines - blank_lines - comment_lines
        
        # Логические строки кода (приблизительная оценка)
        logical_lines = self._count_logical_lines(source_code)
        
        # Метрики длины строк
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines:
            line_lengths = [len(line) for line in non_empty_lines]
            avg_line_length = sum(line_lengths) / len(line_lengths)
            max_line_length = max(line_lengths)
            median_line_length = sorted(line_lengths)[len(line_lengths) // 2]
        else:
            avg_line_length = 0
            max_line_length = 0
            median_line_length = 0
        
        # Метрики отступов
        indentation_metrics = self._calculate_indentation_metrics(lines)
        
        # Метрики символов
        char_metrics = self._calculate_character_metrics(source_code)
        
        return {
            'lines_of_code': code_lines,
            'logical_lines_of_code': logical_lines,
            'total_lines': total_lines,
            'blank_lines': blank_lines,
            'comment_lines': comment_lines,
            'avg_line_length': avg_line_length,
            'max_line_length': max_line_length,
            'median_line_length': median_line_length,
            'comment_ratio': comment_lines / total_lines if total_lines > 0 else 0,
            **indentation_metrics,
            **char_metrics
        }
    
    def _count_logical_lines(self, source_code: str) -> int:
        """Подсчёт логических строк кода"""
        # Удаляем комментарии и пустые строки
        lines = []
        for line in source_code.split('\n'):
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                lines.append(stripped)
        
        # Объединяем продолжения строк
        logical_lines = []
        current_line = ""
        
        for line in lines:
            current_line += line.rstrip('\\')
            
            if not line.endswith('\\'):
                if current_line.strip():
                    logical_lines.append(current_line)
                current_line = ""
        
        if current_line.strip():
            logical_lines.append(current_line)
        
        return len(logical_lines)
    
    def _calculate_indentation_metrics(self, lines: List[str]) -> Dict[str, Any]:
        """Метрики отступов"""
        indentations = []
        
        for line in lines:
            if line.strip():  # Не пустая строка
                indent = len(line) - len(line.lstrip())
                indentations.append(indent)
        
        if indentations:
            max_indent = max(indentations)
            avg_indent = sum(indentations) / len(indentations)
            
            # Оценка уровня вложенности (предполагая 4-пробельные отступы)
            max_nesting = max_indent // 4
            
            # Вариативность отступов
            unique_indents = len(set(indentations))
            
            return {
                'max_indentation': max_indent,
                'avg_indentation': avg_indent,
                'nested_depth': max_nesting,
                'indentation_levels': unique_indents
            }
        else:
            return {
                'max_indentation': 0,
                'avg_indentation': 0,
                'nested_depth': 0,
                'indentation_levels': 0
            }
    
    def _calculate_character_metrics(self, source_code: str) -> Dict[str, Any]:
        """Метрики символов"""
        # Подсчёт различных типов символов
        total_chars = len(source_code)
        
        # Алфавитные символы
        alpha_chars = sum(1 for c in source_code if c.isalpha())
        
        # Цифры
        digit_chars = sum(1 for c in source_code if c.isdigit())
        
        # Пробельные символы
        space_chars = sum(1 for c in source_code if c.isspace())
        
        # Операторы и разделители
        operators = r'[+\-*/=<>!&|^%~]'
        operator_chars = len(re.findall(operators, source_code))
        
        delimiters = r'[(){}[\],;:.]'
        delimiter_chars = len(re.findall(delimiters, source_code))
        
        return {
            'total_characters': total_chars,
            'alpha_characters': alpha_chars,
            'digit_characters': digit_chars,
            'space_characters': space_chars,
            'operator_characters': operator_chars,
            'delimiter_characters': delimiter_chars,
            'alpha_ratio': alpha_chars / total_chars if total_chars > 0 else 0,
            'space_ratio': space_chars / total_chars if total_chars > 0 else 0
        }
    
    def _calculate_ast_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        """Метрики на основе AST"""
        visitor = ASTMetricsVisitor()
        visitor.visit(tree)
        
        return visitor.get_metrics()
    
    def _calculate_structural_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        """Структурные метрики"""
        visitor = StructuralMetricsVisitor()
        visitor.visit(tree)
        
        return visitor.get_metrics()
    
    def _calculate_complexity_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        """Метрики сложности"""
        visitor = ComplexityMetricsVisitor()
        visitor.visit(tree)
        
        return visitor.get_metrics()

class ASTMetricsVisitor(ast.NodeVisitor):
    """Visitor для сбора базовых AST метрик"""
    
    def __init__(self):
        self.node_counts = defaultdict(int)
        self.total_nodes = 0
        self.max_depth = 0
        self.current_depth = 0
    
    def visit(self, node):
        self.node_counts[type(node).__name__] += 1
        self.total_nodes += 1
        
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        
        self.generic_visit(node)
        
        self.current_depth -= 1
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'total_ast_nodes': self.total_nodes,
            'max_ast_depth': self.max_depth,
            'function_count': self.node_counts.get('FunctionDef', 0) + self.node_counts.get('AsyncFunctionDef', 0),
            'class_count': self.node_counts.get('ClassDef', 0),
            'import_count': self.node_counts.get('Import', 0) + self.node_counts.get('ImportFrom', 0),
            'assignment_count': self.node_counts.get('Assign', 0) + self.node_counts.get('AugAssign', 0),
            'call_count': self.node_counts.get('Call', 0),
            'attribute_access_count': self.node_counts.get('Attribute', 0),
            'subscript_count': self.node_counts.get('Subscript', 0)
        }

class StructuralMetricsVisitor(ast.NodeVisitor):
    """Visitor для сбора структурных метрик"""
    
    def __init__(self):
        self.control_structures = defaultdict(int)
        self.nesting_stack = []
        self.max_nesting = 0
        
        # Для анализа функций
        self.current_function = None
        self.function_metrics = {}
        
        # Для анализа классов
        self.current_class = None
        self.class_metrics = {}
    
    def visit_FunctionDef(self, node):
        self._enter_function(node.name)
        self._enter_nesting('function')
        self.generic_visit(node)
        self._exit_nesting()
        self._exit_function()
    
    def visit_AsyncFunctionDef(self, node):
        self._enter_function(node.name)
        self._enter_nesting('async_function')
        self.generic_visit(node)
        self._exit_nesting()
        self._exit_function()
    
    def visit_ClassDef(self, node):
        self._enter_class(node.name)
        self._enter_nesting('class')
        self.generic_visit(node)
        self._exit_nesting()
        self._exit_class()
    
    def visit_If(self, node):
        self.control_structures['if'] += 1
        self._enter_nesting('if')
        self.generic_visit(node)
        self._exit_nesting()
    
    def visit_For(self, node):
        self.control_structures['for'] += 1
        self._enter_nesting('for')
        self.generic_visit(node)
        self._exit_nesting()
    
    def visit_While(self, node):
        self.control_structures['while'] += 1
        self._enter_nesting('while')
        self.generic_visit(node)
        self._exit_nesting()
    
    def visit_Try(self, node):
        self.control_structures['try'] += 1
        self._enter_nesting('try')
        self.generic_visit(node)
        self._exit_nesting()
    
    def visit_With(self, node):
        self.control_structures['with'] += 1
        self._enter_nesting('with')
        self.generic_visit(node)
        self._exit_nesting()
    
    def _enter_function(self, name: str):
        self.current_function = name
        self.function_metrics[name] = {
            'statements': 0,
            'max_nesting': 0,
            'parameters': 0
        }
    
    def _exit_function(self):
        if self.current_function:
            self.function_metrics[self.current_function]['max_nesting'] = len(self.nesting_stack)
        self.current_function = None
    
    def _enter_class(self, name: str):
        self.current_class = name
        self.class_metrics[name] = {
            'methods': 0,
            'attributes': 0
        }
    
    def _exit_class(self):
        self.current_class = None
    
    def _enter_nesting(self, context: str):
        self.nesting_stack.append(context)
        self.max_nesting = max(self.max_nesting, len(self.nesting_stack))
    
    def _exit_nesting(self):
        if self.nesting_stack:
            self.nesting_stack.pop()
    
    def get_metrics(self) -> Dict[str, Any]:
        # Статистика по функциям
        if self.function_metrics:
            avg_function_nesting = sum(f['max_nesting'] for f in self.function_metrics.values()) / len(self.function_metrics)
        else:
            avg_function_nesting = 0
        
        return {
            'nested_depth': self.max_nesting,
            'control_structure_count': sum(self.control_structures.values()),
            'if_statements': self.control_structures['if'],
            'for_loops': self.control_structures['for'],
            'while_loops': self.control_structures['while'],
            'try_blocks': self.control_structures['try'],
            'with_statements': self.control_structures['with'],
            'avg_function_nesting': avg_function_nesting,
            'total_control_structures': len(self.control_structures)
        }

class ComplexityMetricsVisitor(ast.NodeVisitor):
    """Visitor для сбора метрик сложности"""
    
    def __init__(self):
        self.cognitive_complexity = 0
        self.decision_points = 0
        self.loop_nesting = 0
        self.current_nesting = 0
        
        # Для когнитивной сложности
        self.nesting_increments = {
            'if': 1,
            'elif': 1,
            'for': 1,
            'while': 1,
            'try': 1,
            'except': 1,
            'lambda': 1
        }
    
    def visit_If(self, node):
        self.decision_points += 1
        self.cognitive_complexity += 1 + self.current_nesting
        
        self.current_nesting += 1
        self.generic_visit(node)
        self.current_nesting -= 1
    
    def visit_For(self, node):
        self.cognitive_complexity += 1 + self.current_nesting
        
        self.current_nesting += 1
        self.loop_nesting = max(self.loop_nesting, self.current_nesting)
        self.generic_visit(node)
        self.current_nesting -= 1
    
    def visit_While(self, node):
        self.cognitive_complexity += 1 + self.current_nesting
        
        self.current_nesting += 1
        self.loop_nesting = max(self.loop_nesting, self.current_nesting)
        self.generic_visit(node)
        self.current_nesting -= 1
    
    def visit_Try(self, node):
        self.cognitive_complexity += 1 + self.current_nesting
        
        self.current_nesting += 1
        self.generic_visit(node)
        self.current_nesting -= 1
    
    def visit_Lambda(self, node):
        self.cognitive_complexity += 1 + self.current_nesting
        self.generic_visit(node)
    
    def visit_BoolOp(self, node):
        # AND/OR операторы добавляют сложность
        if isinstance(node.op, (ast.And, ast.Or)):
            self.cognitive_complexity += len(node.values) - 1
        self.generic_visit(node)
    
    def visit_Compare(self, node):
        # Цепочки сравнений
        if len(node.comparators) > 1:
            self.cognitive_complexity += len(node.comparators) - 1
        self.generic_visit(node)
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'cognitive_complexity': self.cognitive_complexity,
            'decision_points': self.decision_points,
            'loop_nesting_depth': self.loop_nesting
        }
