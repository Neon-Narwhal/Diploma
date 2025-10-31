"""Извлекатели признаков из AST для машинного обучения"""
import ast
import re
from typing import Dict, Any, List, Set, Optional
import numpy as np
from collections import defaultdict, Counter

class BasicFeatureExtractor(ast.NodeVisitor):
    """Базовый извлекатель признаков AST"""
    
    def __init__(self):
        self.features = defaultdict(int)
        self.node_counts = defaultdict(int)
        self.current_depth = 0
        self.max_depth = 0
        
    def visit(self, node):
        """Общий обход узлов"""
        self.node_counts[type(node).__name__] += 1
        self.features['total_nodes'] += 1
        
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        
        self.generic_visit(node)
        
        self.current_depth -= 1
    
    def visit_FunctionDef(self, node):
        """Обработка определения функций"""
        self.features['function_count'] += 1
        self.features['total_function_args'] += len(node.args.args)
        self.features['decorator_count'] += len(node.decorator_list)
        
        if node.returns:
            self.features['functions_with_return_annotation'] += 1
            
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Обработка определения классов"""
        self.features['class_count'] += 1
        self.features['class_decorator_count'] += len(node.decorator_list)
        self.features['base_class_count'] += len(node.bases)
        
        self.generic_visit(node)
    
    def visit_For(self, node):
        """Обработка for-циклов"""
        self.features['for_loop_count'] += 1
        if node.orelse:
            self.features['for_else_count'] += 1
            
        self.generic_visit(node)
    
    def visit_While(self, node):
        """Обработка while-циклов"""
        self.features['while_loop_count'] += 1
        if node.orelse:
            self.features['while_else_count'] += 1
            
        self.generic_visit(node)
    
    def visit_If(self, node):
        """Обработка условных операторов"""
        self.features['if_count'] += 1
        
        # Подсчет elif и else
        current = node
        elif_count = 0
        has_else = False
        
        while current.orelse:
            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                elif_count += 1
                current = current.orelse[0]
            else:
                has_else = True
                break
        
        self.features['elif_count'] += elif_count
        if has_else:
            self.features['else_count'] += 1
            
        self.generic_visit(node)
    
    def visit_Try(self, node):
        """Обработка try-except блоков"""
        self.features['try_count'] += 1
        self.features['except_handler_count'] += len(node.handlers)
        
        if node.orelse:
            self.features['try_else_count'] += 1
        if node.finalbody:
            self.features['finally_count'] += 1
            
        self.generic_visit(node)
    
    def visit_With(self, node):
        """Обработка context managers"""
        self.features['with_count'] += 1
        self.features['with_items_count'] += len(node.items)
        
        self.generic_visit(node)
    
    def visit_Lambda(self, node):
        """Обработка lambda-функций"""
        self.features['lambda_count'] += 1
        self.features['lambda_args_count'] += len(node.args.args)
        
        self.generic_visit(node)
    
    def visit_ListComp(self, node):
        """Обработка list comprehensions"""
        self.features['list_comp_count'] += 1
        self.features['list_comp_generators'] += len(node.generators)
        
        self.generic_visit(node)
    
    def visit_DictComp(self, node):
        """Обработка dict comprehensions"""
        self.features['dict_comp_count'] += 1
        self.features['dict_comp_generators'] += len(node.generators)
        
        self.generic_visit(node)
    
    def visit_SetComp(self, node):
        """Обработка set comprehensions"""
        self.features['set_comp_count'] += 1
        self.features['set_comp_generators'] += len(node.generators)
        
        self.generic_visit(node)
    
    def visit_GeneratorExp(self, node):
        """Обработка generator expressions"""
        self.features['generator_exp_count'] += 1
        self.features['generator_exp_generators'] += len(node.generators)
        
        self.generic_visit(node)
    
    def get_features(self) -> Dict[str, Any]:
        """Получение извлеченных признаков"""
        features = dict(self.features)
        features['max_ast_depth'] = self.max_depth
        features.update(self.node_counts)
        return features

class ComplexityFeatureExtractor(ast.NodeVisitor):
    """Извлекатель признаков сложности"""
    
    def __init__(self):
        self.features = defaultdict(int)
        self.nesting_stack = []
        self.current_function = None
        self.function_calls = defaultdict(set)
        self.variable_usage = defaultdict(int)
        
    def visit_FunctionDef(self, node):
        """Анализ функций для признаков сложности"""
        prev_function = self.current_function
        self.current_function = node.name
        
        # Анализ параметров
        self.features['max_function_params'] = max(
            self.features['max_function_params'], 
            len(node.args.args)
        )
        
        # Анализ тела функции
        function_complexity = self._analyze_function_body(node.body)
        self.features['max_function_complexity'] = max(
            self.features['max_function_complexity'],
            function_complexity
        )
        
        self.generic_visit(node)
        self.current_function = prev_function
    
    def visit_For(self, node):
        """Анализ циклов"""
        self._enter_loop('for')
        
        # Анализ итерируемого объекта
        self._analyze_iteration_target(node.iter)
        
        self.generic_visit(node)
        self._exit_loop()
    
    def visit_While(self, node):
        """Анализ while-циклов"""
        self._enter_loop('while')
        
        # Анализ условия
        self._analyze_loop_condition(node.test)
        
        self.generic_visit(node)
        self._exit_loop()
    
    def visit_Call(self, node):
        """Анализ вызовов функций"""
        if self.current_function:
            if isinstance(node.func, ast.Name):
                called_func = node.func.id
                self.function_calls[self.current_function].add(called_func)
                
                # Проверка рекурсии
                if called_func == self.current_function:
                    self.features['recursive_calls'] += 1
                
                # Анализ встроенных функций
                self._analyze_builtin_call(called_func, node)
            
            elif isinstance(node.func, ast.Attribute):
                method_name = node.func.attr
                self._analyze_method_call(method_name, node)
        
        self.generic_visit(node)
    
    def visit_Name(self, node):
        """Анализ использования переменных"""
        if isinstance(node.ctx, ast.Load):
            self.variable_usage[node.id] += 1
        
        self.generic_visit(node)
    
    def _enter_loop(self, loop_type: str):
        """Вход в цикл"""
        nesting_level = len(self.nesting_stack)
        self.nesting_stack.append(loop_type)
        
        # Обновляем максимальную вложенность
        self.features['max_loop_nesting'] = max(
            self.features['max_loop_nesting'],
            len(self.nesting_stack)
        )
        
        # Подсчет вложенных циклов
        if nesting_level > 0:
            self.features['nested_loops'] += 1
    
    def _exit_loop(self):
        """Выход из цикла"""
        if self.nesting_stack:
            self.nesting_stack.pop()
    
    def _analyze_function_body(self, body: List[ast.AST]) -> int:
        """Анализ сложности тела функции"""
        complexity_score = 0
        
        for stmt in body:
            if isinstance(stmt, (ast.For, ast.While)):
                complexity_score += 2
            elif isinstance(stmt, ast.If):
                complexity_score += 1
            elif isinstance(stmt, ast.Try):
                complexity_score += 1
        
        return complexity_score
    
    def _analyze_iteration_target(self, iter_node: ast.AST):
        """Анализ цели итерации"""
        if isinstance(iter_node, ast.Call):
            if isinstance(iter_node.func, ast.Name):
                func_name = iter_node.func.id
                if func_name == 'range':
                    self.features['range_iterations'] += 1
                elif func_name in ['enumerate', 'zip']:
                    self.features['complex_iterations'] += 1
                elif func_name in ['sorted', 'reversed']:
                    self.features['sorting_iterations'] += 1
    
    def _analyze_loop_condition(self, condition: ast.AST):
        """Анализ условия цикла"""
        # Подсчет сложности условий
        complexity = self._calculate_expression_complexity(condition)
        self.features['loop_condition_complexity'] += complexity
    
    def _analyze_builtin_call(self, func_name: str, node: ast.Call):
        """Анализ вызовов встроенных функций"""
        if func_name in ['sorted', 'max', 'min', 'sum']:
            self.features['builtin_complex_calls'] += 1
        elif func_name in ['len', 'abs', 'int', 'str']:
            self.features['builtin_simple_calls'] += 1
        elif func_name in ['map', 'filter', 'reduce']:
            self.features['functional_calls'] += 1
    
    def _analyze_method_call(self, method_name: str, node: ast.Call):
        """Анализ вызовов методов"""
        if method_name in ['append', 'insert', 'remove', 'pop']:
            self.features['list_methods'] += 1
        elif method_name in ['get', 'keys', 'values', 'items']:
            self.features['dict_methods'] += 1
        elif method_name in ['add', 'union', 'intersection']:
            self.features['set_methods'] += 1
    
    def _calculate_expression_complexity(self, expr: ast.AST) -> int:
        """Вычисление сложности выражения"""
        if isinstance(expr, ast.BoolOp):
            return len(expr.values)
        elif isinstance(expr, ast.Compare):
            return len(expr.comparators)
        elif isinstance(expr, ast.Call):
            return 2  # Вызов функции добавляет сложности
        else:
            return 1
    
    def get_features(self) -> Dict[str, Any]:
        """Получение признаков сложности"""
        features = dict(self.features)
        
        # Добавляем вычисляемые признаки
        features['total_function_calls'] = sum(len(calls) for calls in self.function_calls.values())
        features['unique_functions_called'] = len(set().union(*self.function_calls.values()) if self.function_calls else set())
        features['most_used_variable_count'] = max(self.variable_usage.values()) if self.variable_usage else 0
        features['total_variable_usage'] = sum(self.variable_usage.values())
        
        return features

class TextualFeatureExtractor:
    """Извлекатель текстовых признаков"""
    
    def __init__(self):
        pass
    
    def extract_features(self, source_code: str) -> Dict[str, Any]:
        """Извлечение текстовых признаков"""
        lines = source_code.split('\n')
        
        features = {}
        
        # Базовые метрики
        features['total_lines'] = len(lines)
        features['non_empty_lines'] = len([line for line in lines if line.strip()])
        features['comment_lines'] = len([line for line in lines if line.strip().startswith('#')])
        features['blank_lines'] = features['total_lines'] - features['non_empty_lines']
        
        # Метрики длины строк
        line_lengths = [len(line) for line in lines]
        features['avg_line_length'] = np.mean(line_lengths) if line_lengths else 0
        features['max_line_length'] = max(line_lengths) if line_lengths else 0
        features['median_line_length'] = np.median(line_lengths) if line_lengths else 0
        
        # Анализ отступов
        indentations = []
        for line in lines:
            if line.strip():  # Не пустая строка
                indent = len(line) - len(line.lstrip())
                indentations.append(indent)
        
        if indentations:
            features['max_indentation'] = max(indentations)
            features['avg_indentation'] = np.mean(indentations)
            features['indentation_variance'] = np.var(indentations)
        else:
            features['max_indentation'] = 0
            features['avg_indentation'] = 0
            features['indentation_variance'] = 0
        
        # Анализ ключевых слов
        keywords = ['def', 'class', 'if', 'else', 'elif', 'for', 'while', 
                   'try', 'except', 'finally', 'with', 'lambda', 'return']
        
        for keyword in keywords:
            pattern = r'\b' + keyword + r'\b'
            features[f'{keyword}_count'] = len(re.findall(pattern, source_code))
        
        # Анализ операторов
        operators = ['+', '-', '*', '/', '//', '%', '**', '==', '!=', '<', '>', '<=', '>=']
        for op in operators:
            features[f'operator_{op.replace("/", "div").replace("*", "mul").replace("<", "lt").replace(">", "gt").replace("=", "eq")}_count'] = source_code.count(op)
        
        # Анализ скобок и разделителей
        features['parentheses_count'] = source_code.count('(')
        features['brackets_count'] = source_code.count('[')
        features['braces_count'] = source_code.count('{')
        features['comma_count'] = source_code.count(',')
        features['colon_count'] = source_code.count(':')
        features['semicolon_count'] = source_code.count(';')
        
        # Анализ строк и чисел
        features['string_literals'] = len(re.findall(r'["\'].*?["\']', source_code))
        features['numeric_literals'] = len(re.findall(r'\b\d+\.?\d*\b', source_code))
        
        # Сложность имен переменных
        variable_names = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', source_code)
        if variable_names:
            name_lengths = [len(name) for name in variable_names]
            features['avg_variable_name_length'] = np.mean(name_lengths)
            features['max_variable_name_length'] = max(name_lengths)
            features['unique_variable_count'] = len(set(variable_names))
        else:
            features['avg_variable_name_length'] = 0
            features['max_variable_name_length'] = 0
            features['unique_variable_count'] = 0
        
        return features

class ASTFeatureExtractor:
    """Главный класс для извлечения всех признаков"""
    
    def __init__(self):
        self.basic_extractor = BasicFeatureExtractor()
        self.complexity_extractor = ComplexityFeatureExtractor()
        self.textual_extractor = TextualFeatureExtractor()
    
    def extract_all_features(self, tree: ast.AST, source_code: str) -> Dict[str, Any]:
        """Извлечение всех типов признаков"""
        # Сброс экстракторов
        self.basic_extractor = BasicFeatureExtractor()
        self.complexity_extractor = ComplexityFeatureExtractor()
        
        # Извлечение AST-признаков
        self.basic_extractor.visit(tree)
        self.complexity_extractor.visit(tree)
        
        # Объединение всех признаков
        all_features = {}
        
        # Базовые признаки
        basic_features = self.basic_extractor.get_features()
        for key, value in basic_features.items():
            all_features[f'basic_{key}'] = value
        
        # Признаки сложности
        complexity_features = self.complexity_extractor.get_features()
        for key, value in complexity_features.items():
            all_features[f'complexity_{key}'] = value
        
        # Текстовые признаки
        textual_features = self.textual_extractor.extract_features(source_code)
        for key, value in textual_features.items():
            all_features[f'textual_{key}'] = value
        
        # Вычисляемые комбинированные признаки
        all_features.update(self._calculate_combined_features(all_features))
        
        return all_features
    
    def _calculate_combined_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Вычисление комбинированных признаков"""
        combined = {}
        
        # Соотношения
        total_lines = features.get('textual_total_lines', 1)
        total_nodes = features.get('basic_total_nodes', 1)
        
        combined['nodes_per_line'] = total_nodes / total_lines
        combined['complexity_density'] = features.get('complexity_max_function_complexity', 0) / total_lines
        
        # Индексы сложности
        loop_count = features.get('basic_for_loop_count', 0) + features.get('basic_while_loop_count', 0)
        condition_count = features.get('basic_if_count', 0)
        function_count = features.get('basic_function_count', 1)
        
        combined['loop_to_function_ratio'] = loop_count / function_count
        combined['condition_to_function_ratio'] = condition_count / function_count
        combined['complexity_index'] = (loop_count * 2 + condition_count) / total_lines
        
        # Индекс структурной сложности
        nesting = features.get('complexity_max_loop_nesting', 0)
        recursion = features.get('complexity_recursive_calls', 0)
        combined['structural_complexity'] = nesting * 2 + min(recursion, 5)
        
        return combined
    
    def get_feature_names(self) -> List[str]:
        """Получение списка имен всех признаков"""
        # Этот метод нужен для ML-моделей
        dummy_tree = ast.parse("def dummy(): pass")
        dummy_features = self.extract_all_features(dummy_tree, "def dummy(): pass")
        return list(dummy_features.keys())
