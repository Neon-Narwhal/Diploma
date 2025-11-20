"""
Извлечение признаков из кода через AST анализ.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import ast
from typing import List, Dict, Any, Optional

# Добавляем корень проекта
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class ComplexityFeatureExtractor:
    """
    Извлечение признаков из кода через AST анализ.
    Гарантированно рабочая версия.
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names
        
        self.all_features = [
            'lines_of_code',
            'num_functions',
            'num_classes',
            'num_methods',
            'num_loops',
            'num_for_loops',
            'num_while_loops',
            'num_conditionals',
            'num_try_blocks',
            'num_with_blocks',
            'max_nesting_depth',
            'num_variables',
            'num_imports',
            'num_binary_ops',
            'num_compare_ops',
            'num_bool_ops',
            'num_return_stmts',
            'num_assignments',
            'num_function_calls',
            'avg_function_length',
        ]
    
    def extract(self, code_samples: List[str]) -> pd.DataFrame:
        """Извлечение признаков из списка кодов"""
        features_list = []
        
        for i, code in enumerate(code_samples):
            if i % 1000 == 0:
                print(f"  Обработано {i}/{len(code_samples)} примеров...", end='\r')
            
            features = self._extract_single(code)
            features_list.append(features)
        
        print(f"  Обработано {len(code_samples)}/{len(code_samples)} примеров")
        
        df = pd.DataFrame(features_list)
        
        if self.feature_names:
            available = [f for f in self.feature_names if f in df.columns]
            df = df[available]
        
        return df
    
    def _extract_single(self, code: str) -> Dict[str, float]:
        """Извлечение признаков из одного кода"""
        features = {name: 0.0 for name in self.all_features}
        
        try:
            tree = ast.parse(code)
            
            # Базовые метрики
            features['lines_of_code'] = len(code.split('\n'))
            
            # Счетчики для функций
            function_lengths = []
            
            # Обход AST
            for node in ast.walk(tree):
                # Функции
                if isinstance(node, ast.FunctionDef):
                    features['num_functions'] += 1
                    # Длина функции
                    if hasattr(node, 'body'):
                        function_lengths.append(len(node.body))
                
                # Классы и методы
                elif isinstance(node, ast.ClassDef):
                    features['num_classes'] += 1
                    # Методы в классе
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            features['num_methods'] += 1
                
                # Циклы
                elif isinstance(node, ast.For):
                    features['num_loops'] += 1
                    features['num_for_loops'] += 1
                
                elif isinstance(node, ast.While):
                    features['num_loops'] += 1
                    features['num_while_loops'] += 1
                
                # Условия
                elif isinstance(node, ast.If):
                    features['num_conditionals'] += 1
                
                # Try блоки
                elif isinstance(node, ast.Try):
                    features['num_try_blocks'] += 1
                
                # With блоки
                elif isinstance(node, ast.With):
                    features['num_with_blocks'] += 1
                
                # Переменные (присваивания)
                elif isinstance(node, ast.Assign):
                    features['num_assignments'] += 1
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            features['num_variables'] += 1
                
                # Импорты
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    features['num_imports'] += 1
                
                # Операторы
                elif isinstance(node, ast.BinOp):
                    features['num_binary_ops'] += 1
                
                elif isinstance(node, ast.Compare):
                    features['num_compare_ops'] += 1
                
                elif isinstance(node, ast.BoolOp):
                    features['num_bool_ops'] += 1
                
                # Return
                elif isinstance(node, ast.Return):
                    features['num_return_stmts'] += 1
                
                # Вызовы функций
                elif isinstance(node, ast.Call):
                    features['num_function_calls'] += 1
            
            # Максимальная глубина вложенности
            features['max_nesting_depth'] = self._compute_max_depth(tree)
            
            # Средняя длина функции
            if function_lengths:
                features['avg_function_length'] = np.mean(function_lengths)
            
        except SyntaxError:
            # Невалидный код - оставляем нули
            pass
        except Exception as e:
            # Любая другая ошибка
            pass
        
        return features
    
    def _compute_max_depth(self, tree: ast.AST) -> int:
        """Вычисление максимальной глубины вложенности"""
        def get_depth(node, current_depth=0):
            max_child_depth = current_depth
            
            for child in ast.iter_child_nodes(node):
                # Увеличиваем глубину для вложенных блоков
                if isinstance(child, (ast.For, ast.While, ast.If, ast.With, 
                                     ast.Try, ast.FunctionDef, ast.ClassDef)):
                    child_depth = get_depth(child, current_depth + 1)
                else:
                    child_depth = get_depth(child, current_depth)
                
                max_child_depth = max(max_child_depth, child_depth)
            
            return max_child_depth
        
        return get_depth(tree)


class TokenFeatureExtractor:
    """Извлечение токен-уровневых признаков"""
    
    def extract(self, code_samples: List[str], tokenizer=None) -> pd.DataFrame:
        features_list = []
        
        for code in code_samples:
            features = {
                'code_length': len(code),
                'num_lines': code.count('\n') + 1,
                'avg_line_length': len(code) / (code.count('\n') + 1) if code else 0,
                'num_whitespaces': sum(c.isspace() for c in code),
                'num_alphanumeric': sum(c.isalnum() for c in code),
            }
            
            if tokenizer:
                tokens = tokenizer.encode(code)
                features['num_tokens'] = len(tokens)
                features['unique_tokens'] = len(set(tokens))
                features['token_diversity'] = len(set(tokens)) / len(tokens) if tokens else 0
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
