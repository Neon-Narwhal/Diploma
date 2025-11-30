"""
Извлечение признаков из кода: AST метрики + TF-IDF.
"""

import sys
import ast
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from complexity_analyzers.analyzers.ast_advanced import AdvancedASTAnalyzer
from complexity_analyzers.core.base import AnalysisContext
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class ComplexityFeatureExtractor:
    """
    Гибридный экстрактор: AST статистика + TF-IDF топ слов.
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None, max_tfidf_features: int = 2000):
        self.feature_names = feature_names
        self.max_tfidf_features = max_tfidf_features
        self.analyzer = AdvancedASTAnalyzer()
        
        # TF-IDF для поиска ключевых слов (sort, recursive, while, nested...)
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=max_tfidf_features,
            stop_words='english',
            analyzer='word',
            token_pattern=r"(?u)\b\w\w+\b"
        )
        self.is_fitted = False
        
        # Базовые AST признаки
        self.ast_features = [
            'lines_of_code', 'num_functions', 'num_classes', 'num_methods',
            'num_loops', 'num_for_loops', 'num_while_loops',
            'num_conditionals', 'num_try_blocks', 'max_nesting_depth',
            'num_return_stmts', 'num_function_calls', 'num_recursion',
            'num_variables', 'num_imports', 'avg_line_length'
        ]
    
    def extract(self, code_samples: List[str]) -> pd.DataFrame:
        """Извлечение признаков из списка кодов"""
        # 1. Извлекаем AST признаки
        ast_data = []
        # print(f"  Извлечение AST признаков ({len(code_samples)} примеров)...")
        for code in code_samples:
            ctx = AnalysisContext(source_code=code)
            result = self.analyzer.analyze(ctx)
            
            # Вытаскиваем плоские метрики для ML из результата
            metrics = {
                'nested_depth': result.metrics.nested_depth,
                'loop_count': result.metrics.loop_count,
                # Вытаскиваем новые фичи из debug_info или loop_analysis
                'has_log_step': int(result.debug_info['loop_analysis']['has_logarithmic_step']),
                'has_dep_loop': int(result.debug_info['loop_analysis']['has_dependent_inner_loop'])
            }
            ast_data.append(metrics)
        
        df_ast = pd.DataFrame(ast_data)
        
        # 2. Извлекаем TF-IDF признаки
        # print(f"  Извлечение TF-IDF признаков...")
        if not self.is_fitted:
            # Обучаем TF-IDF только если еще не обучен (обычно на Train)
            tfidf_matrix = self.tfidf.fit_transform(code_samples)
            self.is_fitted = True
        else:
            # На Val/Test только применяем
            tfidf_matrix = self.tfidf.transform(code_samples)
            
        # Конвертируем в DataFrame
        tfidf_cols = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
        df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols)
        
        # 3. Объединяем
        result = pd.concat([df_ast, df_tfidf], axis=1)
        
        # Если нужны конкретные колонки - фильтруем
        if self.feature_names:
            # Добавляем недостающие колонки нулями (на случай расхождений)
            for col in self.feature_names:
                if col not in result.columns:
                    result[col] = 0.0
            result = result[self.feature_names]
        
        return result
    
    def _extract_ast(self, code: str) -> Dict[str, float]:
        """Извлечение признаков из одного кода"""
        features = {name: 0.0 for name in self.ast_features}
        
        try:
            features['lines_of_code'] = len(code.split('\n'))
            if features['lines_of_code'] > 0:
                features['avg_line_length'] = len(code) / features['lines_of_code']

            tree = ast.parse(code)
            
            # Проверка на рекурсию
            func_names = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    features['num_functions'] += 1
                    func_names.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    features['num_classes'] += 1
                elif isinstance(node, (ast.For, ast.While)):
                    features['num_loops'] += 1
                    if isinstance(node, ast.For): features['num_for_loops'] += 1
                    else: features['num_while_loops'] += 1
                elif isinstance(node, ast.If):
                    features['num_conditionals'] += 1
                elif isinstance(node, ast.Try):
                    features['num_try_blocks'] += 1
                elif isinstance(node, ast.Return):
                    features['num_return_stmts'] += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    features['num_imports'] += 1
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    features['num_variables'] += 1
                    
                elif isinstance(node, ast.Call):
                    features['num_function_calls'] += 1
                    # Проверка рекурсии: вызов функции внутри самой себя
                    if isinstance(node.func, ast.Name) and node.func.id in func_names:
                        features['num_recursion'] = 1
                        
            features['max_nesting_depth'] = self._compute_max_depth(tree)
            
        except:
            # Если синтаксическая ошибка - возвращаем нули
            pass
        
        return features
    
    def _compute_max_depth(self, tree: ast.AST) -> int:
        def get_depth(node, current_depth=0):
            max_d = current_depth
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.For, ast.While, ast.If, ast.FunctionDef, ast.ClassDef)):
                    max_d = max(max_d, get_depth(child, current_depth + 1))
                else:
                    max_d = max(max_d, get_depth(child, current_depth))
            return max_d
        return get_depth(tree)


class TokenFeatureExtractor:
    """
    Извлечение токен-уровневых признаков (Заглушка для совместимости).
    """
    
    def extract(self, code_samples: List[str], tokenizer=None) -> pd.DataFrame:
        """
        Извлечение токен-статистик.
        """
        features_list = []
        
        for code in code_samples:
            features = {
                'code_length': len(code),
                'num_lines': code.count('\n') + 1,
                'avg_line_length': len(code) / (code.count('\n') + 1) if code else 0,
                'num_whitespaces': sum(c.isspace() for c in code),
                'num_alphanumeric': sum(c.isalnum() for c in code),
            }
            features_list.append(features)
        
        return pd.DataFrame(features_list)
