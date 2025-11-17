"""ML-предиктор сложности"""
import joblib
import numpy as np
import pandas as pd
import ast
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from complexity_analyzers.core.base import BaseComplexityAnalyzer, AnalyzerType
from complexity_analyzers.core.result import ComplexityResult, ComplexityClass

class FeatureExtractor:
    """Извлекатель признаков из кода"""
    
    def __init__(self):
        self.feature_names: List[str] = []
        self._initialize_feature_names()
    
    def _initialize_feature_names(self):
        """Инициализация списка признаков"""
        self.feature_names = [
            # AST признаки
            'total_nodes', 'total_functions', 'total_classes',
            'for_loops', 'while_loops', 'nested_depth',
            'if_statements', 'elif_statements', 'else_statements',
            
            # Сложность
            'cyclomatic_complexity', 'cognitive_complexity',
            'halstead_difficulty', 'halstead_volume',
            
            # Рекурсия
            'recursive_functions', 'mutual_recursion',
            'max_recursion_depth',
            
            # Структуры данных
            'list_operations', 'dict_operations', 'set_operations',
            'comprehensions', 'generators',
            
            # Паттерны
            'sorting_patterns', 'search_patterns', 'dp_patterns',
            
            # Код-стиль
            'lines_of_code', 'comment_lines', 'blank_lines',
            'avg_line_length', 'max_line_length',
            
            # Сложные конструкции
            'lambda_count', 'decorator_count', 'exception_handlers',
            'context_managers', 'async_functions'
        ]
    
    def extract_features(self, source_code: str) -> np.ndarray:
        """Извлечение вектора признаков"""
        try:
            import ast
            tree = ast.parse(source_code)
            
            features = {}
            
            # AST признаки
            features.update(self._extract_ast_features(tree))
            
            # Текстовые признаки
            features.update(self._extract_text_features(source_code))
            
            # Метрики сложности
            features.update(self._extract_complexity_metrics(tree))
            
            # Паттерны алгоритмов
            features.update(self._extract_algorithm_patterns(tree))
            
            # Преобразуем в вектор в правильном порядке
            feature_vector = []
            for feature_name in self.feature_names:
                feature_vector.append(features.get(feature_name, 0))
            
            return np.array(feature_vector, dtype=float)
            
        except Exception as e:
            # В случае ошибки возвращаем нулевой вектор
            return np.zeros(len(self.feature_names), dtype=float)
    
    def _extract_ast_features(self, tree: ast.AST) -> Dict[str, Any]:
        """Извлечение AST признаков"""
        class ASTFeatureVisitor(ast.NodeVisitor):
            def __init__(self):
                self.features = {
                    'total_nodes': 0,
                    'total_functions': 0,
                    'total_classes': 0,
                    'for_loops': 0,
                    'while_loops': 0,
                    'if_statements': 0,
                    'elif_statements': 0,
                    'else_statements': 0,
                    'lambda_count': 0,
                    'decorator_count': 0,
                    'exception_handlers': 0,
                    'context_managers': 0,
                    'async_functions': 0,
                    'comprehensions': 0,
                    'generators': 0
                }
                self.current_depth = 0
                self.max_depth = 0
            
            def visit(self, node):
                self.features['total_nodes'] += 1
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
            
            def visit_FunctionDef(self, node):
                self.features['total_functions'] += 1
                self.features['decorator_count'] += len(node.decorator_list)
                self.generic_visit(node)
            
            def visit_AsyncFunctionDef(self, node):
                self.features['async_functions'] += 1
                self.features['total_functions'] += 1
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                self.features['total_classes'] += 1
                self.generic_visit(node)
            
            def visit_For(self, node):
                self.features['for_loops'] += 1
                self.generic_visit(node)
            
            def visit_While(self, node):
                self.features['while_loops'] += 1
                self.generic_visit(node)
            
            def visit_If(self, node):
                self.features['if_statements'] += 1
                if node.orelse:
                    if isinstance(node.orelse[0], ast.If):
                        self.features['elif_statements'] += 1
                    else:
                        self.features['else_statements'] += 1
                self.generic_visit(node)
            
            def visit_Lambda(self, node):
                self.features['lambda_count'] += 1
                self.generic_visit(node)
            
            def visit_Try(self, node):
                self.features['exception_handlers'] += 1
                self.generic_visit(node)
            
            def visit_With(self, node):
                self.features['context_managers'] += 1
                self.generic_visit(node)
            
            def visit_ListComp(self, node):
                self.features['comprehensions'] += 1
                self.generic_visit(node)
            
            def visit_DictComp(self, node):
                self.features['comprehensions'] += 1
                self.generic_visit(node)
            
            def visit_SetComp(self, node):
                self.features['comprehensions'] += 1
                self.generic_visit(node)
            
            def visit_GeneratorExp(self, node):
                self.features['generators'] += 1
                self.generic_visit(node)
        
        visitor = ASTFeatureVisitor()
        visitor.visit(tree)
        visitor.features['nested_depth'] = visitor.max_depth
        
        return visitor.features
    
    def _extract_text_features(self, source_code: str) -> Dict[str, Any]:
        """Извлечение текстовых признаков"""
        lines = source_code.split('\n')
        
        total_lines = len(lines)
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        blank_lines = sum(1 for line in lines if not line.strip())
        code_lines = total_lines - comment_lines - blank_lines
        
        line_lengths = [len(line) for line in lines if line.strip()]
        avg_line_length = np.mean(line_lengths) if line_lengths else 0
        max_line_length = max(line_lengths) if line_lengths else 0
        
        return {
            'lines_of_code': code_lines,
            'comment_lines': comment_lines,
            'blank_lines': blank_lines,
            'avg_line_length': avg_line_length,
            'max_line_length': max_line_length
        }
    
    def _extract_complexity_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        """Извлечение метрик сложности"""
        # Здесь можно интегрировать существующие анализаторы
        return {
            'cyclomatic_complexity': 0,  # Будет заполнено из других анализаторов
            'cognitive_complexity': 0,
            'halstead_difficulty': 0,
            'halstead_volume': 0
        }
    
    def _extract_algorithm_patterns(self, tree: ast.AST) -> Dict[str, Any]:
        """Извлечение паттернов алгоритмов"""
        # Упрощенная реализация
        return {
            'sorting_patterns': 0,
            'search_patterns': 0,
            'dp_patterns': 0,
            'recursive_functions': 0,
            'mutual_recursion': 0,
            'max_recursion_depth': 0,
            'list_operations': 0,
            'dict_operations': 0,
            'set_operations': 0
        }

class MLComplexityPredictor(BaseComplexityAnalyzer):
    """ML-предиктор сложности алгоритмов"""
    
    def __init__(self):
        super().__init__("ml_predictor", AnalyzerType.ML_PREDICTOR)
        self.feature_extractor = FeatureExtractor()
        self.models: Dict[str, Any] = {}
        self.scaler: Optional[StandardScaler] = None
        self.is_trained: bool = False
        self.class_names: List[str] = [cls.value for cls in ComplexityClass]
    
    def is_available(self) -> bool:
        """Проверка доступности"""
        try:
            import sklearn
            import xgboost
            return True
        except ImportError:
            return False
    
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Инициализация предиктора"""
        if not super().initialize(config):
            return False
        
        # Инициализируем модели
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        # Скейлер
        self.scaler = StandardScaler()
        
        # Пытаемся загрузить предобученные модели
        self._load_pretrained_models()
        
        return True
    
    def train(self, training_data: List[Dict[str, Any]], 
              validation_split: float = 0.2) -> Dict[str, Any]:
        """Обучение моделей"""
        if not training_data:
            raise ValueError("No training data provided")
        
        # Подготовка данных
        X, y = self._prepare_training_data(training_data)
        
        # Разделение на обучение и валидацию
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Нормализация признаков
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Обучение моделей
        training_results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Обучение
            model.fit(X_train_scaled, y_train)
            
            # Валидация
            train_pred = model.predict(X_train_scaled)
            val_pred = model.predict(X_val_scaled)
            
            train_accuracy = accuracy_score(y_train, train_pred)
            val_accuracy = accuracy_score(y_val, val_pred)
            
            # Кросс-валидация
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            training_results[name] = {
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_val, val_pred)
            }
            
            print(f"{name} - Val Accuracy: {val_accuracy:.3f}, CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        self.is_trained = True
        
        # Сохранение моделей
        self._save_models()
        
        return training_results
    
    def analyze(self, context) -> ComplexityResult:
        """Предсказание сложности"""
        if not self.is_trained:
            return ComplexityResult(
                complexity_class=ComplexityClass.UNKNOWN,
                confidence=0.0,
                analyzer_name=self.name,
                errors=["Model not trained"]
            )
        
        try:
            # Извлечение признаков
            features = self.feature_extractor.extract_features(context.source_code)
            features_scaled = self.scaler.transform([features])
            
            # Предсказания от всех моделей
            predictions = {}
            confidences = {}
            
            for name, model in self.models.items():
                pred = model.predict(features_scaled)[0]
                pred_proba = model.predict_proba(features_scaled)[0]
                
                predictions[name] = pred
                confidences[name] = max(pred_proba)
            
            # Ансамблевое предсказание
            final_prediction = self._ensemble_predict(predictions, confidences)
            final_confidence = np.mean(list(confidences.values()))
            
            # Преобразование строки в ComplexityClass
            complexity_class = ComplexityClass.UNKNOWN
            for cls in ComplexityClass:
                if cls.value == final_prediction:
                    complexity_class = cls
                    break
            
            return ComplexityResult(
                complexity_class=complexity_class,
                confidence=final_confidence,
                analyzer_name=self.name,
                ml_predictions={
                    'individual_predictions': predictions,
                    'individual_confidences': confidences,
                    'ensemble_prediction': final_prediction,
                    'feature_vector': features.tolist()
                }
            )
            
        except Exception as e:
            return ComplexityResult(
                complexity_class=ComplexityClass.UNKNOWN,
                confidence=0.0,
                analyzer_name=self.name,
                errors=[f"ML prediction error: {e}"]
            )
    
    def _prepare_training_data(self, training_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка данных для обучения"""
        X = []
        y = []
        
        for sample in training_data:
            source_code = sample.get('source_code', '')
            complexity_label = sample.get('complexity_class', 'unknown')
            
            if source_code and complexity_label != 'unknown':
                features = self.feature_extractor.extract_features(source_code)
                X.append(features)
                y.append(complexity_label)
        
        return np.array(X), np.array(y)
    
    def _ensemble_predict(self, predictions: Dict[str, str], 
                         confidences: Dict[str, float]) -> str:
        """Ансамблевое предсказание"""
        # Взвешенное голосование
        votes = {}
        total_weight = 0
        
        for model_name, prediction in predictions.items():
            confidence = confidences[model_name]
            weight = confidence  # Используем уверенность как вес
            
            if prediction not in votes:
                votes[prediction] = 0
            votes[prediction] += weight
            total_weight += weight
        
        # Нормализация и выбор лучшего
        if votes:
            return max(votes, key=votes.get)
        
        return 'unknown'
    
    def _save_models(self):
        """Сохранение обученных моделей"""
        try:
            for name, model in self.models.items():
                joblib.dump(model, f'models/{name}_complexity.pkl')
            
            joblib.dump(self.scaler, 'models/complexity_scaler.pkl')
            
            # Сохраняем информацию о признаках
            joblib.dump(self.feature_extractor.feature_names, 'models/feature_names.pkl')
            
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def _load_pretrained_models(self):
        """Загрузка предобученных моделей"""
        try:
            for name in self.models.keys():
                try:
                    model = joblib.load(f'models/{name}_complexity.pkl')
                    self.models[name] = model
                except FileNotFoundError:
                    pass  # Модель не найдена, используем неообученную
            
            try:
                self.scaler = joblib.load('models/complexity_scaler.pkl')
                self.is_trained = True
            except FileNotFoundError:
                pass
                
        except Exception as e:
            print(f"Error loading models: {e}")

class DatasetLoader:
    """Загрузчик датасетов для обучения"""
    
    @staticmethod
    def load_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
        """Загрузка из JSONL файла"""
        import json
        
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        sample = json.loads(line)
                        data.append(sample)
        except Exception as e:
            print(f"Error loading dataset: {e}")
        
        return data
    
    @staticmethod
    def load_bigobench_dataset() -> List[Dict[str, Any]]:
        """Загрузка BigO-Bench датасета"""
        # Интеграция с существующим bigobench_dataset.py
        try:
            from training.bigobench_dataset import BigOBenchDataset
            dataset = BigOBenchDataset()
            return dataset.get_all_samples()
        except ImportError:
            return []

class ModelEvaluator:
    """Оценщик качества моделей"""
    
    def __init__(self, predictor: MLComplexityPredictor):
        self.predictor = predictor
    
    def evaluate_on_dataset(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Оценка на тестовом датасете"""
        if not self.predictor.is_trained:
            return {'error': 'Model not trained'}
        
        results = {
            'total_samples': len(test_data),
            'correct_predictions': 0,
            'predictions_by_class': {},
            'confusion_matrix': {},
            'detailed_results': []
        }
        
        for sample in test_data:
            source_code = sample.get('source_code', '')
            true_complexity = sample.get('complexity_class', 'unknown')
            
            if not source_code or true_complexity == 'unknown':
                continue
            
            # Предсказание
            from complexity_analyzers.base.analyzer import AnalysisContext
            context = AnalysisContext(source_code=source_code)
            prediction_result = self.predictor.analyze(context)
            
            predicted_complexity = prediction_result.complexity_class.value
            confidence = prediction_result.confidence
            
            # Подсчет результатов
            if predicted_complexity == true_complexity:
                results['correct_predictions'] += 1
            
            # Статистика по классам
            if true_complexity not in results['predictions_by_class']:
                results['predictions_by_class'][true_complexity] = {
                    'total': 0, 'correct': 0
                }
            
            results['predictions_by_class'][true_complexity]['total'] += 1
            if predicted_complexity == true_complexity:
                results['predictions_by_class'][true_complexity]['correct'] += 1
            
            # Матрица ошибок
            confusion_key = f"{true_complexity}->{predicted_complexity}"
            results['confusion_matrix'][confusion_key] = results['confusion_matrix'].get(confusion_key, 0) + 1
            
            # Детальные результаты
            results['detailed_results'].append({
                'true_complexity': true_complexity,
                'predicted_complexity': predicted_complexity,
                'confidence': confidence,
                'correct': predicted_complexity == true_complexity
            })
        
        # Вычисление accuracy
        if results['total_samples'] > 0:
            results['accuracy'] = results['correct_predictions'] / results['total_samples']
        else:
            results['accuracy'] = 0.0
        
        # Accuracy по классам
        for class_name, stats in results['predictions_by_class'].items():
            if stats['total'] > 0:
                stats['accuracy'] = stats['correct'] / stats['total']
            else:
                stats['accuracy'] = 0.0
        
        return results
