"""Гибридный ансамблевый анализатор"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from collections import defaultdict, Counter
from complexity_analyzers.base.analyzer import BaseComplexityAnalyzer, AnalyzerType, AnalysisContext
from complexity_analyzers.base.result import ComplexityResult, ComplexityClass, ResultAggregator

class WeightingStrategy:
    """Стратегия весов для ансамбля"""
    
    @staticmethod
    def uniform_weights(analyzers: List[str]) -> Dict[str, float]:
        """Равномерные веса"""
        weight = 1.0 / len(analyzers)
        return {name: weight for name in analyzers}
    
    @staticmethod
    def confidence_based_weights(results: Dict[str, ComplexityResult]) -> Dict[str, float]:
        """Веса на основе уверенности"""
        total_confidence = sum(r.confidence for r in results.values() if r.confidence > 0)
        
        if total_confidence == 0:
            return WeightingStrategy.uniform_weights(list(results.keys()))
        
        weights = {}
        for name, result in results.items():
            weights[name] = result.confidence / total_confidence
        
        return weights
    
    @staticmethod
    def performance_based_weights(analyzer_performance: Dict[str, float]) -> Dict[str, float]:
        """Веса на основе исторической производительности"""
        total_performance = sum(analyzer_performance.values())
        
        if total_performance == 0:
            return WeightingStrategy.uniform_weights(list(analyzer_performance.keys()))
        
        weights = {}
        for name, performance in analyzer_performance.items():
            weights[name] = performance / total_performance
        
        return weights
    
    @staticmethod
    def adaptive_weights(results: Dict[str, ComplexityResult], 
                        historical_performance: Dict[str, float]) -> Dict[str, float]:
        """Адаптивные веса (комбинация уверенности и производительности)"""
        confidence_weights = WeightingStrategy.confidence_based_weights(results)
        performance_weights = WeightingStrategy.performance_based_weights(historical_performance)
        
        adaptive_weights = {}
        for name in results.keys():
            conf_weight = confidence_weights.get(name, 0)
            perf_weight = performance_weights.get(name, 0)
            # Комбинируем с коэффициентами 0.7 и 0.3
            adaptive_weights[name] = 0.7 * conf_weight + 0.3 * perf_weight
        
        # Нормализация
        total = sum(adaptive_weights.values())
        if total > 0:
            adaptive_weights = {k: v/total for k, v in adaptive_weights.items()}
        
        return adaptive_weights

class VotingStrategy:
    """Стратегии голосования"""
    
    @staticmethod
    def majority_voting(results: Dict[str, ComplexityResult]) -> ComplexityClass:
        """Мажоритарное голосование"""
        votes = [r.complexity_class for r in results.values() if r.is_valid()]
        
        if not votes:
            return ComplexityClass.UNKNOWN
        
        vote_counts = Counter(votes)
        return vote_counts.most_common(1)[0][0]
    
    @staticmethod
    def weighted_voting(results: Dict[str, ComplexityResult], 
                       weights: Dict[str, float]) -> ComplexityClass:
        """Взвешенное голосование"""
        complexity_votes = defaultdict(float)
        
        for name, result in results.items():
            if result.is_valid():
                weight = weights.get(name, 0)
                complexity_votes[result.complexity_class] += weight
        
        if not complexity_votes:
            return ComplexityClass.UNKNOWN
        
        return max(complexity_votes, key=complexity_votes.get)
    
    @staticmethod
    def confidence_weighted_voting(results: Dict[str, ComplexityResult]) -> ComplexityClass:
        """Голосование с весами по уверенности"""
        complexity_votes = defaultdict(float)
        
        for result in results.values():
            if result.is_valid():
                complexity_votes[result.complexity_class] += result.confidence
        
        if not complexity_votes:
            return ComplexityClass.UNKNOWN
        
        return max(complexity_votes, key=complexity_votes.get)
    
    @staticmethod
    def rank_based_voting(results: Dict[str, ComplexityResult], 
                         analyzer_rankings: Dict[str, int]) -> ComplexityClass:
        """Голосование на основе ранжирования анализаторов"""
        complexity_votes = defaultdict(float)
        
        for name, result in results.items():
            if result.is_valid():
                # Чем меньше ранг, тем больше вес (1-й ранг = максимальный вес)
                rank = analyzer_rankings.get(name, len(analyzer_rankings) + 1)
                weight = 1.0 / rank
                complexity_votes[result.complexity_class] += weight
        
        if not complexity_votes:
            return ComplexityClass.UNKNOWN
        
        return max(complexity_votes, key=complexity_votes.get)

class ConflictResolver:
    """Решатель конфликтов между анализаторами"""
    
    def __init__(self):
        self.complexity_hierarchy = {
            ComplexityClass.CONSTANT: 1,
            ComplexityClass.LOGARITHMIC: 2,
            ComplexityClass.LINEAR: 3,
            ComplexityClass.LINEARITHMIC: 4,
            ComplexityClass.QUADRATIC: 5,
            ComplexityClass.CUBIC: 6,
            ComplexityClass.POLYNOMIAL: 7,
            ComplexityClass.EXPONENTIAL: 8,
            ComplexityClass.FACTORIAL: 9,
            ComplexityClass.UNKNOWN: 0
        }
    
    def resolve_conflicts(self, results: Dict[str, ComplexityResult]) -> ComplexityResult:
        """Разрешение конфликтов между результатами"""
        valid_results = {k: v for k, v in results.items() if v.is_valid()}
        
        if not valid_results:
            return ComplexityResult(
                complexity_class=ComplexityClass.UNKNOWN,
                confidence=0.0,
                analyzer_name="conflict_resolver",
                errors=["No valid results to resolve conflicts"]
            )
        
        # Анализ конфликтов
        complexity_classes = [r.complexity_class for r in valid_results.values()]
        unique_classes = set(complexity_classes)
        
        if len(unique_classes) == 1:
            # Нет конфликтов
            return self._merge_consistent_results(valid_results)
        
        # Есть конфликты - применяем стратегии разрешения
        conflict_analysis = self._analyze_conflicts(valid_results)
        
        if conflict_analysis['severity'] == 'low':
            # Небольшие различия - используем консервативную оценку
            return self._conservative_resolution(valid_results)
        elif conflict_analysis['severity'] == 'medium':
            # Средние различия - взвешенное голосование по уверенности
            return self._confidence_based_resolution(valid_results)
        else:
            # Высокие различия - экспертная система
            return self._expert_system_resolution(valid_results, conflict_analysis)
    
    def _analyze_conflicts(self, results: Dict[str, ComplexityResult]) -> Dict[str, Any]:
        """Анализ конфликтов"""
        complexity_orders = [
            self.complexity_hierarchy[r.complexity_class] 
            for r in results.values()
        ]
        
        max_order = max(complexity_orders)
        min_order = min(complexity_orders)
        order_range = max_order - min_order
        
        # Определение серьезности конфликта
        if order_range <= 1:
            severity = 'low'
        elif order_range <= 3:
            severity = 'medium'
        else:
            severity = 'high'
        
        # Анализ надежности анализаторов
        high_confidence_results = [
            name for name, result in results.items() 
            if result.confidence > 0.8
        ]
        
        return {
            'severity': severity,
            'order_range': order_range,
            'min_complexity': min([self.complexity_hierarchy[r.complexity_class] for r in results.values()]),
            'max_complexity': max([self.complexity_hierarchy[r.complexity_class] for r in results.values()]),
            'high_confidence_analyzers': high_confidence_results,
            'analyzer_count': len(results)
        }
    
    def _conservative_resolution(self, results: Dict[str, ComplexityResult]) -> ComplexityResult:
        """Консервативное разрешение (выбираем более высокую сложность)"""
        # Выбираем результат с максимальной сложностью среди высоконадежных
        high_conf_results = {k: v for k, v in results.items() if v.confidence > 0.7}
        
        if not high_conf_results:
            high_conf_results = results
        
        best_result = max(
            high_conf_results.values(),
            key=lambda r: self.complexity_hierarchy[r.complexity_class]
        )
        
        return ComplexityResult(
            complexity_class=best_result.complexity_class,
            confidence=best_result.confidence * 0.9,  # Снижаем уверенность из-за конфликта
            analyzer_name="conservative_resolver",
            debug_info={
                'resolution_strategy': 'conservative',
                'original_results': len(results),
                'high_confidence_results': len(high_conf_results)
            }
        )
    
    def _confidence_based_resolution(self, results: Dict[str, ComplexityResult]) -> ComplexityResult:
        """Разрешение на основе уверенности"""
        # Взвешенное голосование по уверенности
        complexity_votes = defaultdict(float)
        total_confidence = 0
        
        for result in results.values():
            vote_weight = result.confidence
            complexity_votes[result.complexity_class] += vote_weight
            total_confidence += vote_weight
        
        best_complexity = max(complexity_votes, key=complexity_votes.get)
        confidence_score = complexity_votes[best_complexity] / total_confidence if total_confidence > 0 else 0
        
        return ComplexityResult(
            complexity_class=best_complexity,
            confidence=confidence_score,
            analyzer_name="confidence_resolver",
            debug_info={
                'resolution_strategy': 'confidence_based',
                'complexity_votes': dict(complexity_votes),
                'total_confidence': total_confidence
            }
        )
    
    def _expert_system_resolution(self, results: Dict[str, ComplexityResult], 
                                 conflict_analysis: Dict[str, Any]) -> ComplexityResult:
        """Экспертная система для разрешения сложных конфликтов"""
        # Правила экспертной системы
        rules = []
        
        # Правило 1: Если есть анализаторы с очень высокой уверенностью, доверяем им
        high_conf_results = {k: v for k, v in results.items() if v.confidence > 0.9}
        if high_conf_results:
            rules.append(('high_confidence', high_conf_results))
        
        # Правило 2: Если ML-анализатор уверен, и есть поддержка от AST, доверяем
        ml_result = results.get('ml_predictor')
        ast_result = results.get('ast_advanced')
        if (ml_result and ast_result and 
            ml_result.confidence > 0.8 and ast_result.confidence > 0.7 and
            ml_result.complexity_class == ast_result.complexity_class):
            rules.append(('ml_ast_agreement', {
                'ml_predictor': ml_result,
                'ast_advanced': ast_result
            }))
        
        # Правило 3: Если runtime анализ показывает сложность, и есть статическое подтверждение
        runtime_result = results.get('runtime_profiler')
        if runtime_result and runtime_result.confidence > 0.8:
            supporting_static = [
                v for k, v in results.items() 
                if k.startswith(('ast_', 'cfg_', 'metrics_')) and 
                v.complexity_class == runtime_result.complexity_class
            ]
            if supporting_static:
                rules.append(('runtime_with_static_support', {
                    'runtime': runtime_result,
                    'static_support': supporting_static
                }))
        
        # Применяем правила по приоритету
        if rules:
            best_rule_name, best_rule_results = rules[0]  # Берем первое правило
            
            if isinstance(best_rule_results, dict) and len(best_rule_results) == 1:
                # Один результат
                result = list(best_rule_results.values())[0]
                final_complexity = result.complexity_class
                final_confidence = result.confidence * 0.95
            else:
                # Несколько результатов - голосование
                if isinstance(best_rule_results, dict):
                    rule_results = best_rule_results.values()
                else:
                    rule_results = best_rule_results
                
                complexity_votes = Counter(r.complexity_class for r in rule_results)
                final_complexity = complexity_votes.most_common(1)[0][0]
                final_confidence = sum(r.confidence for r in rule_results) / len(rule_results)
            
            return ComplexityResult(
                complexity_class=final_complexity,
                confidence=final_confidence,
                analyzer_name="expert_system_resolver",
                debug_info={
                    'resolution_strategy': 'expert_system',
                    'applied_rule': best_rule_name,
                    'conflict_severity': conflict_analysis['severity'],
                    'available_rules': len(rules)
                }
            )
        
        # Если правила не сработали, используем консервативный подход
        return self._conservative_resolution(results)
    
    def _merge_consistent_results(self, results: Dict[str, ComplexityResult]) -> ComplexityResult:
        """Объединение согласованных результатов"""
        # Все результаты имеют одинаковый класс сложности
        complexity_class = list(results.values())[0].complexity_class
        
        # Средняя уверенность (взвешенная)
        total_confidence = sum(r.confidence for r in results.values())
        avg_confidence = total_confidence / len(results)
        
        # Объединяем все дополнительные данные
        merged_ast_features = {}
        merged_runtime_data = {}
        merged_ml_predictions = {}
        
        for result in results.values():
            merged_ast_features.update(result.ast_features)
            merged_runtime_data.update(result.runtime_data)
            merged_ml_predictions.update(result.ml_predictions)
        
        return ComplexityResult(
            complexity_class=complexity_class,
            confidence=min(avg_confidence * 1.1, 1.0),  # Повышаем уверенность за согласованность
            analyzer_name="consistent_merger",
            ast_features=merged_ast_features,
            runtime_data=merged_runtime_data,
            ml_predictions=merged_ml_predictions,
            debug_info={
                'resolution_strategy': 'consistent_merge',
                'participating_analyzers': list(results.keys()),
                'consensus_achieved': True
            }
        )

class HybridComplexityAnalyzer(BaseComplexityAnalyzer):
    """Гибридный анализатор сложности"""
    
    def __init__(self):
        super().__init__("hybrid_ensemble", AnalyzerType.HYBRID_ENSEMBLE)
        
        # Реестр анализаторов
        self.analyzers: Dict[str, BaseComplexityAnalyzer] = {}
        
        # Компоненты ансамбля
        self.weighting_strategy = WeightingStrategy()
        self.voting_strategy = VotingStrategy()
        self.conflict_resolver = ConflictResolver()
        
        # Конфигурация
        self.enabled_analyzers: List[str] = []
        self.weights: Dict[str, float] = {}
        self.historical_performance: Dict[str, float] = {}
        
        # Статистика
        self.analysis_history: List[Dict[str, Any]] = []
    
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Инициализация гибридного анализатора"""
        if not super().initialize(config):
            return False
        
        # Загрузка конфигурации
        config = config or {}
        self.enabled_analyzers = config.get('enabled_analyzers', [
            'ast_advanced', 'runtime_profiler', 'cfg_analyzer', 
            'ml_predictor', 'dynamic_tracer'
        ])
        
        # Инициализация анализаторов
        from complexity_analyzers.ast_analyzers.advanced_analyzer import AdvancedASTAnalyzer
        from complexity_analyzers.runtime.profiler import RuntimeProfiler
        from complexity_analyzers.cfg.analyzer import CFGComplexityAnalyzer
        from complexity_analyzers.ml.predictor import MLComplexityPredictor
        from complexity_analyzers.dynamic.tracer import DynamicComplexityTracer
        
        analyzer_classes = {
            'ast_advanced': AdvancedASTAnalyzer,
            'runtime_profiler': RuntimeProfiler,
            'cfg_analyzer': CFGComplexityAnalyzer,
            'ml_predictor': MLComplexityPredictor,
            'dynamic_tracer': DynamicComplexityTracer
        }
        
        # Создание экземпляров анализаторов
        for analyzer_name in self.enabled_analyzers:
            if analyzer_name in analyzer_classes:
                try:
                    analyzer = analyzer_classes[analyzer_name]()
                    if analyzer.is_available() and analyzer.initialize():
                        self.analyzers[analyzer_name] = analyzer
                    else:
                        print(f"Warning: {analyzer_name} not available or failed to initialize")
                except Exception as e:
                    print(f"Error initializing {analyzer_name}: {e}")
        
        # Инициализация весов
        self.weights = config.get('weights', {})
        if not self.weights:
            self.weights = self.weighting_strategy.uniform_weights(list(self.analyzers.keys()))
        
        # Загрузка исторической производительности
        self.historical_performance = config.get('historical_performance', {})
        
        return len(self.analyzers) > 0
    
    def is_available(self) -> bool:
        """Проверка доступности"""
        return len(self.analyzers) > 0
    
    def analyze(self, context: AnalysisContext) -> ComplexityResult:
        """Гибридный анализ сложности"""
        if not self.analyzers:
            return ComplexityResult(
                complexity_class=ComplexityClass.UNKNOWN,
                confidence=0.0,
                analyzer_name=self.name,
                errors=["No analyzers available"]
            )
        
        start_time = time.time()
        
        try:
            # Параллельный запуск анализаторов
            individual_results = self._run_analyzers_parallel(context)
            
            if not individual_results:
                return ComplexityResult(
                    complexity_class=ComplexityClass.UNKNOWN,
                    confidence=0.0,
                    analyzer_name=self.name,
                    errors=["All analyzers failed"]
                )
            
            # Разрешение конфликтов
            final_result = self.conflict_resolver.resolve_conflicts(individual_results)
            
            # Обновление метаданных
            final_result.analyzer_name = self.name
            final_result.analysis_time = time.time() - start_time
            
            # Добавление отладочной информации
            final_result.debug_info.update({
                'hybrid_analysis': {
                    'participating_analyzers': list(individual_results.keys()),
                    'successful_analyzers': len([r for r in individual_results.values() if r.is_valid()]),
                    'failed_analyzers': len([r for r in individual_results.values() if not r.is_valid()]),
                    'weights_used': self.weights,
                    'analysis_time_breakdown': {k: v.analysis_time for k, v in individual_results.items()}
                }
            })
            
            # Сохранение в историю для обучения
            self._update_analysis_history(context, individual_results, final_result)
            
            return final_result
            
        except Exception as e:
            return ComplexityResult(
                complexity_class=ComplexityClass.UNKNOWN,
                confidence=0.0,
                analyzer_name=self.name,
                analysis_time=time.time() - start_time,
                errors=[f"Hybrid analysis error: {e}"]
            )
    
    def _run_analyzers_parallel(self, context: AnalysisContext) -> Dict[str, ComplexityResult]:
        """Параллельный запуск анализаторов"""
        import concurrent.futures
        import threading
        
        results = {}
        timeout = context.timeout or 30
        
        def run_analyzer(name: str, analyzer: BaseComplexityAnalyzer) -> Tuple[str, ComplexityResult]:
            """Запуск одного анализатора"""
            try:
                result = analyzer.analyze(context)
                return name, result
            except Exception as e:
                error_result = ComplexityResult(
                    complexity_class=ComplexityClass.UNKNOWN,
                    confidence=0.0,
                    analyzer_name=name,
                    errors=[f"Analyzer error: {e}"]
                )
                return name, error_result
        
        # Последовательный запуск (можно заменить на параллельный)
        for name, analyzer in self.analyzers.items():
            try:
                analyzer_name, result = run_analyzer(name, analyzer)
                results[analyzer_name] = result
            except Exception as e:
                results[name] = ComplexityResult(
                    complexity_class=ComplexityClass.UNKNOWN,
                    confidence=0.0,
                    analyzer_name=name,
                    errors=[f"Runner error: {e}"]
                )
        
        return results
    
    def _update_analysis_history(self, context: AnalysisContext, 
                                individual_results: Dict[str, ComplexityResult],
                                final_result: ComplexityResult):
        """Обновление истории анализов для обучения"""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'source_code_hash': hashlib.md5(context.source_code.encode()).hexdigest(),
            'individual_results': {k: v.to_dict() for k, v in individual_results.items()},
            'final_result': final_result.to_dict(),
            'context': {
                'language': context.language,
                'timeout': context.timeout,
                'debug_mode': context.debug_mode
            }
        }
        
        self.analysis_history.append(history_entry)
        
        # Ограничиваем размер истории
        if len(self.analysis_history) > 1000:
            self.analysis_history = self.analysis_history[-500:]
    
    def get_analyzer_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Статистика производительности анализаторов"""
        stats = {}
        
        for analyzer_name in self.analyzers.keys():
            analyzer_results = []
            
            for entry in self.analysis_history:
                if analyzer_name in entry['individual_results']:
                    result_data = entry['individual_results'][analyzer_name]
                    analyzer_results.append({
                        'confidence': result_data.get('confidence', 0),
                        'analysis_time': result_data.get('analysis_time', 0),
                        'errors': len(result_data.get('errors', [])),
                        'valid': result_data.get('complexity_class') != 'unknown'
                    })
            
            if analyzer_results:
                stats[analyzer_name] = {
                    'avg_confidence': np.mean([r['confidence'] for r in analyzer_results]),
                    'avg_analysis_time': np.mean([r['analysis_time'] for r in analyzer_results]),
                    'success_rate': np.mean([r['valid'] for r in analyzer_results]),
                    'error_rate': np.mean([r['errors'] > 0 for r in analyzer_results]),
                    'total_analyses': len(analyzer_results)
                }
        
        return stats
    
    def optimize_weights(self, validation_data: List[Dict[str, Any]] = None) -> Dict[str, float]:
        """Оптимизация весов анализаторов"""
        if not validation_data and not self.analysis_history:
            return self.weights
        
        # Используем историю для оптимизации весов
        performance_stats = self.get_analyzer_performance_stats()
        
        # Вычисляем новые веса на основе производительности
        new_weights = {}
        total_score = 0
        
        for analyzer_name, stats in performance_stats.items():
            # Комбинированный скор: уверенность * успешность / время
            score = (stats['avg_confidence'] * stats['success_rate']) / max(stats['avg_analysis_time'], 0.001)
            new_weights[analyzer_name] = score
            total_score += score
        
        # Нормализация весов
        if total_score > 0:
            new_weights = {k: v/total_score for k, v in new_weights.items()}
        
        self.weights.update(new_weights)
        return self.weights
    
    def add_analyzer(self, name: str, analyzer: BaseComplexityAnalyzer, weight: float = None) -> bool:
        """Добавление нового анализатора в ансамбль"""
        if not analyzer.is_available():
            return False
        
        try:
            if analyzer.initialize():
                self.analyzers[name] = analyzer
                
                # Добавляем вес
                if weight is not None:
                    self.weights[name] = weight
                else:
                    # Равномерно перераспределяем веса
                    uniform_weight = 1.0 / len(self.analyzers)
                    self.weights = {k: uniform_weight for k in self.analyzers.keys()}
                
                return True
        except Exception as e:
            print(f"Error adding analyzer {name}: {e}")
        
        return False
    
    def remove_analyzer(self, name: str) -> bool:
        """Удаление анализатора из ансамбля"""
        if name in self.analyzers:
            del self.analyzers[name]
            
            if name in self.weights:
                del self.weights[name]
                
                # Перенормализация весов
                if self.weights:
                    total_weight = sum(self.weights.values())
                    if total_weight > 0:
                        self.weights = {k: v/total_weight for k, v in self.weights.items()}
            
            return True
        return False
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Информация о составе ансамбля"""
        return {
            'total_analyzers': len(self.analyzers),
            'enabled_analyzers': list(self.analyzers.keys()),
            'weights': self.weights.copy(),
            'historical_performance': self.historical_performance.copy(),
            'analysis_history_size': len(self.analysis_history),
            'performance_stats': self.get_analyzer_performance_stats()
        }

class AdaptiveEnsemble(HybridComplexityAnalyzer):
    """Адаптивный ансамбль с автоматической настройкой"""
    
    def __init__(self):
        super().__init__()
        self.name = "adaptive_ensemble"
        self.adaptation_frequency = 10  # Частота адаптации весов
        self.min_history_size = 5  # Минимальный размер истории для адаптации
        self.adaptation_counter = 0
    
    def analyze(self, context: AnalysisContext) -> ComplexityResult:
        """Анализ с адаптивной настройкой"""
        # Адаптация весов если необходимо
        if self._should_adapt():
            self._adapt_ensemble()
        
        # Обычный анализ
        result = super().analyze(context)
        
        self.adaptation_counter += 1
        return result
    
    def _should_adapt(self) -> bool:
        """Проверка необходимости адаптации"""
        return (len(self.analysis_history) >= self.min_history_size and
                self.adaptation_counter % self.adaptation_frequency == 0)
    
    def _adapt_ensemble(self):
        """Адаптация ансамбля"""
        print("Adapting ensemble weights...")
        
        # Оптимизация весов
        old_weights = self.weights.copy()
        new_weights = self.optimize_weights()
        
        # Логирование изменений
        weight_changes = {
            k: new_weights.get(k, 0) - old_weights.get(k, 0)
            for k in set(list(new_weights.keys()) + list(old_weights.keys()))
        }
        
        print(f"Weight changes: {weight_changes}")
        
        # Можно добавить логику удаления плохо работающих анализаторов
        performance_stats = self.get_analyzer_performance_stats()
        
        for analyzer_name, stats in performance_stats.items():
            if stats['success_rate'] < 0.3 and stats['total_analyses'] > 10:
                print(f"Considering removal of underperforming analyzer: {analyzer_name}")
                # Здесь можно добавить логику удаления

class SpecializedEnsemble(HybridComplexityAnalyzer):
    """Специализированный ансамбль для определенных типов алгоритмов"""
    
    def __init__(self, algorithm_type: str = "general"):
        super().__init__()
        self.algorithm_type = algorithm_type
        self.name = f"specialized_ensemble_{algorithm_type}"
        
        # Специализированные веса для разных типов алгоритмов
        self.algorithm_weights = {
            "sorting": {
                'ast_advanced': 0.4,
                'runtime_profiler': 0.3,
                'cfg_analyzer': 0.2,
                'ml_predictor': 0.1
            },
            "searching": {
                'ast_advanced': 0.3,
                'runtime_profiler': 0.4,
                'ml_predictor': 0.2,
                'cfg_analyzer': 0.1
            },
            "recursive": {
                'dynamic_tracer': 0.4,
                'ast_advanced': 0.3,
                'ml_predictor': 0.2,
                'runtime_profiler': 0.1
            },
            "iterative": {
                'cfg_analyzer': 0.4,
                'ast_advanced': 0.3,
                'runtime_profiler': 0.2,
                'ml_predictor': 0.1
            },
            "general": {
                'ast_advanced': 0.25,
                'runtime_profiler': 0.25,
                'cfg_analyzer': 0.2,
                'ml_predictor': 0.2,
                'dynamic_tracer': 0.1
            }
        }
    
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Инициализация с учетом специализации"""
        if super().initialize(config):
            # Применяем специализированные веса
            if self.algorithm_type in self.algorithm_weights:
                specialized_weights = self.algorithm_weights[self.algorithm_type]
                
                # Оставляем только веса для доступных анализаторов
                self.weights = {
                    k: v for k, v in specialized_weights.items()
                    if k in self.analyzers
                }
                
                # Перенормализация
                total_weight = sum(self.weights.values())
                if total_weight > 0:
                    self.weights = {k: v/total_weight for k, v in self.weights.items()}
            
            return True
        return False
    
    def detect_algorithm_type(self, context: AnalysisContext) -> str:
        """Автоматическое определение типа алгоритма"""
        source_code = context.source_code.lower()
        
        # Простые эвристики
        if any(keyword in source_code for keyword in ['sort', 'bubble', 'merge', 'quick']):
            return "sorting"
        elif any(keyword in source_code for keyword in ['search', 'find', 'binary', 'linear']):
            return "searching"
        elif any(keyword in source_code for keyword in ['def ', 'return ']) and 'def ' in source_code:
            # Проверяем на рекурсию
            import re
            func_names = re.findall(r'def\s+(\w+)', source_code)
            for func_name in func_names:
                if func_name in source_code.replace(f'def {func_name}', ''):
                    return "recursive"
        elif any(keyword in source_code for keyword in ['for', 'while']):
            return "iterative"
        
        return "general"
    
    def analyze(self, context: AnalysisContext) -> ComplexityResult:
        """Анализ с автоматическим определением типа алгоритма"""
        # Определяем тип алгоритма
        detected_type = self.detect_algorithm_type(context)
        
        if detected_type != self.algorithm_type and detected_type in self.algorithm_weights:
            # Временно меняем веса для этого анализа
            original_weights = self.weights.copy()
            
            specialized_weights = self.algorithm_weights[detected_type]
            temp_weights = {
                k: v for k, v in specialized_weights.items()
                if k in self.analyzers
            }
            
            total_weight = sum(temp_weights.values())
            if total_weight > 0:
                self.weights = {k: v/total_weight for k, v in temp_weights.items()}
            
            # Выполняем анализ
            result = super().analyze(context)
            
            # Восстанавливаем веса
            self.weights = original_weights
            
            # Добавляем информацию о детекции
            result.debug_info['algorithm_type_detection'] = {
                'detected_type': detected_type,
                'original_type': self.algorithm_type,
                'weights_adapted': True
            }
            
            return result
        else:
            return super().analyze(context)

# Фабрика ансамблей
class EnsembleFactory:
    """Фабрика для создания различных типов ансамблей"""
    
    @staticmethod
    def create_hybrid_ensemble(config: Dict[str, Any] = None) -> HybridComplexityAnalyzer:
        """Создание стандартного гибридного ансамбля"""
        ensemble = HybridComplexityAnalyzer()
        ensemble.initialize(config)
        return ensemble
    
    @staticmethod
    def create_adaptive_ensemble(config: Dict[str, Any] = None) -> AdaptiveEnsemble:
        """Создание адаптивного ансамбля"""
        ensemble = AdaptiveEnsemble()
        ensemble.initialize(config)
        return ensemble
    
    @staticmethod
    def create_specialized_ensemble(algorithm_type: str, 
                                  config: Dict[str, Any] = None) -> SpecializedEnsemble:
        """Создание специализированного ансамбля"""
        ensemble = SpecializedEnsemble(algorithm_type)
        ensemble.initialize(config)
        return ensemble
    
    @staticmethod
    def create_lightweight_ensemble(config: Dict[str, Any] = None) -> HybridComplexityAnalyzer:
        """Создание облегченного ансамбля (только быстрые анализаторы)"""
        lightweight_config = config or {}
        lightweight_config['enabled_analyzers'] = ['ast_advanced', 'metrics_calculator']
        
        ensemble = HybridComplexityAnalyzer()
        ensemble.initialize(lightweight_config)
        return ensemble
    
    @staticmethod
    def create_comprehensive_ensemble(config: Dict[str, Any] = None) -> HybridComplexityAnalyzer:
        """Создание полного ансамбля (все доступные анализаторы)"""
        comprehensive_config = config or {}
        comprehensive_config['enabled_analyzers'] = [
            'ast_advanced', 'runtime_profiler', 'cfg_analyzer',
            'ml_predictor', 'dynamic_tracer', 'metrics_calculator',
            'tools_integration'
        ]
        
        ensemble = HybridComplexityAnalyzer()
        ensemble.initialize(comprehensive_config)
        return ensemble

