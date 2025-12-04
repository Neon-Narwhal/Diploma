"""
Пайплайн обработки для AST анализа с передачей ground truth.
"""

from typing import List, Dict, Any, Optional
from shared.data_loader.dataset import Dataset
from shared.processing import BatchProcessor, ProcessingResult
from ast_analysis.core.analyzer_factory import ASTAnalyzerFactory
from ast_analysis.core.base_analyzer import BaseASTAnalyzer
from ast_analysis.core.result import ASTAnalysisResult


class ASTPipeline:
    """
    Пайплайн для AST анализа кода с предсказанием сложности.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Конфигурация эксперимента
        """
        self.config = config
        
        # Создание анализаторов
        analyzer_configs = config.get('analyzers', [])
        self.analyzers = ASTAnalyzerFactory.create_multiple(analyzer_configs)
        
        if not self.analyzers:
            raise ValueError("No analyzers created")
        
        print(f"Created {len(self.analyzers)} analyzers:")
        for analyzer in self.analyzers:
            print(f"  - {analyzer.name}")
        
        # Настройка обработки
        processing_config = config.get('processing', {})
        self.batch_processor = BatchProcessor(
            batch_size=processing_config.get('batch_size', 100),
            n_workers=processing_config.get('n_workers', 4),
            timeout_per_sample=processing_config.get('timeout_per_sample', 5),
            show_progress=True,
            skip_on_error=processing_config.get('skip_on_error', True)
        )
    
    def process(self, dataset: Dataset) -> Dict[str, List[Dict[str, Any]]]:
        """
        Обработка датасета через все анализаторы с сохранением ground truth.
        
        Args:
            dataset: Датасет с кодом и метками
        
        Returns:
            Словарь результатов для каждого сплита
        """
        all_results = {}
        
        # Обработка train
        print("\n" + "=" * 60)
        print("PROCESSING TRAIN SET")
        print("=" * 60)
        train_results = self._process_codes(
            dataset.train_codes, 
            dataset.train_labels,
            "train"
        )
        all_results['train'] = train_results
        
        # Обработка val
        print("\n" + "=" * 60)
        print("PROCESSING VAL SET")
        print("=" * 60)
        val_results = self._process_codes(
            dataset.val_codes,
            dataset.val_labels,
            "val"
        )
        all_results['val'] = val_results
        
        # Обработка test
        print("\n" + "=" * 60)
        print("PROCESSING TEST SET")
        print("=" * 60)
        test_results = self._process_codes(
            dataset.test_codes,
            dataset.test_labels,
            "test"
        )
        all_results['test'] = test_results
        
        return all_results
    
    def _process_codes(self, 
                      codes: List[str], 
                      labels: List[str],
                      split_name: str) -> List[Dict[str, Any]]:
        """
        Обработка списка кодов через все анализаторы с ground truth.
        
        Args:
            codes: Список кодов
            labels: Ground truth метки
            split_name: Имя сплита
        
        Returns:
            Список результатов с полем ground_truth
        """
        results = []
        
        for analyzer in self.analyzers:
            print(f"\nRunning analyzer: {analyzer.name}")
            
            # Обработка через BatchProcessor
            processing_results = self.batch_processor.process(
                items=codes,
                process_fn=analyzer.analyze,
                parallel=self.config.get('processing', {}).get('parallel', True)
            )
            
            # Преобразование ProcessingResult -> ASTAnalysisResult
            analyzer_results = []
            for pr in processing_results:
                if pr.success and isinstance(pr.data, ASTAnalysisResult):
                    analyzer_results.append(pr.data)
                else:
                    # Создаём результат с ошибкой
                    analyzer_results.append(ASTAnalysisResult(
                        success=False,
                        features={},
                        error=pr.error or "Unknown error",
                        analyzer_name=analyzer.name
                    ))
            
            # Сохраняем результаты этого анализатора
            if not results:
                # Первый анализатор - создаём структуру с ground truth
                results = [
                    {
                        'code_idx': i,
                        'ground_truth': labels[i] if i < len(labels) else None,
                        'analyzers': {}
                    }
                    for i in range(len(codes))
                ]
            
            # Добавляем результаты анализатора
            for i, result in enumerate(analyzer_results):
                results[i]['analyzers'][analyzer.name] = result.to_dict()
        
        return results
    
    def get_aggregated_features(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Агрегация признаков от всех анализаторов.
        
        Args:
            results: Результаты обработки
        
        Returns:
            Список агрегированных признаков
        """
        aggregated = []
        
        for result in results:
            features = {}
            
            # Собираем признаки от всех анализаторов
            for analyzer_name, analyzer_result in result.get('analyzers', {}).items():
                if analyzer_result.get('success'):
                    analyzer_features = analyzer_result.get('features', {})
                    # Добавляем префикс имени анализатора
                    for feat_name, feat_value in analyzer_features.items():
                        features[f"{analyzer_name}_{feat_name}"] = feat_value
            
            aggregated.append(features)
        
        return aggregated
