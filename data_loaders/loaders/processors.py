"""Обработчики и фильтры для датасетов (БЕЗ маппинга)"""
from typing import Dict, List, Set, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class ComplexityClassFilter:
    """Фильтрация по классам сложности"""
    
    def __init__(self, 
                 min_samples: int = 100,
                 target_time_classes: Optional[Set[str]] = None,
                 target_space_classes: Optional[Set[str]] = None):
        self.min_samples = min_samples
        self.target_time_classes = target_time_classes
        self.target_space_classes = target_space_classes
    
    def analyze_distribution(self, samples: List[Dict]) -> tuple:
        """Анализ распределения классов"""
        time_dist = Counter()
        space_dist = Counter()
        
        for sample in samples:
            time_cls = sample.get('time_complexity')
            space_cls = sample.get('space_complexity')
            
            if time_cls:
                time_dist[time_cls] += 1
            if space_cls:
                space_dist[space_cls] += 1
        
        return time_dist, space_dist
    
    def auto_select_classes(self, distribution: Counter) -> Set[str]:
        """Автоматический выбор классов по threshold"""
        return {
            cls for cls, count in distribution.items()
            if count >= self.min_samples and cls is not None
        }
    
    def filter(self, samples: List[Dict]) -> tuple:
        """Фильтрация образцов"""
        if self.target_time_classes is None or self.target_space_classes is None:
            time_dist, space_dist = self.analyze_distribution(samples)
            
            if self.target_time_classes is None:
                self.target_time_classes = self.auto_select_classes(time_dist)
            
            if self.target_space_classes is None:
                self.target_space_classes = self.auto_select_classes(space_dist)
        
        filtered_samples = []
        stats = {
            'total': len(samples),
            'none_time': 0,
            'none_space': 0,
            'filtered_time': 0,
            'filtered_space': 0,
            'kept': 0
        }
        
        for sample in samples:
            time_cls = sample.get('time_complexity')
            space_cls = sample.get('space_complexity')
            
            if time_cls is None:
                stats['none_time'] += 1
                continue
            
            if space_cls is None:
                stats['none_space'] += 1
                continue
            
            if time_cls not in self.target_time_classes:
                stats['filtered_time'] += 1
                continue
            
            if space_cls not in self.target_space_classes:
                stats['filtered_space'] += 1
                continue
            
            filtered_samples.append(sample)
            stats['kept'] += 1
        
        return filtered_samples, stats


class DatasetJoiner:
    """Джойн нескольких источников данных"""
    
    @staticmethod
    def join_complexity_and_solutions(
        complexity_dataset,
        solutions_map: Dict[str, Dict]
    ) -> List[Dict]:
        """Джойн меток сложности и кода решений"""
        samples = []
        missing_count = 0
        
        for item in complexity_dataset:
            solution_id = item['solution_id']
            
            if solution_id not in solutions_map:
                missing_count += 1
                continue
            
            solution_data = solutions_map[solution_id]
            
            sample = {
                'solution_id': solution_id,
                'problem_id': item['problem_id'],
                'problem_name': item.get('problem_name', solution_data['problem_name']),
                'code': solution_data['code'],
                'time_complexity': item.get('time_complexity'),
                'space_complexity': item.get('space_complexity'),
                'time_curve_coefficient': item.get('time_curve_coefficient'),
                'space_curve_coefficient': item.get('space_curve_coefficient')
            }
            
            samples.append(sample)
        
        if missing_count > 0:
            logger.warning(f"Отсутствует код для {missing_count} решений")
        
        return samples


class DataValidator:
    """Валидация данных"""
    
    @staticmethod
    def validate_sample(sample: Dict) -> tuple:
        """Валидация одного образца"""
        errors = []
        
        required_fields = ['solution_id', 'problem_id', 'code', 
                          'time_complexity', 'space_complexity']
        
        for field in required_fields:
            if field not in sample or sample[field] is None:
                errors.append(f"Отсутствует {field}")
        
        if 'code' in sample:
            code = sample['code']
            if not isinstance(code, str):
                errors.append("code должен быть строкой")
            elif len(code) == 0:
                errors.append("code пустой")
            elif len(code) > 100000:
                errors.append("code слишком длинный (>100k символов)")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_dataset(samples: List[Dict]) -> tuple:
        """Валидация датасета"""
        valid_samples = []
        stats = {
            'total': len(samples),
            'valid': 0,
            'invalid': 0,
            'errors': Counter()
        }
        
        for sample in samples:
            is_valid, errors = DataValidator.validate_sample(sample)
            
            if is_valid:
                valid_samples.append(sample)
                stats['valid'] += 1
            else:
                stats['invalid'] += 1
                for error in errors:
                    stats['errors'][error] += 1
        
        return valid_samples, stats
