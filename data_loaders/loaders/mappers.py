"""
DEPRECATED: Этот модуль больше не используется в основном pipeline.

Маппинг теперь происходит в complexity_analyzers/processors.py через
функцию normalize_complexity(), которая сохраняет точность выражений.

Этот код оставлен только для:
1. Обучения ML-моделей на упрощенных классах (если требуется)
2. Обратной совместимости со старым кодом

НЕ используй в production анализе!
"""

from typing import Dict, Optional, Set
import logging

logger = logging.getLogger(__name__)


class ComplexityMapper:
    """
    Маппинг fine-grained классов сложности на базовые категории
    
    Теоретическое обоснование:
    - O(n+m) ≈ O(n) при m << n (доминирование в Big-O нотации)
    - O(nlogn+m) ≈ O(nlogn) при n >> m
    - Классы в одной группе асимптотически эквивалентны
    """
    
    # Базовые классы временной сложности
    TIME_MAPPING = {
        # Constant - O(1)
        'O(1)': 'constant',
        
        # Logarithmic - O(logn)
        'O(logn)': 'logarithmic',
        'O(logn*logm)': 'logarithmic',
        'O(logn*logm*logk)': 'logarithmic',
        'O(logn*logm*logk*logl)': 'logarithmic',
        
        # Linear - O(n) — ТОЛЬКО одномерные
        'O(n)': 'linear',

        # Bivariate - O(n*m), O(n+m) — двумерные линейные
        'O(n+m)': 'bivariate',
        'O(n+m+k)': 'bivariate',
        'O(n*m)': 'bivariate',  # НЕ linear
        'O(n+m**2)': 'bivariate',  # двумерная структура
        'O(n**2+m)': 'bivariate',  # квадратичная по n, но двумерная

        
        # Linearithmic - O(nlogn)
        'O(nlogn)': 'linearithmic',
        'O(n+m)log(n+m)': 'linearithmic',
        'O(nlogn+m)': 'linearithmic',
        'O(nlogn+mlogm)': 'linearithmic',
        'O(nlogn*m)': 'linearithmic',  # при m=O(1)
        'O(nlogn**2)': 'linearithmic',
        'O(n+mlogm)': 'linearithmic',
        'O(nlogn+m**2)': 'linearithmic',
        
        # Quadratic - O(n²)
        'O(n**2)': 'quadratic',
        'O(n**2+m)': 'quadratic',
        'O(n**2+m**2)': 'quadratic',
        'O(n**2+m**2+k**2)': 'quadratic',
        'O(n**2*m)': 'quadratic',  # при m=O(1)
        'O(n**2+mlogm)': 'quadratic',
        
        # Polynomial (higher than quadratic)
        'O(n**3)': 'polynomial',
        'O(n**2*m**2)': 'polynomial',
        'O(n*m**2)': 'polynomial',  # при m=O(n)
        'O(n**2*mlogm)': 'polynomial',
    }
    
    # Базовые классы пространственной сложности
    SPACE_MAPPING = {
        # Constant - O(1)
        'O(1)': 'constant',
        
        # Logarithmic - O(logn)
        'O(logn)': 'logarithmic',
        'O(logn*logm)': 'logarithmic',
        'O(logn*logm*logk)': 'logarithmic',
        'O(logn*logm*logk*logl)': 'logarithmic',
        
        # Linear - O(n)
        'O(n)': 'linear',
        'O(n+m)': 'linear',  # для space O(n+m) можно оставить linear
        'O(n+m+k)': 'linear',
        'O(nlogn)': 'linear',  # space обычно O(n) для sorting
        'O(n+m)log(n+m)': 'linear',

        
        # Product - O(n*m)
        'O(n*m)': 'product',
        'O(n*m*k)': 'product',
        
        # Quadratic - O(n²)
        'O(n**2)': 'quadratic',
        'O(n**2+m)': 'quadratic',
        'O(n**2+m**2)': 'quadratic',
        'O(n**2*m**2)': 'quadratic',
    }
    
    def __init__(self, 
                 time_mapping: Optional[Dict[str, str]] = None,
                 space_mapping: Optional[Dict[str, str]] = None):
        """
        Args:
            time_mapping: Кастомный маппинг для временной сложности
            space_mapping: Кастомный маппинг для пространственной сложности
        """
        self.time_mapping = time_mapping or self.TIME_MAPPING
        self.space_mapping = space_mapping or self.SPACE_MAPPING
    
    def map_time_complexity(self, complexity: str) -> Optional[str]:
        """
        Маппинг временной сложности
        
        Args:
            complexity: Fine-grained класс (например, 'O(n+m)')
            
        Returns:
            Базовый класс (например, 'linear') или None если не найден
        """
        return self.time_mapping.get(complexity)
    
    def map_space_complexity(self, complexity: str) -> Optional[str]:
        """Маппинг пространственной сложности"""
        return self.space_mapping.get(complexity)
    
    def map_sample(self, sample: Dict, in_place: bool = False) -> Dict:
        """
        Применение маппинга к одному образцу
        
        Args:
            sample: Образец с полями time_complexity, space_complexity
            in_place: Модифицировать исходный dict или создать копию
            
        Returns:
            Образец с дополнительными полями *_mapped и *_original
        """
        if not in_place:
            sample = sample.copy()
        
        # Временная сложность
        time_orig = sample.get('time_complexity')
        if time_orig:
            time_mapped = self.map_time_complexity(time_orig)
            sample['time_complexity_mapped'] = time_mapped
            sample['time_complexity_original'] = time_orig
        
        # Пространственная сложность
        space_orig = sample.get('space_complexity')
        if space_orig:
            space_mapped = self.map_space_complexity(space_orig)
            sample['space_complexity_mapped'] = space_mapped
            sample['space_complexity_original'] = space_orig
        
        return sample
    
    def map_dataset(self, samples: list) -> list:
        """
        Применение маппинга ко всему датасету
        
        Args:
            samples: Список образцов
            
        Returns:
            Список образцов с маппингом (только те, что успешно замаплены)
        """
        mapped_samples = []
        skipped_time = 0
        skipped_space = 0
        
        for sample in samples:
            mapped = self.map_sample(sample, in_place=False)
            
            # Фильтрация: сохраняем только если оба маппинга успешны
            if mapped.get('time_complexity_mapped') is None:
                skipped_time += 1
                continue
            
            if mapped.get('space_complexity_mapped') is None:
                skipped_space += 1
                continue
            
            mapped_samples.append(mapped)
        
        logger.info(f"Маппинг завершен:")
        logger.info(f"  Исходных образцов: {len(samples)}")
        logger.info(f"  Замапленных: {len(mapped_samples)}")
        logger.info(f"  Пропущено (time): {skipped_time}")
        logger.info(f"  Пропущено (space): {skipped_space}")
        
        return mapped_samples
    
    def get_base_classes(self, complexity_type: str = 'time') -> Set[str]:
        """
        Получение множества базовых классов
        
        Args:
            complexity_type: 'time' или 'space'
            
        Returns:
            Set базовых классов
        """
        if complexity_type == 'time':
            return set(self.time_mapping.values())
        elif complexity_type == 'space':
            return set(self.space_mapping.values())
        else:
            raise ValueError(f"Unknown complexity_type: {complexity_type}")
    
    def get_mapping_statistics(self, samples: list) -> Dict:
        """
        Статистика маппинга
        
        Args:
            samples: Образцы (до или после маппинга)
            
        Returns:
            Dict со статистикой
        """
        from collections import Counter
        
        # Оригинальные классы
        time_orig = Counter(s.get('time_complexity_original') or s.get('time_complexity') 
                           for s in samples)
        space_orig = Counter(s.get('space_complexity_original') or s.get('space_complexity')
                            for s in samples)
        
        # Замапленные классы
        time_mapped = Counter(s.get('time_complexity_mapped') for s in samples 
                             if s.get('time_complexity_mapped'))
        space_mapped = Counter(s.get('space_complexity_mapped') for s in samples
                              if s.get('space_complexity_mapped'))
        
        return {
            'original': {
                'time_classes': len(time_orig),
                'space_classes': len(space_orig),
                'time_distribution': dict(time_orig),
                'space_distribution': dict(space_orig),
            },
            'mapped': {
                'time_classes': len(time_mapped),
                'space_classes': len(space_mapped),
                'time_distribution': dict(time_mapped),
                'space_distribution': dict(space_mapped),
            }
        }


def create_label_encoder(samples: list, complexity_type: str = 'time') -> Dict:
    """
    Создание энкодера label → int для PyTorch
    
    Args:
        samples: Образцы с замапленными классами
        complexity_type: 'time' или 'space'
        
    Returns:
        Dict {label: int_id}
    """
    field = f'{complexity_type}_complexity_mapped'
    unique_labels = sorted(set(s[field] for s in samples if s.get(field)))
    
    return {label: idx for idx, label in enumerate(unique_labels)}
