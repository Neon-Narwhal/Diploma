"""Loss функции для обучения"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss для мультиклассовой классификации с дисбалансом классов
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
        https://arxiv.org/abs/1708.02002
    
    Формула:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    где:
        p_t - predicted probability для истинного класса
        α_t - балансирующий вес для класса t
        γ - focusing parameter (обычно 2)
    
    Интуиция:
        - Когда образец легко классифицируется (p_t → 1), вклад в loss → 0
        - Когда образец сложный (p_t → 0), вклад в loss большой
        - Автоматически фокусируется на hard examples
    """
    
    def __init__(self, 
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        """
        Args:
            alpha: Tensor весов для каждого класса [num_classes]
                   Если None, используется uniform weighting
            gamma: Focusing parameter. Чем больше, тем сильнее down-weighting
                   Рекомендуется 2.0
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Focal loss value
        """
        # Вычисление cross-entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Вычисление p_t (predicted probability для истинного класса)
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Применение alpha весов если заданы
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = focal_weight * alpha_t
        
        # Итоговый loss
        focal_loss = focal_weight * ce_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss
    
    def extra_repr(self) -> str:
        """String representation для print(model)"""
        return f'gamma={self.gamma}, alpha={self.alpha is not None}'


def compute_class_weights(samples: list, 
                          complexity_type: str = 'time',
                          method: str = 'inverse_freq') -> torch.Tensor:
    """
    Вычисление весов классов для Focal Loss
    
    Args:
        samples: Список образцов с замапленными классами
        complexity_type: 'time' или 'space'
        method: 'inverse_freq' | 'effective_num'
            - inverse_freq: weight = total / (num_classes * class_count)
            - effective_num: weight = (1 - beta) / (1 - beta^n)
            
    Returns:
        Tensor весов [num_classes]
    """
    from collections import Counter
    
    field = f'{complexity_type}_complexity_mapped'
    class_counts = Counter(s[field] for s in samples if s.get(field))
    
    # Сортировка для consistent ordering
    sorted_classes = sorted(class_counts.keys())
    counts = torch.tensor([class_counts[cls] for cls in sorted_classes], dtype=torch.float32)
    
    if method == 'inverse_freq':
        # Inverse frequency: weight = N / (K * n_k)
        total = counts.sum()
        num_classes = len(counts)
        weights = total / (num_classes * counts)
        
    elif method == 'effective_num':
        # Effective number of samples: (1 - beta) / (1 - beta^n)
        # beta обычно 0.999 или 0.9999
        beta = 0.9999
        effective_num = (1.0 - torch.pow(beta, counts)) / (1.0 - beta)
        weights = 1.0 / effective_num
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Нормализация весов
    weights = weights / weights.sum() * len(weights)
    
    return weights


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss с автоматическим вычислением весов классов
    
    Wrapper для удобства использования
    """
    
    def __init__(self,
                 samples: list,
                 complexity_type: str = 'time',
                 gamma: float = 2.0,
                 weighting_method: str = 'inverse_freq'):
        """
        Args:
            samples: Обучающая выборка для вычисления весов
            complexity_type: 'time' или 'space'
            gamma: Focusing parameter
            weighting_method: Метод вычисления весов
        """
        super().__init__()
        
        # Вычисление весов
        alpha = compute_class_weights(samples, complexity_type, weighting_method)
        
        # Создание Focal Loss
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.complexity_type = complexity_type
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal_loss(inputs, targets)
