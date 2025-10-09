# =============================================================================
# models/base_model.py
# Абстрактный базовый класс для всех моделей (гарантирует интерфейс)
# =============================================================================

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Tuple, Optional


class BaseLanguageModel(nn.Module, ABC):
    """
    Абстрактный базовый класс для всех языковых моделей.
    Определяет обязательный интерфейс, который должен быть у каждой модели.
    Trainer работает только через этот интерфейс.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass модели.
        
        Args:
            idx: Input token IDs [batch_size, seq_len]
            targets: Target token IDs [batch_size, seq_len] (опционально)
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            loss: scalar tensor (если targets предоставлены)
        """
        pass
    
    @abstractmethod
    def generate(
        self, 
        idx: torch.Tensor, 
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Генерация новых токенов.
        
        Args:
            idx: Начальная последовательность [batch_size, seq_len]
            max_new_tokens: Количество токенов для генерации
            temperature: Температура сэмплирования
            top_k: Top-K фильтрация (опционально)
        
        Returns:
            Сгенерированная последовательность [batch_size, seq_len + max_new_tokens]
        """
        pass
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Подсчет количества параметров в модели.
        
        Args:
            non_embedding: Если True, не считает embedding параметры
        
        Returns:
            Количество параметров
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self, 'token_embedding_table'):
            n_params -= self.token_embedding_table.weight.numel()
        return n_params
    
    def configure_optimizers(self, config):
        """
        Настройка оптимизатора (может быть переопределена в подклассах).
        По умолчанию возвращает AdamW.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )
        return optimizer
    
    def get_model_specific_metrics(self) -> dict:
        """
        Возвращает model-specific метрики для логирования.
        Может быть переопределена в подклассах.
        """
        return {}