# =============================================================================
# utils/configs/gpt_config.py
# Конфигурация для GPT модели - только архитектурные параметры
# =============================================================================

from dataclasses import dataclass
from typing import Optional
from utils.configs.base_config import BaseTrainingConfig


@dataclass
class GPTConfig(BaseTrainingConfig):
    """
    Конфигурация для GPT модели.
    Наследует ВСЕ параметры из BaseTrainingConfig и добавляет только архитектуру.
    """
    
    # ============================================================================
    # АРХИТЕКТУРА GPT (единственное что добавляем)
    # ============================================================================
    
    n_embd: int = 256
    """Размерность эмбеддингов. Должна быть кратна n_head."""
    
    n_layer: int = 6
    """Количество transformer layers."""
    
    n_head: int = 4
    """Количество attention heads. n_embd должно быть кратно n_head."""
    
    n_inner: Optional[int] = None
    """FFN hidden size. Если None, автоматически = 4 * n_embd."""
    
    # ============================================================================
    # DROPOUT (GPT-специфичные, переопределяем базовый dropout)
    # ============================================================================
    
    attention_dropout: float = 0.1
    """Dropout в attention слое."""
    
    residual_dropout: float = 0.1
    """Dropout в residual connections."""
    
    embedding_dropout: float = 0.1
    """Dropout в эмбеддингах."""
    
    # ============================================================================
    # ИНИЦИАЛИЗАЦИЯ И ПРОЧЕЕ
    # ============================================================================
    
    initializer_range: float = 0.02
    """Std для инициализации весов (Normal distribution)."""
    
    layer_norm_epsilon: float = 1e-5
    """Epsilon для layer normalization."""
    
    use_bias: bool = True
    """Использовать bias в Linear слоях."""
    
    activation_function: str = "gelu"
    """Activation функция: 'gelu', 'relu', 'silu'."""
    
    # ============================================================================
    # OVERRIDE БАЗОВЫХ ЗНАЧЕНИЙ (если нужно для GPT)
    # ============================================================================
    
    experiment_name: str = "gpt_bigobench_training"
    model_name: str = "gpt_model"
    
    def __post_init__(self):
        """Дополнительная валидация для GPT."""
        
        # Вызываем родительский __post_init__
        super().__post_init__()
        
        # Автоматически устанавливаем n_inner
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd
        
        # Валидация: n_embd кратно n_head
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) должно быть кратно n_head ({self.n_head}). "
                f"Текущее: {self.n_embd / self.n_head}"
            )
        
        # Для совместимости со старым кодом
        self.model_type = 'gpt'
        
        # Оценка параметров
        self._estimate_parameters()
    
    def _estimate_parameters(self):
        """Оценка количества параметров модели."""
        token_emb = self.vocab_size * self.n_embd
        pos_emb = self.block_size * self.n_embd
        
        layer_params = (
            2 * self.n_embd +                  # LN 1
            4 * self.n_embd * self.n_embd +    # Attention
            2 * self.n_embd +                  # LN 2
            2 * self.n_embd * self.n_inner     # FFN
        )
        
        final_ln = 2 * self.n_embd
        
        total = token_emb + pos_emb + (layer_params * self.n_layer) + final_ln
        
        self.estimated_parameters = total
        self.estimated_parameters_millions = total / 1e6
    
    def get_head_dim(self) -> int:
        """Размерность каждого attention head."""
        return self.n_embd // self.n_head
