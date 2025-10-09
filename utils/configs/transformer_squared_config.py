# =============================================================================
# utils/configs/transformer_squared_config.py
# ПОЛНАЯ конфигурация для Transformer² - все поля для совместимости с моделью
# =============================================================================

from dataclasses import dataclass, field
from typing import Optional, List
import torch
from utils.configs.base_config import BaseTrainingConfig


@dataclass
class TransformerSquaredConfig(BaseTrainingConfig):
    """
    ПОЛНАЯ конфигурация для Transformer² модели.
    Включает ВСЕ поля необходимые для модели.
    """
    
    # ============================================================================
    # БАЗОВАЯ АРХИТЕКТУРА
    # ============================================================================
    
    n_embd: int = 256
    """Размерность эмбеддингов."""
    
    n_layer: int = 6
    """Количество transformer layers."""
    
    n_head: int = 4
    """Количество attention heads."""
    
    n_inner: Optional[int] = None
    """FFN hidden size. Если None, = 4 * n_embd."""

    ffn_expansion_factor: int = 4
    """Множитель для FFN размера (hidden_dim = n_embd * ffn_expansion_factor).
    Стандартное значение: 4 (как в оригинальном Transformer)."""
    
    # ============================================================================
    # SVF (Singular Value Fine-tuning) ПАРАМЕТРЫ
    # ============================================================================
    
    use_svf: bool = True
    """Включить SVF механизм."""
    
    svf_rank: int = 8
    """Фиксированный ранг SVF (если используется)."""
    
    svf_rank_ratio: float = 0.25
    """Отношение rank к размеру слоя (rank = min(in, out) * ratio).
    Используется если svf_rank не задан явно."""
    
    svf_alpha: float = 16.0
    """Scaling фактор для SVF обновлений."""
    
    svf_dropout: float = 0.1
    """Dropout для SVF."""
    
    svf_bias: bool = False
    """Использовать bias в SVF слоях."""
    
    apply_svf_to: List[str] = field(default_factory=lambda: ['q', 'v'])
    """К каким матрицам применять SVF: 'q', 'k', 'v', 'o', 'ffn'."""
    
    svf_initializer_range: float = 0.01
    """Std для инициализации SVF весов."""
    
    # ============================================================================
    # EXPERT VECTORS
    # ============================================================================
    
    num_experts: int = 4
    """Количество task-specific expert vectors."""
    
    expert_dim: int = 64
    """Размерность каждого expert vector."""
    
    expert_dropout: float = 0.1
    """Dropout для expert vectors."""

    expert_vector_init_std: float = 0.01 
    """Standard deviation для инициализации expert vectors.
    Обычно меньше чем для основных весов (0.01 vs 0.02)."""
    
    # ============================================================================
    # ADAPTATION STRATEGY
    # ============================================================================
    
    adaptation_strategy: str = 'mixture'
    """Стратегия адаптации: 'prompt', 'classifier', 'mixture'."""
    
    mixture_temperature: float = 1.0
    """Температура для softmax при mixture strategy."""
    
    enable_expert_composition: bool = True
    """Позволяет комбинировать expert vectors."""
    
    composition_dropout: float = 0.05
    """Dropout при композиции experts."""

    enable_task_detector: bool = False  # 🆕 ДОБАВЬТЕ ЭТО
    """Включить автоматический task detector для выбора expert vectors.
    Если True, модель сама определяет тип задачи и выбирает подходящих экспертов.
    Если False, используется adaptation_strategy."""

    task_detector_hidden_dim: int = 128
    """Размер скрытого слоя в task detector."""
    
    # ============================================================================
    # ROTARY POSITION EMBEDDING (RoPE)
    # ============================================================================
    
    use_rope: bool = True
    """Использовать RoPE вместо learned positional embeddings."""
    
    rope_theta: float = 10000.0
    """Base theta для RoPE."""
    
    rope_scaling: Optional[float] = None
    """Масштабирование RoPE для длинных контекстов."""
    
    # ============================================================================
    # ADAPTIVE ATTENTION
    # ============================================================================
    
    use_adaptive_attention: bool = True
    """Использовать adaptive attention span."""
    
    adaptive_span_enabled: bool = False
    """Полный adaptive span механизм."""
    
    max_adaptive_span: int = 1024
    """Максимальный attention span."""
    
    min_adaptive_span: int = 32
    """Минимальный attention span."""
    
    # ============================================================================
    # DROPOUT ПАРАМЕТРЫ
    # ============================================================================
    
    attention_dropout: float = 0.1
    """Dropout в attention."""
    
    residual_dropout: float = 0.1
    """Dropout в residual connections."""
    
    embedding_dropout: float = 0.1
    """Dropout в embeddings."""
    
    hidden_dropout: float = 0.1
    """Dropout в hidden layers."""
    
    # ============================================================================
    # BIAS ПАРАМЕТРЫ
    # ============================================================================
    
    use_bias: bool = True
    """Глобальный параметр для bias (используется если specific не задан)."""
    
    attention_bias: bool = False
    """Bias в attention проекциях."""
    
    mlp_bias: bool = False
    """Bias в MLP слоях."""
    
    embedding_bias: bool = False
    """Bias в embeddings."""
    
    # ============================================================================
    # ИНИЦИАЛИЗАЦИЯ
    # ============================================================================
    
    initializer_range: float = 0.02
    """Std для инициализации основных весов."""

    weight_init_std: float = 0.02 
    """Standard deviation для инициализации весов (используется в _init_weights)."""

    embedding_init_std: float = 0.02  
    """Standard deviation для embeddings инициализации."""

    layernorm_init_std: float = 1.0  
    """Standard deviation для LayerNorm весов (обычно 1.0)."""
    
    # ============================================================================
    # LAYER NORMALIZATION
    # ============================================================================
    
    layer_norm_epsilon: float = 1e-5
    """Epsilon для layer normalization."""
    
    layer_norm_type: str = 'layernorm'
    """Тип normalization: 'layernorm', 'rmsnorm'."""

    use_pre_norm: bool = True  # 🆕 ДОБАВЬТЕ ЭТО!
    """Использовать Pre-LN (Layer Norm перед sublayer) вместо Post-LN.
    Pre-LN: более стабильное обучение (GPT-2, современные модели)
    Post-LN: оригинальный Transformer
    Рекомендуется: True (Pre-LN)"""
    
    # ============================================================================
    # ACTIVATION
    # ============================================================================
    
    activation_function: str = "gelu"
    """Activation функция: 'gelu', 'relu', 'silu', 'swish'."""
    
    # ============================================================================
    # DTYPE
    # ============================================================================
    
    model_dtype: str = 'float32'
    """Тип данных модели: 'float32', 'float16', 'bfloat16'."""
    
    # ============================================================================
    # MEMORY OPTIMIZATION
    # ============================================================================
    
    use_gradient_checkpointing: bool = False
    """Использовать gradient checkpointing для экономии памяти."""
    
    use_flash_attention: bool = False
    """Использовать Flash Attention (если доступно)."""
    
    # ============================================================================
    # REINFORCEMENT LEARNING (для обучения experts)
    # ============================================================================
    
    use_rl_for_experts: bool = False
    """Использовать RL для обучения expert vectors."""
    
    rl_reward_type: str = "task_accuracy"
    """Тип reward: 'task_accuracy', 'perplexity', 'custom'."""
    
    # ============================================================================
    # OVERRIDE БАЗОВЫХ ЗНАЧЕНИЙ
    # ============================================================================
    
    experiment_name: str = "transformer_squared_bigobench_training"
    model_name: str = "transformer_squared_model"
    learning_rate: float = 2e-4
    weight_decay: float = 0.02

    # ============================================================================
    # COMPLEXITY PREDICTION (для BigOBench задачи)
    # ============================================================================

    enable_complexity_head: bool = False 
    """Добавить дополнительный head для предсказания time/space complexity.
    Используется для multi-task learning на BigOBench датасете."""

    num_complexity_classes: int = 10
    """Количество классов сложности (например: O(1), O(log n), O(n), O(n log n), ...).
    Используется только если enable_complexity_head=True."""

    complexity_head_hidden_dim: int = 256
    """Размер скрытого слоя в complexity prediction head."""
    
    # ============================================================================
    # МЕТОДЫ
    # ============================================================================
    
    def __post_init__(self):
        """Валидация и автоматические вычисления."""
        
        # Вызываем родительский __post_init__
        super().__post_init__()
        
        # Автоматически устанавливаем n_inner
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd
        
        # Валидация: n_embd кратно n_head
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) должно быть кратно n_head ({self.n_head})"
            )
        
        # Валидация: expert_dim разумный
        if self.expert_dim > self.n_embd:
            import warnings
            warnings.warn(
                f"expert_dim ({self.expert_dim}) > n_embd ({self.n_embd}). "
                f"Рекомендуется: expert_dim <= n_embd/2"
            )
        
        # Валидация: apply_svf_to
        valid_targets = {'q', 'k', 'v', 'o', 'ffn'}
        for target in self.apply_svf_to:
            if target not in valid_targets:
                raise ValueError(
                    f"Неизвестная SVF target: {target}. Допустимые: {valid_targets}"
                )
        
        # Валидация: adaptive span
        if self.adaptive_span_enabled:
            if self.max_adaptive_span > self.block_size:
                raise ValueError(
                    f"max_adaptive_span ({self.max_adaptive_span}) > "
                    f"block_size ({self.block_size})"
                )
            if self.min_adaptive_span >= self.max_adaptive_span:
                raise ValueError(
                    "min_adaptive_span должен быть < max_adaptive_span"
                )
        
        # Если bias параметры не заданы явно, используем use_bias
        if not hasattr(self, '_bias_set'):
            if self.attention_bias is None:
                self.attention_bias = self.use_bias
            if self.mlp_bias is None:
                self.mlp_bias = self.use_bias
        
        # Для совместимости
        self.model_type = 'transformer_squared'
        self.num_expert_vectors = self.num_experts
        
        # Оценка параметров
        self._estimate_parameters()
    
    def _estimate_parameters(self):
        """Оценка количества параметров."""
        # Базовые параметры
        token_emb = self.vocab_size * self.n_embd
        pos_emb = 0 if self.use_rope else self.block_size * self.n_embd
        
        base_layer_params = (
            2 * self.n_embd +                  # LN 1
            4 * self.n_embd * self.n_embd +    # Attention
            2 * self.n_embd +                  # LN 2
            2 * self.n_embd * self.n_inner     # FFN
        )
        
        # SVF параметры
        svf_params_per_layer = 0
        if self.use_svf:
            num_svf_matrices = len(self.apply_svf_to)
            # SVF использует low-rank decomposition
            avg_dim = self.n_embd
            svf_params_per_layer = num_svf_matrices * (avg_dim * self.svf_rank * 2)
        
        # Expert parameters
        expert_params = self.num_experts * self.expert_dim
        
        # Classifier
        classifier_params = 0
        if self.adaptation_strategy == 'classifier':
            classifier_params = self.n_embd * self.num_experts
        
        final_ln = 2 * self.n_embd
        
        total_base = token_emb + pos_emb + (base_layer_params * self.n_layer) + final_ln
        total_svf = svf_params_per_layer * self.n_layer
        total_experts = expert_params + classifier_params
        
        total = total_base + total_svf + total_experts
        
        self.estimated_parameters = total
        self.estimated_parameters_millions = total / 1e6
        self.estimated_base_parameters = total_base
        self.estimated_svf_parameters = total_svf
        self.estimated_expert_parameters = total_experts
    
    def get_head_dim(self) -> int:
        """Размерность каждого attention head."""
        return self.n_embd // self.n_head
    
    def get_dtype(self) -> torch.dtype:
        """Конвертация строки dtype в torch.dtype."""
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }
        return dtype_map.get(self.model_dtype, torch.float32)
    
    def validate(self):
        """Явная валидация для обратной совместимости."""
        # Вся валидация в __post_init__, этот метод для совместимости
        pass
