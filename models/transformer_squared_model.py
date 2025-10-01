"""
Transformer² Model - Sakana AI Innovations
Uses universal configuration from utils.config
No separate config classes - everything in ModelConfig
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Dict, Any, List, Tuple
from metods.rope import RotaryPositionalEmbedding

class SVFLinear(nn.Module):
    """Singular Value Fine-tuning Linear Layer"""

    def __init__(self, in_features: int, out_features: int, config):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # Calculate rank from config
        self.rank = max(1, int(min(in_features, out_features) * config.svf_rank_ratio))

        # SVD components
        self.U = nn.Linear(self.rank, out_features, bias=False, dtype=self.config.model_dtype)
        self.V = nn.Linear(in_features, self.rank, bias=False, dtype=self.config.model_dtype)

        # Expert vector for task adaptation
        self.z_vector = nn.Parameter(torch.ones(1, self.rank, dtype=self.config.model_dtype))

        # Optional bias
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=self.config.model_dtype)) if config.svf_bias else None

        self._init_weights()

    def _init_weights(self):
        """Initialize weights according to config"""
        nn.init.normal_(self.U.weight, std=self.config.weight_init_std)
        nn.init.normal_(self.V.weight, std=self.config.weight_init_std)
        nn.init.ones_(self.z_vector)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.V(x)
        x = x * self.z_vector
        x = self.U(x)
        if self.bias is not None:
            x = x + self.bias
        return x

    def set_expert_vector(self, expert_weights: torch.Tensor):
        """Update expert vector for task adaptation"""
        with torch.no_grad():
            if expert_weights.numel() >= self.rank:
                self.z_vector.copy_(expert_weights[:self.rank].unsqueeze(0))

class StandardLinear(nn.Module):
    """Standard linear layer wrapper for consistency"""

    def __init__(self, in_features: int, out_features: int, config, bias: bool = True):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=self.config.model_dtype)
        self.config = config
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.linear.weight, std=self.config.weight_init_std)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def set_expert_vector(self, expert_weights: torch.Tensor):
        """Compatibility method - does nothing for standard linear"""
        pass

class AdaptiveAttentionHead(nn.Module):
    """Configurable attention head с RoPE поддержкой"""

    def __init__(self, head_size: int, config, rope: Optional[RotaryPositionalEmbedding] = None):
        super().__init__()
        self.head_size = head_size
        self.config = config
        self.rope = rope  # Добавляем RoPE

        # Choose linear layer type based on config
        linear_cls = SVFLinear if config.use_svf else StandardLinear

        self.key = linear_cls(config.n_embd, head_size, config)
        self.query = linear_cls(config.n_embd, head_size, config) 
        self.value = linear_cls(config.n_embd, head_size, config)

        # Causal mask
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size, dtype=self.config.model_dtype)))
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
    
    
        k = self.key(x)    # [B, T, head_size] 
        q = self.query(x)  # [B, T, head_size]
        v = self.value(x)  # [B, T, head_size]

        # Применяем RoPE к query и key (НЕ к value!)
        if self.rope is not None:
            q = self.rope(q, T)
            k = self.rope(k, T)

        # Scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1, dtype=self.config.model_dtype)
        wei = self.dropout(wei)

        out = wei @ v
        return out


class AdaptiveMultiHeadAttention(nn.Module):
    """Multi-head attention с RoPE"""

    def __init__(self, config, rope: Optional[RotaryPositionalEmbedding] = None):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.config = config
        head_size = config.n_embd // config.n_head

        self.heads = nn.ModuleList([
            AdaptiveAttentionHead(head_size, config, rope) for _ in range(config.n_head)
        ])

        # Output projection
        linear_cls = SVFLinear if config.use_svf else StandardLinear
        self.proj = linear_cls(config.n_embd, config.n_embd, config)
        self.dropout = nn.Dropout(config.residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class AdaptiveFeedForward(nn.Module):
    """Feed-forward network with configurable expansion and components"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        hidden_dim = config.n_embd * config.ffn_expansion_factor
        linear_cls = SVFLinear if config.use_svf else StandardLinear

        self.net = nn.Sequential(
            linear_cls(config.n_embd, hidden_dim, config),
            nn.GELU(),
            linear_cls(hidden_dim, config.n_embd, config),
            nn.Dropout(config.residual_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerSquaredBlock(nn.Module):
    """Transformer block с RoPE поддержкой"""

    def __init__(self, config, rope: Optional[RotaryPositionalEmbedding] = None):
        super().__init__()
        self.config = config

        # Layer norms
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        # Attention и feed-forward с RoPE
        self.sa = AdaptiveMultiHeadAttention(config, rope)
        self.ffwd = AdaptiveFeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.use_pre_norm:
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
        else:
            x = self.ln1(x + self.sa(x))
            x = self.ln2(x + self.ffwd(x))
        return x


class TaskDetector(nn.Module):
    """Configurable task detection module"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = StandardLinear(config.n_embd, config.num_expert_vectors, config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(hidden_states.transpose(-1, -2)).squeeze(-1)
        return self.classifier(pooled)

class ExpertVectorManager(nn.Module):
    """Configurable expert vector management system"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create expert vectors based on config
        expert_names = ['language_modeling', 'complexity_analysis', 'code_understanding']
        self.task_experts = nn.ParameterDict()

        for i, name in enumerate(expert_names[:config.num_expert_vectors]):
            self.task_experts[name] = nn.Parameter(
                torch.randn(config.n_layer, config.n_embd, dtype=self.config.model_dtype)
            )

        self._init_expert_vectors()

    def _init_expert_vectors(self):
        """Initialize expert vectors according to config"""
        for expert in self.task_experts.values():
            nn.init.normal_(expert, std=self.config.expert_vector_init_std)

    def get_expert_vector(self, task_type: str, layer_idx: int) -> torch.Tensor:
        """Get expert vector for specific task and layer"""
        if task_type not in self.task_experts:
            # Fallback to first available expert
            task_type = list(self.task_experts.keys())[0]

        return self.task_experts[task_type][layer_idx]

    def get_available_tasks(self) -> List[str]:
        """Get list of available task types"""
        return list(self.task_experts.keys())

class TransformerSquared(nn.Module):
    """
    Universal Transformer² модель с полной поддержкой RoPE
    Работает как стандартный трансформер или с расширенными возможностями
    Все контролируется универсальным ModelConfig
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Валидация конфигурации
        config.validate()
        
        # Embeddings - токенные эмбеддинги всегда нужны
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd, dtype=self.config.model_dtype)
        
        # Создаем RoPE если включен в конфиге, иначе используем стандартные позиционные эмбеддинги
        self.rope = None
        if getattr(config, 'use_rope', False):
            head_dim = config.n_embd // config.n_head
            self.rope = RotaryPositionalEmbedding(
                dim=head_dim,
                max_seq_len=config.block_size,
                base=getattr(config, 'rope_base', 10000.0)
            )
            # Не создаем позиционные эмбеддинги при использовании RoPE
            self.position_embedding_table = None
        else:
            # Стандартные позиционные эмбеддинги
            self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd, dtype=self.config.model_dtype)

        # Transformer блоки с передачей RoPE
        self.blocks = nn.ModuleList([
            TransformerSquaredBlock(config, self.rope) for _ in range(config.n_layer)
        ])

        # Финальная layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Task-specific головы
        self.lm_head = StandardLinear(config.n_embd, config.vocab_size, config, bias=False)

        # Опциональная голова для анализа сложности
        self.complexity_head = None
        if config.enable_complexity_head:
            self.complexity_head = StandardLinear(config.n_embd, config.num_complexity_classes, config)

        # Опциональный детектор задач
        self.task_detector = None
        if config.enable_task_detector:
            self.task_detector = TaskDetector(config)

        # Менеджер экспертных векторов (всегда создается, но используется только если SVF включен)
        self.expert_manager = ExpertVectorManager(config)

        # Текущее состояние задачи
        self.current_task = 'language_modeling'

        self._init_weights()

    def _init_weights(self):
        """Инициализация всех весов модели согласно конфигу"""
        # Эмбеддинги
        nn.init.normal_(self.token_embedding_table.weight, std=self.config.embedding_init_std)
        if self.position_embedding_table is not None:
            nn.init.normal_(self.position_embedding_table.weight, std=self.config.embedding_init_std)

        # Layer norms
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    def set_task_mode(self, task_type: str):
        """Устанавливает текущую задачу и обновляет экспертные векторы соответственно"""
        available_tasks = self.expert_manager.get_available_tasks()
        if task_type not in available_tasks:
            raise ValueError(f"Задача '{task_type}' недоступна. Доступные задачи: {available_tasks}")

        self.current_task = task_type

        # Обновляем экспертные векторы только если SVF включен
        if self.config.use_svf:
            self._update_expert_vectors(task_type)

    def _update_expert_vectors(self, task_type: str):
        """Обновляет экспертные векторы во всех адаптивных компонентах"""
        for layer_idx, block in enumerate(self.blocks):
            expert_vector = self.expert_manager.get_expert_vector(task_type, layer_idx)

            # Обновляем головы внимания
            for head in block.sa.heads:
                head.key.set_expert_vector(expert_vector)
                head.query.set_expert_vector(expert_vector)
                head.value.set_expert_vector(expert_vector)

            # Обновляем проекцию
            block.sa.proj.set_expert_vector(expert_vector)

            # Обновляем feed-forward слои
            if hasattr(block.ffwd.net[0], 'set_expert_vector'):
                block.ffwd.net[0].set_expert_vector(expert_vector)
            if hasattr(block.ffwd.net[2], 'set_expert_vector'):
                block.ffwd.net[2].set_expert_vector(expert_vector)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, 
                task_type: Optional[str] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass с конфигурируемой обработкой задач"""

        # Устанавливаем режим задачи если указан
        if task_type is not None and task_type != self.current_task:
            self.set_task_mode(task_type)

        B, T = idx.shape
        device = idx.device

        # Эмбеддинги
        tok_emb = self.token_embedding_table(idx)
        
        if self.rope is not None:
            # Используем только токенные эмбеддинги, RoPE применится в attention
            x = tok_emb
        else:
            # Стандартный путь с позиционными эмбеддингами
            pos_emb = self.position_embedding_table(torch.arange(T, device=device, dtype=self.config.model_dtype))
            x = tok_emb + pos_emb

        # Transformer блоки
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # Task-specific выход
        if (self.current_task == 'complexity_analysis' and 
            self.complexity_head is not None):
            # Глобальный pooling для классификации
            pooled = x.mean(dim=1)
            logits = self.complexity_head(pooled)
        else:
            # Языковое моделирование (token-level)
            logits = self.lm_head(x)

        # Вычисляем loss если targets предоставлены
        loss = None
        if targets is not None:
            if self.current_task == 'complexity_analysis':
                loss = F.cross_entropy(logits, targets)
            else:
                B, T, C = logits.shape
                logits_flat = logits.view(B*T, C)
                targets_flat = targets.view(B*T)
                loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: Optional[int] = None, 
                 temperature: Optional[float] = None) -> torch.Tensor:
        """Генерация новых токенов используя параметры конфига"""
        max_new_tokens = max_new_tokens or self.config.max_generation_length
        temperature = temperature or self.config.generation_temperature

        # Убеждаемся что мы в режиме языкового моделирования
        original_task = self.current_task
        if 'language_modeling' in self.expert_manager.get_available_tasks():
            self.set_task_mode('language_modeling')

        self.eval()
        with torch.no_grad():
            # AMP context для генерации если включен
            amp_context = torch.amp.autocast(
                device_type=self.config.device.split(':')[0],
                dtype=getattr(self.config, 'amp_dtype', torch.float32),
                enabled=getattr(self.config, 'use_amp', False)
            )
            
            with amp_context:
                for _ in range(max_new_tokens):
                    idx_cond = idx[:, -self.config.block_size:]
                    logits, _ = self(idx_cond)
                    logits = logits[:, -1, :] / temperature
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    idx = torch.cat((idx, idx_next), dim=1)

        # Восстанавливаем оригинальную задачу
        if original_task != self.current_task:
            self.set_task_mode(original_task)

        return idx

    def detect_task(self, idx: torch.Tensor) -> str:
        """Определяет тип задачи из входа (если детектор задач включен)"""
        if self.task_detector is None:
            return self.current_task

        self.eval()
        with torch.no_grad():
            # Быстрый проход через эмбеддинги и первый блок
            tok_emb = self.token_embedding_table(idx)
            
            if self.rope is not None:
                x = tok_emb
            else:
                pos_emb = self.position_embedding_table(torch.arange(idx.size(1), device=idx.device))
                x = tok_emb + pos_emb

            # Частичный forward pass
            x = self.blocks[0](x)

            task_logits = self.task_detector(x)
            task_id = torch.argmax(task_logits, dim=-1)[0]

            available_tasks = self.expert_manager.get_available_tasks()
            return available_tasks[task_id.item() % len(available_tasks)]

    def get_model_info(self) -> Dict[str, Any]:
        """Получает исчерпывающую информацию о модели"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        svf_params = 0
        standard_params = 0
        rope_params = 0
        
        for module in self.modules():
            if isinstance(module, SVFLinear):
                svf_params += sum(p.numel() for p in module.parameters())
            elif isinstance(module, StandardLinear):
                standard_params += sum(p.numel() for p in module.parameters())
            elif isinstance(module, RotaryPositionalEmbedding):
                rope_params += sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'svf_parameters': svf_params,
            'standard_parameters': standard_params,
            'rope_parameters': rope_params,
            'available_tasks': self.expert_manager.get_available_tasks(),
            'current_task': self.current_task,
            'model_type': 'Transformer²' if self.config.use_transformer_squared else 'Standard Transformer',
            'supports_complexity_analysis': self.complexity_head is not None,
            'supports_task_detection': self.task_detector is not None,
            'use_svf': self.config.use_svf,
            'use_rope': self.rope is not None,
            'rope_enabled': getattr(self.config, 'use_rope', False),
            'rope_base': getattr(self.config, 'rope_base', 10000.0),
            'model_name': self.config.model_name,
            'config_info': self.config.get_model_info()
        }

    def is_transformer_squared(self) -> bool:
        """Проверяет использует ли модель возможности Transformer²"""
        return (self.config.use_transformer_squared and 
                (self.config.use_svf or 
                 self.complexity_head is not None or 
                 self.task_detector is not None or
                 self.rope is not None))

    def get_available_tasks(self) -> List[str]:
        """Получает список доступных задач"""
        return self.expert_manager.get_available_tasks()

    def get_rope_info(self) -> Dict[str, Any]:
        """Получает информацию о RoPE если включен"""
        if self.rope is None:
            return {'enabled': False}
        
        return {
            'enabled': True,
            'dim': self.rope.dim,
            'max_seq_len': self.rope.max_seq_len,
            'base': self.rope.base,
            'cached_positions': self.rope.cos_cached.size(0) if hasattr(self.rope, 'cos_cached') else 0
        }

    def switch_to_rope(self, base: float = 10000.0):
        """Динамически переключается на использование RoPE"""
        if self.rope is not None:
            print("RoPE уже включен")
            return
        
        # Создаем RoPE
        head_dim = self.config.n_embd // self.config.n_head
        self.rope = RotaryPositionalEmbedding(
            dim=head_dim,
            max_seq_len=self.config.block_size,
            base=base
        ).to(next(self.parameters()).device)
        
        # Обновляем блоки для использования RoPE
        for block in self.blocks:
            block.sa.rope = self.rope
            for head in block.sa.heads:
                head.rope = self.rope
        
        # Удаляем позиционные эмбеддинги (опционально)
        # self.position_embedding_table = None
        
        # Обновляем конфиг
        self.config.use_rope = True
        self.config.rope_base = base
        
        print(f"RoPE включен с base={base}")

    def switch_to_standard_pe(self):
        """Переключается обратно на стандартные позиционные эмбеддинги"""
        if self.rope is None:
            print("RoPE не включен")
            return
        
        # Удаляем RoPE из блоков
        for block in self.blocks:
            block.sa.rope = None
            for head in block.sa.heads:
                head.rope = None
        
        # Создаем позиционные эмбеддинги если их нет
        if self.position_embedding_table is None:
            self.position_embedding_table = nn.Embedding(
                self.config.block_size, 
                self.config.n_embd
            ).to(next(self.parameters()).device)
            nn.init.normal_(self.position_embedding_table.weight, std=self.config.embedding_init_std)
        
        # Удаляем RoPE
        self.rope = None
        
        # Обновляем конфиг
        self.config.use_rope = False
        
        print("Переключено на стандартные позиционные эмбеддинги")