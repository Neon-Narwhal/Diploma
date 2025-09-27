"""
Transformer² Model - Sakana AI Innovations
Uses universal configuration from utils.config
No separate config classes - everything in ModelConfig
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Dict, Any, List, Tuple

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
        self.U = nn.Linear(self.rank, out_features, bias=False)
        self.V = nn.Linear(in_features, self.rank, bias=False)

        # Expert vector for task adaptation
        self.z_vector = nn.Parameter(torch.ones(1, self.rank))

        # Optional bias
        self.bias = nn.Parameter(torch.zeros(out_features)) if config.svf_bias else None

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
        self.linear = nn.Linear(in_features, out_features, bias=bias)
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
    """Configurable attention head with optional SVF"""

    def __init__(self, head_size: int, config):
        super().__init__()
        self.head_size = head_size
        self.config = config

        # Choose linear layer type based on config
        linear_cls = SVFLinear if config.use_svf else StandardLinear

        self.key = linear_cls(config.n_embd, head_size, config)
        self.query = linear_cls(config.n_embd, head_size, config) 
        self.value = linear_cls(config.n_embd, head_size, config)

        # Causal mask
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        return out

class AdaptiveMultiHeadAttention(nn.Module):
    """Multi-head attention with configurable components"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.config = config
        head_size = config.n_embd // config.n_head

        self.heads = nn.ModuleList([
            AdaptiveAttentionHead(head_size, config) for _ in range(config.n_head)
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
    """Transformer block with configurable architecture"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Layer norms
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        # Attention and feed-forward
        self.sa = AdaptiveMultiHeadAttention(config)
        self.ffwd = AdaptiveFeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.use_pre_norm:
            # Pre-norm (more stable)
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
        else:
            # Post-norm (original transformer)
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
                torch.randn(config.n_layer, config.n_embd)
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
    Universal Transformer model with optional Sakana AI innovations
    Works with standard transformer config or enhanced Transformer² features
    All controlled by single universal ModelConfig
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Validate configuration
        config.validate()

        # Embeddings
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerSquaredBlock(config) for _ in range(config.n_layer)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Task-specific heads
        self.lm_head = StandardLinear(config.n_embd, config.vocab_size, config, bias=False)

        # Optional complexity analysis head
        self.complexity_head = None
        if config.enable_complexity_head:
            self.complexity_head = StandardLinear(config.n_embd, config.num_complexity_classes, config)

        # Optional task detector
        self.task_detector = None
        if config.enable_task_detector:
            self.task_detector = TaskDetector(config)

        # Expert vector manager (always created, but only used if SVF is enabled)
        self.expert_manager = ExpertVectorManager(config)

        # Current task state
        self.current_task = 'language_modeling'

        self._init_weights()

    def _init_weights(self):
        """Initialize all model weights according to config"""
        # Embeddings
        nn.init.normal_(self.token_embedding_table.weight, std=self.config.embedding_init_std)
        nn.init.normal_(self.position_embedding_table.weight, std=self.config.embedding_init_std)

        # Layer norms
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    def set_task_mode(self, task_type: str):
        """Set current task and update expert vectors accordingly"""
        available_tasks = self.expert_manager.get_available_tasks()
        if task_type not in available_tasks:
            raise ValueError(f"Task '{task_type}' not available. Available tasks: {available_tasks}")

        self.current_task = task_type

        # Update expert vectors only if SVF is enabled
        if self.config.use_svf:
            self._update_expert_vectors(task_type)

    def _update_expert_vectors(self, task_type: str):
        """Update expert vectors in all adaptive components"""
        for layer_idx, block in enumerate(self.blocks):
            expert_vector = self.expert_manager.get_expert_vector(task_type, layer_idx)

            # Update attention heads
            for head in block.sa.heads:
                head.key.set_expert_vector(expert_vector)
                head.query.set_expert_vector(expert_vector)
                head.value.set_expert_vector(expert_vector)

            # Update projection
            block.sa.proj.set_expert_vector(expert_vector)

            # Update feed-forward layers
            if hasattr(block.ffwd.net[0], 'set_expert_vector'):
                block.ffwd.net[0].set_expert_vector(expert_vector)
            if hasattr(block.ffwd.net[2], 'set_expert_vector'):
                block.ffwd.net[2].set_expert_vector(expert_vector)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, 
                task_type: Optional[str] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with configurable task handling"""

        # Set task mode if specified
        if task_type is not None and task_type != self.current_task:
            self.set_task_mode(task_type)

        B, T = idx.shape
        device = idx.device

        # Embeddings
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # Task-specific output
        if (self.current_task == 'complexity_analysis' and 
            self.complexity_head is not None):
            # Global pooling for classification
            pooled = x.mean(dim=1)
            logits = self.complexity_head(pooled)
        else:
            # Language modeling (token-level)
            logits = self.lm_head(x)

        # Calculate loss if targets provided
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
        """Generate new tokens using config parameters"""
        max_new_tokens = max_new_tokens or self.config.max_generation_length
        temperature = temperature or self.config.generation_temperature

        # Ensure we're in language modeling mode
        original_task = self.current_task
        if 'language_modeling' in self.expert_manager.get_available_tasks():
            self.set_task_mode('language_modeling')

        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)

        # Restore original task
        if original_task != self.current_task:
            self.set_task_mode(original_task)

        return idx

    def detect_task(self, idx: torch.Tensor) -> str:
        """Detect task type from input (if task detector enabled)"""
        if self.task_detector is None:
            return self.current_task

        self.eval()
        with torch.no_grad():
            # Quick pass through embeddings and first block
            tok_emb = self.token_embedding_table(idx)
            pos_emb = self.position_embedding_table(torch.arange(idx.size(1), device=idx.device))
            x = tok_emb + pos_emb

            # Partial forward pass
            x = self.blocks[0](x)

            task_logits = self.task_detector(x)
            task_id = torch.argmax(task_logits, dim=-1)[0]

            available_tasks = self.expert_manager.get_available_tasks()
            return available_tasks[task_id.item() % len(available_tasks)]

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        svf_params = 0
        standard_params = 0
        for module in self.modules():
            if isinstance(module, SVFLinear):
                svf_params += sum(p.numel() for p in module.parameters())
            elif isinstance(module, StandardLinear):
                standard_params += sum(p.numel() for p in module.parameters())

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'svf_parameters': svf_params,
            'standard_parameters': standard_params,
            'available_tasks': self.expert_manager.get_available_tasks(),
            'current_task': self.current_task,
            'model_type': 'Transformer²' if self.config.use_transformer_squared else 'Standard Transformer',
            'supports_complexity_analysis': self.complexity_head is not None,
            'supports_task_detection': self.task_detector is not None,
            'use_svf': self.config.use_svf,
            'model_name': self.config.model_name,
            'config_info': self.config.get_model_info()
        }

    def is_transformer_squared(self) -> bool:
        """Check if model is using Transformer² features"""
        return (self.config.use_transformer_squared and 
                (self.config.use_svf or 
                 self.complexity_head is not None or 
                 self.task_detector is not None))

    def get_available_tasks(self) -> List[str]:
        """Get list of available tasks"""
        return self.expert_manager.get_available_tasks()

# Factory function for easy model creation
def create_model(config_type: str = "standard", **kwargs):
    """
    Factory function to create models with different configurations

    Args:
        config_type: 'standard', 'transformer_squared', 'code_analysis', 'small', 'large'
        **kwargs: Additional config overrides
    """
    # Import here to avoid circular dependency
    from utils.config import ModelConfig, create_config_for_task

    if config_type == "standard":
        config = ModelConfig.get_standard_config()
    elif config_type == "transformer_squared":
        config = ModelConfig.get_comparison_config()
    elif config_type == "code_analysis":
        config = ModelConfig.get_code_analysis_config()
    elif config_type == "small":
        config = ModelConfig.get_small_research_config()
    elif config_type == "large":
        config = ModelConfig.get_large_production_config()
    else:
        # Default to medium transformer_squared
        config = ModelConfig()
        config.enable_transformer_squared_features()

    # Apply any overrides
    config.update_from_dict(kwargs)

    return TransformerSquared(config)