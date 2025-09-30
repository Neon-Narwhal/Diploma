import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding для Transformer²"""
    
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Создаем обратные частоты
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Предвычисляем cos и sin
        self._set_cos_sin_cache(max_seq_len)
    
    def _set_cos_sin_cache(self, seq_len: int):
        """Предвычисляет cos и sin для заданной длины последовательности"""
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        angles = position * self.inv_freq.unsqueeze(0)
        
        # Дублируем углы для пар (cos, sin) 
        angles = torch.cat([angles, angles], dim=-1)
        
        self.register_buffer('cos_cached', angles.cos())
        self.register_buffer('sin_cached', angles.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        if seq_len is None:
            seq_len = x.size(-2)
            
        # Расширяем кэш если нужно
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)
        
        return self.apply_rotary_pos_emb(x, seq_len)
    
    def apply_rotary_pos_emb(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Применяет ротационное позиционное кодирование"""
        cos = self.cos_cached[:seq_len].unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0)
        
        # Разделяем на пары и применяем вращение
        x1, x2 = x.chunk(2, dim=-1)
        
        rotated = torch.cat([
            x1 * cos[..., :x1.size(-1)] - x2 * sin[..., x1.size(-1):],
            x1 * sin[..., :x1.size(-1)] + x2 * cos[..., x1.size(-1):]
        ], dim=-1)
        
        return rotated
