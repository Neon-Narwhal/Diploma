"""GPT-подобная модель"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import time
from typing import Optional
from utils.config import ModelConfig
from utils.logging import GenerationLogger
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096, base=10000):
        """
        RoPE implementation
        Args:
            dim: размер эмбеддинга (обычно head_dim = n_embd // n_head)
            max_seq_len: максимальная длина последовательности  
            base: база для геометрической прогрессии
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Создаем частоты для вращения
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Предвычисляем cos и sin для максимальной длины последовательности
        self._set_cos_sin_cache(max_seq_len)
    
    def _set_cos_sin_cache(self, seq_len):
        """Предвычисляет cos и sin для заданной длины последовательности"""
        position = torch.arange(seq_len).unsqueeze(1)  # [seq_len, 1]
        angles = position * self.inv_freq.unsqueeze(0)  # [seq_len, dim//2]
        
        # Дублируем углы для пар (cos, sin)
        angles = torch.cat([angles, angles], dim=-1)  # [seq_len, dim]
        
        self.register_buffer('cos_cached', angles.cos())
        self.register_buffer('sin_cached', angles.sin())
    
    def forward(self, x, seq_len=None):
        """
        Применяет RoPE к входному тензору
        Args:
            x: тензор формы [batch, seq_len, n_heads, head_dim]
            seq_len: длина последовательности (если отличается от x.size(1))
        """
        if seq_len is None:
            seq_len = x.size(1)
            
        # Если последовательность длиннее кэша, пересчитываем
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)
        
        # Получаем cos и sin для текущей длины
        cos = self.cos_cached[:seq_len].unsqueeze(0)  # [1, seq_len, dim]
        sin = self.sin_cached[:seq_len].unsqueeze(0)  # [1, seq_len, dim]
        
        return self.apply_rotary_pos_emb(x, cos, sin)
    
    def apply_rotary_pos_emb(self, x, cos, sin):
        """Применяет ротационное позиционное кодирование"""
        # Разделяем на пары и применяем вращение
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([
            x1 * cos[..., :x1.size(-1)] - x2 * sin[..., x1.size(-1):],
            x1 * sin[..., :x1.size(-1)] + x2 * cos[..., x1.size(-1):]
        ], dim=-1)
        
        return rotated


class Head(nn.Module):
    """Одна голова внимания"""
    
    def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float, rope):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.rope = rope
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Применяем RoPE к query и key
        q = self.rope(q.unsqueeze(2), T).squeeze(2)  # [B, T, head_size]
        k = self.rope(k.unsqueeze(2), T).squeeze(2)  # [B, T, head_size]

        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiheadAttention(nn.Module):
    """Многоголовое внимание"""
    
    def __init__(self, num_heads: int, head_size: int, n_embd: int, block_size: int, dropout: float, rope):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout, rope) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """Прямое распространение"""
    
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.LeakyReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Блок трансформера"""
    
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, rope):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiheadAttention(n_head, head_size, n_embd, block_size, dropout, rope)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLikeModel(nn.Module):
    """GPT-подобная языковая модель с подробным логированием генерации"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        #self.position_embeding_table = nn.Embedding(config.block_size, config.n_embd)

        head_dim = config.n_embd // config.n_head
        self.rope = RotaryPositionalEmbedding(head_dim, config.block_size)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.blocks = nn.Sequential(*[
            Block(config.n_embd, config.n_head, config.block_size, config.dropout, self.rope) 
            for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.logger = GenerationLogger()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        #pos_emb = self.position_embeding_table(torch.arange(T, device=self.config.device))
        x = tok_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else: 
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate_with_logging(self, idx, max_new_token: int, tokenizer, temperature: float = 1.0):
        """Генерация с подробным логированием каждого шага"""
        start_time = time.time()
        
        # Логируем начало генерации
        self.logger.log_generation_start(idx, max_new_token)
        
        generated_tokens = []
        
        for step in range(max_new_token):
            # Обрезаем контекст до максимального размера блока
            idx_cond = idx[:, -self.config.block_size:]
            
            # Получаем предсказания
            with torch.no_grad():
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                
                # Применяем softmax и семплируем
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Получаем ID токена и декодируем его
                token_id = idx_next[0, 0].item()
                decoded_token = tokenizer.decode([token_id])
                
                # Логируем шаг
                self.logger.log_generation_step(step + 1, token_id, decoded_token)
                
                # Добавляем токен к последовательности
                idx = torch.cat((idx, idx_next), dim=1)
                generated_tokens.append(token_id)
        
        # Декодируем полную сгенерированную последовательность
        full_sequence = idx[0].tolist()
        generated_text = tokenizer.decode(full_sequence)
        
        execution_time = time.time() - start_time
        self.logger.log_generation_complete(generated_text, len(generated_tokens), execution_time)
        
        return idx
    
    def generate(self, idx, max_new_token: int):
        """Обычная генерация без логирования (для совместимости)"""
        for _ in range(max_new_token):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
