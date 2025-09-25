"""GPT-подобная модель"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import time
from typing import Optional
from utils.config import ModelConfig
from utils.logging import GenerationLogger


class Head(nn.Module):
    """Одна голова внимания"""
    
    def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiheadAttention(nn.Module):
    """Многоголовое внимание"""
    
    def __init__(self, num_heads: int, head_size: int, n_embd: int, block_size: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
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
    
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiheadAttention(n_head, head_size, n_embd, block_size, dropout)
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
        self.position_embeding_table = nn.Embedding(config.block_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.blocks = nn.Sequential(*[
            Block(config.n_embd, config.n_head, config.block_size, config.dropout) 
            for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.logger = GenerationLogger()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embeding_table(torch.arange(T, device=self.config.device))
        x = tok_emb + pos_emb
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
