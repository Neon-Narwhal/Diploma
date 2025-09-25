import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import mlflow
import mlflow.pytorch
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import argparse
import json

torch.manual_seed(1337)

@dataclass
class ModelConfig:
    # Основные параметры модели
    batch_size: int = 64
    block_size: int = 256
    vocab_size: int = 3000
    max_iters: int = 2500
    eval_interval: int = 500
    learning_rate: float = 3e-4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters: int = 200
    n_embd: int = 256
    n_layer: int = 6
    n_head: int = 4
    epoch_count: int = 1
    overfit_line: float = 0.05
    dropout: float = 0.1
    model_name = "my_model_v1"
    
    # Параметры токенизации
    tokenizer_type: str = 'bpe'  # 'bpe' или 'char'
    experiment_name: str = "transformer_language_model"
    data_file: str = "data/input.txt"

# Базовый класс для токенизаторов
class BaseTokenizer:
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError
    
    def decode(self, tokens: List[int]) -> str:
        raise NotImplementedError
    
    def get_vocab_size(self) -> int:
        raise NotImplementedError
    
    def save(self, path: str):
        raise NotImplementedError

# BPE токенизатор
class BPETokenizer(BaseTokenizer):
    def __init__(self, data: str, vocab_size: int):
        self.tokenizer = Tokenizer(models.BPE())
        
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        )
        
        self.tokenizer.pre_tokenizer = pre_tokenizers.Split(
            pattern=r"(\s)",
            behavior="merged_with_previous"
        )
        
        lines = data.splitlines(keepends=True)
        self.tokenizer.train_from_iterator(lines, trainer)
        
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids
    
    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
    
    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
    
    def save(self, path: str):
        self.tokenizer.save(path)

# Символьный токенизатор
class CharTokenizer(BaseTokenizer):
    def __init__(self, data: str):
        chars = sorted(set(data))

        self.chars = chars
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
    
    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(ch, 0) for ch in text]
    
    def decode(self, tokens: List[int]) -> str:
        return "".join([self.itos.get(i, '') for i in tokens])
    
    def get_vocab_size(self) -> int:
        return self.vocab_size
    
    def save(self, path: str):
        tokenizer_data = {
            'type': 'char',
            'chars': self.chars,
            'stoi': self.stoi,
            'itos': {str(k): v for k, v in self.itos.items()}
        }
        with open(path, 'w') as f:
            json.dump(tokenizer_data, f)

# Ваша модель Transformer (без изменений)
class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
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
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
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
    def __init__(self, n_embd, dropout):
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
    def __init__(self, n_embd, n_head, block_size, dropout):
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

class BigramLanguageModel(nn.Module):
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
    
    def generate(self, idx, max_new_token):
        for _ in range(max_new_token):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Функции для работы с данными и обучения
def load_data(file_path: str) -> str:
    """Загрузка данных из файла"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Файл {file_path} не найден. Используем тестовые данные.")
        return "Hello world! This is a test dataset for training language models. " * 100

def create_tokenizer(data: str, config: ModelConfig) -> BaseTokenizer:
    """Создание токенизатора в зависимости от конфигурации"""
    if config.tokenizer_type == 'bpe':
        return BPETokenizer(data, config.vocab_size)
    elif config.tokenizer_type == 'char':
        return CharTokenizer(data, config.vocab_size)
    else:
        raise ValueError(f"Unknown tokenizer type: {config.tokenizer_type}")

def get_batch(data: torch.Tensor, config: ModelConfig, split: str = "train"):
    """Получение батча для обучения"""
    n = int(0.9 * len(data))
    train_data = data[:n] if split == "train" else data[n:]
    
    ix = torch.randint(len(train_data) - config.block_size, (config.batch_size,))
    x = torch.stack([train_data[i:i + config.block_size] for i in ix])
    y = torch.stack([train_data[i + 1:i + config.block_size + 1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y

@torch.no_grad()
def estimate_loss(model, data: torch.Tensor, config: ModelConfig):
    """Оценка loss на train/val"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, y = get_batch(data, config, split)
            logits, loss = model(X, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def run_experiment(config: ModelConfig, run_name: str = None):
    """Запуск одного эксперимента с MLflow tracking"""
    
    # Загружаем данные
    data = load_data(config.data_file)
    
    # Создаем токенизатор
    tokenizer = create_tokenizer(data, config)
    actual_vocab_size = tokenizer.get_vocab_size()
    
    # Обновляем vocab_size в конфигурации
    config.vocab_size = actual_vocab_size
    
    # Токенизируем данные
    encoded_data = tokenizer.encode(data)
    data_torch = torch.tensor(encoded_data, dtype=torch.long)
    
    # Начинаем MLflow run
    with mlflow.start_run(run_name=run_name):
        # Логируем параметры
        mlflow.log_params(asdict(config))
        mlflow.log_param("actual_vocab_size", actual_vocab_size)
        mlflow.log_param("data_length", len(data_torch))
        
        # Создаем и инициализируем модель
        model = BigramLanguageModel(config)
        model = model.to(config.device)
        
        # Оптимизатор
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        
        # Обучение
        losses_val = []
        model.train()
        
        for iter_num in range(config.max_iters):
            # Оценка loss
            if iter_num % config.eval_interval == 0:
                losses = estimate_loss(model, data_torch, config)
                losses_val.append(losses['val'].item())
                
                print(f"step {iter_num}: train loss {losses['train']:.4f} val loss {losses['val']:.4f}")
                
                # Логируем метрики
                mlflow.log_metrics({
                    "train_loss": losses['train'].item(),
                    "val_loss": losses['val'].item(),
                }, step=iter_num)
                
                # Проверка на переобучение
                if len(losses_val) >= 2 and losses_val[-1] - losses_val[-2] > config.overfit_line:
                    print(f"Early stopping at iteration {iter_num}")
                    mlflow.log_param("early_stopped_at", iter_num)
                    break
            
            # Шаг обучения
            xb, yb = get_batch(data_torch, config, "train")
            logits, loss = model(xb, yb)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        # Финальные метрики
        final_losses = estimate_loss(model, data_torch, config)
        mlflow.log_metrics({
            "final_train_loss": final_losses['train'].item(),
            "final_val_loss": final_losses['val'].item(),
        })
        
        # Сохраняем модель и токенизатор
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",  # Это корректно для MLflow 2.x
            registered_model_name=config.model_name  # Добавьте это для версионирования
        )

        #tokenizer.save("tokenizer.json")
        #mlflow.log_artifact("tokenizer.json")
        
        # Генерируем тестовый текст
        model.eval()
        context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
        generated_tokens = model.generate(context, max_new_token=100)[0].tolist()
        generated_text = tokenizer.decode(generated_tokens)
        mlflow.log_text(generated_text, "generated_sample.txt")
        
        return final_losses['val'].item()

def run_vocab_size_experiments():
    """Запуск экспериментов с разными размерами словаря"""
    
    # Настройка MLflow
    mlflow.set_experiment("vocab_size_comparison")
    
    vocab_sizes = [100, 200, 300]
    tokenizer_types = ['bpe']
    
    results = []
    
    for tokenizer_type in tokenizer_types:
        for vocab_size in vocab_sizes:
            config = ModelConfig(
                vocab_size=vocab_size,
                tokenizer_type=tokenizer_type,
                max_iters=ModelConfig.max_iters,  
                eval_interval=ModelConfig.eval_interval,
            )
            
            run_name = f"{tokenizer_type}_vocab_{vocab_size}"
            print(f"\n🚀 Starting experiment: {run_name}")
            
            try:
                final_val_loss = run_experiment(config, run_name)
                results.append({
                    'tokenizer_type': tokenizer_type,
                    'vocab_size': vocab_size,
                    'final_val_loss': final_val_loss
                })
                print(f"✅ Completed {run_name}: val_loss = {final_val_loss:.4f}")
                
            except Exception as e:
                print(f"❌ Error in {run_name}: {e}")
                continue
    
    # Выводим сводку результатов
    print("\n📊 РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ:")
    print("=" * 50)
    for result in results:
        print(f"{result['tokenizer_type'].upper()} vocab_size={result['vocab_size']}: "
              f"val_loss={result['final_val_loss']:.4f}")
    
    return results

if __name__ == "__main__":
    # Запуск экспериментов
    results = run_vocab_size_experiments()
    
    # Можно также запустить один эксперимент:
    # config = ModelConfig(vocab_size=1000, tokenizer_type='char')
    # run_experiment(config, "single_test")
