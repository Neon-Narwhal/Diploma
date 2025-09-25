"""Конфигурация модели и параметров обучения"""

from dataclasses import dataclass
import torch


@dataclass
class ModelConfig:
    """Конфигурация параметров модели и обучения"""
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
    model_name: str = "my_model_v1"
    seed: int = 1337

    # Параметры токенизации
    tokenizer_type: str = 'bpe'  # 'bpe' или 'char'
    experiment_name: str = "transformer_language_model"
    data_file: str = "data/input.txt"

    # Параметры генерации
    max_generation_chars: int = 500
    generation_temperature: float = 1.0
    generation_top_k: int = None
