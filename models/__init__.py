"""Модели нейронных сетей"""

from .gpt_model import GPTLikeModel, Head, MultiheadAttention, FeedForward, Block
from .transformer_squared_model import TransformerSquared
__all__ = ['GPTLikeModel', 'Head', 'MultiheadAttention', 'FeedForward', 'Block', 'TransformerSquared']
