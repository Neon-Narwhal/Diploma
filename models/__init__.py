"""Модели нейронных сетей"""

from .gpt_model import GPTLikeModel, Head, MultiheadAttention, FeedForward, Block

__all__ = ['GPTLikeModel', 'Head', 'MultiheadAttention', 'FeedForward', 'Block']
