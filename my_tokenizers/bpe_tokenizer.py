"""BPE токенизатор"""

from typing import List
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from my_tokenizers.base_tokenizer import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """BPE (Byte Pair Encoding) токенизатор"""
    
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
