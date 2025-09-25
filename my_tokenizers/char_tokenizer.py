"""Символьный токенизатор"""

from typing import List, Dict
import json
from my_tokenizers.base_tokenizer import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    """Символьный токенизатор (character-level)"""
    
    def __init__(self, data: str):
        chars = sorted(set(data))
        self.chars = chars
        self.vocab_size = len(chars)
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}
    
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
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
