"""Загрузчики различных датасетов"""

from datasets.loaders.bigobench_loader import (
    BigOBenchLoader,
    BigOBenchSample,
    BigOBenchIterator
)

__all__ = [
    'BigOBenchLoader',
    'BigOBenchSample', 
    'BigOBenchIterator',
]
