"""
Конфигурация AST анализатора.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ASTAnalyzerConfig:
    """
    Конфигурация одного AST анализатора.
    """
    name: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.type,
            'params': self.params,
            'enabled': self.enabled
        }
