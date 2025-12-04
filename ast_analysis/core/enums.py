"""
Перечисления для AST модуля.
"""

from enum import Enum


class ComplexityClass(Enum):
    """Классы вычислительной сложности (согласованные с датасетом)"""
    CONSTANT = "constant"        # O(1)
    LOGARITHMIC = "logarithmic"  # O(log N)
    LINEAR = "linear"            # O(N)
    LINEARITHMIC = "linearithmic"# O(N log N)
    QUADRATIC = "quadratic"      # O(N^2)
    CUBIC = "cubic"              # O(N^3) - в датасете может быть 'polynomial'
    EXPONENTIAL = "exponential"  # O(2^N)
    FACTORIAL = "factorial"      # O(N!)
    UNKNOWN = "unknown"
    
    # Добавим маппинг для полиномиальных сложностей
    POLYNOMIAL = "polynomial"    # O(N^k), k > 2

    
    @property
    def order(self) -> int:
        """Порядок сложности для сравнения"""
        order_map = {
            self.CONSTANT: 1,
            self.LOGARITHMIC: 2,
            self.LINEAR: 3,
            self.LINEARITHMIC: 4,
            self.QUADRATIC: 5,
            self.CUBIC: 6,
            self.EXPONENTIAL: 7,
            self.FACTORIAL: 8,
            self.UNKNOWN: 0
        }
        return order_map[self]
    
    def __lt__(self, other):
        """Сравнение сложностей"""
        if not isinstance(other, ComplexityClass):
            return NotImplemented
        return self.order < other.order
    
    def __le__(self, other):
        if not isinstance(other, ComplexityClass):
            return NotImplemented
        return self.order <= other.order
    
    def __gt__(self, other):
        if not isinstance(other, ComplexityClass):
            return NotImplemented
        return self.order > other.order
    
    def __ge__(self, other):
        if not isinstance(other, ComplexityClass):
            return NotImplemented
        return self.order >= other.order
