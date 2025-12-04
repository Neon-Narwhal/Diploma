"""
Подбор сложности (Curve Fitting).
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import List, Tuple, Dict
from ast_analysis.core.enums import ComplexityClass

class ComplexityFitter:
    """
    Подбирает класс сложности O(...) под точки (N, Time).
    """
    
    def __init__(self):
        # Функции моделей: y = f(x, a, b)
        self.models = {
            ComplexityClass.CONSTANT: lambda n, a, b: a * np.ones_like(n) + b,
            ComplexityClass.LOGARITHMIC: lambda n, a, b: a * np.log(n + 1) + b, # log(n+1) to avoid log(0)
            ComplexityClass.LINEAR: lambda n, a, b: a * n + b,
            ComplexityClass.LINEARITHMIC: lambda n, a, b: a * n * np.log(n + 1) + b,
            ComplexityClass.QUADRATIC: lambda n, a, b: a * n**2 + b,
            # ComplexityClass.CUBIC: lambda n, a, b: a * n**3 + b, # Слишком редко для тестов
        }
        
    def fit(self, ns: List[int], times: List[float]) -> Tuple[ComplexityClass, float]:
        """
        Определяет наилучший класс сложности.
        Returns: (ComplexityClass, Error_MSE)
        """
        if len(ns) < 3:
            return ComplexityClass.UNKNOWN, float('inf')
            
        x_data = np.array(ns, dtype=float)
        y_data = np.array(times, dtype=float)
        
        # Нормализация для стабильности фиттинга
        y_max = np.max(y_data)
        if y_max > 1e-9:
            y_data = y_data / y_max
        else:
            return ComplexityClass.CONSTANT, 0.0
            
        best_complexity = ComplexityClass.UNKNOWN
        min_error = float('inf')
        
        for comp_class, func in self.models.items():
            try:
                popt, _ = curve_fit(func, x_data, y_data, maxfev=2000)
                y_pred = func(x_data, *popt)
                mse = np.mean((y_data - y_pred) ** 2)
                
                # Штрафы (Occam's razor)
                penalty = 0.0
                if comp_class == ComplexityClass.CONSTANT: penalty = 0.0
                elif comp_class == ComplexityClass.LINEAR: penalty = 0.005
                elif comp_class == ComplexityClass.QUADRATIC: penalty = 0.02
                
                score = mse + penalty
                
                if score < min_error:
                    min_error = score
                    best_complexity = comp_class
                    
            except Exception:
                continue
                
        return best_complexity, min_error
