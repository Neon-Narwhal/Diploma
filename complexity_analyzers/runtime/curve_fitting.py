"""Подгонка кривых сложности к результатам бенчмарков"""
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from complexity_analyzers.base.enums import ComplexityClass

class FittingMethod(Enum):
    """Методы подгонки кривых"""
    LEAST_SQUARES = "least_squares"
    ROBUST = "robust"
    WEIGHTED = "weighted"
    BAYESIAN = "bayesian"

@dataclass
class FitResult:
    """Результат подгонки кривой"""
    complexity_class: ComplexityClass
    parameters: List[float]
    r_squared: float
    mse: float
    confidence: float
    method: FittingMethod
    function_name: str
    
class ComplexityFunction:
    """Базовый класс для функций сложности"""
    
    def __init__(self, name: str, complexity_class: ComplexityClass):
        self.name = name
        self.complexity_class = complexity_class
        self.param_count = 0
    
    def __call__(self, x: np.ndarray, *params) -> np.ndarray:
        """Вычисление функции"""
        raise NotImplementedError
    
    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        """Начальное приближение параметров"""
        return [1.0] * self.param_count
    
    def bounds(self) -> Tuple[List[float], List[float]]:
        """Границы параметров (lower, upper)"""
        return ([-np.inf] * self.param_count, [np.inf] * self.param_count)

class ConstantFunction(ComplexityFunction):
    """O(1) - константная сложность"""
    
    def __init__(self):
        super().__init__("O(1)", ComplexityClass.CONSTANT)
        self.param_count = 1
    
    def __call__(self, x: np.ndarray, a: float) -> np.ndarray:
        return np.full_like(x, a, dtype=float)
    
    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        return [float(np.mean(y))]

class LogarithmicFunction(ComplexityFunction):
    """O(log n) - логарифмическая сложность"""
    
    def __init__(self):
        super().__init__("O(log n)", ComplexityClass.LOGARITHMIC)
        self.param_count = 2
    
    def __call__(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        # Избегаем log(0)
        x_safe = np.maximum(x, 1e-10)
        return a * np.log(x_safe) + b
    
    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        # Линейная регрессия на log(x)
        x_safe = np.maximum(x, 1e-10)
        log_x = np.log(x_safe)
        if len(x) > 1:
            a = (y[-1] - y[0]) / (log_x[-1] - log_x[0])
            b = y[0] - a * log_x[0]
        else:
            a, b = 1.0, float(y[0])
        return [a, b]

class LinearFunction(ComplexityFunction):
    """O(n) - линейная сложность"""
    
    def __init__(self):
        super().__init__("O(n)", ComplexityClass.LINEAR)
        self.param_count = 2
    
    def __call__(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * x + b
    
    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        if len(x) > 1:
            a = (y[-1] - y[0]) / (x[-1] - x[0])
            b = y[0] - a * x[0]
        else:
            a, b = 1.0, float(y[0])
        return [a, b]

class LinearithmicFunction(ComplexityFunction):
    """O(n log n) - линеарифметическая сложность"""
    
    def __init__(self):
        super().__init__("O(n log n)", ComplexityClass.LINEARITHMIC)
        self.param_count = 2
    
    def __call__(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        x_safe = np.maximum(x, 1e-10)
        return a * x_safe * np.log(x_safe) + b
    
    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        x_safe = np.maximum(x, 1e-10)
        nlogn = x_safe * np.log(x_safe)
        if len(x) > 1 and nlogn[-1] != nlogn[0]:
            a = (y[-1] - y[0]) / (nlogn[-1] - nlogn[0])
            b = y[0] - a * nlogn[0]
        else:
            a, b = 1.0, float(y[0])
        return [a, b]

class QuadraticFunction(ComplexityFunction):
    """O(n²) - квадратичная сложность"""
    
    def __init__(self):
        super().__init__("O(n²)", ComplexityClass.QUADRATIC)
        self.param_count = 2
    
    def __call__(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * x**2 + b
    
    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        if len(x) > 1:
            x_sq = x**2
            a = (y[-1] - y[0]) / (x_sq[-1] - x_sq[0])
            b = y[0] - a * x_sq[0]
        else:
            a, b = 1.0, float(y[0])
        return [a, b]

class CubicFunction(ComplexityFunction):
    """O(n³) - кубическая сложность"""
    
    def __init__(self):
        super().__init__("O(n³)", ComplexityClass.CUBIC)
        self.param_count = 2
    
    def __call__(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * x**3 + b
    
    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        if len(x) > 1:
            x_cub = x**3
            if x_cub[-1] != x_cub[0]:
                a = (y[-1] - y[0]) / (x_cub[-1] - x_cub[0])
                b = y[0] - a * x_cub[0]
            else:
                a, b = 1.0, float(y[0])
        else:
            a, b = 1.0, float(y[0])
        return [a, b]

class ExponentialFunction(ComplexityFunction):
    """O(2^n) - экспоненциальная сложность"""
    
    def __init__(self):
        super().__init__("O(2^n)", ComplexityClass.EXPONENTIAL)
        self.param_count = 2
    
    def __call__(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        # Ограничиваем экспоненту для предотвращения переполнения
        exp_arg = np.minimum(x * b, 100)
        return a * np.exp(exp_arg)
    
    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        y_safe = np.maximum(y, 1e-10)
        if len(x) > 1 and y_safe[-1] > y_safe[0]:
            try:
                log_y = np.log(y_safe)
                b = (log_y[-1] - log_y[0]) / (x[-1] - x[0])
                a = y_safe[0] / np.exp(b * x[0])
                return [float(a), float(b)]
            except:
                pass
        return [1.0, 0.1]
    
    def bounds(self) -> Tuple[List[float], List[float]]:
        return ([1e-10, 1e-10], [1e10, 1.0])

class PolynomialFunction(ComplexityFunction):
    """O(n^k) - полиномиальная сложность"""
    
    def __init__(self, degree: int = 4):
        super().__init__(f"O(n^{degree})", ComplexityClass.POLYNOMIAL)
        self.degree = degree
        self.param_count = 2
    
    def __call__(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * x**self.degree + b
    
    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        if len(x) > 1:
            x_pow = x**self.degree
            if x_pow[-1] != x_pow[0]:
                a = (y[-1] - y[0]) / (x_pow[-1] - x_pow[0])
                b = y[0] - a * x_pow[0]
            else:
                a, b = 1.0, float(y[0])
        else:
            a, b = 1.0, float(y[0])
        return [a, b]

class ComplexityCurveFitter:
    """Основной класс для подгонки кривых сложности"""
    
    def __init__(self):
        self.functions = [
            ConstantFunction(),
            LogarithmicFunction(),
            LinearFunction(),
            LinearithmicFunction(),
            QuadraticFunction(),
            CubicFunction(),
            PolynomialFunction(4),
            PolynomialFunction(5),
            ExponentialFunction()
        ]
        
    def fit_complexity(self, sizes: List[int], times: List[float], 
                      method: FittingMethod = FittingMethod.LEAST_SQUARES) -> FitResult:
        """Подгонка кривой сложности к данным"""
        if len(sizes) < 2 or len(times) < 2:
            return FitResult(
                complexity_class=ComplexityClass.UNKNOWN,
                parameters=[],
                r_squared=0.0,
                mse=float('inf'),
                confidence=0.0,
                method=method,
                function_name="unknown"
            )
        
        x = np.array(sizes, dtype=float)
        y = np.array(times, dtype=float)
        
        # Фильтрация некорректных данных
        valid_mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        x = x[valid_mask]
        y = y[valid_mask]
        
        if len(x) < 2:
            return FitResult(
                complexity_class=ComplexityClass.UNKNOWN,
                parameters=[],
                r_squared=0.0,
                mse=float('inf'),
                confidence=0.0,
                method=method,
                function_name="unknown"
            )
        
        best_fit = None
        best_score = -float('inf')
        
        for func in self.functions:
            try:
                fit_result = self._fit_function(func, x, y, method)
                
                # Комбинированный скор: R² - штраф за сложность
                complexity_penalty = func.complexity_class.complexity_order * 0.05
                score = fit_result.r_squared - complexity_penalty
                
                if score > best_score:
                    best_score = score
                    best_fit = fit_result
                    
            except Exception:
                continue
        
        if best_fit is None:
            return FitResult(
                complexity_class=ComplexityClass.UNKNOWN,
                parameters=[],
                r_squared=0.0,
                mse=float('inf'),
                confidence=0.0,
                method=method,
                function_name="unknown"
            )
        
        return best_fit
    
    def _fit_function(self, func: ComplexityFunction, x: np.ndarray, y: np.ndarray,
                     method: FittingMethod) -> FitResult:
        """Подгонка конкретной функции"""
        try:
            from scipy.optimize import curve_fit
            
            # Начальное приближение
            initial_guess = func.initial_guess(x, y)
            bounds = func.bounds()
            
            # Подгонка параметров
            if method == FittingMethod.LEAST_SQUARES:
                popt, pcov = curve_fit(
                    func, x, y, 
                    p0=initial_guess,
                    bounds=bounds,
                    maxfev=1000
                )
            elif method == FittingMethod.ROBUST:
                # Robust fitting с использованием loss='soft_l1'
                from scipy.optimize import least_squares
                
                def residuals(params):
                    return func(x, *params) - y
                
                result = least_squares(
                    residuals, 
                    initial_guess,
                    bounds=bounds,
                    loss='soft_l1',
                    max_nfev=1000
                )
                popt = result.x
                pcov = None
            else:
                # Fallback к обычному методу наименьших квадратов
                popt, pcov = curve_fit(
                    func, x, y,
                    p0=initial_guess,
                    bounds=bounds,
                    maxfev=1000
                )
            
            # Вычисление метрик качества
            y_pred = func(x, *popt)
            r_squared = self._calculate_r_squared(y, y_pred)
            mse = np.mean((y - y_pred)**2)
            
            # Оценка уверенности
            confidence = self._calculate_confidence(r_squared, len(x), func.param_count)
            
            return FitResult(
                complexity_class=func.complexity_class,
                parameters=popt.tolist(),
                r_squared=r_squared,
                mse=mse,
                confidence=confidence,
                method=method,
                function_name=func.name
            )
            
        except Exception as e:
            return FitResult(
                complexity_class=func.complexity_class,
                parameters=[],
                r_squared=-1.0,
                mse=float('inf'),
                confidence=0.0,
                method=method,
                function_name=func.name
            )
    
    def _calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Вычисление коэффициента детерминации R²"""
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        return float(r_squared)
    
    def _calculate_confidence(self, r_squared: float, n_points: int, n_params: int) -> float:
        """Вычисление уверенности в подгонке"""
        if r_squared < 0:
            return 0.0
        
        # Скорректированный R²
        if n_points > n_params + 1:
            adjusted_r_squared = 1 - (1 - r_squared) * (n_points - 1) / (n_points - n_params - 1)
        else:
            adjusted_r_squared = r_squared
        
        # Базовая уверенность на основе скорректированного R²
        base_confidence = max(0, adjusted_r_squared)
        
        # Штраф за малое количество точек
        if n_points < 5:
            base_confidence *= 0.5
        elif n_points < 3:
            base_confidence *= 0.2
        
        return min(1.0, base_confidence)
    
    def compare_fits(self, sizes: List[int], times: List[float]) -> List[FitResult]:
        """Сравнение всех подгонок"""
        x = np.array(sizes, dtype=float)
        y = np.array(times, dtype=float)
        
        # Фильтрация данных
        valid_mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        x = x[valid_mask]
        y = y[valid_mask]
        
        if len(x) < 2:
            return []
        
        results = []
        
        for func in self.functions:
            try:
                fit_result = self._fit_function(func, x, y, FittingMethod.LEAST_SQUARES)
                if fit_result.r_squared >= 0:  # Только валидные результаты
                    results.append(fit_result)
            except:
                continue
        
        # Сортировка по качеству подгонки
        results.sort(key=lambda r: r.r_squared, reverse=True)
        
        return results
    
    def predict_time(self, fit_result: FitResult, size: int) -> Optional[float]:
        """Предсказание времени выполнения для заданного размера"""
        if not fit_result.parameters:
            return None
        
        # Находим соответствующую функцию
        target_func = None
        for func in self.functions:
            if func.complexity_class == fit_result.complexity_class:
                target_func = func
                break
        
        if target_func is None:
            return None
        
        try:
            x = np.array([size], dtype=float)
            prediction = target_func(x, *fit_result.parameters)
            return float(prediction[0])
        except:
            return None
    
    def extrapolate_performance(self, fit_result: FitResult, target_sizes: List[int]) -> Dict[int, float]:
        """Экстраполяция производительности на большие размеры"""
        predictions = {}
        
        for size in target_sizes:
            predicted_time = self.predict_time(fit_result, size)
            if predicted_time is not None:
                predictions[size] = predicted_time
        
        return predictions

class AdvancedCurveFitter(ComplexityCurveFitter):
    """Продвинутый анализатор с дополнительными функциями"""
    
    def __init__(self):
        super().__init__()
        
        # Добавляем специальные функции
        self.functions.extend([
            self._create_factorial_function(),
            self._create_sqrt_function(),
            self._create_log_squared_function()
        ])
    
    def _create_factorial_function(self) -> ComplexityFunction:
        """Приближение факториальной сложности"""
        class FactorialApproxFunction(ComplexityFunction):
            def __init__(self):
                super().__init__("O(n!)", ComplexityClass.FACTORIAL)
                self.param_count = 2
            
            def __call__(self, x, a, b):
                # Аппроксимация факториала через формулу Стирлинга
                x_safe = np.maximum(x, 1)
                # log(n!) ≈ n*log(n) - n + 0.5*log(2πn)
                log_factorial = x_safe * np.log(x_safe) - x_safe + 0.5 * np.log(2 * np.pi * x_safe)
                # Ограничиваем для предотвращения переполнения
                log_factorial = np.minimum(log_factorial, 100)
                return a * np.exp(log_factorial) + b
            
            def initial_guess(self, x, y):
                return [1e-10, 0.0]
            
            def bounds(self):
                return ([1e-15, -1e10], [1e10, 1e10])
        
        return FactorialApproxFunction()
    
    def _create_sqrt_function(self) -> ComplexityFunction:
        """O(√n) - корень из n"""
        class SqrtFunction(ComplexityFunction):
            def __init__(self):
                super().__init__("O(√n)", ComplexityClass.LINEAR)  # Между константой и линейной
                self.param_count = 2
            
            def __call__(self, x, a, b):
                return a * np.sqrt(np.maximum(x, 0)) + b
            
            def initial_guess(self, x, y):
                sqrt_x = np.sqrt(np.maximum(x, 1e-10))
                if len(x) > 1:
                    a = (y[-1] - y[0]) / (sqrt_x[-1] - sqrt_x[0])
                    b = y[0] - a * sqrt_x[0]
                else:
                    a, b = 1.0, float(y[0])
                return [a, b]
        
        return SqrtFunction()
    
    def _create_log_squared_function(self) -> ComplexityFunction:
        """O(log²n) - квадрат логарифма"""
        class LogSquaredFunction(ComplexityFunction):
            def __init__(self):
                super().__init__("O(log²n)", ComplexityClass.LOGARITHMIC)
                self.param_count = 2
            
            def __call__(self, x, a, b):
                x_safe = np.maximum(x, 1e-10)
                log_x = np.log(x_safe)
                return a * log_x**2 + b
            
            def initial_guess(self, x, y):
                x_safe = np.maximum(x, 1e-10)
                log_sq_x = np.log(x_safe)**2
                if len(x) > 1 and log_sq_x[-1] != log_sq_x[0]:
                    a = (y[-1] - y[0]) / (log_sq_x[-1] - log_sq_x[0])
                    b = y[0] - a * log_sq_x[0]
                else:
                    a, b = 1.0, float(y[0])
                return [a, b]
        
        return LogSquaredFunction()
