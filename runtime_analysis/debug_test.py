"""
Отладочный скрипт для проверки RuntimeAnalyzer.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from runtime_analysis.core.generators import InputGenerator
from runtime_analysis.core.execution import CodeExecutor

# Простой тестовый код
test_code = """
def test_func(arr):
    total = 0
    for x in arr:
        total += x
    return total
"""

def main():
    print("Testing RuntimeAnalyzer components...")
    
    # 1. Тест генератора
    print("\n1. Testing InputGenerator...")
    gen = InputGenerator()
    data = gen.generate("list_int", 10)
    print(f"   Generated data (n=10): {data[:5]}... (length={len(data)})")
    
    # 2. Тест executor
    print("\n2. Testing CodeExecutor...")
    executor = CodeExecutor(timeout=2.0)
    
    for n in [10, 100, 1000]:
        data = gen.generate("list_int", n)
        t = executor.measure_time(test_code, data)
        print(f"   N={n:4d}: time={t:.6f}s")
        
    # 3. Тест через RuntimeAnalyzer
    print("\n3. Testing RuntimeAnalyzer...")
    import runtime_analysis.core.analyzer
    from runtime_analysis.core.analyzer import RuntimeAnalyzer
    
    analyzer = RuntimeAnalyzer(timeout=2.0)
    result = analyzer.analyze(test_code)
    
    print(f"   Success: {result.success}")
    print(f"   Prediction: {result.prediction}")
    print(f"   Confidence: {result.confidence}")
    print(f"   Metadata: {result.prediction_metadata}")
    
if __name__ == "__main__":
    main()
