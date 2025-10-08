# complexity_analyzer/analyzers.py (ОБНОВЛЕННЫЙ - объединяет baseline и enhanced)
"""
Главный файл анализаторов - предоставляет единый интерфейс к baseline и enhanced версиям.
"""

from analyzers_baseline import get_baseline_analyzer, BASELINE_ANALYZER_FACTORY
#from analyzers_enhanced import get_enhanced_analyzer, ENHANCED_ANALYZER_FACTORY
from config import ANALYZER_MODE


def get_analyzer(tool_name: str):
    """
    Получить анализатор (baseline или enhanced) в зависимости от конфигурации.
    
    Args:
        tool_name: Имя инструмента ('radon', 'lizard', 'mccabe')
    """
    if ANALYZER_MODE == 'baseline':
        return get_baseline_analyzer(f'{tool_name}_baseline')
    elif ANALYZER_MODE == 'enhanced':
        return get_enhanced_analyzer(f'{tool_name}_enhanced')
    else:
        raise ValueError(f"Unknown ANALYZER_MODE: {ANALYZER_MODE}")


# Экспортируем обе фабрики
ANALYZER_FACTORY = {
    **BASELINE_ANALYZER_FACTORY#,
    #**ENHANCED_ANALYZER_FACTORY
}
