"""
Пайплайн для CFG анализа.
Использует общую инфраструктуру, так как интерфейс совместим.
"""
from ast_analysis.processing.pipeline import ASTPipeline as CFGPipeline

__all__ = ['CFGPipeline']
