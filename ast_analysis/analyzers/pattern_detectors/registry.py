import ast
from typing import Dict, Optional, Any
from .base import PatternDetector
from .sorting import SortingPatternDetector
from .search import SearchPatternDetector
from .dynamic_programming import DynamicProgrammingDetector
from .data_structures import DataStructurePatternDetector

class PatternDetectorRegistry:
    """Реестр детекторов паттернов"""
    
    def __init__(self):
        self.detectors: Dict[str, PatternDetector] = {}
        self._register_default_detectors()
    
    def _register_default_detectors(self):
        self.register(SortingPatternDetector())
        self.register(SearchPatternDetector())
        self.register(DynamicProgrammingDetector())
        self.register(DataStructurePatternDetector())
    
    def register(self, detector: PatternDetector):
        self.detectors[detector.name] = detector
    
    def detect_all(self, tree: ast.AST) -> Dict[str, Any]:
        results = {}
        for name, detector in self.detectors.items():
            try:
                results[name] = detector.detect(tree)
            except Exception as e:
                results[name] = {'error': str(e)}
        return results
