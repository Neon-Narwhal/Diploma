"""Конфигурации для анализаторов сложности"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from complexity_analyzers.base.enums import AnalyzerType, ComplexityClass

@dataclass
class AnalyzerConfig:
    """Базовая конфигурация анализатора"""
    name: str
    enabled: bool = True
    timeout: int = 30
    priority: int = 1
    weights: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ASTAnalyzerConfig(AnalyzerConfig):
    """Конфигурация AST-анализатора"""
    enable_pattern_detection: bool = True
    enable_feature_extraction: bool = True
    pattern_detectors: List[str] = field(default_factory=lambda: [
        'sorting_patterns', 'search_patterns', 'dp_patterns', 'data_structure_patterns'
    ])
    feature_extractors: List[str] = field(default_factory=lambda: [
        'basic_features', 'complexity_features', 'textual_features'
    ])

@dataclass
class RuntimeAnalyzerConfig(AnalyzerConfig):
    """Конфигурация runtime-анализатора"""
    test_sizes: List[int] = field(default_factory=lambda: [10, 50, 100, 500, 1000])
    iterations_per_size: int = 5
    warmup_iterations: int = 2
    use_subprocess: bool = True
    measure_memory: bool = True
    curve_fitting_method: str = 'least_squares'

@dataclass
class CFGAnalyzerConfig(AnalyzerConfig):
    """Конфигурация CFG-анализатора"""
    include_exception_edges: bool = True
    simplify_graph: bool = False
    calculate_dominance: bool = True
    metrics_to_calculate: List[str] = field(default_factory=lambda: [
        'cyclomatic_complexity', 'nesting_depth', 'path_complexity'
    ])

@dataclass
class MLAnalyzerConfig(AnalyzerConfig):
    """Конфигурация ML-анализатора"""
    models_to_use: List[str] = field(default_factory=lambda: [
        'random_forest', 'xgboost', 'neural_net'
    ])
    feature_selection: bool = True
    ensemble_method: str = 'weighted_voting'
    confidence_threshold: float = 0.5
    retrain_threshold: int = 100  # Количество новых образцов для переобучения

@dataclass
class DynamicAnalyzerConfig(AnalyzerConfig):
    """Конфигурация динамического анализатора"""
    trace_method: str = 'safe_subprocess'  # 'direct', 'subprocess', 'safe_subprocess'
    trace_timeout: int = 10
    max_recursion_depth: int = 100
    enable_memory_tracking: bool = True
    test_data_types: List[str] = field(default_factory=lambda: ['list', 'matrix', 'graph'])

@dataclass
class HybridAnalyzerConfig(AnalyzerConfig):
    """Конфигурация гибридного анализатора"""
    enabled_analyzers: List[str] = field(default_factory=lambda: [
        'ast_advanced', 'runtime_profiler', 'cfg_analyzer', 'ml_predictor'
    ])
    weighting_strategy: str = 'adaptive'  # 'uniform', 'confidence_based', 'performance_based', 'adaptive'
    voting_strategy: str = 'weighted'  # 'majority', 'weighted', 'confidence_weighted'
    conflict_resolution: str = 'expert_system'  # 'conservative', 'confidence_based', 'expert_system'
    min_analyzers_required: int = 2

class ConfigManager:
    """Менеджер конфигураций анализаторов"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path("utils/configs")
        self.configs: Dict[str, AnalyzerConfig] = {}
        self._load_default_configs()
    
    def _load_default_configs(self):
        """Загрузка конфигураций по умолчанию"""
        # AST анализатор
        self.configs['ast_advanced'] = ASTAnalyzerConfig(
            name='ast_advanced',
            enabled=True,
            timeout=30,
            priority=1,
            weights={'pattern_detection': 0.4, 'feature_extraction': 0.6}
        )
        
        # Runtime анализатор
        self.configs['runtime_profiler'] = RuntimeAnalyzerConfig(
            name='runtime_profiler',
            enabled=True,
            timeout=60,
            priority=2,
            test_sizes=[10, 50, 100, 500, 1000, 2000],
            iterations_per_size=3,
            use_subprocess=True
        )
        
        # CFG анализатор
        self.configs['cfg_analyzer'] = CFGAnalyzerConfig(
            name='cfg_analyzer',
            enabled=True,
            timeout=45,
            priority=3,
            include_exception_edges=True,
            calculate_dominance=False  # Может быть медленным
        )
        
        # ML анализатор
        self.configs['ml_predictor'] = MLAnalyzerConfig(
            name='ml_predictor',
            enabled=True,
            timeout=20,
            priority=4,
            models_to_use=['random_forest', 'xgboost'],
            confidence_threshold=0.6
        )
        
        # Динамический анализатор
        self.configs['dynamic_tracer'] = DynamicAnalyzerConfig(
            name='dynamic_tracer',
            enabled=False,  # По умолчанию отключен (может быть небезопасным)
            timeout=30,
            priority=5,
            trace_method='safe_subprocess',
            trace_timeout=5
        )
        
        # Гибридный анализатор
        self.configs['hybrid_ensemble'] = HybridAnalyzerConfig(
            name='hybrid_ensemble',
            enabled=True,
            timeout=120,
            priority=6,
            enabled_analyzers=['ast_advanced', 'runtime_profiler', 'cfg_analyzer', 'ml_predictor'],
            weighting_strategy='adaptive',
            min_analyzers_required=2
        )
    
    def get_config(self, analyzer_name: str) -> Optional[AnalyzerConfig]:
        """Получение конфигурации анализатора"""
        return self.configs.get(analyzer_name)
    
    def set_config(self, analyzer_name: str, config: AnalyzerConfig):
        """Установка конфигурации анализатора"""
        self.configs[analyzer_name] = config
    
    def update_config(self, analyzer_name: str, updates: Dict[str, Any]):
        """Обновление конфигурации анализатора"""
        if analyzer_name in self.configs:
            config = self.configs[analyzer_name]
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    def get_enabled_analyzers(self) -> List[str]:
        """Получение списка включенных анализаторов"""
        return [name for name, config in self.configs.items() if config.enabled]
    
    def disable_analyzer(self, analyzer_name: str):
        """Отключение анализатора"""
        if analyzer_name in self.configs:
            self.configs[analyzer_name].enabled = False
    
    def enable_analyzer(self, analyzer_name: str):
        """Включение анализатора"""
        if analyzer_name in self.configs:
            self.configs[analyzer_name].enabled = True
    
    def save_config_to_file(self, filename: str):
        """Сохранение конфигураций в файл"""
        import json
        
        config_data = {}
        for name, config in self.configs.items():
            if hasattr(config, '__dict__'):
                config_data[name] = {
                    'type': type(config).__name__,
                    'data': config.__dict__
                }
        
        filepath = self.config_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, default=str)
    
    def load_config_from_file(self, filename: str):
        """Загрузка конфигураций из файла"""
        import json
        
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            for name, config_info in config_data.items():
                config_type = config_info.get('type')
                config_data_dict = config_info.get('data', {})
                
                # Создаем конфигурацию соответствующего типа
                if config_type == 'ASTAnalyzerConfig':
                    config = ASTAnalyzerConfig(**config_data_dict)
                elif config_type == 'RuntimeAnalyzerConfig':
                    config = RuntimeAnalyzerConfig(**config_data_dict)
                elif config_type == 'CFGAnalyzerConfig':
                    config = CFGAnalyzerConfig(**config_data_dict)
                elif config_type == 'MLAnalyzerConfig':
                    config = MLAnalyzerConfig(**config_data_dict)
                elif config_type == 'DynamicAnalyzerConfig':
                    config = DynamicAnalyzerConfig(**config_data_dict)
                elif config_type == 'HybridAnalyzerConfig':
                    config = HybridAnalyzerConfig(**config_data_dict)
                else:
                    config = AnalyzerConfig(**config_data_dict)
                
                self.configs[name] = config
                
        except Exception as e:
            print(f"Error loading config from {filename}: {e}")

class PresetConfigurations:
    """Предустановленные конфигурации для разных сценариев"""
    
    @staticmethod
    def get_fast_analysis_config() -> Dict[str, AnalyzerConfig]:
        """Быстрый анализ (только статические методы)"""
        return {
            'ast_advanced': ASTAnalyzerConfig(
                name='ast_advanced',
                enabled=True,
                timeout=15,
                enable_pattern_detection=True,
                enable_feature_extraction=False
            ),
            'cfg_analyzer': CFGAnalyzerConfig(
                name='cfg_analyzer',
                enabled=True,
                timeout=20,
                include_exception_edges=False,
                calculate_dominance=False
            ),
            'hybrid_ensemble': HybridAnalyzerConfig(
                name='hybrid_ensemble',
                enabled=True,
                enabled_analyzers=['ast_advanced', 'cfg_analyzer'],
                weighting_strategy='uniform'
            )
        }
    
    @staticmethod
    def get_comprehensive_analysis_config() -> Dict[str, AnalyzerConfig]:
        """Всесторонний анализ (все методы)"""
        return {
            'ast_advanced': ASTAnalyzerConfig(
                name='ast_advanced',
                enabled=True,
                timeout=60
            ),
            'runtime_profiler': RuntimeAnalyzerConfig(
                name='runtime_profiler',
                enabled=True,
                timeout=120,
                test_sizes=[10, 50, 100, 500, 1000, 2000, 5000],
                iterations_per_size=5
            ),
            'cfg_analyzer': CFGAnalyzerConfig(
                name='cfg_analyzer',
                enabled=True,
                timeout=60,
                calculate_dominance=True
            ),
            'ml_predictor': MLAnalyzerConfig(
                name='ml_predictor',
                enabled=True,
                timeout=40,
                models_to_use=['random_forest', 'xgboost', 'neural_net']
            ),
            'dynamic_tracer': DynamicAnalyzerConfig(
                name='dynamic_tracer',
                enabled=True,
                timeout=60,
                trace_method='safe_subprocess'
            ),
            'hybrid_ensemble': HybridAnalyzerConfig(
                name='hybrid_ensemble',
                enabled=True,
                timeout=300,
                enabled_analyzers=['ast_advanced', 'runtime_profiler', 'cfg_analyzer', 'ml_predictor', 'dynamic_tracer'],
                weighting_strategy='adaptive',
                conflict_resolution='expert_system'
            )
        }
    
    @staticmethod
    def get_research_config() -> Dict[str, AnalyzerConfig]:
        """Конфигурация для исследований (максимальная детализация)"""
        return {
            'ast_advanced': ASTAnalyzerConfig(
                name='ast_advanced',
                enabled=True,
                timeout=90,
                enable_pattern_detection=True,
                enable_feature_extraction=True,
                pattern_detectors=[
                    'sorting_patterns', 'search_patterns', 'dp_patterns',
                    'data_structure_patterns', 'recursive_patterns', 'iterative_patterns'
                ]
            ),
            'runtime_profiler': RuntimeAnalyzerConfig(
                name='runtime_profiler',
                enabled=True,
                timeout=180,
                test_sizes=[5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
                iterations_per_size=10,
                measure_memory=True,
                curve_fitting_method='robust'
            ),
            'cfg_analyzer': CFGAnalyzerConfig(
                name='cfg_analyzer',
                enabled=True,
                timeout=120,
                include_exception_edges=True,
                calculate_dominance=True,
                metrics_to_calculate=[
                    'cyclomatic_complexity', 'nesting_depth', 'path_complexity',
                    'fan_in_out', 'structural_complexity'
                ]
            )
        }
    
    @staticmethod
    def get_production_config() -> Dict[str, AnalyzerConfig]:
        """Конфигурация для продакшена (надежность и скорость)"""
        return {
            'ast_advanced': ASTAnalyzerConfig(
                name='ast_advanced',
                enabled=True,
                timeout=30,
                enable_pattern_detection=True,
                enable_feature_extraction=False
            ),
            'ml_predictor': MLAnalyzerConfig(
                name='ml_predictor',
                enabled=True,
                timeout=20,
                models_to_use=['random_forest'],  # Только быстрая модель
                confidence_threshold=0.7
            ),
            'hybrid_ensemble': HybridAnalyzerConfig(
                name='hybrid_ensemble',
                enabled=True,
                timeout=60,
                enabled_analyzers=['ast_advanced', 'ml_predictor'],
                weighting_strategy='performance_based',
                min_analyzers_required=1
            )
        }

# Глобальный менеджер конфигураций
config_manager = ConfigManager()

def get_analyzer_config(analyzer_name: str) -> Optional[AnalyzerConfig]:
    """Получение конфигурации анализатора"""
    return config_manager.get_config(analyzer_name)

def set_preset_configuration(preset_name: str):
    """Установка предустановленной конфигурации"""
    presets = {
        'fast': PresetConfigurations.get_fast_analysis_config(),
        'comprehensive': PresetConfigurations.get_comprehensive_analysis_config(),
        'research': PresetConfigurations.get_research_config(),
        'production': PresetConfigurations.get_production_config()
    }
    
    if preset_name in presets:
        for name, config in presets[preset_name].items():
            config_manager.set_config(name, config)
    else:
        raise ValueError(f"Unknown preset: {preset_name}")
