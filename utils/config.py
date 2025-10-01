"""Конфигурация модели и параметров обучения"""

from dataclasses import dataclass
import torch
from typing import Optional, Dict, Any

@dataclass
class ModelConfig:
    """Универсальная конфигурация для всех типов моделей"""

    # ========== ОСНОВНЫЕ ПАРАМЕТРЫ МОДЕЛИ ==========
    batch_size: int = 64
    block_size: int = 256
    vocab_size: int = 3000
    max_iters: int = 2500
    eval_interval: int = 500
    learning_rate: float = 3e-4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters: int = 200
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256
    epoch_count: int = 1
    overfit_line: float = 0.05
    dropout: float = 0.1
    model_name: str = "my_model_v1"
    experiment_name: str = "experiment_name"
    seed: int = 1337

    # ========== ПАРАМЕТРЫ РАЗДЕЛЕНИЯ ДАННЫХ ==========
    train_val_split: float = 0.9
    split_flag: bool = True

    # ========== ПАРАМЕТРЫ ТОКЕНИЗАЦИИ ==========
    tokenizer_type: str = 'bpe'  # 'bpe' или 'char'
    experiment_name: str = "transformer_language_model"
    data_file: str = "data/input.txt"

    # ========== типы данных ==========
    use_amp = True  # Использовать automatic mixed precision
    amp_dtype = torch.bfloat16  # Тип для AMP (bfloat16 или float16)
    model_dtype = torch.bfloat16  # Базовый тип весов модели

    # ========== ПАРАМЕТРЫ ГЕНЕРАЦИИ ==========
    max_generation_chars: int = 500
    generation_temperature: float = 1.0
    generation_top_k: Optional[int] = None
    max_generation_length: int = 1000

    # ========== TRANSFORMER² / SAKANA AI ПАРАМЕТРЫ ==========
    # Главный флаг для включения Transformer² архитектуры
    use_transformer_squared: bool = False

    # SVF (Singular Value Fine-tuning) параметры
    use_svf: bool = True
    svf_rank_ratio: float = 0.1
    svf_bias: bool = True

    # Expert system параметры
    num_expert_vectors: int = 3
    expert_vector_init_std: float = 0.02

    # Task-specific параметры
    num_complexity_classes: int = 7
    enable_complexity_head: bool = True
    enable_task_detector: bool = True

    # Инициализация модели
    weight_init_std: float = 0.02
    embedding_init_std: float = 0.02

    # Конфигурация архитектуры
    use_pre_norm: bool = True
    use_bias_in_attention: bool = False
    use_bias_in_ffn: bool = True
    ffn_expansion_factor: int = 4
    use_rope: bool = True

    # Attention конфигурация
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1

    # ========== МЕТОДЫ КОНФИГУРАЦИИ ==========

    def enable_transformer_squared_features(self):
        """Включает все возможности Transformer² / Sakana AI"""
        self.use_transformer_squared = True
        self.use_svf = True
        self.enable_complexity_head = True
        self.enable_task_detector = True
        return self

    def disable_transformer_squared_features(self):
        """Отключает все возможности Transformer² (стандартный трансформер)"""
        self.use_transformer_squared = False
        self.use_svf = False
        self.enable_complexity_head = False
        self.enable_task_detector = False
        return self

    def set_complexity_analysis_mode(self):
        """Настройка для анализа сложности кода"""
        self.use_transformer_squared = True
        self.enable_complexity_head = True
        self.enable_task_detector = True
        self.vocab_size = 8000  # Больший словарь для кода
        self.block_size = 1024  # Длинные последовательности
        self.svf_rank_ratio = 0.2  # Больший ранг для сложных паттернов
        self.ffn_expansion_factor = 6  # Больше FFN для анализа
        self.attention_dropout = 0.05  # Меньший dropout для структуры
        self.residual_dropout = 0.05
        self.num_complexity_classes = 12  # Расширенные классы сложности
        return self

    def set_small_model_config(self):
        """Конфигурация для небольшой модели (тестирование, разработка)"""
        self.n_embd = 128
        self.n_layer = 4
        self.n_head = 4
        self.vocab_size = 1000
        self.block_size = 128
        self.batch_size = 16
        return self

    def set_medium_model_config(self):
        """Конфигурация для средней модели (эксперименты)"""
        self.n_embd = 384
        self.n_layer = 8
        self.n_head = 6
        self.vocab_size = 3000
        self.block_size = 256
        self.batch_size = 32
        return self

    def set_large_model_config(self):
        """Конфигурация для большой модели (продакшен)"""
        self.n_embd = 768
        self.n_layer = 12
        self.n_head = 12
        self.vocab_size = 5000
        self.block_size = 512
        self.batch_size = 16  # Меньший batch для больших моделей
        self.svf_rank_ratio = 0.1  # Оптимальный ранг для больших моделей
        return self

    def set_training_params(self, learning_rate: float = None, max_iters: int = None, 
                           batch_size: int = None, eval_interval: int = None):
        """Установка параметров обучения"""
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if max_iters is not None:
            self.max_iters = max_iters
        if batch_size is not None:
            self.batch_size = batch_size
        if eval_interval is not None:
            self.eval_interval = eval_interval
        return self

    def set_svf_params(self, use_svf: bool = True, rank_ratio: float = 0.1, bias: bool = True):
        """Настройка параметров SVF"""
        self.use_svf = use_svf
        self.svf_rank_ratio = rank_ratio
        self.svf_bias = bias
        return self

    def set_expert_system(self, num_experts: int = 3, init_std: float = 0.02):
        """Настройка системы экспертных векторов"""
        self.num_expert_vectors = num_experts
        self.expert_vector_init_std = init_std
        return self

    def set_dropout_rates(self, base_dropout: float = None, attention_dropout: float = None, 
                         residual_dropout: float = None):
        """Настройка уровней dropout"""
        if base_dropout is not None:
            self.dropout = base_dropout
        if attention_dropout is not None:
            self.attention_dropout = attention_dropout
        if residual_dropout is not None:
            self.residual_dropout = residual_dropout
        return self

    def set_generation_params(self, temperature: float = None, max_length: int = None, 
                            top_k: Optional[int] = None):
        """Настройка параметров генерации"""
        if temperature is not None:
            self.generation_temperature = temperature
        if max_length is not None:
            self.max_generation_length = max_length
        if top_k is not None:
            self.generation_top_k = top_k
        return self

    # ========== ВАЛИДАЦИЯ И УТИЛИТЫ ==========

    def validate(self) -> bool:
        """Валидация конфигурации"""
        assert self.n_embd % self.n_head == 0, "n_embd должно быть кратно n_head"
        assert self.vocab_size > 0, "vocab_size должен быть положительным"
        assert self.block_size > 0, "block_size должен быть положительным"
        assert 0.0 <= self.dropout <= 1.0, "dropout должен быть в диапазоне [0, 1]"
        assert 0.0 < self.svf_rank_ratio <= 1.0, "svf_rank_ratio должен быть в диапазоне (0, 1]"
        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Получить информацию о конфигурации модели"""
        return {
            'model_name': self.model_name,
            'model_type': 'Transformer²' if self.use_transformer_squared else 'Standard Transformer',
            'parameters_estimate': self._estimate_parameters(),
            'n_layers': self.n_layer,
            'n_heads': self.n_head,
            'n_embd': self.n_embd,
            'vocab_size': self.vocab_size,
            'block_size': self.block_size,
            'use_svf': self.use_svf,
            'svf_rank_ratio': self.svf_rank_ratio if self.use_svf else None,
            'num_expert_vectors': self.num_expert_vectors,
            'supports_complexity_analysis': self.enable_complexity_head,
            'supports_task_detection': self.enable_task_detector,
        }

    def _estimate_parameters(self) -> int:
        """Приблизительная оценка количества параметров"""
        # Embeddings
        embedding_params = self.vocab_size * self.n_embd + self.block_size * self.n_embd

        # Transformer blocks
        # Attention: 4 * n_embd^2 (Q, K, V, proj)
        # FFN: 2 * n_embd * (ffn_expansion_factor * n_embd)
        attention_params = 4 * self.n_embd * self.n_embd
        ffn_params = 2 * self.n_embd * (self.ffn_expansion_factor * self.n_embd)
        block_params = (attention_params + ffn_params) * self.n_layer

        # Output head
        output_params = self.n_embd * self.vocab_size

        # SVF adjustment (приблизительно)
        if self.use_svf:
            # SVF использует меньше параметров
            svf_reduction = 1 - self.svf_rank_ratio
            block_params = int(block_params * (1 - svf_reduction * 0.5))  # Приблизительно

        # Complexity head (если включена)
        complexity_params = 0
        if self.enable_complexity_head:
            complexity_params = self.n_embd * self.num_complexity_classes

        total = embedding_params + block_params + output_params + complexity_params
        return int(total)

    def copy(self):
        """Создать копию конфигурации"""
        return ModelConfig(**self.__dict__)

    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Обновить конфигурацию из словаря"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Конвертировать в словарь"""
        return {k: v for k, v in self.__dict__.items()}

    def save_to_file(self, filepath: str):
        """Сохранить конфигурацию в файл"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str):
        """Загрузить конфигурацию из файла"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    # ========== ПРЕДУСТАНОВЛЕННЫЕ КОНФИГУРАЦИИ ==========

    @classmethod
    def get_comparison_config(cls):
        """Конфигурация для сравнения с оригинальной моделью"""
        config = cls()
        config.enable_transformer_squared_features()
        config.model_name = "transformer_squared_comparison"
        return config

    @classmethod
    def get_code_analysis_config(cls):
        """Оптимизированная конфигурация для анализа сложности кода"""
        config = cls()
        config.set_complexity_analysis_mode()
        config.model_name = "code_complexity_analyzer"
        return config

    @classmethod
    def get_small_research_config(cls):
        """Небольшая модель для исследований"""
        config = cls()
        config.set_small_model_config()
        config.enable_transformer_squared_features()
        config.model_name = "small_transformer_squared"
        return config

    @classmethod
    def get_large_production_config(cls):
        """Большая модель для продакшена"""
        config = cls()
        config.set_large_model_config()
        config.enable_transformer_squared_features()
        config.model_name = "large_transformer_squared"
        return config

    @classmethod
    def get_standard_config(cls):
        """Стандартная конфигурация трансформера (без T² возможностей)"""
        config = cls()
        config.disable_transformer_squared_features()
        config.model_name = "standard_transformer"
        return config

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========

def create_config_for_task(task: str, model_size: str = "medium") -> ModelConfig:
    """
    Создать конфигурацию для конкретной задачи

    Args:
        task: 'language_modeling', 'code_analysis', 'comparison', 'standard'
        model_size: 'small', 'medium', 'large'
    """
    if model_size == "small":
        config = ModelConfig.get_small_research_config()
    elif model_size == "large":
        config = ModelConfig.get_large_production_config()
    else:  # medium
        config = ModelConfig()
        config.set_medium_model_config()

    if task == "code_analysis":
        config.set_complexity_analysis_mode()
    elif task == "comparison":
        config.enable_transformer_squared_features()
    elif task == "standard":
        config.disable_transformer_squared_features()
    else:  # language_modeling
        config.enable_transformer_squared_features()

    return config

def merge_configs(base_config: ModelConfig, override_dict: Dict[str, Any]) -> ModelConfig:
    """Создать новую конфигурацию, объединив базовую с переопределениями"""
    new_config = base_config.copy()
    new_config.update_from_dict(override_dict)
    return new_config