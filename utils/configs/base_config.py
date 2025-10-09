# =============================================================================
# utils/configs/base_config.py
# Базовый конфиг для всех моделей - содержит ВСЕ общие параметры
# =============================================================================

from dataclasses import dataclass, asdict
from typing import Optional, Literal
import torch
from pathlib import Path


@dataclass
class BaseTrainingConfig:
    """
    Базовая конфигурация для обучения любых моделей.
    
    Содержит ВСЕ общие параметры обучения, оптимизации, данных и системы.
    Model-specific параметры (n_embd, n_layer) добавляются в наследниках.
    
    Философия: Base конфиг = всё что НЕ зависит от архитектуры модели.
    """
    
    # ============================================================================
    # ДАННЫЕ
    # ============================================================================
    
    data_path: str = "data/complexity_labels_full/complexity_labels_full_0-49.json"
    """Путь к данным для обучения (JSONL файл для BigOBench)."""
    
    train_split: float = 0.9
    """Доля данных для тренировки (остальное - валидация)."""
    
    filter_missing_complexity: bool = True
    """Пропускать примеры без time/space complexity в BigOBench."""
    
    use_dataclass_code: bool = False
    """Использовать query_dataclass_code вместо query_code в BigOBench."""
    
    # ============================================================================
    # ТОКЕНИЗАЦИЯ
    # ============================================================================
    
    tokenizer_type: str = 'bpe'
    """Тип токенизатора: 'bpe' или 'char'."""
    
    vocab_size: int = 5000
    """Размер словаря. Будет обновлен после создания токенизатора."""
    
    block_size: int = 512
    """Максимальная длина последовательности (context window)."""
    
    # ============================================================================
    # ОБУЧЕНИЕ - ОСНОВНЫЕ ПАРАМЕТРЫ
    # ============================================================================
    
    batch_size: int = 8
    """Batch size для обучения (для RTX 3060 12GB: 4-16)."""
    
    learning_rate: float = 3e-4
    """Learning rate. Типичные значения: 1e-4 до 5e-4."""
    
    weight_decay: float = 0.01
    """L2 regularization (weight decay). Стандарт: 0.01."""
    
    max_epochs: int = 10
    """Максимальное количество эпох обучения."""
    
    max_iters: Optional[int] = None
    """Максимальное количество итераций (если задано, приоритет над max_epochs)."""
    
    # ============================================================================
    # ОПТИМИЗАТОР
    # ============================================================================
    
    optimizer_type: str = 'adamw'
    """Тип оптимизатора: 'adamw', 'adam', 'sgd'."""
    
    beta1: float = 0.9
    """Beta1 для Adam/AdamW оптимизаторов."""
    
    beta2: float = 0.95
    """Beta2 для Adam/AdamW оптимизаторов (0.95 для GPT, 0.999 стандарт)."""
    
    epsilon: float = 1e-8
    """Epsilon для Adam/AdamW (численная стабильность)."""

    model_dtype = torch.bfloat16
    
    # ============================================================================
    # LEARNING RATE SCHEDULER
    # ============================================================================
    
    scheduler_type: str = 'cosine'
    """Тип scheduler: 'cosine', 'linear', 'constant', 'warmup_cosine'."""
    
    warmup_steps: int = 0
    """Количество warmup шагов (0 = без warmup)."""
    
    warmup_ratio: float = 0.0
    """Доля от total_steps для warmup (альтернатива warmup_steps)."""
    
    min_lr: float = 1e-6
    """Минимальный learning rate для cosine scheduler."""
    
    # ============================================================================
    # ГРАДИЕНТЫ
    # ============================================================================
    
    gradient_accumulation_steps: int = 4
    """Накопление градиентов (эффективный batch = batch_size * grad_accum)."""
    
    max_grad_norm: float = 1.0
    """Максимальная норма градиента для clipping (стабилизация обучения)."""
    
    # ============================================================================
    # MIXED PRECISION & MEMORY
    # ============================================================================
    
    use_amp: bool = True
    """Использовать Automatic Mixed Precision (FP16). Экономит память и ускоряет."""
    
    gradient_checkpointing: bool = False
    """Gradient checkpointing (экономит память, замедляет ~20%)."""
    
    # ============================================================================
    # РЕГУЛЯРИЗАЦИЯ
    # ============================================================================
    
    dropout: float = 0.1
    """Общий dropout rate (если модель не определяет специфичные)."""
    
    label_smoothing: float = 0.0
    """Label smoothing для loss функции (0.0 = выключен)."""
    
    # ============================================================================
    # ВАЛИДАЦИЯ И ЛОГИРОВАНИЕ
    # ============================================================================
    
    eval_interval: int = 500
    """Интервал валидации в шагах (0 = только после каждой эпохи)."""
    
    eval_iters: int = 200
    """Количество батчей для валидации (для оценки метрик)."""
    
    log_interval: int = 10
    """Интервал логирования training loss в шагах."""
    
    # ============================================================================
    # СОХРАНЕНИЕ МОДЕЛЕЙ
    # ============================================================================
    
    save_every_epochs: int = 2
    """Сохранять checkpoint каждые N эпох."""
    
    save_total_limit: int = 3
    """Максимальное количество сохраненных checkpoints (старые удаляются)."""
    
    checkpoint_dir: str = "checkpoints"
    """Директория для сохранения checkpoints."""
    
    # ============================================================================
    # EARLY STOPPING
    # ============================================================================
    
    early_stopping_patience: int = 5
    """Остановить обучение после N эпох без улучшения val_loss."""
    
    early_stopping_threshold: float = 0.0
    """Минимальное улучшение для учета (0.0 = любое улучшение)."""
    
    # ============================================================================
    # ГЕНЕРАЦИЯ (для Language Models)
    # ============================================================================
    
    generate_samples: bool = True
    """Генерировать примеры текста во время обучения."""
    
    generation_interval: int = 2
    """Генерировать примеры каждые N эпох."""
    
    max_generation_tokens: int = 100
    """Максимальное количество токенов для генерации."""
    
    generation_temperature: float = 0.8
    """Температура для sampling при генерации."""
    
    generation_top_k: Optional[int] = None
    """Top-K sampling (None = не использовать)."""
    
    # ============================================================================
    # MLFLOW
    # ============================================================================
    
    experiment_name: str = "base_experiment"
    """Имя MLflow эксперимента."""
    
    run_name: Optional[str] = None
    """Имя MLflow run (если None, генерируется автоматически)."""
    
    mlflow_tracking_uri: Optional[str] = None
    """MLflow tracking URI (None = использовать локальный)."""
    
    log_model: bool = True
    """Логировать модель в MLflow после обучения."""
    
    # ============================================================================
    # СИСТЕМА
    # ============================================================================
    
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    """Устройство для обучения: 'cuda', 'cpu', 'mps' (для Apple Silicon)."""
    
    num_workers: int = 2
    """Количество workers для DataLoader."""
    
    pin_memory: bool = True
    """Pin memory для DataLoader (ускоряет на GPU)."""
    
    seed: int = 42
    """Random seed для воспроизводимости."""
    
    compile_model: bool = False
    """Использовать torch.compile (PyTorch 2.0+, ускоряет ~2x)."""
    
    # ============================================================================
    # РАЗНОЕ
    # ============================================================================
    
    model_name: str = "base_model"
    """Базовое имя модели (используется для сохранения)."""
    
    log_level: str = "INFO"
    """Уровень логирования: DEBUG, INFO, WARNING, ERROR."""
    
    notes: str = ""
    """Заметки об эксперименте (логируются в MLflow)."""
    
    # ============================================================================
    # МЕТОДЫ
    # ============================================================================
    
    def __post_init__(self):
        """Валидация и автоматические вычисления после инициализации."""
        
        # Валидация путей
        self.data_path = Path(self.data_path)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        
        # Валидация train_split
        if not 0 < self.train_split < 1:
            raise ValueError(f"train_split должен быть между 0 и 1, получено: {self.train_split}")
        
        # Валидация batch_size
        if self.batch_size <= 0:
            raise ValueError(f"batch_size должен быть > 0, получено: {self.batch_size}")
        
        # Валидация learning_rate
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate должен быть > 0, получено: {self.learning_rate}")
        
        # Автоматически устанавливаем pin_memory для CPU
        if self.device == 'cpu':
            self.pin_memory = False
        
        # Предупреждение о compile для старых версий PyTorch
        if self.compile_model:
            torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
            if torch_version < (2, 0):
                import warnings
                warnings.warn(
                    f"torch.compile требует PyTorch >= 2.0, текущая версия: {torch.__version__}. "
                    f"Отключаем compile_model."
                )
                self.compile_model = False
        
        # Вычисляем эффективный batch size
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
    
    def to_dict(self) -> dict:
        """
        Конвертация конфига в словарь для MLflow логирования.
        
        Returns:
            Словарь со всеми параметрами конфига
        """
        config_dict = asdict(self)
        
        # Конвертируем Path объекты в строки
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        # Добавляем вычисляемые поля
        config_dict['effective_batch_size'] = self.effective_batch_size
        config_dict['device_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        config_dict['pytorch_version'] = torch.__version__
        
        return config_dict
    
    def update(self, **kwargs):
        """
        Обновление параметров конфига.
        
        Args:
            **kwargs: Параметры для обновления
        
        Raises:
            ValueError: Если параметр не существует в конфиге
        
        Example:
            config.update(batch_size=16, learning_rate=5e-4)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"BaseTrainingConfig не имеет атрибута '{key}'. "
                    f"Доступные атрибуты: {list(self.__dataclass_fields__.keys())}"
                )
        
        # Перезапускаем __post_init__ для валидации
        self.__post_init__()
    
    def save(self, filepath: str):
        """
        Сохранение конфига в JSON файл.
        
        Args:
            filepath: Путь для сохранения
        """
        import json
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str):
        """
        Загрузка конфига из JSON файла.
        
        Args:
            filepath: Путь к конфиг файлу
        
        Returns:
            Объект конфига
        """
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
