# =============================================================================
# utils/logging.py
# Утилиты для логирования генерации текста и метрик обучения
# =============================================================================

import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import sys


# =============================================================================
# НАСТРОЙКА БАЗОВОГО ЛОГИРОВАНИЯ
# =============================================================================

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Настройка базового Python логирования для всего проекта.
    
    Args:
        log_level: Уровень логирования ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Путь к файлу для логов (если None, только в консоль)
        log_format: Формат логов (если None, используется дефолтный)
    
    Returns:
        Настроенный logger
    
    Example:
        >>> logger = setup_logging('INFO', 'training.log')
        >>> logger.info('Training started')
    """
    # Удаляем существующие handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Устанавливаем уровень
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(level)
    
    # Формат логов
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (опционально)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={log_level}, file={log_file}")
    
    return root_logger


# =============================================================================
# ГЕНЕРАЦИЯ ТЕКСТА - ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ
# =============================================================================

@dataclass
class GenerationStep:
    """Информация об одном шаге генерации."""
    step: int
    token_id: int
    token_text: str
    logit_max: float = 0.0
    logit_mean: float = 0.0
    probability: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class GenerationSession:
    """Полная сессия генерации текста."""
    session_id: str
    prompt_tokens: List[int]
    prompt_text: str
    generated_tokens: List[int] = field(default_factory=list)
    generated_text: str = ""
    steps: List[GenerationStep] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_time: Optional[float] = None
    tokens_per_second: Optional[float] = None
    
    def finalize(self):
        """Завершение сессии и вычисление метрик."""
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
        if self.generated_tokens:
            self.tokens_per_second = len(self.generated_tokens) / self.total_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для MLflow."""
        return {
            'session_id': self.session_id,
            'prompt_length': len(self.prompt_tokens),
            'generated_length': len(self.generated_tokens),
            'total_length': len(self.prompt_tokens) + len(self.generated_tokens),
            'total_time': self.total_time,
            'tokens_per_second': self.tokens_per_second,
            'generated_text_preview': self.generated_text[:200] + '...' if len(self.generated_text) > 200 else self.generated_text
        }


class GenerationLogger:
    """
    Логгер для детального отслеживания процесса генерации текста.
    
    Используется в моделях для логирования каждого шага генерации.
    Полезно для отладки и анализа поведения модели.
    
    Example:
        >>> logger = GenerationLogger()
        >>> session = logger.start_session(prompt_tokens=[1, 2, 3], prompt_text="Hello")
        >>> logger.log_step(session, step=0, token_id=42, token_text=" world")
        >>> logger.end_session(session, generated_text="Hello world!")
    """
    
    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: Если True, выводит информацию в консоль
        """
        self.verbose = verbose
        self.logger = logging.getLogger(f"{__name__}.GenerationLogger")
        self.sessions: List[GenerationSession] = []
        self.current_session: Optional[GenerationSession] = None
    
    def start_session(
        self,
        prompt_tokens: List[int],
        prompt_text: str,
        session_id: Optional[str] = None
    ) -> GenerationSession:
        """
        Начало новой сессии генерации.
        
        Args:
            prompt_tokens: Токены входного промпта
            prompt_text: Текст промпта
            session_id: ID сессии (если None, генерируется автоматически)
        
        Returns:
            Объект GenerationSession
        """
        if session_id is None:
            session_id = f"gen_{int(time.time() * 1000)}"
        
        session = GenerationSession(
            session_id=session_id,
            prompt_tokens=prompt_tokens,
            prompt_text=prompt_text
        )
        
        self.current_session = session
        self.sessions.append(session)
        
        if self.verbose:
            self.logger.info(f"[{session_id}] Generation started")
            self.logger.info(f"[{session_id}] Prompt: {prompt_text[:100]}...")
        
        return session
    
    def log_step(
        self,
        session: GenerationSession,
        step: int,
        token_id: int,
        token_text: str,
        logit_max: float = 0.0,
        logit_mean: float = 0.0,
        probability: float = 0.0
    ):
        """
        Логирование одного шага генерации.
        
        Args:
            session: Текущая сессия
            step: Номер шага (0-indexed)
            token_id: ID сгенерированного токена
            token_text: Текст токена
            logit_max: Максимальный logit
            logit_mean: Средний logit
            probability: Вероятность выбранного токена
        """
        generation_step = GenerationStep(
            step=step,
            token_id=token_id,
            token_text=token_text,
            logit_max=logit_max,
            logit_mean=logit_mean,
            probability=probability
        )
        
        session.steps.append(generation_step)
        session.generated_tokens.append(token_id)
        
        if self.verbose and step % 10 == 0:  # Логируем каждые 10 токенов
            self.logger.debug(
                f"[{session.session_id}] Step {step}: "
                f"token_id={token_id}, text='{token_text}', prob={probability:.4f}"
            )
    
    def end_session(
        self,
        session: GenerationSession,
        generated_text: str
    ) -> Dict[str, Any]:
        """
        Завершение сессии генерации.
        
        Args:
            session: Завершаемая сессия
            generated_text: Полный сгенерированный текст
        
        Returns:
            Словарь с метриками сессии
        """
        session.generated_text = generated_text
        session.finalize()
        
        if self.verbose:
            self.logger.info(f"[{session.session_id}] Generation completed")
            self.logger.info(f"[{session.session_id}] Generated {len(session.generated_tokens)} tokens")
            self.logger.info(f"[{session.session_id}] Time: {session.total_time:.2f}s")
            self.logger.info(f"[{session.session_id}] Speed: {session.tokens_per_second:.2f} tokens/s")
            self.logger.info(f"[{session.session_id}] Text: {generated_text[:200]}...")
        
        self.current_session = None
        
        return session.to_dict()
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Получить summary конкретной сессии."""
        for session in self.sessions:
            if session.session_id == session_id:
                return session.to_dict()
        return None
    
    def get_all_sessions_summary(self) -> List[Dict[str, Any]]:
        """Получить summary всех сессий."""
        return [session.to_dict() for session in self.sessions]
    
    def clear_history(self):
        """Очистить историю сессий."""
        self.sessions.clear()
        self.current_session = None


# =============================================================================
# МЕТРИКИ ОБУЧЕНИЯ - ФОРМАТИРОВАНИЕ ДЛЯ ВЫВОДА
# =============================================================================

class MetricsFormatter:
    """
    Форматирование метрик обучения для красивого вывода.
    
    Example:
        >>> formatter = MetricsFormatter()
        >>> print(formatter.format_epoch_metrics(epoch=1, train_loss=2.5, val_loss=2.3))
        Epoch 1/10 | Train Loss: 2.5000 | Val Loss: 2.3000 | PPL: 9.97
    """
    
    @staticmethod
    def format_epoch_metrics(
        epoch: int,
        max_epochs: int,
        train_loss: float,
        val_loss: float,
        train_ppl: Optional[float] = None,
        val_ppl: Optional[float] = None,
        learning_rate: Optional[float] = None
    ) -> str:
        """Форматирование метрик для одной эпохи."""
        parts = [f"Epoch {epoch}/{max_epochs}"]
        parts.append(f"Train Loss: {train_loss:.4f}")
        parts.append(f"Val Loss: {val_loss:.4f}")
        
        if train_ppl is not None:
            parts.append(f"Train PPL: {train_ppl:.2f}")
        if val_ppl is not None:
            parts.append(f"Val PPL: {val_ppl:.2f}")
        if learning_rate is not None:
            parts.append(f"LR: {learning_rate:.2e}")
        
        return " | ".join(parts)
    
    @staticmethod
    def format_step_metrics(
        step: int,
        total_steps: int,
        loss: float,
        learning_rate: float
    ) -> str:
        """Форматирование метрик для одного шага."""
        return f"Step {step}/{total_steps} | Loss: {loss:.4f} | LR: {learning_rate:.2e}"
    
    @staticmethod
    def format_table(data: Dict[str, List[Any]], headers: List[str]) -> str:
        """
        Форматирование данных в ASCII таблицу.
        
        Args:
            data: Словарь {column_name: [values]}
            headers: Список заголовков колонок
        
        Returns:
            Форматированная таблица
        """
        # Вычисляем ширину колонок
        col_widths = {}
        for header in headers:
            col_widths[header] = max(len(header), max(len(str(v)) for v in data[header]))
        
        # Создаем строки
        lines = []
        
        # Заголовок
        header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
        lines.append(header_line)
        lines.append("-" * len(header_line))
        
        # Данные
        num_rows = len(data[headers[0]])
        for i in range(num_rows):
            row = " | ".join(str(data[h][i]).ljust(col_widths[h]) for h in headers)
            lines.append(row)
        
        return "\n".join(lines)


# =============================================================================
# ЭКСПОРТ ФУНКЦИЙ
# =============================================================================

__all__ = [
    'setup_logging',
    'GenerationLogger',
    'GenerationSession',
    'GenerationStep',
    'MetricsFormatter',
]


# =============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =============================================================================

if __name__ == "__main__":
    # 1. Настройка базового логирования
    logger = setup_logging('DEBUG', 'test.log')
    
    # 2. Пример использования GenerationLogger
    gen_logger = GenerationLogger(verbose=True)
    
    # Начало сессии
    session = gen_logger.start_session(
        prompt_tokens=[1, 2, 3],
        prompt_text="Hello, how are"
    )
    
    # Симуляция генерации
    import random
    for i in range(10):
        token_id = random.randint(10, 1000)
        token_text = f" token_{i}"
        prob = random.random()
        
        gen_logger.log_step(
            session=session,
            step=i,
            token_id=token_id,
            token_text=token_text,
            probability=prob
        )
    
    # Завершение
    metrics = gen_logger.end_session(session, "Hello, how are you doing today?")
    print("\nSession metrics:", metrics)
    
    # 3. Пример форматирования метрик
    formatter = MetricsFormatter()
    
    print("\n" + "="*80)
    print(formatter.format_epoch_metrics(
        epoch=5,
        max_epochs=10,
        train_loss=2.456,
        val_loss=2.389,
        train_ppl=11.65,
        val_ppl=10.90,
        learning_rate=3e-4
    ))
    print("="*80)
    
    # 4. Пример таблицы
    table_data = {
        'Model': ['GPT', 'T²', 'BERT'],
        'Loss': [2.45, 2.39, 2.67],
        'PPL': [11.6, 10.9, 14.4]
    }
    print("\n" + formatter.format_table(table_data, ['Model', 'Loss', 'PPL']))
