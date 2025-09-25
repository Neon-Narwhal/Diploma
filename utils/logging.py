"""Система логирования для генерации текста"""

import torch
import logging
from typing import List, Optional
from datetime import datetime


class GenerationLogger:
    """Класс для логирования процесса генерации текста"""

    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger("GenerationLogger")
        self.logger.setLevel(getattr(logging, log_level))

        # Создаем обработчик консоли если он не существует
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_generation_start(self, context: torch.Tensor, max_chars: int):
        """Логирование начала генерации"""
        self.logger.info(f"🚀 Начало генерации текста")
        self.logger.info(f"   Контекст: {context.shape}")
        self.logger.info(f"   Максимум символов: {max_chars}")
        self.logger.info(f"   Время: {datetime.now().strftime('%H:%M:%S')}")

    def log_generation_step(self, step: int, token_id: int, decoded_token: str):
        """Логирование каждого шага генерации"""
        # Экранируем специальные символы для красивого вывода
        display_token = repr(decoded_token) if decoded_token in ['\n', '\t', ' '] else decoded_token
        self.logger.debug(f"   Шаг {step:3d}: token_id={token_id:4d} -> '{display_token}'")

    def log_generation_complete(self, generated_text: str, total_tokens: int, execution_time: float):
        """Логирование завершения генерации"""
        self.logger.info(f"✅ Генерация завершена")
        self.logger.info(f"   Всего токенов: {total_tokens}")
        self.logger.info(f"   Время выполнения: {execution_time:.2f}с")
        self.logger.info(f"   Символов в секунду: {len(generated_text)/execution_time:.1f}")
        self.logger.info(f"\n📝 Сгенерированный текст:")
        self.logger.info(f"{'='*50}")
        self.logger.info(generated_text)
        self.logger.info(f"{'='*50}")

    def log_tokenizer_info(self, tokenizer_type: str, vocab_size: int):
        """Логирование информации о токенизаторе"""
        self.logger.info(f"🔤 Токенизатор: {tokenizer_type.upper()}")
        self.logger.info(f"   Размер словаря: {vocab_size}")

    def log_model_info(self, model, device: str):
        """Логирование информации о модели"""
        param_count = sum(p.numel() for p in model.parameters())
        self.logger.info(f"🧠 Модель загружена")
        self.logger.info(f"   Параметров: {param_count:,}")
        self.logger.info(f"   Устройство: {device}")
