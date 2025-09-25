"""
🏋️‍♂️ ТРЕНЕР МОДЕЛИ - ПОЛНЫЙ ГИБКИЙ КОД
====================================

Максимально гибкая система обучения с:
- Продвинутыми метриками и early stopping
- Поддержкой разных оптимизаторов и schedulers  
- Гибкими callbacks и логированием
- Автоматической валидацией и чекпоинтами
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import time
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from pathlib import Path
import json

# 🔧 БЕЗОПАСНЫЕ ИМПОРТЫ
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

try:
    from utils.config import ModelConfig
    from utils.data_utils import get_batch
    from utils.logging import GenerationLogger
except ImportError as e:
    print(f"❌ Ошибка импорта в trainer: {e}")
    sys.exit(1)


@torch.no_grad()
def estimate_loss(model: nn.Module, 
                  data: torch.Tensor, 
                  config: ModelConfig,
                  splits: List[str] = None) -> Dict[str, torch.Tensor]:
    """
    📊 ОЦЕНКА LOSS

    Рассчитывает loss на разных выборках с дополнительными метриками

    Args:
        model: Модель для оценки
        data: Данные для оценки
        config: Конфигурация
        splits: Список разделов для оценки ['train', 'val', 'test']

    Returns:
        Словарь с loss для каждого раздела
    """
    if splits is None:
        splits = ['train', 'val']

    out = {}
    model.eval()

    for split in splits:
        losses = torch.zeros(config.eval_iters, device=config.device)

        for k in range(config.eval_iters):
            try:
                X, y = get_batch(data, config, split)
                logits, loss = model(X, y)
                losses[k] = loss.item()
            except Exception as e:
                # Если не хватает данных для батча, используем среднее
                losses[k] = losses[:k].mean() if k > 0 else torch.tensor(float('inf'))
                break

        out[split] = losses.mean()

    model.train()
    return out


def calculate_perplexity(loss: float) -> float:
    """📏 Расчет perplexity из loss"""
    return torch.exp(torch.tensor(loss)).item()


def calculate_tokens_per_second(tokens_processed: int, elapsed_time: float) -> float:
    """⚡ Расчет скорости обработки токенов"""
    return tokens_processed / elapsed_time if elapsed_time > 0 else 0.0


class EarlyStopping:
    """
    ⏹️ ПРОДВИНУТЫЙ EARLY STOPPING

    Поддерживает разные стратегии остановки:
    - По validation loss
    - По overfit индикатору  
    - По отсутствию улучшений за N эпох
    """

    def __init__(self, 
                 patience: int = 5,
                 min_delta: float = 0.001,
                 mode: str = 'min',
                 restore_best_weights: bool = True):
        """
        Args:
            patience: Количество эпох без улучшения
            min_delta: Минимальное изменение для считающегося улучшением
            mode: 'min' для loss, 'max' для accuracy
            restore_best_weights: Восстанавливать лучшие веса при остановке
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

        self.compare = np.less if mode == 'min' else np.greater
        self.min_delta *= -1 if mode == 'min' else 1

    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Проверка условия early stopping

        Args:
            score: Текущая метрика (loss или accuracy)
            model: Модель для сохранения весов

        Returns:
            bool: True если нужно остановить обучение
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        elif self.compare(score, self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)

        return self.early_stop


class MetricsTracker:
    """
    📈 ТРЕКЕР МЕТРИК

    Отслеживает и рассчитывает различные метрики обучения
    """

    def __init__(self):
        self.metrics_history = {}
        self.start_time = time.time()

    def update(self, metrics: Dict[str, float], step: int = None):
        """Обновление метрик"""
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            self.metrics_history[name].append({
                'step': step,
                'value': value,
                'timestamp': time.time() - self.start_time
            })

    def get_latest(self, metric_name: str) -> Optional[float]:
        """Получение последнего значения метрики"""
        if metric_name in self.metrics_history and self.metrics_history[metric_name]:
            return self.metrics_history[metric_name][-1]['value']
        return None

    def get_trend(self, metric_name: str, window: int = 5) -> str:
        """Анализ тренда метрики"""
        if metric_name not in self.metrics_history or len(self.metrics_history[metric_name]) < window:
            return "insufficient_data"

        recent_values = [entry['value'] for entry in self.metrics_history[metric_name][-window:]]
        if len(recent_values) < 2:
            return "stable"

        # Простой анализ тренда
        slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]

        if abs(slope) < 0.001:
            return "stable"
        elif slope > 0:
            return "increasing" if metric_name != 'loss' else "worsening"
        else:
            return "decreasing" if metric_name != 'loss' else "improving"

    def get_summary(self) -> Dict[str, Any]:
        """Получение сводки по всем метрикам"""
        summary = {}
        for name, history in self.metrics_history.items():
            if history:
                values = [entry['value'] for entry in history]
                summary[name] = {
                    'current': values[-1],
                    'best': min(values) if 'loss' in name else max(values),
                    'worst': max(values) if 'loss' in name else min(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'trend': self.get_trend(name)
                }
        return summary


class AdvancedModelTrainer:
    """
    🏋️‍♂️ ПРОДВИНУТЫЙ ТРЕНЕР МОДЕЛИ

    Полнофункциональный класс для обучения с:
    - Гибкими callbacks и метриками
    - Early stopping и learning rate scheduling
    - Автоматическим сохранением чекпоинтов
    - Подробным логированием и анализом
    """

    def __init__(self, 
                 model: nn.Module,
                 optimizer: Optimizer,
                 config: ModelConfig,
                 scheduler: Optional[_LRScheduler] = None,
                 early_stopping: Optional[EarlyStopping] = None,
                 checkpoint_dir: Optional[str] = None):
        """
        Args:
            model: Модель для обучения
            optimizer: Оптимизатор
            config: Конфигурация обучения
            scheduler: Learning rate scheduler
            early_stopping: Early stopping стратегия
            checkpoint_dir: Директория для сохранения чекпоинтов
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.scheduler = scheduler
        self.early_stopping = early_stopping or EarlyStopping(patience=10)

        self.logger = GenerationLogger()
        self.metrics_tracker = MetricsTracker()

        # Настройка чекпоинтов
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None

        # Счетчики и статистика
        self.step = 0
        self.epoch = 0
        self.tokens_processed = 0
        self.training_start_time = None

    def train_step(self, data: torch.Tensor) -> Dict[str, float]:
        """
        🔄 ПРОДВИНУТЫЙ ШАГ ОБУЧЕНИЯ

        Returns:
            Словарь с метриками шага
        """
        step_start_time = time.time()

        # Получаем батч
        xb, yb = get_batch(data, self.config, "train")
        batch_size, seq_len = xb.shape

        # Прямой проход
        self.model.train()
        logits, loss = self.model(xb, yb)

        # Обратный проход
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping (опционально)
        if hasattr(self.config, 'gradient_clip_val') and self.config.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)

        self.optimizer.step()

        # Обновление scheduler
        if self.scheduler:
            self.scheduler.step()

        # Статистика
        step_time = time.time() - step_start_time
        tokens_in_batch = batch_size * seq_len
        self.tokens_processed += tokens_in_batch

        # Метрики шага
        step_metrics = {
            'train_loss': loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'step_time': step_time,
            'tokens_per_second': tokens_in_batch / step_time,
            'batch_size': batch_size,
            'sequence_length': seq_len
        }

        # Дополнительные метрики
        if hasattr(self.config, 'compute_additional_metrics') and self.config.compute_additional_metrics:
            step_metrics.update(self._compute_additional_metrics(logits, yb))

        self.step += 1
        return step_metrics

    def _compute_additional_metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """📊 Дополнительные метрики обучения"""
        with torch.no_grad():
            # Accuracy
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == targets).float().mean().item()

            # Perplexity
            loss_unreduced = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                reduction='none'
            )
            perplexity = torch.exp(loss_unreduced.mean()).item()

            return {
                'accuracy': accuracy,
                'perplexity': perplexity
            }

    def evaluate(self, data: torch.Tensor, 
                splits: List[str] = None) -> Dict[str, float]:
        """
        📊 ПРОДВИНУТАЯ ОЦЕНКА МОДЕЛИ

        Returns:
            Подробные метрики по всем split'ам
        """
        eval_start_time = time.time()

        # Базовые loss'ы
        losses = estimate_loss(self.model, data, self.config, splits)

        # Преобразуем в float и добавляем perplexity
        eval_metrics = {}
        for split, loss_tensor in losses.items():
            loss_val = loss_tensor.item()
            eval_metrics[f'{split}_loss'] = loss_val
            eval_metrics[f'{split}_perplexity'] = calculate_perplexity(loss_val)

        # Дополнительные метрики
        if 'train' in losses and 'val' in losses:
            train_loss = losses['train'].item()
            val_loss = losses['val'].item()
            eval_metrics['overfit_indicator'] = val_loss - train_loss
            eval_metrics['generalization_gap'] = val_loss / train_loss if train_loss > 0 else float('inf')

        eval_metrics['eval_time'] = time.time() - eval_start_time

        return eval_metrics

    def save_checkpoint(self, 
                       filepath: Optional[str] = None, 
                       include_optimizer: bool = True) -> str:
        """💾 Сохранение чекпоинта"""
        if self.checkpoint_dir is None and filepath is None:
            raise ValueError("Не указана директория для чекпоинтов и путь к файлу")

        if filepath is None:
            filepath = self.checkpoint_dir / f"checkpoint_epoch_{self.epoch}_step_{self.step}.pt"

        checkpoint_data = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'metrics_history': self.metrics_tracker.metrics_history,
            'tokens_processed': self.tokens_processed
        }

        if include_optimizer:
            checkpoint_data['optimizer_state_dict'] = self.optimizer.state_dict()
            if self.scheduler:
                checkpoint_data['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint_data, filepath)
        self.logger.logger.info(f"💾 Чекпоинт сохранен: {filepath}")

        return str(filepath)

    def load_checkpoint(self, filepath: str, load_optimizer: bool = True) -> Dict[str, Any]:
        """📁 Загрузка чекпоинта"""
        checkpoint = torch.load(filepath, map_location=self.config.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint.get('epoch', 0)
        self.step = checkpoint.get('step', 0)
        self.tokens_processed = checkpoint.get('tokens_processed', 0)

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Восстанавливаем историю метрик
        if 'metrics_history' in checkpoint:
            self.metrics_tracker.metrics_history = checkpoint['metrics_history']

        self.logger.logger.info(f"📁 Чекпоинт загружен: {filepath}")
        return checkpoint

    def train_epoch(self, 
                   data: torch.Tensor,
                   callback: Optional[Callable] = None,
                   save_checkpoints: bool = False,
                   checkpoint_freq: int = 500) -> Dict[str, Any]:
        """
        🚀 ПОЛНОЕ ОБУЧЕНИЕ ЭПОХИ

        Args:
            data: Данные для обучения
            callback: Функция callback для логирования
            save_checkpoints: Сохранять чекпоинты
            checkpoint_freq: Частота сохранения чекпоинтов

        Returns:
            Подробные результаты обучения
        """

        if self.training_start_time is None:
            self.training_start_time = time.time()

        self.epoch += 1
        epoch_start_time = time.time()

        self.logger.logger.info(f"🚀 Начало эпохи {self.epoch}")

        # История метрик эпохи
        epoch_train_losses = []

        for iter_num in range(self.config.max_iters):

            # Шаг обучения
            step_metrics = self.train_step(data)
            epoch_train_losses.append(step_metrics['train_loss'])

            # Обновляем трекер метрик
            self.metrics_tracker.update(step_metrics, self.step)

            # Оценка и логирование
            if iter_num % self.config.eval_interval == 0:

                # Оценка модели
                eval_metrics = self.evaluate(data)

                # Объединяем метрики
                combined_metrics = {**step_metrics, **eval_metrics}

                # Обновляем трекер
                self.metrics_tracker.update(eval_metrics, self.step)

                # Логирование
                val_loss = eval_metrics.get('val_loss', float('inf'))
                train_loss = step_metrics['train_loss']

                self.logger.logger.info(
                    f"📊 Step {iter_num}: "
                    f"train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} "
                    f"lr={step_metrics['learning_rate']:.2e} "
                    f"tokens/s={step_metrics['tokens_per_second']:.0f}"
                )

                # Callback для внешнего логирования (MLflow)
                if callback:
                    try:
                        callback(iter_num, combined_metrics)
                    except Exception as e:
                        self.logger.logger.warning(f"⚠️ Ошибка в callback: {e}")

                # Early stopping
                if self.early_stopping and self.early_stopping(val_loss, self.model):
                    self.logger.logger.info(f"⏹️ Early stopping на шаге {iter_num}")

                    return {
                        'early_stopped_at': iter_num,
                        'final_metrics': combined_metrics,
                        'epoch_losses': epoch_train_losses,
                        'metrics_summary': self.metrics_tracker.get_summary(),
                        'total_tokens_processed': self.tokens_processed,
                        'training_time': time.time() - epoch_start_time
                    }

            # Сохранение чекпоинтов
            if save_checkpoints and self.checkpoint_dir and (iter_num + 1) % checkpoint_freq == 0:
                self.save_checkpoint()

        # Финальная оценка
        final_metrics = self.evaluate(data)

        training_time = time.time() - epoch_start_time
        total_training_time = time.time() - self.training_start_time

        # Финальное логирование эпохи
        self.logger.logger.info(f"✅ Эпоха {self.epoch} завершена за {training_time:.1f}с")
        self.logger.logger.info(f"📈 Финальные метрики: {final_metrics}")

        return {
            'final_metrics': final_metrics,
            'epoch_losses': epoch_train_losses,
            'metrics_summary': self.metrics_tracker.get_summary(),
            'total_tokens_processed': self.tokens_processed,
            'epoch_training_time': training_time,
            'total_training_time': total_training_time,
            'tokens_per_second_avg': self.tokens_processed / total_training_time
        }


# 🔄 ОБРАТНАЯ СОВМЕСТИМОСТЬ С ОРИГИНАЛЬНЫМ КОДОМ
class ModelTrainer:
    """Простой тренер для обратной совместимости"""

    def __init__(self, model, optimizer, config: ModelConfig):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.losses_val = []
        self.step = 0

    def train_step(self, data: torch.Tensor) -> float:
        """Один шаг обучения"""
        xb, yb = get_batch(data, self.config, "train")
        logits, loss = self.model(xb, yb)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, data: torch.Tensor) -> Dict[str, float]:
        """Оценка модели"""
        losses = estimate_loss(self.model, data, self.config)
        return {k: v.item() for k, v in losses.items()}

    def should_early_stop(self, val_loss: float) -> bool:
        """Проверка условия раннего останова"""
        self.losses_val.append(val_loss)

        if len(self.losses_val) >= 2:
            return self.losses_val[-1] - self.losses_val[-2] > self.config.overfit_line

        return False

    def train_epoch(self, data: torch.Tensor, callback=None) -> Dict[str, Any]:
        """Обучение на протяжении одной эпохи"""
        self.model.train()
        epoch_losses = []

        for iter_num in range(self.config.max_iters):
            # Шаг обучения
            loss = self.train_step(data)
            epoch_losses.append(loss)

            # Оценка и логирование
            if iter_num % self.config.eval_interval == 0:
                eval_losses = self.evaluate(data)

                print(f"step {iter_num}: train loss {eval_losses['train']:.4f} val loss {eval_losses['val']:.4f}")

                # Вызываем callback если он есть
                if callback:
                    callback(iter_num, eval_losses)

                # Проверка на переобучение
                if self.should_early_stop(eval_losses['val']):
                    print(f"Early stopping at iteration {iter_num}")
                    return {
                        'early_stopped_at': iter_num,
                        'final_losses': eval_losses,
                        'epoch_losses': epoch_losses
                    }

        # Финальная оценка
        final_losses = self.evaluate(data)

        return {
            'final_losses': final_losses,
            'epoch_losses': epoch_losses
        }