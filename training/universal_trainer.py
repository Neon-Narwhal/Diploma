# =============================================================================
# training/universal_trainer.py
# ЕДИНСТВЕННЫЙ универсальный тренер для всех моделей
# =============================================================================

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import time
import logging
from pathlib import Path
from dataclasses import asdict

from models.base_model import BaseLanguageModel
from utils.configs.base_config import BaseTrainingConfig
from utils.logging import setup_logging, GenerationLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniversalTrainer:
    """
    Универсальный тренер для любых моделей, реализующих BaseLanguageModel.
    
    Особенности:
    - Model-agnostic: работает с любой моделью через общий интерфейс
    - Config-driven: все параметры берутся из конфига
    - Extensible: поддерживает hooks для кастомизации
    - Production-ready: MLflow, checkpointing, early stopping, mixed precision
    """
    
    def __init__(
        self,
        model: BaseLanguageModel,
        tokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: BaseTrainingConfig
    ):
        """
        Args:
            model: Любая модель, наследующая BaseLanguageModel
            tokenizer: Токенизатор (должен иметь encode/decode методы)
            train_loader: DataLoader для тренировки
            val_loader: DataLoader для валидации
            config: Объект конфига (BaseTrainingConfig или его подкласс)
        """
        self.config = config
        self.device = torch.device(config.device)

        setup_logging(config.log_level)
        
        # Создаем generation logger
        self.gen_logger = GenerationLogger(verbose=True)
        
        # Модель и токенизатор
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        
        # DataLoaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Оптимизатор (модель может предоставить свой через configure_optimizers)
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler(enabled=config.use_amp)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # MLflow run name
        self.run_name = config.run_name or f"{model.__class__.__name__}_{time.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"[UniversalTrainer] Инициализирован для {model.__class__.__name__}")
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"[UniversalTrainer] Параметров: {num_params:,} (~{num_params/1e6:.1f}")
        logger.info(f"[UniversalTrainer] Device: {self.device}, AMP: {config.use_amp}")
    
    def _create_optimizer(self):
        """Создание оптимизатора на основе конфига"""
        # Даем модели возможность предоставить свой оптимизатор
        if hasattr(self.model, 'configure_optimizers'):
            return self.model.configure_optimizers(self.config)
        
        # Иначе используем стандартный из конфига
        if self.config.optimizer_type == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.95)
            )
        elif self.config.optimizer_type == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer_type}")
    
    def _create_scheduler(self):
        """Создание learning rate scheduler"""
        total_steps = len(self.train_loader) * self.config.max_epochs
        
        if self.config.scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=1e-6
            )
        elif self.config.scheduler_type == 'linear':
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps
            )
        elif self.config.scheduler_type == 'constant':
            return torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler_type}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Один эпох тренировки"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.max_epochs} [Train]", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            # Forward pass с AMP
            with autocast(enabled=self.config.use_amp):
                logits, loss = self.model(input_ids, targets=target_ids)
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Scheduler step
                self.scheduler.step()
                
                self.global_step += 1
            
            # Метрики
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Progress bar
            pbar.set_postfix({
                'loss': f"{loss.item() * self.config.gradient_accumulation_steps:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        metrics = {
            'train_loss': avg_loss,
            'train_perplexity': perplexity,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        # Добавляем model-specific метрики
        model_metrics = self.model.get_model_specific_metrics()
        if model_metrics:
            metrics.update({f'train_{k}': v for k, v in model_metrics.items()})
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Валидация модели"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.config.max_epochs} [Val]", leave=False)
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            with autocast(enabled=self.config.use_amp):
                logits, loss = self.model(input_ids, targets=target_ids)
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'val_loss': avg_loss,
            'val_perplexity': perplexity
        }
    
    def generate_sample(self):
        """Генерация с детальным логированием."""
        self.model.eval()
        context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        
        # Начинаем сессию логирования
        session = self.gen_logger.start_session(
            prompt_tokens=context[0].tolist(),
            prompt_text=""
        )
        
        # Генерация (модель может логировать шаги через gen_logger)
        with torch.no_grad():
            generated = self.model.generate(
                context, 
                max_new_tokens=self.config.max_generation_tokens,
                temperature=self.config.generation_temperature,
                gen_logger=self.gen_logger,  # Передаем logger в модель
                session=session
            )
        
        # Завершаем сессию
        generated_text = self.tokenizer.decode(generated[0].tolist())
        metrics = self.gen_logger.end_session(session, generated_text)
        
        return generated_text
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> str:
        """Сохранение чекпоинта"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        checkpoint_path = checkpoint_dir / f"{self.run_name}_epoch{epoch}.pt"
        
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'config': self.config.to_dict()
        }, checkpoint_path)
        
        logger.info(f"[Trainer] Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def train(self) -> Dict[str, float]:
        """
        Главный цикл обучения.
        Полностью управляется конфигом.
        """
        try:
            mlflow.set_experiment(self.config.experiment_name)
        except:
            logger.warning("[Trainer] MLflow недоступен, логирование отключено")
        
        try:
            with mlflow.start_run(run_name=self.run_name):
                # Логируем конфиг
                mlflow.log_params(self.config.to_dict())
                mlflow.log_param('model_class', self.model.__class__.__name__)
                num_params = sum(p.numel() for p in self.model.parameters())
                logger.info(f"[UniversalTrainer] Параметров: {num_params:,} (~{num_params/1e6:.1f}M)")
                
                logger.info(f"[Trainer] Начинаем обучение на {self.config.max_epochs} эпох")
                
                for epoch in range(1, self.config.max_epochs + 1):
                    self.current_epoch = epoch
                    
                    # Тренировка
                    train_metrics = self.train_epoch(epoch)
                    
                    # Валидация
                    val_metrics = self.validate(epoch)
                    
                    # Объединяем метрики
                    all_metrics = {**train_metrics, **val_metrics}
                    
                    # MLflow logging
                    try:
                        mlflow.log_metrics(all_metrics, step=epoch)
                    except:
                        pass
                    
                    # Логирование
                    logger.info(
                        f"[Trainer] Epoch {epoch}/{self.config.max_epochs} | "
                        f"Train Loss: {train_metrics['train_loss']:.4f} | "
                        f"Val Loss: {val_metrics['val_loss']:.4f} | "
                        f"Train PPL: {train_metrics['train_perplexity']:.2f} | "
                        f"Val PPL: {val_metrics['val_perplexity']:.2f}"
                    )
                    
                    # Генерация примеров
                    if self.config.generate_samples and epoch % self.config.generation_interval == 0:
                        try:
                            sample = self.generate_sample()
                            mlflow.log_text(sample, f"sample_epoch{epoch}.txt")
                            logger.info(f"[Trainer] Generated sample:\n{sample[:200]}...")
                        except Exception as e:
                            logger.warning(f"[Trainer] Generation error: {e}")
                    
                    # Сохранение чекпоинта
                    if epoch % self.config.save_every_epochs == 0:
                        checkpoint_path = self.save_checkpoint(epoch, all_metrics)
                        try:
                            mlflow.log_artifact(checkpoint_path)
                        except:
                            pass
                    
                    # Early stopping
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.patience_counter = 0
                        # Сохраняем лучшую модель
                        best_path = self.save_checkpoint(epoch, all_metrics)
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.config.early_stopping_patience:
                            logger.info(f"[Trainer] Early stopping triggered at epoch {epoch}")
                            try:
                                mlflow.log_param('early_stopped_at_epoch', epoch)
                            except:
                                pass
                            break
                
                # Финальное сохранение
                final_path = self.save_checkpoint(epoch, all_metrics)
                try:
                    mlflow.pytorch.log_model(self.model, "model")
                    mlflow.log_artifact(final_path)
                except:
                    pass
                
                logger.info(f"[Trainer] Training completed! Best val loss: {self.best_val_loss:.4f}")
                
                return {
                    'best_val_loss': self.best_val_loss,
                    'final_epoch': epoch,
                    **all_metrics
                }
        
        except Exception as e:
            logger.error(f"[Trainer] Training error: {e}")
            raise