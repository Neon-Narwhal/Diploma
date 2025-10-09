import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Callable
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import time
from pathlib import Path

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedCodeTrainer:
    """
    Универсальный тренер для GPT и TransformerSquared моделей.
    Поддерживает gradient accumulation, mixed precision, early stopping.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        train_loader,
        val_loader,
        config: Dict,
        experiment_name: str = "bigobench_training",
        run_name: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Args:
            model: модель (GPT или TransformerSquared)
            tokenizer: токенизатор
            train_loader: DataLoader для тренировки
            val_loader: DataLoader для валидации
            config: словарь с гиперпараметрами
            experiment_name: имя MLflow эксперимента
            run_name: имя MLflow run'а
            device: устройство ('cuda' или 'cpu')
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.run_name = run_name or f"{model.__class__.__name__}_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Оптимизатор
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 3e-4),
            weight_decay=config.get('weight_decay', 0.01),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('max_epochs', 10) * len(train_loader),
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Mixed precision scaler
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Gradient accumulation
        self.grad_accum_steps = config.get('gradient_accumulation_steps', 4)
        
        # Early stopping
        self.patience = config.get('early_stopping_patience', 5)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # MLflow
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"Инициализирован тренер для {model.__class__.__name__}")
        logger.info(f"Параметры: LR={config.get('learning_rate')}, Batch={config.get('batch_size')}, "
                   f"Grad Accum={self.grad_accum_steps}, AMP={self.use_amp}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Один эпох тренировки"""
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            # Forward pass с mixed precision
            with autocast(enabled=self.use_amp):
                logits, loss = self.model(input_ids, targets=target_ids)
                loss = loss / self.grad_accum_steps  # Нормализация для grad accumulation
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.config.get('max_grad_norm', 1.0)
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Scheduler step
                self.scheduler.step()
            
            # Метрики
            total_loss += loss.item() * self.grad_accum_steps
            total_tokens += input_ids.numel()
            
            # Обновление progress bar
            pbar.set_postfix({
                'loss': f"{loss.item() * self.grad_accum_steps:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'train_loss': avg_loss,
            'train_perplexity': perplexity,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Валидация модели"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            with autocast(enabled=self.use_amp):
                logits, loss = self.model(input_ids, targets=target_ids)
            
            total_loss += loss.item()
            total_tokens += input_ids.numel()
            
            pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.val_loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'val_loss': avg_loss,
            'val_perplexity': perplexity
        }
    
    def should_stop_early(self, val_loss: float) -> bool:
        """Проверка условий early stopping"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
                return True
        return False
    
    def save_checkpoint(self, epoch: int, metrics: Dict, checkpoint_dir: str = "checkpoints"):
        """Сохранение чекпоинта"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        checkpoint_path = checkpoint_dir / f"{self.run_name}_epoch{epoch}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }, checkpoint_path)
        
        logger.info(f"Чекпоинт сохранен: {checkpoint_path}")
        return str(checkpoint_path)
    
    def generate_sample(self, prompt: str = "", max_tokens: int = 100, temperature: float = 0.8):
        """Генерация примера кода"""
        self.model.eval()
        
        if not prompt:
            # Начинаем с пустой последовательности
            context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        else:
            tokens = self.tokenizer.encode(prompt)
            context = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            generated = self.model.generate(context, max_tokens)
            generated_tokens = generated[0].tolist()
            generated_text = self.tokenizer.decode(generated_tokens)
        
        return generated_text
    
    def train(self) -> Dict[str, float]:
        """
        Основной цикл обучения
        
        Returns:
            Финальные метрики
        """
        max_epochs = self.config.get('max_epochs', 10)
        save_every = self.config.get('save_every_epochs', 2)
        generate_samples = self.config.get('generate_samples', True)
        
        with mlflow.start_run(run_name=self.run_name):
            # Логируем конфигурацию
            mlflow.log_params(self.config)
            mlflow.log_param('model_class', self.model.__class__.__name__)
            mlflow.log_param('total_parameters', sum(p.numel() for p in self.model.parameters()))
            mlflow.log_param('trainable_parameters', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
            
            logger.info(f"Начинаем обучение на {max_epochs} эпох")
            
            for epoch in range(1, max_epochs + 1):
                # Тренировка
                train_metrics = self.train_epoch(epoch)
                
                # Валидация
                val_metrics = self.validate(epoch)
                
                # Объединяем метрики
                all_metrics = {**train_metrics, **val_metrics}
                
                # Логируем в MLflow
                mlflow.log_metrics(all_metrics, step=epoch)
                
                # Вывод метрик
                logger.info(f"Epoch {epoch}/{max_epochs} | "
                          f"Train Loss: {train_metrics['train_loss']:.4f} | "
                          f"Val Loss: {val_metrics['val_loss']:.4f} | "
                          f"Train PPL: {train_metrics['train_perplexity']:.2f} | "
                          f"Val PPL: {val_metrics['val_perplexity']:.2f}")
                
                # Генерация примеров
                if generate_samples and epoch % 2 == 0:
                    sample_text = self.generate_sample(max_tokens=100, temperature=0.8)
                    mlflow.log_text(sample_text, f"generated_sample_epoch{epoch}.txt")
                    logger.info(f"Generated sample (epoch {epoch}):\n{sample_text[:200]}...")
                
                # Сохранение чекпоинта
                if epoch % save_every == 0:
                    checkpoint_path = self.save_checkpoint(epoch, all_metrics)
                    mlflow.log_artifact(checkpoint_path)
                
                # Early stopping
                if self.should_stop_early(val_metrics['val_loss']):
                    mlflow.log_param('early_stopped_at_epoch', epoch)
                    break
            
            # Финальное сохранение модели
            final_checkpoint = self.save_checkpoint(epoch, all_metrics, checkpoint_dir="final_models")
            mlflow.pytorch.log_model(self.model, "model")
            mlflow.log_artifact(final_checkpoint)
            
            # Сохраняем токенизатор
            tokenizer_path = Path("final_models") / f"{self.run_name}_tokenizer.json"
            self.tokenizer.save(str(tokenizer_path))
            mlflow.log_artifact(str(tokenizer_path))
            
            logger.info(f"Обучение завершено! Best val loss: {self.best_val_loss:.4f}")
            
            return {
                'best_val_loss': self.best_val_loss,
                'final_epoch': epoch,
                **all_metrics
            }