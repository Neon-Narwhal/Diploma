"""
Universal Training Script for Transformer² models
Uses unified ModelConfig from utils.config
Compatible with existing MLflow infrastructure
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
from typing import Dict, Any, Optional, Tuple
import time
import sys

# Import universal config and model
from utils.data_utils import load_data, prepare_data, get_batch, split_data
from my_tokenizers.char_tokenizer import CharTokenizer
from my_tokenizers.bpe_tokenizer import BPETokenizer
from utils.config import ModelConfig
from models.transformer_squared_model import TransformerSquared, create_model

class LanguageModelingDataset(Dataset):
    """Custom dataset for language modeling task"""

    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

class UniversalTransformerTrainer:
    """
    Universal trainer class for Transformer models with MLflow integration
    Works with any ModelConfig - standard or Transformer²
    """

    def __init__(self, config: ModelConfig, experiment_name: str = "transformer_experiments"):
        self.config = config
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def create_dataloader(self, data: torch.Tensor, batch_size: Optional[int] = None, 
                         shuffle: bool = True) -> DataLoader:
        """Create DataLoader for language modeling data"""
        batch_size = batch_size or self.config.batch_size
        dataset = LanguageModelingDataset(data, self.config.block_size)

        # Custom collate function to transfer data to device
        def collate_fn(batch):
            x_batch = torch.stack([x for x, y in batch]).to(self.config.device)
            y_batch = torch.stack([y for x, y in batch]).to(self.config.device)
            return x_batch, y_batch

        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=collate_fn,
            pin_memory=False,
            num_workers=0  # Keep 0 for compatibility with CUDA
        )

    def prepare_complexity_data(self, code_samples: list, complexity_labels: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for complexity analysis task"""
        # This would be implemented based on your tokenization strategy
        # For now, returning dummy data structure
        tokenized_data = torch.randint(0, self.config.vocab_size, (len(code_samples), self.config.block_size))
        labels = torch.tensor(complexity_labels, dtype=torch.long)
        return tokenized_data, labels

    def train_step(self, model: TransformerSquared, batch: torch.Tensor, targets: torch.Tensor, 
                   optimizer: torch.optim.Optimizer, task_type: str = 'language_modeling') -> float:
        """Single training step"""
        model.train()
        optimizer.zero_grad()

        logits, loss = model(batch, targets, task_type=task_type)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        return loss.item()

    def evaluate_model(self, model: TransformerSquared, val_dataloader: DataLoader, 
                      task_type: str = 'language_modeling') -> Dict[str, float]:
        """Evaluate model on validation data using DataLoader"""
        model.eval()
        eval_losses = []

        with torch.no_grad():
            for batch, targets in val_dataloader:
                # Data is already on correct device thanks to collate_fn
                _, loss = model(batch, targets, task_type=task_type)
                if loss is not None:
                    eval_losses.append(loss.item())

        avg_loss = sum(eval_losses) / len(eval_losses) if eval_losses else 0.0
        return {
            'eval_loss': avg_loss,
            'perplexity': torch.exp(torch.tensor(avg_loss)).item() if eval_losses else 0.0
        }

    def run_training(self, train_data: torch.Tensor, val_data: Optional[torch.Tensor] = None, 
                    max_iters: Optional[int] = None, eval_interval: Optional[int] = None, 
                    learning_rate: Optional[float] = None, run_name: Optional[str] = None) -> TransformerSquared:
        """
        Main training loop with MLflow logging and DataLoader
        Uses config parameters by default, allows overrides
        """

        # Use config parameters with optional overrides
        max_iters = max_iters or self.config.max_iters
        eval_interval = eval_interval or self.config.eval_interval
        learning_rate = learning_rate or self.config.learning_rate

        # Create model using unified config
        model = TransformerSquared(self.config).to(self.config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # Create DataLoaders
        train_dataloader = self.create_dataloader(train_data, shuffle=True)
        val_dataloader = None
        if val_data is not None:
            val_dataloader = self.create_dataloader(val_data, shuffle=False)

        # MLflow run
        with mlflow.start_run(run_name=run_name):
            # Log configuration - convert to dict for MLflow
            config_dict = self.config.to_dict()

            # Log basic config params
            for key, value in config_dict.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(f"config_{key}", value)

            # Log training params
            mlflow.log_param("max_iters", max_iters)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("eval_interval", eval_interval)

            # Log model info
            model_info = model.get_model_info()
            for key, value in model_info.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(f"model_{key}", value)
                elif isinstance(value, list):
                    mlflow.log_param(f"model_{key}", str(value))

            best_val_loss = float('inf')
            train_losses = []
            global_step = 0

            for epoch in range(max_iters):
                for batch, targets in train_dataloader:
                    # Data is already on correct device thanks to collate_fn
                    loss = self.train_step(model, batch, targets, optimizer, 'language_modeling')
                    train_losses.append(loss)

                    global_step += 1

                    # Evaluation
                    if global_step % eval_interval == 0 or global_step >= max_iters:
                        if val_dataloader is not None:
                            eval_metrics = self.evaluate_model(model, val_dataloader, 'language_modeling')
                            val_loss = eval_metrics['eval_loss']

                            # MLflow logging
                            mlflow.log_metrics({
                                'train_loss': sum(train_losses[-eval_interval:]) / min(len(train_losses), eval_interval),
                                'val_loss': val_loss,
                                'perplexity': eval_metrics['perplexity']
                            }, step=global_step)

                            # Early stopping check
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss

                            print(f"Step {global_step}: train_loss={loss:.4f}, val_loss={val_loss:.4f}")
                        else:
                            mlflow.log_metric('train_loss', loss, step=global_step)
                            print(f"Step {global_step}: train_loss={loss:.4f}")

                    if global_step >= max_iters:
                        break

                if global_step >= max_iters:
                    break

            # Save model
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=self.config.model_name
            )

            # Final evaluation and generation sample
            model.eval()
            with torch.no_grad():
                # Generate sample text
                context = torch.zeros(1, 1, dtype=torch.long, device=self.config.device)

                # Use config generation parameters
                generated = model.generate(
                    context, 
                    max_new_tokens=min(100, self.config.max_generation_length),
                    temperature=self.config.generation_temperature
                )

                # Log generation sample (would need tokenizer for proper decoding)
                mlflow.log_text(f"Generated tokens: {generated[0].tolist()}", "generation_sample.txt")

            return model

    def run_complexity_fine_tuning(self, model: TransformerSquared, code_data: list, 
                                 complexity_labels: list, max_iters: int = 1000, 
                                 learning_rate: float = 1e-4, run_name: Optional[str] = None) -> TransformerSquared:
        """Fine-tune model for complexity analysis"""

        # Check if model supports complexity analysis
        if not model.complexity_head:
            print("Warning: Model does not have complexity head enabled. Enable with config.enable_complexity_head = True")
            return model

        # Prepare complexity data
        x_complexity, y_complexity = self.prepare_complexity_data(code_data, complexity_labels)

        # Create DataLoader for complexity data
        complexity_dataset = TensorDataset(x_complexity, y_complexity)

        def complexity_collate_fn(batch):
            x_batch = torch.stack([x for x, y in batch]).to(self.config.device)
            y_batch = torch.stack([y for x, y in batch]).to(self.config.device)
            return x_batch, y_batch

        complexity_dataloader = DataLoader(
            complexity_dataset, 
            batch_size=min(16, len(complexity_dataset)), 
            shuffle=True,
            collate_fn=complexity_collate_fn
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_param("task", "complexity_analysis")
            mlflow.log_param("fine_tuning_iters", max_iters)
            mlflow.log_param("fine_tuning_lr", learning_rate)
            mlflow.log_param("complexity_classes", self.config.num_complexity_classes)

            for iter_num in range(max_iters):
                for x_batch, y_batch in complexity_dataloader:
                    # Data is already on correct device thanks to collate_fn
                    loss = self.train_step(model, x_batch, y_batch, optimizer, 'complexity_analysis')

                    if iter_num % 100 == 0:
                        mlflow.log_metric('complexity_loss', loss, step=iter_num)
                        print(f"Complexity fine-tuning iter {iter_num}: loss={loss:.4f}")

            # Evaluate complexity classification
            model.eval()
            with torch.no_grad():
                eval_batch = next(iter(complexity_dataloader))
                x_eval, y_eval = eval_batch
                logits, _ = model(x_eval, task_type='complexity_analysis')
                predictions = torch.argmax(logits, dim=-1)
                accuracy = (predictions == y_eval).float().mean().item()

                mlflow.log_metric("complexity_accuracy", accuracy)
                print(f"Final complexity accuracy: {accuracy:.4f}")

            return model

    def compare_models(self, train_data: torch.Tensor, val_data: torch.Tensor, 
                      comparison_configs: Dict[str, ModelConfig], 
                      experiment_suffix: str = "") -> Dict[str, TransformerSquared]:
        """
        Compare multiple model configurations

        Args:
            train_data: Training data
            val_data: Validation data  
            comparison_configs: Dict of {name: config} to compare
            experiment_suffix: Suffix for experiment names

        Returns:
            Dict of {name: trained_model}
        """
        results = {}

        for config_name, config in comparison_configs.items():
            print(f"\nTraining {config_name}...")

            # Create trainer for this config
            trainer = UniversalTransformerTrainer(
                config, 
                experiment_name=f"{self.experiment_name}_{config_name}_{experiment_suffix}"
            )

            # Train model
            model = trainer.run_training(
                train_data, 
                val_data,
                run_name=f"{config_name}_baseline"
            )

            results[config_name] = model

            # Log comparison info
            model_info = model.get_model_info()
            print(f"{config_name} completed:")
            print(f"  Parameters: {model_info['total_parameters']}")
            print(f"  Model type: {model_info['model_type']}")

        return results

# Convenience functions for common training scenarios

def train_standard_model(config: ModelConfig, train_data: torch.Tensor, 
                        val_data: Optional[torch.Tensor] = None) -> TransformerSquared:
    """Train a standard transformer model"""
    config.disable_transformer_squared_features()
    trainer = UniversalTransformerTrainer(config, "standard_transformer")
    return trainer.run_training(train_data, val_data, run_name="standard_baseline")

def train_transformer_squared(config: ModelConfig, train_data: torch.Tensor, 
                            val_data: Optional[torch.Tensor] = None) -> TransformerSquared:
    """Train a Transformer² model"""
    config.enable_transformer_squared_features()
    trainer = UniversalTransformerTrainer(config, "transformer_squared")
    return trainer.run_training(train_data, val_data, run_name="transformer_squared_baseline")

def train_code_analysis_model(config: ModelConfig, train_data: torch.Tensor, 
                            val_data: Optional[torch.Tensor] = None,
                            code_samples: Optional[list] = None,
                            complexity_labels: Optional[list] = None) -> TransformerSquared:
    """Train a model optimized for code complexity analysis"""
    config.set_complexity_analysis_mode()
    trainer = UniversalTransformerTrainer(config, "code_analysis")

    # First train on language modeling
    model = trainer.run_training(train_data, val_data, run_name="code_lm_baseline")

    # Then fine-tune on complexity analysis if data provided
    if code_samples and complexity_labels:
        model = trainer.run_complexity_fine_tuning(
            model, code_samples, complexity_labels, run_name="complexity_fine_tuning"
        )

    return model

def run_comparison_experiment():
    """Run comprehensive comparison between different model types"""

    # Base configuration
    base_config = ModelConfig()
    base_config.vocab_size = 1000
    base_config.block_size = 256
    base_config.n_embd = 256
    base_config.n_layer = 6
    base_config.n_head = 4
    base_config.batch_size = 32
    base_config.max_iters = 1000
    base_config.eval_interval = 200

    # Create comparison configurations
    comparison_configs = {
        'standard_transformer': base_config.copy().disable_transformer_squared_features(),
        'transformer_squared': base_config.copy().enable_transformer_squared_features(),
        'code_analysis': base_config.copy().set_complexity_analysis_mode(),
        'small_t2': base_config.copy().set_small_model_config().enable_transformer_squared_features()
    }

    # Set model names
    for name, config in comparison_configs.items():
        config.model_name = f"{name}_comparison"

    # Dummy data for testing
    dummy_train_data = torch.randint(0, base_config.vocab_size, (10000,))
    dummy_val_data = torch.randint(0, base_config.vocab_size, (2000,))

    # Run comparison
    print("Starting comprehensive model comparison...")
    trainer = UniversalTransformerTrainer(base_config, "model_comparison")

    results = trainer.compare_models(
        dummy_train_data, 
        dummy_val_data, 
        comparison_configs,
        experiment_suffix="comprehensive"
    )

    # Print final comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"{'Model':<20} {'Parameters':<12} {'Type':<20} {'SVF':<5} {'Complexity'}")
    print("-"*80)

    for name, model in results.items():
        info = model.get_model_info()
        print(f"{name:<20} {info['total_parameters']:<12} {info['model_type']:<20} "
              f"{'Yes' if info['use_svf'] else 'No':<5} "
              f"{'Yes' if info['supports_complexity_analysis'] else 'No'}")

    # Fine-tune code analysis model if it exists
    if 'code_analysis' in results:
        print("\nFine-tuning code analysis model...")
        dummy_code_samples = [f"code_sample_{i}" for i in range(100)]
        dummy_complexity_labels = torch.randint(0, 7, (100,)).tolist()

        trainer_code = UniversalTransformerTrainer(
            comparison_configs['code_analysis'], 
            "code_analysis_fine_tune"
        )

        trainer_code.run_complexity_fine_tuning(
            results['code_analysis'],
            dummy_code_samples,
            dummy_complexity_labels,
            max_iters=500,
            run_name="complexity_fine_tuning"
        )

    print("\nComparison experiment completed!")
    print("Check MLflow UI for results: mlflow ui")

    return results

def run_laptop_test():
    """Quick test of the universal trainer"""
    print("Running laptop test...")

    # Create simple config
    config = ModelConfig(
        batch_size=8,
        block_size=8,
        max_iters=500,
        eval_interval=100,
        eval_iters=40,
        n_embd=48,
        n_head=6,
        n_layer=3,
        overfit_line=1)
    config.tokenizer_type = 'char'
    config.enable_transformer_squared_features()
    config.model_name = "laptop_model"

       # Подготовка данных
    data = load_data(config.data_file)

    if config.tokenizer_type == 'char':
        tokenizer = CharTokenizer(data)
    else:
        tokenizer = BPETokenizer(data)

    config.vocab_size = tokenizer.get_vocab_size()

    # Подготавливаем данные для обучения
    train_data, val_data  = prepare_data(data, tokenizer, config, split_flag=config.split_flag)


    # Train model
    model = train_transformer_squared(config, train_data, val_data)

    print(f"Quick test completed! Model has {model.get_model_info()['total_parameters']} parameters")
    return model

def run_quick_test():
    """Quick test of the universal trainer"""
    print("Running quick test...")

    # Create simple config
    config = ModelConfig()
    config.tokenizer_type = 'char'
    config.enable_transformer_squared_features()
    config.model_name = "quick_test_model"

       # Подготовка данных
    data = load_data(config.data_file)

    if config.tokenizer_type == 'char':
        tokenizer = CharTokenizer(data)
    else:
        tokenizer = BPETokenizer(data)

    config.vocab_size = tokenizer.get_vocab_size()

    # Подготавливаем данные для обучения
    train_data, val_data  = prepare_data(data, tokenizer, config, split_flag=config.split_flag)


    # Train model
    model = train_transformer_squared(config, train_data, val_data)

    print(f"Quick test completed! Model has {model.get_model_info()['total_parameters']} parameters")
    return model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Universal Transformer Trainer")
    parser.add_argument("--mode", choices=["comparison", "quick_test"], 
                       default="comparison", help="Training mode")

    args = parser.parse_args()

    if args.mode == "comparison":
        run_comparison_experiment()
    elif args.mode == "quick_test":
        run_quick_test()
