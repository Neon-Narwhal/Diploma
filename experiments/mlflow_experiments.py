"""
🔬 ЭКСПЕРИМЕНТЫ С MLFLOW - ПОЛНЫЙ ГИБКИЙ КОД
==============================================

Максимально гибкая система экспериментов для GPT-модели с:
- MLflow tracking для всех метрик
- Поддержка разных токенизаторов  
- Логирование генерации текста
- Early stopping и продвинутые метрики
- Гибкая конфигурация экспериментов
"""

import torch
import mlflow
import mlflow.pytorch
import numpy as np
from dataclasses import asdict
from typing import Dict, Any, List, Optional, Callable, Union
import json
import tempfile
import os
from pathlib import Path

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)


from models.gpt_model import GPTLikeModel
from utils.config import ModelConfig
from utils.data_utils import load_data, get_batch, prepare_data
from utils.logging import GenerationLogger
from my_tokenizers.char_tokenizer import CharTokenizer
from training.trainer import ModelTrainer, estimate_loss


from my_tokenizers.bpe_tokenizer import BPETokenizer
HAS_BPE = True



class ExperimentRunner:
    """
    🎯 ГЛАВНЫЙ КЛАСС ДЛЯ УПРАВЛЕНИЯ ЭКСПЕРИМЕНТАМИ

    Гибко настраиваемый runner для экспериментов с:
    - Множественными конфигурациями
    - Автоматическим логированием
    - Продвинутыми метриками
    - Генерацией и сохранением результатов
    """

    def __init__(self, 
                 experiment_name: str = "gpt_experiments",
                 tracking_uri: Optional[str] = None,
                 auto_log: bool = True):
        """
        Args:
            experiment_name: Название эксперимента в MLflow
            tracking_uri: URI для MLflow tracking server
            auto_log: Автоматическое логирование PyTorch метрик
        """
        self.experiment_name = experiment_name
        self.logger = GenerationLogger()

        # Настройка MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

        if auto_log:
            mlflow.pytorch.autolog(log_models=False)  # Отключаем автологирование моделей для контроля

    def create_tokenizer(self, data: str, config: ModelConfig):
        """
        🔤 УМНОЕ СОЗДАНИЕ ТОКЕНИЗАТОРА

        Автоматически выбирает лучший доступный токенизатор
        """
        if config.tokenizer_type == 'char':
            return CharTokenizer(data)
        elif config.tokenizer_type == 'bpe' :
            return BPETokenizer(data, config.vocab_size)

        else:
            raise ValueError(f"Неизвестный тип токенизатора: {config.tokenizer_type}")

    def run_experiment(self, 
                      config: ModelConfig, 
                      run_name: Optional[str] = None,
                      generation_samples: int = 3,
                      save_model: bool = True,
                      custom_metrics: Optional[Dict[str, Callable]] = None) -> float:
        """
        🚀 ЗАПУСК ОДНОГО ЭКСПЕРИМЕНТА

        Полнофункциональный запуск эксперимента с MLflow tracking

        Args:
            config: Конфигурация модели
            run_name: Название запуска
            generation_samples: Количество примеров генерации
            save_model: Сохранять ли модель в MLflow
            custom_metrics: Дополнительные метрики для логирования

        Returns:
            float: Финальный validation loss
        """

        # 📊 ПОДГОТОВКА ДАННЫХ
        self.logger.logger.info(f"🚀 Начало эксперимента: {run_name or 'Unnamed'}")

        data = load_data(config.data_file)
        tokenizer = self.create_tokenizer(data, config)
        actual_vocab_size = tokenizer.get_vocab_size()

        # Обновляем конфигурацию с реальным размером словаря
        config.vocab_size = actual_vocab_size

        # Подготовка данных для обучения
        data_torch = prepare_data(data, tokenizer, config)

        # 🎯 MLflow RUN
        with mlflow.start_run(run_name=run_name):

            # Логирование параметров
            params_dict = asdict(config)
            params_dict.update({
                "actual_vocab_size": actual_vocab_size,
                "data_length": len(data_torch),
                "data_chars": len(data),
                "tokenizer_efficiency": len(data_torch) / len(data)  # токенов на символ
            })
            mlflow.log_params(params_dict)

            # 🧠 СОЗДАНИЕ МОДЕЛИ
            model = GPTLikeModel(config)
            model = model.to(config.device)

            param_count = sum(p.numel() for p in model.parameters())
            model_size_mb = param_count * 4 / (1024 * 1024)  # примерный размер в MB

            mlflow.log_params({
                "model_parameters": param_count,
                "model_size_mb": round(model_size_mb, 2)
            })

            self.logger.logger.info(f"📊 Модель создана: {param_count:,} параметров ({model_size_mb:.1f}MB)")

            # 🏃‍♂️ ОБУЧЕНИЕ
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            trainer = ModelTrainer(model, optimizer, config)

            # Callback для MLflow логирования
            def mlflow_callback(iter_num: int, losses: Dict[str, float]):
                metrics = {
                    "train_loss": losses['train'],
                    "val_loss": losses['val'],
                    "loss_diff": losses['val'] - losses['train'],  # overfit indicator
                    "learning_rate": optimizer.param_groups[0]['lr']
                }

                # Добавляем пользовательские метрики
                if custom_metrics:
                    for name, metric_func in custom_metrics.items():
                        try:
                            value = metric_func(model, data_torch, config)
                            metrics[f"custom_{name}"] = value
                        except Exception as e:
                            self.logger.logger.warning(f"Ошибка в метрике {name}: {e}")

                mlflow.log_metrics(metrics, step=iter_num)

            # Запуск обучения
            training_results = trainer.train_epoch(data_torch, callback=mlflow_callback)

            # 📈 ФИНАЛЬНЫЕ МЕТРИКИ
            final_losses = training_results['final_losses']

            final_metrics = {
                "final_train_loss": final_losses['train'],
                "final_val_loss": final_losses['val'],
                "final_overfit": final_losses['val'] - final_losses['train'],
                "epochs_completed": 1,
                "training_efficiency": final_losses['train'] / training_results.get('epoch_losses', [1])[0] if training_results.get('epoch_losses') else 1.0
            }

            # Early stopping информация
            if 'early_stopped_at' in training_results:
                final_metrics["early_stopped_at"] = training_results['early_stopped_at']
                final_metrics["training_completed"] = False
            else:
                final_metrics["training_completed"] = True

            mlflow.log_metrics(final_metrics)

            # 💾 СОХРАНЕНИЕ МОДЕЛИ
            if save_model:
                try:
                    mlflow.pytorch.log_model(
                        pytorch_model=model,
                        artifact_path="model",
                        registered_model_name=config.model_name,
                        signature=None  # TODO: можно добавить input/output signature
                    )
                    self.logger.logger.info("✅ Модель сохранена в MLflow")
                except Exception as e:
                    self.logger.logger.error(f"❌ Ошибка сохранения модели: {e}")

            # 💾 СОХРАНЕНИЕ ТОКЕНИЗАТОРА
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                tokenizer_path = f.name
                tokenizer.save(tokenizer_path)
                mlflow.log_artifact(tokenizer_path, "tokenizer")
                os.unlink(tokenizer_path)  # удаляем временный файл

            # 🎨 ГЕНЕРАЦИЯ ОБРАЗЦОВ ТЕКСТА
            self._generate_and_log_samples(model, tokenizer, config, generation_samples)

            # 📊 ДОПОЛНИТЕЛЬНАЯ АНАЛИТИКА
            self._log_model_analysis(model, data_torch, config)

            self.logger.logger.info(f"✅ Эксперимент завершен: val_loss = {final_losses['val']:.4f}")

            return final_losses['val']

    def _generate_and_log_samples(self, model, tokenizer, config, num_samples: int):
        """🎨 Генерация и логирование образцов текста"""

        model.eval()
        self.logger.logger.info(f"🎨 Генерация {num_samples} образцов текста...")

        samples = {}

        for i in range(num_samples):
            context = torch.zeros((1, 1), dtype=torch.long, device=config.device)

            # Генерация с логированием для первого образца
            if i == 0:
                with torch.no_grad():
                    generated = model.generate_with_logging(
                        context, 
                        max_new_token=100, 
                        tokenizer=tokenizer,
                        temperature=0.8
                    )
                    sample_text = tokenizer.decode(generated[0].tolist())
                    samples[f"sample_{i+1}_with_logging"] = sample_text
            else:
                # Обычная генерация для остальных
                with torch.no_grad():
                    generated = model.generate(context, max_new_token=100)
                    sample_text = tokenizer.decode(generated[0].tolist())
                    samples[f"sample_{i+1}"] = sample_text

        # Логирование в MLflow
        for name, text in samples.items():
            mlflow.log_text(text, f"generated/{name}.txt")

        # Сводная статистика по генерации
        avg_length = np.mean([len(text) for text in samples.values()])
        unique_chars = len(set(''.join(samples.values())))

        mlflow.log_metrics({
            "generation_avg_length": avg_length,
            "generation_unique_chars": unique_chars,
            "generation_samples": num_samples
        })

    def _log_model_analysis(self, model, data_torch, config):
        """📊 Дополнительный анализ модели"""

        try:
            # Анализ весов
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Статистика весов
            all_weights = []
            for param in model.parameters():
                if param.requires_grad:
                    all_weights.extend(param.data.cpu().numpy().flatten())

            weights_array = np.array(all_weights)

            analysis_metrics = {
                "model_total_params": total_params,
                "model_trainable_params": trainable_params,
                "weights_mean": float(np.mean(weights_array)),
                "weights_std": float(np.std(weights_array)),
                "weights_min": float(np.min(weights_array)),
                "weights_max": float(np.max(weights_array)),
                "data_vocab_coverage": len(torch.unique(data_torch)) / config.vocab_size
            }

            mlflow.log_metrics(analysis_metrics)

        except Exception as e:
            self.logger.logger.warning(f"⚠️  Ошибка в анализе модели: {e}")

    def run_vocab_size_experiments(self,
                                  vocab_sizes: List[int] = None,
                                  tokenizer_types: List[str] = None,
                                  base_config: ModelConfig = None) -> List[Dict[str, Any]]:
        """
        📊 ЗАПУСК ЭКСПЕРИМЕНТОВ С РАЗНЫМИ РАЗМЕРАМИ СЛОВАРЯ

        Args:
            vocab_sizes: Размеры словарей для тестирования
            tokenizer_types: Типы токенизаторов
            base_config: Базовая конфигурация (будет скопирована и изменена)

        Returns:
            Список результатов экспериментов
        """

        if vocab_sizes is None:
            vocab_sizes = [100, 200, 500, 1000]

        if tokenizer_types is None:
            tokenizer_types = ['char'] + (['bpe'] if HAS_BPE else [])

        if base_config is None:
            base_config = ModelConfig(
                max_iters=500,
                eval_interval=100,
                n_embd=128,
                n_layer=3,
                n_head=4
            )

        results = []
        total_experiments = len(vocab_sizes) * len(tokenizer_types)
        current_experiment = 0

        self.logger.logger.info(f"🔬 Запуск {total_experiments} экспериментов...")

        for tokenizer_type in tokenizer_types:
            for vocab_size in vocab_sizes:
                current_experiment += 1

                # Создаем копию конфигурации для каждого эксперимента
                config = ModelConfig(
                    vocab_size=vocab_size,
                    tokenizer_type=tokenizer_type,
                    max_iters=base_config.max_iters,
                    eval_interval=base_config.eval_interval,
                    n_embd=base_config.n_embd,
                    n_layer=base_config.n_layer,
                    n_head=base_config.n_head,
                    learning_rate=base_config.learning_rate,
                    model_name=f"gpt_model_{tokenizer_type}_{vocab_size}"
                )

                run_name = f"{tokenizer_type}_vocab_{vocab_size}"

                self.logger.logger.info(f"\n🚀 [{current_experiment}/{total_experiments}] Запуск: {run_name}")

                try:
                    final_val_loss = self.run_experiment(
                        config, 
                        run_name=run_name,
                        generation_samples=2,  # меньше образцов для скорости
                        save_model=True
                    )

                    result = {
                        'tokenizer_type': tokenizer_type,
                        'vocab_size': vocab_size,
                        'final_val_loss': final_val_loss,
                        'model_name': config.model_name,
                        'success': True
                    }

                    results.append(result)
                    self.logger.logger.info(f"✅ [{current_experiment}/{total_experiments}] Завершен {run_name}: val_loss = {final_val_loss:.4f}")

                except Exception as e:
                    self.logger.logger.error(f"❌ [{current_experiment}/{total_experiments}] Ошибка в {run_name}: {e}")
                    results.append({
                        'tokenizer_type': tokenizer_type,
                        'vocab_size': vocab_size,
                        'final_val_loss': float('inf'),
                        'model_name': config.model_name,
                        'success': False,
                        'error': str(e)
                    })

        # Сводка результатов
        self._print_experiment_summary(results)
        self._log_experiment_summary(results)

        return results

    def _print_experiment_summary(self, results: List[Dict[str, Any]]):
        """📊 Печать сводки результатов"""

        print("\n" + "="*80)
        print("📊 СВОДКА РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТОВ")
        print("="*80)

        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]

        print(f"✅ Успешно: {len(successful_results)}/{len(results)}")
        print(f"❌ Ошибки: {len(failed_results)}/{len(results)}")

        if successful_results:
            print("\n🏆 ЛУЧШИЕ РЕЗУЛЬТАТЫ:")
            sorted_results = sorted(successful_results, key=lambda x: x['final_val_loss'])

            for i, result in enumerate(sorted_results[:5], 1):
                print(f"  {i}. {result['tokenizer_type'].upper()} vocab={result['vocab_size']:4d}: "
                      f"val_loss={result['final_val_loss']:.4f}")

        if failed_results:
            print("\n❌ НЕУДАЧНЫЕ ЭКСПЕРИМЕНТЫ:")
            for result in failed_results:
                print(f"  • {result['tokenizer_type'].upper()} vocab={result['vocab_size']:4d}: {result['error']}")

        print("="*80)

    def _log_experiment_summary(self, results: List[Dict[str, Any]]):
        """📈 Логирование сводки в MLflow"""

        try:
            # Создаем отдельный run для сводки
            with mlflow.start_run(run_name="experiment_summary"):
                successful_results = [r for r in results if r['success']]

                if successful_results:
                    losses = [r['final_val_loss'] for r in successful_results]
                    best_result = min(successful_results, key=lambda x: x['final_val_loss'])

                    summary_metrics = {
                        "summary_total_experiments": len(results),
                        "summary_successful_experiments": len(successful_results),
                        "summary_failed_experiments": len(results) - len(successful_results),
                        "summary_best_val_loss": best_result['final_val_loss'],
                        "summary_worst_val_loss": max(losses),
                        "summary_mean_val_loss": np.mean(losses),
                        "summary_std_val_loss": np.std(losses)
                    }

                    mlflow.log_metrics(summary_metrics)

                    # Логируем лучшую конфигурацию
                    mlflow.log_params({
                        "best_tokenizer": best_result['tokenizer_type'],
                        "best_vocab_size": best_result['vocab_size'],
                        "best_model_name": best_result['model_name']
                    })

                    # Сохраняем подробные результаты как JSON
                    results_json = json.dumps(results, indent=2, ensure_ascii=False)
                    mlflow.log_text(results_json, "detailed_results.json")

        except Exception as e:
            self.logger.logger.warning(f"⚠️  Ошибка логирования сводки: {e}")


def run_single_experiment(vocab_size: int = 1000, 
                         tokenizer_type: str = 'char',
                         experiment_name: str = "single_test_runs") -> float:
    """
    🧪 БЫСТРЫЙ ЗАПУСК ОДНОГО ЭКСПЕРИМЕНТА

    Удобная функция для тестирования одной конфигурации
    """

    runner = ExperimentRunner(experiment_name=experiment_name)

    config = ModelConfig(
        vocab_size=vocab_size,
        tokenizer_type=tokenizer_type,
        max_iters=200,
        eval_interval=50,
        n_embd=64,
        n_layer=2,
        n_head=2,
        model_name=f"test_model_{tokenizer_type}_{vocab_size}"
    )

    run_name = f"test_{tokenizer_type}_vocab_{vocab_size}"

    print(f"🧪 Быстрый тест: {run_name}")
    final_val_loss = runner.run_experiment(config, run_name, generation_samples=1)

    print(f"🎯 Тест завершен: val_loss = {final_val_loss:.4f}")
    return final_val_loss


# 🚀 ГЛАВНЫЕ ФУНКЦИИ ДЛЯ СОВМЕСТИМОСТИ С ОРИГИНАЛЬНЫМ КОДОМ
def run_experiment(config: ModelConfig, run_name: str = None) -> float:
    """Запуск одного эксперимента - совместимость с оригинальным API"""
    runner = ExperimentRunner()
    return runner.run_experiment(config, run_name)


def run_vocab_size_experiments() -> List[Dict[str, Any]]:
    """Запуск экспериментов с разными размерами словаря - совместимость"""
    runner = ExperimentRunner()
    return runner.run_vocab_size_experiments()


def create_tokenizer(data: str, config: ModelConfig):
    """Создание токенизатора - совместимость"""
    runner = ExperimentRunner()
    return runner.create_tokenizer(data, config)


if __name__ == "__main__":
    # 🎯 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ

    print("🚀 DIPLOMA PROJECT - MLFLOW EXPERIMENTS")
    print("="*60)
    print("Доступные режимы:")
    print("1. 🧪 Быстрый тест")
    print("2. 🔬 Полные эксперименты")
    print("3. ⚙️  Кастомный эксперимент")

    mode = input("\nВыберите режим (1-3) или Enter для теста: ").strip()

    if mode == "1" or mode == "":
        # Быстрый тест
        test_loss = run_single_experiment(vocab_size=500, tokenizer_type='char')

    elif mode == "2":
        # Полные эксперименты
        runner = ExperimentRunner(experiment_name="vocab_size_comparison")
        results = runner.run_vocab_size_experiments()
        print(f"\n🎉 Завершено {len(results)} экспериментов!")

    elif mode == "3":
        # Кастомный эксперимент
        runner = ExperimentRunner(experiment_name="custom_experiments")

        config = ModelConfig(
            vocab_size=1000,
            tokenizer_type='char',
            max_iters=1000,
            eval_interval=200,
            n_embd=256,
            n_layer=6,
            n_head=4,
            model_name="custom_gpt_model"
        )

        final_loss = runner.run_experiment(
            config, 
            run_name="custom_experiment",
            generation_samples=5,
            save_model=True
        )

        print(f"\n🎯 Кастомный эксперимент завершен: val_loss = {final_loss:.4f}")

    print("\n🎉 Все эксперименты завершены!")
    print("📊 Результаты доступны в MLflow UI: mlflow ui")
