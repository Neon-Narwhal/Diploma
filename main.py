"""Главный файл для демонстрации генерации с логированием, обучения и MLflow"""

import torch
import sys
import os

# Добавляем текущую папку в путь Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Прямые импорты из папок в корне
from models.gpt_model import GPTLikeModel
from utils.config import ModelConfig
from my_tokenizers.char_tokenizer import CharTokenizer
from my_tokenizers.bpe_tokenizer import BPETokenizer
from utils.logging import GenerationLogger
from utils.data_utils import load_data, prepare_data

from training.transformer_squared_training import run_quick_test, run_laptop_test
from training.trainer import ModelTrainer, AdvancedModelTrainer




def demo_basic_training():
    """🏋️‍♂️ Демонстрация базового обучения без MLflow"""

    print("\n" + "="*60)
    print("🏋️‍♂️ ДЕМОНСТРАЦИЯ БАЗОВОГО ОБУЧЕНИЯ")
    print("="*60)

    # Конфигурация для быстрого обучения
    config = ModelConfig(
        tokenizer_type='char',
        max_iters=50,          
        eval_interval=10,
        n_embd=32,             # Маленькая модель
        n_layer=2,
        n_head=2,
        learning_rate=3e-4
    )

    # Подготовка данных
    data = load_data(config.data_file)
    tokenizer = CharTokenizer(data)
    config.vocab_size = tokenizer.get_vocab_size()

    # Подготавливаем данные для обучения
    data_tensor = prepare_data(data, tokenizer, config)


    # Создаем модель
    model = GPTLikeModel(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print(f"\n🧠 Модель создана:")
    print(f"   Параметры: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Размер данных: {len(data_tensor):,} токенов")

    # Простой тренер
    trainer = ModelTrainer(model, optimizer, config)

    print("\n🚀 Начинаем обучение...")

    # Обучение с простым callback
    def training_callback(step, metrics):
        if step % 20 == 0:
            print(f"  📊 Шаг {step}: train={metrics.get('train', 0):.3f}, val={metrics.get('val', 0):.3f}")

    results = trainer.train_epoch(data_tensor, callback=training_callback)

    print(f"\n📈 Результаты обучения:")
    final_losses = results.get('final_losses', {})
    print(f"   Финальный train loss: {final_losses.get('train', 'N/A'):.4f}")
    print(f"   Финальный val loss: {final_losses.get('val', 'N/A'):.4f}")

    if 'early_stopped_at' in results:
        print(f"   ⏹️ Early stopping на шаге: {results['early_stopped_at']}")

    print("✅ Базовое обучение завершено!")

    return model, tokenizer, config


def demo_generation_after_training(model, tokenizer, config):
    """🎨 Демонстрация генерации после обучения"""

    if model is None:
        print("⚠️ Модель недоступна, пропускаем генерацию после обучения")
        return

    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)

    print("🎯 Генерация с обученной моделью (с логированием):")
    print("-" * 50)

    with torch.no_grad():
        generated = model.generate_with_logging(
            context, 
            max_new_token=80, 
            tokenizer=tokenizer,
            temperature=0.9
        )

    print("-" * 50)
    print("✅ Генерация после обучения завершена!")



if __name__ == "__main__":
    torch.cuda.empty_cache()
    #main()
    run_quick_test() ## sakana
    #run_laptop_test()