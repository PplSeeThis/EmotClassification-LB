import torch
import pandas as pd
import itertools
import time
from transformers import BertTokenizer, BertConfig # Нужны для создания конфигураций и токенизатора

# Предполагается, что эти модули находятся в той же директории или доступны для импорта
# from bert_lite_model import створити_bert_lite_модель
# from training import run_training, get_device
# from torch.utils.data import DataLoader

# --- Заглушки для импортируемых функций и классов (замените реальными) ---
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Заглушка для функции создания модели BERT-lite
def створити_bert_lite_модель(num_classes, config_bert_lite, tokenizer_name_for_vocab):
    # config_bert_lite здесь это словарь с параметрами
    # tokenizer_name_for_vocab используется для получения vocab_size
    
    # Создаем BertConfig на основе словаря config_bert_lite и tokenizer_name_for_vocab
    # Это упрощенная версия того, что делает реальная функция
    try:
        base_config = BertConfig.from_pretrained(tokenizer_name_for_vocab)
    except OSError: # Если токенизатор не найден локально и нет сети
        print(f"ПРЕДУПРЕЖДЕНИЕ (ЗАГЛУШКА): Не удалось загрузить конфигурацию для {tokenizer_name_for_vocab}. Используется BertConfig() по умолчанию.")
        base_config = BertConfig() # Базовая конфигурация по умолчанию

    actual_config = BertConfig(
        vocab_size=base_config.vocab_size, # Важно
        hidden_size=config_bert_lite.get('hidden_size', 256), # Меньше для заглушки
        num_hidden_layers=config_bert_lite.get('num_hidden_layers', 2),
        num_attention_heads=config_bert_lite.get('num_attention_heads', 4),
        intermediate_size=config_bert_lite.get('intermediate_size', 1024),
        hidden_dropout_prob=config_bert_lite.get('hidden_dropout_prob', 0.1),
        attention_probs_dropout_prob=config_bert_lite.get('attention_probs_dropout_prob', 0.1),
        # dropout_rate_head не является частью BertConfig, а передается в BertLiteClassifier
    )
    print(f"ЗАГЛУШКА: Создание BERT-lite модели с num_classes={num_classes}, config={actual_config}")
    # Возвращаем простую nn.Module для имитации
    # Вход BertLiteClassifier обычно hidden_size из BertConfig
    model = torch.nn.Linear(actual_config.hidden_size, num_classes)
    # Добавим атрибут config для возможного использования в training.py
    model.config = actual_config
    return model

# Заглушка для функции обучения (та же, что и в LSTM экспериментах)
def run_training(model, model_type, train_iterator, val_iterator, n_epochs, device,
                 model_save_path, model_name,
                 lr_lstm=0.001, lr_bert=2e-5,
                 patience_early_stopping=5, patience_lr_scheduler=3,
                 bert_warmup_steps=0, bert_total_steps=0): # bert_total_steps нужно вычислить
    print(f"ЗАГЛУШКА: Запуск обучения для {model_name} (тип: {model_type}) с lr={lr_lstm if model_type=='lstm' else lr_bert}")
    if model_type == 'bert' and bert_total_steps == 0 and hasattr(train_iterator, '__len__'):
        bert_total_steps = len(train_iterator) * n_epochs
        print(f"ЗАГЛУШКА: bert_total_steps оценен в {bert_total_steps}")
    if model_type == 'bert' and bert_warmup_steps == 0 and bert_total_steps > 0:
        bert_warmup_steps = int(0.1 * bert_total_steps) # 10% warmup
        print(f"ЗАГЛУШКА: bert_warmup_steps установлен в {bert_warmup_steps}")

    history = {
        'train_loss': [0.5/i for i in range(1, n_epochs+1)],
        'train_acc': [0.7 + i*0.02 for i in range(n_epochs)],
        'train_f1': [0.65 + i*0.02 for i in range(n_epochs)],
        'val_loss': [0.6/i for i in range(1, n_epochs+1)],
        'val_acc': [0.65 + i*0.015 for i in range(n_epochs)],
        'val_f1': [0.6 + i*0.015 for i in range(n_epochs)],
    }
    best_val_f1 = history['val_f1'][-1] if history['val_f1'] else 0
    print(f"ЗАГЛУШКА: Обучение завершено. Лучшая Val F1: {best_val_f1:.4f}")
    return model, history

# Заглушка для DataLoader для BERT
from torch.utils.data import Dataset, DataLoader
class DummyBertDataset(Dataset):
    def __init__(self, num_samples, seq_len, vocab_size_bert, num_classes):
        self.num_samples = num_samples
        # Данные для BERT: (input_ids, attention_mask, labels)
        # token_type_ids опциональны для классификации одного предложения
        self.input_ids = torch.randint(0, vocab_size_bert, (num_samples, seq_len))
        self.attention_masks = torch.ones((num_samples, seq_len), dtype=torch.long)
        # Упрощенно: если есть паддинг, маска должна быть 0 там
        # for i in range(num_samples):
        #    pad_start_idx = torch.randint(seq_len // 2, seq_len + 1, (1,)).item()
        #    if pad_start_idx < seq_len:
        #        self.input_ids[i, pad_start_idx:] = 0 # Пример паддинг токена
        #        self.attention_masks[i, pad_start_idx:] = 0
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]

# ---------------------------------------------------------------------------

def run_bert_arch_experiments(train_loader, val_loader, num_classes, tokenizer_name, device, base_bert_config_dict, n_epochs_exp=5):
    """
    Проводит эксперименты с архитектурными параметрами BERT-lite.
    (Количество слоев, количество голов внимания, размер скрытого состояния)
    """
    print("\n--- Начало экспериментов с архитектурными параметрами BERT-lite ---")
    results = []

    # Параметры из Таблицы 15 / Рис. 8/15/18 (стр. 53)
    num_hidden_layers_options = [4, 6, 8]
    num_attention_heads_options = [4, 8, 12]
    hidden_size_options = [384, 512, 768]
    
    # Базовые параметры, которые не меняются в этих экспериментах (или берутся из tokenizer_name)
    # base_config_dict содержит dropout_rate_head и другие параметры для модели

    # Эксперимент 1: Влияние количества слоев (при heads=8, hidden_size=512)
    print("\nЭксперимент 1: Влияние количества слоев (heads=8, hidden_size=512)")
    for n_layers in num_hidden_layers_options:
        current_exp_config = base_bert_config_dict.copy()
        current_exp_config['num_hidden_layers'] = n_layers
        current_exp_config['num_attention_heads'] = 8
        current_exp_config['hidden_size'] = 512
        current_exp_config['intermediate_size'] = 4 * 512 # Стандартно

        experiment_name = f"bert_l{n_layers}_h512_heads8"
        print(f"\nЗапуск: {experiment_name}, Config_dict: {current_exp_config}")
        
        # `створити_bert_lite_модель` ожидает словарь для `config_bert_lite`
        model = створити_bert_lite_модель(num_classes, current_exp_config, tokenizer_name)
        
        total_steps = len(train_loader) * n_epochs_exp
        _, history = run_training(model, 'bert', train_loader, val_loader, n_epochs_exp, device,
                                  './models_exp_bert', experiment_name, 
                                  lr_bert=current_exp_config.get('lr', 2e-5),
                                  bert_total_steps=total_steps)
        
        best_f1 = max(history['val_f1']) if history['val_f1'] else 0
        results.append({'experiment': experiment_name, 'n_layers': n_layers, 'n_heads': 8, 
                        'hidden_size': 512, 'val_f1': best_f1, 'config_lr': current_exp_config.get('lr', 2e-5)})
        print(f"Результат для {experiment_name}: Лучшая Val F1 = {best_f1:.4f}")

    # Эксперимент 2: Влияние количества голов внимания (при layers=6, hidden_size=512)
    print("\nЭксперимент 2: Влияние количества голов внимания (layers=6, hidden_size=512)")
    for n_heads in num_attention_heads_options:
        current_exp_config = base_bert_config_dict.copy()
        current_exp_config['num_hidden_layers'] = 6
        current_exp_config['num_attention_heads'] = n_heads
        current_exp_config['hidden_size'] = 512
        current_exp_config['intermediate_size'] = 4 * 512

        experiment_name = f"bert_l6_h512_heads{n_heads}"
        print(f"\nЗапуск: {experiment_name}, Config_dict: {current_exp_config}")

        model = створити_bert_lite_модель(num_classes, current_exp_config, tokenizer_name)
        total_steps = len(train_loader) * n_epochs_exp
        _, history = run_training(model, 'bert', train_loader, val_loader, n_epochs_exp, device,
                                  './models_exp_bert', experiment_name,
                                  lr_bert=current_exp_config.get('lr', 2e-5),
                                  bert_total_steps=total_steps)
        
        best_f1 = max(history['val_f1']) if history['val_f1'] else 0
        results.append({'experiment': experiment_name, 'n_layers': 6, 'n_heads': n_heads, 
                        'hidden_size': 512, 'val_f1': best_f1, 'config_lr': current_exp_config.get('lr', 2e-5)})
        print(f"Результат для {experiment_name}: Лучшая Val F1 = {best_f1:.4f}")

    # Эксперимент 3: Влияние размерности скрытого состояния (при layers=6, heads=8)
    print("\nЭксперимент 3: Влияние размерности скрытого состояния (layers=6, heads=8)")
    for h_size in hidden_size_options:
        current_exp_config = base_bert_config_dict.copy()
        current_exp_config['num_hidden_layers'] = 6
        current_exp_config['num_attention_heads'] = 8
        current_exp_config['hidden_size'] = h_size
        current_exp_config['intermediate_size'] = 4 * h_size

        experiment_name = f"bert_l6_h{h_size}_heads8"
        print(f"\nЗапуск: {experiment_name}, Config_dict: {current_exp_config}")

        model = створити_bert_lite_модель(num_classes, current_exp_config, tokenizer_name)
        total_steps = len(train_loader) * n_epochs_exp
        _, history = run_training(model, 'bert', train_loader, val_loader, n_epochs_exp, device,
                                  './models_exp_bert', experiment_name,
                                  lr_bert=current_exp_config.get('lr', 2e-5),
                                  bert_total_steps=total_steps)
        
        best_f1 = max(history['val_f1']) if history['val_f1'] else 0
        results.append({'experiment': experiment_name, 'n_layers': 6, 'n_heads': 8, 
                        'hidden_size': h_size, 'val_f1': best_f1, 'config_lr': current_exp_config.get('lr', 2e-5)})
        print(f"Результат для {experiment_name}: Лучшая Val F1 = {best_f1:.4f}")
        
    return pd.DataFrame(results)


def run_bert_learning_hp_experiments(train_loader, val_loader, num_classes, tokenizer_name, device, base_arch_config_dict, n_epochs_exp=5):
    """
    Проводит эксперименты с гиперпараметрами обучения BERT-lite (скорость обучения).
    Размер батча фиксирован через DataLoader.
    """
    print("\n--- Начало экспериментов с гиперпараметрами обучения BERT-lite (скорость обучения) ---")
    results = []

    learning_rates = [5e-6, 2e-5, 5e-5] # Из Рис. 10 (стр. 30)
    
    print(f"Базовая архитектура для эксперимента со скоростью обучения: {base_arch_config_dict}")

    for lr in learning_rates:
        current_exp_config = base_arch_config_dict.copy()
        # lr будет передан в run_training напрямую
        
        experiment_name = f"bert_optArch_lr{lr}"
        print(f"\nЗапуск: {experiment_name}, LR: {lr}")
        
        model = створити_bert_lite_модель(num_classes, current_exp_config, tokenizer_name)
        total_steps = len(train_loader) * n_epochs_exp
        _, history = run_training(model, 'bert', train_loader, val_loader, n_epochs_exp, device,
                                  './models_exp_bert', experiment_name, lr_bert=lr,
                                  bert_total_steps=total_steps)

        best_f1 = max(history['val_f1']) if history['val_f1'] else 0
        results.append({'experiment': experiment_name, 'learning_rate': lr, 'val_f1': best_f1,
                        'architecture': base_arch_config_dict})
        print(f"Результат для {experiment_name}: Лучшая Val F1 = {best_f1:.4f}")
        
    return pd.DataFrame(results)


if __name__ == '__main__':
    DEVICE = get_device()
    print(f"Используется устройство: {DEVICE}")

    NUM_CLASSES = 7
    TOKENIZER_NAME_FOR_BERT = 'bert-base-multilingual-cased' # Для vocab_size и токенизации
    
    # Получаем vocab_size из реального токенизатора
    try:
        tokenizer_for_vocab = BertTokenizer.from_pretrained(TOKENIZER_NAME_FOR_BERT)
        VOCAB_SIZE_BERT = tokenizer_for_vocab.vocab_size
    except OSError:
        print(f"ПРЕДУПРЕЖДЕНИЕ: Не удалось загрузить токенизатор {TOKENIZER_NAME_FOR_BERT}. Используется фиктивный vocab_size=30000.")
        VOCAB_SIZE_BERT = 30000 # Фиктивный, если нет сети/токенизатора

    SEQ_LEN_BERT_DUMMY = 64
    BATCH_SIZE_BERT_EXP = 16 # Меньше для BERT из-за памяти (Рис. 10 использует 16, 32, 64)
    N_EPOCHS_PER_BERT_EXPERIMENT = 2 # Очень мало для быстрого теста

    # Создание фиктивных DataLoader'ов для BERT
    train_dummy_dataset_bert = DummyBertDataset(num_samples=BATCH_SIZE_BERT_EXP * 5, seq_len=SEQ_LEN_BERT_DUMMY, vocab_size_bert=VOCAB_SIZE_BERT, num_classes=NUM_CLASSES)
    val_dummy_dataset_bert = DummyBertDataset(num_samples=BATCH_SIZE_BERT_EXP * 2, seq_len=SEQ_LEN_BERT_DUMMY, vocab_size_bert=VOCAB_SIZE_BERT, num_classes=NUM_CLASSES)
    
    train_loader_bert_stub = DataLoader(train_dummy_dataset_bert, batch_size=BATCH_SIZE_BERT_EXP, shuffle=True)
    val_loader_bert_stub = DataLoader(val_dummy_dataset_bert, batch_size=BATCH_SIZE_BERT_EXP)

    # Базовая конфигурация для BERT-lite экспериментов (архитектурные параметры будут меняться)
    # Содержит параметры, не относящиеся к архитектуре энкодера BERT, но нужные для модели/обучения
    base_bert_exp_config = {
        'dropout_rate_head': 0.3, # Из курсовой (стр. 17)
        'lr': 2e-5, # Базовая скорость обучения для BERT
        # Архитектурные (num_hidden_layers, hidden_size, num_attention_heads, intermediate_size) 
        # будут установлены в циклах экспериментов.
        # hidden_dropout_prob и attention_probs_dropout_prob можно добавить, если они варьируются,
        # иначе они возьмутся из BertConfig по умолчанию или из tokenizer_name.
        'hidden_dropout_prob': 0.1, # Стандартный для BERT
        'attention_probs_dropout_prob': 0.1 # Стандартный для BERT
    }

    # Запуск экспериментов с архитектурой BERT-lite
    bert_arch_results_df = run_bert_arch_experiments(
        train_loader_bert_stub, val_loader_bert_stub, NUM_CLASSES, TOKENIZER_NAME_FOR_BERT, DEVICE,
        base_bert_exp_config, n_epochs_exp=N_EPOCHS_PER_BERT_EXPERIMENT
    )
    print("\n--- Результаты экспериментов с архитектурой BERT-lite ---")
    if not bert_arch_results_df.empty:
        print(bert_arch_results_df.sort_values(by='val_f1', ascending=False))
    else:
        print("Нет результатов для отображения (архитектурные эксперименты BERT).")

    # Определение "оптимальной" архитектуры BERT-lite на основе курсовой (стр. 27)
    # BERT-lite с 6 шарами трансформера, 8 головками уваги та розмірністю прихованого стану 512.
    optimal_arch_config_bert = base_bert_exp_config.copy()
    optimal_arch_config_bert.update({
        'num_hidden_layers': 6,
        'num_attention_heads': 8,
        'hidden_size': 512,
        'intermediate_size': 4 * 512
    })

    # Запуск экспериментов с гиперпараметрами обучения BERT-lite (скорость обучения)
    bert_learning_hp_results_df = run_bert_learning_hp_experiments(
        train_loader_bert_stub, val_loader_bert_stub, NUM_CLASSES, TOKENIZER_NAME_FOR_BERT, DEVICE,
        optimal_arch_config_bert, n_epochs_exp=N_EPOCHS_PER_BERT_EXPERIMENT
    )
    print("\n--- Результаты экспериментов с гиперпараметрами обучения BERT-lite (LR) ---")
    if not bert_learning_hp_results_df.empty:
        print(bert_learning_hp_results_df.sort_values(by='val_f1', ascending=False))
    else:
        print("Нет результатов для отображения (эксперименты с LR для BERT).")

    print("\nВсе эксперименты с BERT-lite завершены.")
