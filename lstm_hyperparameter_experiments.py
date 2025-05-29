import torch
import pandas as pd
import itertools # Для комбинаций гиперпараметров
import time

# Предполагается, что эти модули находятся в той же директории или доступны для импорта
# from preprocessing import # ... функции для загрузки и подготовки данных (например, создание словаря)
# from lstm_model import створити_lstm_модель
# from training import run_training, get_device, plot_training_history # plot_training_history для отдельных запусков
# from torch.utils.data import DataLoader # или ваш способ создания итераторов

# --- Заглушки для импортируемых функций и классов (замените реальными) ---
# Это нужно, чтобы файл мог быть проанализирован, даже если реальные зависимости отсутствуют.
# В реальном проекте эти заглушки нужно удалить и обеспечить корректный импорт.

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StubVocab: # Простая заглушка для словаря
    def __init__(self, tokens, unk_token='<unk>', pad_token='<pad>'):
        self.stoi = {token: i for i, token in enumerate(tokens)}
        self.itos = {i: token for i, token in enumerate(tokens)}
        self.unk_index = self.stoi.get(unk_token)
        self.pad_index = self.stoi.get(pad_token)
        self.pad_token = pad_token
        self.unk_token = unk_token
    def __len__(self):
        return len(self.stoi)
    def get(self, token, default=None):
        return self.stoi.get(token, default if default is not None else self.unk_index)


# Заглушка для функции создания модели LSTM
def створити_lstm_модель(vocab_size, output_dim, config, embedding_weights=None):
    print(f"ЗАГЛУШКА: Создание LSTM модели с vocab_size={vocab_size}, output_dim={output_dim}, config={config}")
    # Возвращаем простую nn.Module для имитации
    model = torch.nn.Linear(config.get('embedding_dim',100), output_dim) # Очень упрощенная модель
    # Добавим атрибуты, которые могут использоваться в training.py или других местах
    model.pad_idx = config.get('pad_idx', 1)
    model.embedding_dim = config.get('embedding_dim', 100) # Пример
    return model

# Заглушка для функции обучения
def run_training(model, model_type, train_iterator, val_iterator, n_epochs, device,
                 model_save_path, model_name,
                 lr_lstm=0.001, lr_bert=2e-5,
                 patience_early_stopping=5, patience_lr_scheduler=3,
                 bert_warmup_steps=0, bert_total_steps=0):
    print(f"ЗАГЛУШКА: Запуск обучения для {model_name} (тип: {model_type}) с lr={lr_lstm if model_type=='lstm' else lr_bert}")
    # Имитируем историю обучения
    history = {
        'train_loss': [0.5/i for i in range(1, n_epochs+1)],
        'train_acc': [0.7 + i*0.02 for i in range(n_epochs)],
        'train_f1': [0.65 + i*0.02 for i in range(n_epochs)],
        'val_loss': [0.6/i for i in range(1, n_epochs+1)],
        'val_acc': [0.65 + i*0.015 for i in range(n_epochs)],
        'val_f1': [0.6 + i*0.015 for i in range(n_epochs)], # Ключевая метрика
    }
    # Возвращаем модель и историю, а также лучшую F1 (для простоты берем последнюю)
    best_val_f1 = history['val_f1'][-1] if history['val_f1'] else 0
    print(f"ЗАГЛУШКА: Обучение завершено. Лучшая Val F1: {best_val_f1:.4f}")
    return model, history # , best_val_f1 # run_training возвращает model, history

# Заглушка для DataLoader
from torch.utils.data import Dataset, DataLoader
class DummyDataset(Dataset):
    def __init__(self, num_samples, seq_len, vocab_size, num_classes, pad_idx):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.pad_idx = pad_idx
        # Данные для LSTM: (text_indices, text_lengths, labels)
        self.texts = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.lengths = torch.randint(int(seq_len/2), seq_len + 1, (num_samples,))
        self.labels = torch.randint(0, num_classes, (num_samples,))
        # Убедимся, что длины не превышают seq_len и не равны 0
        self.lengths = torch.clamp(self.lengths, min=1, max=seq_len)
        # Применяем padding на основе длин (упрощенно)
        for i in range(num_samples):
            self.texts[i, self.lengths[i]:] = self.pad_idx


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Для LSTM модели, как ожидает train_epoch_lstm/evaluate_lstm
        # text_indices, text_lengths, labels
        return self.texts[idx], self.lengths[idx], self.labels[idx]

# ---------------------------------------------------------------------------

def run_lstm_arch_experiments(train_loader, val_loader, vocab_size, output_dim, pad_idx, device, base_config, n_epochs_exp=10):
    """
    Проводит эксперименты с архитектурными параметрами LSTM.
    (Размер скрытого состояния, количество слоев, направленность)
    """
    print("\n--- Начало экспериментов с архитектурными параметрами LSTM ---")
    
    results = []

    # Параметры из Таблицы 11 / Рис. 13 (стр. 46)
    hidden_dims = [128, 256, 512]
    n_layers_options = [1, 2, 3]
    bidirectional_options = [True, False] # True для BiLSTM, False для однонаправленного

    # Эксперимент 1: Влияние размерности скрытого состояния (при n_layers=2, bidirectional=True)
    print("\nЭксперимент 1: Влияние размерности скрытого состояния (n_layers=2, BiLSTM)")
    for h_dim in hidden_dims:
        config = base_config.copy()
        config['hidden_dim'] = h_dim
        config['n_layers'] = 2
        config['bidirectional'] = True
        
        experiment_name = f"lstm_h{h_dim}_l2_bi"
        print(f"\nЗапуск: {experiment_name}, Config: {config}")
        
        model = створити_lstm_модель(vocab_size, output_dim, config)
        _, history = run_training(model, 'lstm', train_loader, val_loader, n_epochs_exp, device,
                                  './models_exp', experiment_name, lr_lstm=config.get('lr', 0.001))
        
        best_f1 = max(history['val_f1']) if history['val_f1'] else 0
        results.append({'experiment': experiment_name, 'hidden_dim': h_dim, 'n_layers': 2, 
                        'bidirectional': True, 'val_f1': best_f1, 'config_lr': config.get('lr', 0.001)})
        print(f"Результат для {experiment_name}: Лучшая Val F1 = {best_f1:.4f}")

    # Эксперимент 2: Влияние количества слоев (при hidden_dim=256, bidirectional=True)
    print("\nЭксперимент 2: Влияние количества слоев (hidden_dim=256, BiLSTM)")
    for n_l in n_layers_options:
        config = base_config.copy()
        config['hidden_dim'] = 256
        config['n_layers'] = n_l
        config['bidirectional'] = True
        
        experiment_name = f"lstm_h256_l{n_l}_bi"
        print(f"\nЗапуск: {experiment_name}, Config: {config}")
        
        model = створити_lstm_модель(vocab_size, output_dim, config)
        _, history = run_training(model, 'lstm', train_loader, val_loader, n_epochs_exp, device,
                                  './models_exp', experiment_name, lr_lstm=config.get('lr', 0.001))

        best_f1 = max(history['val_f1']) if history['val_f1'] else 0
        results.append({'experiment': experiment_name, 'hidden_dim': 256, 'n_layers': n_l, 
                        'bidirectional': True, 'val_f1': best_f1, 'config_lr': config.get('lr', 0.001)})
        print(f"Результат для {experiment_name}: Лучшая Val F1 = {best_f1:.4f}")

    # Эксперимент 3: Влияние направленности (при hidden_dim=256, n_layers=2)
    print("\nЭксперимент 3: Влияние направленности (hidden_dim=256, n_layers=2)")
    for bi_opt in bidirectional_options:
        config = base_config.copy()
        config['hidden_dim'] = 256
        config['n_layers'] = 2
        config['bidirectional'] = bi_opt
        
        direction_str = "bi" if bi_opt else "uni"
        experiment_name = f"lstm_h256_l2_{direction_str}"
        print(f"\nЗапуск: {experiment_name}, Config: {config}")

        model = створити_lstm_модель(vocab_size, output_dim, config)
        _, history = run_training(model, 'lstm', train_loader, val_loader, n_epochs_exp, device,
                                  './models_exp', experiment_name, lr_lstm=config.get('lr', 0.001))
        
        best_f1 = max(history['val_f1']) if history['val_f1'] else 0
        results.append({'experiment': experiment_name, 'hidden_dim': 256, 'n_layers': 2, 
                        'bidirectional': bi_opt, 'val_f1': best_f1, 'config_lr': config.get('lr', 0.001)})
        print(f"Результат для {experiment_name}: Лучшая Val F1 = {best_f1:.4f}")
        
    return pd.DataFrame(results)


def run_lstm_learning_hp_experiments(train_loader, val_loader, vocab_size, output_dim, pad_idx, device, base_arch_config, n_epochs_exp=10):
    """
    Проводит эксперименты с гиперпараметрами обучения LSTM.
    (Скорость обучения, размер батча - хотя размер батча влияет на DataLoader, не на модель напрямую)
    Размер батча здесь не перебирается, т.к. он задается при создании DataLoader'ов.
    Этот эксперимент сфокусируется на скорости обучения.
    """
    print("\n--- Начало экспериментов с гиперпараметрами обучения LSTM (скорость обучения) ---")
    results = []

    learning_rates = [0.0001, 0.001, 0.01] # Из Рис. 9 (стр. 29)

    # Используем "оптимальную" архитектуру из предыдущих экспериментов или курсовой
    # Например, BiLSTM, 2 слоя, hidden_dim=256
    
    print(f"Базовая архитектура для эксперимента со скоростью обучения: {base_arch_config}")

    for lr in learning_rates:
        config = base_arch_config.copy()
        config['lr'] = lr # Добавляем LR в конфиг для передачи в run_training (если он это поддерживает)
                          # или просто передаем lr_lstm в run_training.
        
        experiment_name = f"lstm_optArch_lr{lr}"
        print(f"\nЗапуск: {experiment_name}, LR: {lr}")
        
        model = створити_lstm_модель(vocab_size, output_dim, config) # config здесь для архитектуры
        _, history = run_training(model, 'lstm', train_loader, val_loader, n_epochs_exp, device,
                                  './models_exp', experiment_name, lr_lstm=lr) # Передаем lr явно

        best_f1 = max(history['val_f1']) if history['val_f1'] else 0
        results.append({'experiment': experiment_name, 'learning_rate': lr, 'val_f1': best_f1, 
                        'architecture': base_arch_config})
        print(f"Результат для {experiment_name}: Лучшая Val F1 = {best_f1:.4f}")
        
    return pd.DataFrame(results)


if __name__ == '__main__':
    # --- Настройка для примера ---
    DEVICE = get_device()
    print(f"Используется устройство: {DEVICE}")

    # Параметры для фиктивных данных и модели
    VOCAB_SIZE = 1000
    NUM_CLASSES = 7
    PAD_IDX = 1
    EMBEDDING_DIM_DEFAULT = 100 # Для заглушки модели
    SEQ_LEN_DUMMY = 50
    BATCH_SIZE_EXP = 32 # Размер батча для экспериментов (Рис. 9 варьирует и его, но здесь фиксируем для DataLoader)
    N_EPOCHS_PER_EXPERIMENT = 3 # Уменьшаем для быстрого теста, в реальности 10-30

    # Создание фиктивного словаря
    dummy_tokens = [f'word{i}' for i in range(VOCAB_SIZE - 2)] + ['<unk>', '<pad>']
    vocab_stub = StubVocab(dummy_tokens)

    # Создание фиктивных DataLoader'ов
    # Важно: DataLoader для LSTM должен возвращать (text_indices, text_lengths, labels)
    train_dummy_dataset = DummyDataset(num_samples=BATCH_SIZE_EXP * 5, seq_len=SEQ_LEN_DUMMY, vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES, pad_idx=PAD_IDX)
    val_dummy_dataset = DummyDataset(num_samples=BATCH_SIZE_EXP * 2, seq_len=SEQ_LEN_DUMMY, vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES, pad_idx=PAD_IDX)
    
    # Убедимся, что DummyDataset возвращает то, что ожидает train_epoch_lstm
    # sample_batch_text, sample_batch_lengths, sample_batch_labels = next(iter(DataLoader(train_dummy_dataset, batch_size=BATCH_SIZE_EXP)))
    # print("Пример батча из DummyDataset (текст):", sample_batch_text.shape)
    # print("Пример батча из DummyDataset (длины):", sample_batch_lengths.shape)
    # print("Пример батча из DummyDataset (метки):", sample_batch_labels.shape)


    train_loader_stub = DataLoader(train_dummy_dataset, batch_size=BATCH_SIZE_EXP, shuffle=True)
    val_loader_stub = DataLoader(val_dummy_dataset, batch_size=BATCH_SIZE_EXP)


    # Базовая конфигурация для LSTM (можно менять)
    # Эта конфигурация будет использоваться как основа, и отдельные параметры будут варьироваться.
    base_lstm_config = {
        'embedding_dim': EMBEDDING_DIM_DEFAULT, # Должно соответствовать вашей модели
        'dropout_lstm': 0.3,    # Из курсовой
        'pad_idx': PAD_IDX,
        'lr': 0.001 # Базовая скорость обучения
        # hidden_dim, n_layers, bidirectional будут варьироваться в экспериментах
    }

    # Запуск экспериментов с архитектурой
    arch_results_df = run_lstm_arch_experiments(
        train_loader_stub, val_loader_stub, VOCAB_SIZE, NUM_CLASSES, PAD_IDX, DEVICE,
        base_lstm_config, n_epochs_exp=N_EPOCHS_PER_EXPERIMENT
    )
    print("\n--- Результаты экспериментов с архитектурой LSTM ---")
    if not arch_results_df.empty:
        print(arch_results_df.sort_values(by='val_f1', ascending=False))
    else:
        print("Нет результатов для отображения (архитектурные эксперименты).")

    # Определение "оптимальной" архитектуры на основе результатов или курсовой
    # В курсовой (стр. 26): BiLSTM, 2 слоя, hidden_dim=256
    optimal_arch_config_lstm = base_lstm_config.copy()
    optimal_arch_config_lstm.update({
        'hidden_dim': 256 if EMBEDDING_DIM_DEFAULT < 256 else EMBEDDING_DIM_DEFAULT, # hidden_dim не может быть < embedding_dim если они разные
        'n_layers': 2,
        'bidirectional': True
    })
    if EMBEDDING_DIM_DEFAULT > 256: # Корректировка для заглушки
        optimal_arch_config_lstm['hidden_dim'] = 256


    # Запуск экспериментов с гиперпараметрами обучения (скорость обучения)
    learning_hp_results_df = run_lstm_learning_hp_experiments(
        train_loader_stub, val_loader_stub, VOCAB_SIZE, NUM_CLASSES, PAD_IDX, DEVICE,
        optimal_arch_config_lstm, n_epochs_exp=N_EPOCHS_PER_EXPERIMENT
    )
    print("\n--- Результаты экспериментов с гиперпараметрами обучения LSTM (LR) ---")
    if not learning_hp_results_df.empty:
        print(learning_hp_results_df.sort_values(by='val_f1', ascending=False))
    else:
        print("Нет результатов для отображения (эксперименты с LR).")

    print("\nВсе эксперименты с LSTM завершены.")

