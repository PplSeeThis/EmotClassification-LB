import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # Для создания подвыборок
from sklearn.preprocessing import LabelEncoder # Для кодирования меток

# Предполагается, что эти модули находятся в той же директории или доступны для импорта
# from preprocessing import препроцесинг_тексту as препроцесинг_тексту_lstm_базовий, ukrainian_stopwords, розділення_даних as повне_розділення_даних
# from lstm_model import створити_lstm_модель
# from bert_lite_model import створити_bert_lite_модель
# from bert_data_preparation import завантажити_bert_токенізатор, підготувати_дані_для_bert_з_df # или отдельные компоненты
# from training import run_training, get_device, display_classification_report_and_confusion_matrix

# --- Заглушки для импортируемых функций и классов (замените реальными) ---
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StubVocab:
    def __init__(self, tokens, unk_token='<unk>', pad_token='<pad>'):
        self.stoi = {token: i for i, token in enumerate(tokens)}
        self.itos = {i: token for i, token in enumerate(tokens)}
        self.unk_index = self.stoi.get(unk_token)
        self.pad_index = self.stoi.get(pad_token)
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.vectors = None # Для имитации возможности загрузки предобученных весов
    def __len__(self):
        return len(self.stoi)
    def get(self, token, default=None):
        return self.stoi.get(token, default if default is not None else self.unk_index)
    def set_vectors(self, stoi, vectors, dim): # Метод для установки векторов, если нужно
        self.vectors = torch.randn(len(stoi), dim) # Фиктивные векторы
        print("ЗАГЛУШКА: Установлены фиктивные векторы для словаря.")


def створити_lstm_модель(vocab_size, output_dim, config, embedding_weights=None):
    print(f"ЗАГЛУШКА: Создание LSTM модели (data_size_comp) с vocab_size={vocab_size}, output_dim={output_dim}")
    model = torch.nn.Linear(config.get('embedding_dim',100), output_dim)
    model.pad_idx = config.get('pad_idx', 1)
    return model

from transformers import BertConfig # Нужен для заглушки створити_bert_lite_модель
def створити_bert_lite_модель(num_classes, config_bert_lite, tokenizer_name_for_vocab):
    try:
        base_config = BertConfig.from_pretrained(tokenizer_name_for_vocab)
    except OSError:
        base_config = BertConfig()
    actual_config = BertConfig(
        vocab_size=base_config.vocab_size,
        hidden_size=config_bert_lite.get('hidden_size', 256),
        num_hidden_layers=config_bert_lite.get('num_hidden_layers', 2),
        num_attention_heads=config_bert_lite.get('num_attention_heads', 4),
        intermediate_size=config_bert_lite.get('intermediate_size', 1024)
    )
    print(f"ЗАГЛУШКА: Создание BERT-lite (data_size_comp) с num_classes={num_classes}")
    model = torch.nn.Linear(actual_config.hidden_size, num_classes)
    model.config = actual_config
    return model

def run_training(model, model_type, train_iterator, val_iterator, n_epochs, device,
                 model_save_path, model_name, lr_lstm=0.001, lr_bert=2e-5,
                 patience_early_stopping=5, patience_lr_scheduler=3,
                 bert_warmup_steps=0, bert_total_steps=0):
    print(f"ЗАГЛУШКА: Запуск обучения (data_size_comp) для {model_name} (тип: {model_type})")
    if model_type == 'bert' and bert_total_steps == 0 and hasattr(train_iterator, '__len__'):
        bert_total_steps = len(train_iterator) * n_epochs
    if model_type == 'bert' and bert_warmup_steps == 0 and bert_total_steps > 0:
        bert_warmup_steps = int(0.1 * bert_total_steps)

    history = {
        'val_f1': [0.5 + 0.05 * i + (0.01 if model_type == 'bert' else 0) for i in range(n_epochs)],
        # Добавляем остальные ключи, чтобы не было ошибок при обращении к ним
        'train_loss': [1.0 / (i + 1) for i in range(n_epochs)],
        'train_acc': [0.6 + 0.05 * i for i in range(n_epochs)],
        'train_f1': [0.55 + 0.05 * i for i in range(n_epochs)],
        'val_loss': [0.9 / (i + 1) for i in range(n_epochs)],
        'val_acc': [0.65 + 0.05 * i for i in range(n_epochs)],
    }
    best_val_f1 = max(history['val_f1']) if history['val_f1'] else 0
    print(f"ЗАГЛУШКА: Обучение завершено для {model_name}. Лучшая Val F1: {best_val_f1:.4f}")
    return model, history

from torch.utils.data import Dataset, DataLoader
# Заглушка для DataLoader для LSTM
class DummyDatasetLSTM(Dataset):
    def __init__(self, texts_pd_series, labels_pd_series, vocab, max_len, pad_idx, num_classes_unused):
        self.texts_pd_series = texts_pd_series
        self.labels_pd_series = labels_pd_series # Ожидаем числовые метки
        self.vocab = vocab
        self.max_len = max_len
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.texts_pd_series)
    def __getitem__(self, idx):
        text_str = self.texts_pd_series.iloc[idx]
        # Упрощенная токенизация и индексация для заглушки
        tokens = text_str.lower().split()[:self.max_len]
        indexed = [self.vocab.get(t, self.vocab.unk_index) for t in tokens]
        length = len(indexed)
        padded = indexed + [self.pad_idx] * (self.max_len - length)
        return torch.tensor(padded, dtype=torch.long), torch.tensor(length, dtype=torch.long), torch.tensor(self.labels_pd_series.iloc[idx], dtype=torch.long)

# Заглушка для DataLoader для BERT
from transformers import BertTokenizer
class DummyDatasetBERT(Dataset):
    def __init__(self, texts_pd_series, labels_pd_series, tokenizer, max_len, num_classes_unused):
        self.texts_pd_series = texts_pd_series
        self.labels_pd_series = labels_pd_series # Ожидаем числовые метки
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts_pd_series)
    def __getitem__(self, idx):
        text_str = self.texts_pd_series.iloc[idx]
        encoded = self.tokenizer.encode_plus(
            text_str, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0), torch.tensor(self.labels_pd_series.iloc[idx], dtype=torch.long)

# --- Конец заглушек ---

def create_data_subset(full_train_df, text_col, label_col, subset_fraction, random_state=42):
    """
    Создает стратифицированную подвыборку из обучающего DataFrame.
    """
    if subset_fraction == 1.0:
        return full_train_df
    
    # train_test_split используется для получения стратифицированной подвыборки
    # Мы берем 'train' часть нужного размера, 'test' часть игнорируем
    subset_df, _ = train_test_split(
        full_train_df,
        train_size=subset_fraction, # или test_size = 1.0 - subset_fraction
        random_state=random_state,
        stratify=full_train_df[label_col]
    )
    return subset_df

def run_data_size_comparison(full_train_df_original: pd.DataFrame, 
                             val_df_original: pd.DataFrame, 
                             text_col: str, label_col: str,
                             num_classes: int, device: torch.device,
                             # LSTM specific
                             vocab_lstm: StubVocab, pad_idx_lstm: int, max_len_lstm: int, 
                             config_lstm_optimal: dict, lr_lstm_optimal: float,
                             # BERT specific
                             tokenizer_bert: BertTokenizer, max_len_bert: int,
                             config_bert_optimal: dict, lr_bert_optimal: float,
                             # Common
                             batch_size: int, n_epochs_per_subset: int,
                             data_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
                            ):
    """
    Проводит сравнение моделей LSTM и BERT-lite на разных объемах обучающих данных.
    """
    print("\n--- Начало сравнения моделей на разных объемах данных ---")
    results = []
    
    # Кодирование меток один раз для всего датасета, если они текстовые
    # В курсовой не указано, но это стандартная практика.
    # Предположим, что label_col уже содержит числовые метки или LabelEncoder применен ранее.
    # Если нет, нужно добавить:
    # le = LabelEncoder()
    # full_train_df_original[label_col + '_encoded'] = le.fit_transform(full_train_df_original[label_col])
    # val_df_original[label_col + '_encoded'] = le.transform(val_df_original[label_col])
    # label_col_encoded = label_col + '_encoded'
    # Вместо этого, мы будем передавать числовые метки в DummyDataset

    # Валидационные данные остаются неизменными для всех экспериментов
    # Создаем валидационные DataLoader'ы один раз
    val_dataset_lstm = DummyDatasetLSTM(val_df_original[text_col], val_df_original[label_col], vocab_lstm, max_len_lstm, pad_idx_lstm, num_classes)
    val_loader_lstm = DataLoader(val_dataset_lstm, batch_size=batch_size)

    val_dataset_bert = DummyDatasetBERT(val_df_original[text_col], val_df_original[label_col], tokenizer_bert, max_len_bert, num_classes)
    val_loader_bert = DataLoader(val_dataset_bert, batch_size=batch_size)

    for fraction in data_fractions:
        print(f"\n--- Обучение на {fraction*100:.0f}% обучающих данных ---")
        
        current_train_df = create_data_subset(full_train_df_original, text_col, label_col, fraction)
        print(f"Размер текущей обучающей выборки: {len(current_train_df)}")

        # --- LSTM ---
        print("\nМодель: LSTM")
        # Создаем DataLoader для текущей обучающей подвыборки LSTM
        train_dataset_lstm_subset = DummyDatasetLSTM(current_train_df[text_col], current_train_df[label_col], vocab_lstm, max_len_lstm, pad_idx_lstm, num_classes)
        train_loader_lstm_subset = DataLoader(train_dataset_lstm_subset, batch_size=batch_size, shuffle=True)

        model_lstm = створити_lstm_модель(len(vocab_lstm), num_classes, config_lstm_optimal, embedding_weights=vocab_lstm.vectors)
        
        # Убедимся, что bert_total_steps и bert_warmup_steps не передаются для LSTM
        _, history_lstm = run_training(model_lstm, 'lstm', train_loader_lstm_subset, val_loader_lstm,
                                       n_epochs_per_subset, device, './models_datasize', f"lstm_datafrac{fraction:.2f}",
                                       lr_lstm=lr_lstm_optimal)
        best_f1_lstm = max(history_lstm['val_f1']) if history_lstm['val_f1'] else 0
        results.append({'model': 'LSTM', 'data_fraction': fraction, 'val_f1': best_f1_lstm})
        print(f"LSTM на {fraction*100:.0f}% данных: Val F1 = {best_f1_lstm:.4f}")

        # --- BERT-lite ---
        print("\nМодель: BERT-lite")
        # Создаем DataLoader для текущей обучающей подвыборки BERT
        train_dataset_bert_subset = DummyDatasetBERT(current_train_df[text_col], current_train_df[label_col], tokenizer_bert, max_len_bert, num_classes)
        train_loader_bert_subset = DataLoader(train_dataset_bert_subset, batch_size=batch_size, shuffle=True)
        
        model_bert = створити_bert_lite_модель(num_classes, config_bert_optimal, tokenizer_bert.name_or_path)
        
        total_steps_bert = len(train_loader_bert_subset) * n_epochs_per_subset
        warmup_steps_bert = int(0.1 * total_steps_bert) # 10% warmup

        _, history_bert = run_training(model_bert, 'bert', train_loader_bert_subset, val_loader_bert,
                                       n_epochs_per_subset, device, './models_datasize', f"bert_datafrac{fraction:.2f}",
                                       lr_bert=lr_bert_optimal, 
                                       bert_total_steps=total_steps_bert, bert_warmup_steps=warmup_steps_bert)
        best_f1_bert = max(history_bert['val_f1']) if history_bert['val_f1'] else 0
        results.append({'model': 'BERT-lite', 'data_fraction': fraction, 'val_f1': best_f1_bert})
        print(f"BERT-lite на {fraction*100:.0f}% данных: Val F1 = {best_f1_bert:.4f}")

    return pd.DataFrame(results)

def plot_data_size_comparison_results(df_results):
    """
    Визуализирует результаты сравнения моделей на разных объемах данных (как Рис. 16).
    """
    plt.figure(figsize=(10, 6))
    
    for model_name in df_results['model'].unique():
        model_data = df_results[df_results['model'] == model_name]
        plt.plot(model_data['data_fraction'] * 100, model_data['val_f1'], marker='o', linestyle='-', label=model_name)
        # Добавление аннотаций F1-меры на график (как на Рис. 16)
        for i, row in model_data.iterrows():
            plt.annotate(f"{row['val_f1']:.3f}", 
                         (row['data_fraction'] * 100, row['val_f1']),
                         textcoords="offset points", xytext=(0,5), ha='center')


    plt.title('Залежність F1-міри від обсягу навчальних даних')
    plt.xlabel('Відсоток навчальних даних, %')
    plt.ylabel('F1-міра (на валідації)')
    plt.xticks([f * 100 for f in df_results['data_fraction'].unique()]) # Устанавливаем метки на оси X
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(bottom=min(0.6, df_results['val_f1'].min() - 0.05) if not df_results.empty else 0.6, 
             top=max(0.9, df_results['val_f1'].max() + 0.05) if not df_results.empty else 0.9) # Динамический Y-диапазон
    plt.show()


if __name__ == '__main__':
    DEVICE = get_device()
    print(f"Используется устройство: {DEVICE}")

    # --- Параметры для заглушек и примера ---
    NUM_CLASSES = 7
    TEXT_COL = 'text'
    LABEL_COL = 'emotion_encoded' # Предполагаем, что метки уже закодированы в числа
    
    # Создание фиктивного полного обучающего и валидационного DataFrame
    # В реальном проекте вы загрузите свои данные
    N_TOTAL_TRAIN_SAMPLES = 1000 # Уменьшено для быстрого теста
    N_VAL_SAMPLES = 200
    
    emotion_labels_numeric = np.random.randint(0, NUM_CLASSES, N_TOTAL_TRAIN_SAMPLES)
    full_train_data = pd.DataFrame({
        TEXT_COL: [f"Пример обучающего текста номер {i} для разных данных." for i in range(N_TOTAL_TRAIN_SAMPLES)],
        LABEL_COL: emotion_labels_numeric
    })
    val_data = pd.DataFrame({
        TEXT_COL: [f"Пример валидационного текста номер {i}." for i in range(N_VAL_SAMPLES)],
        LABEL_COL: np.random.randint(0, NUM_CLASSES, N_VAL_SAMPLES)
    })

    # --- LSTM: Настройка оптимальной конфигурации ---
    # (из курсовой или предыдущих экспериментов)
    # Фиктивный словарь для LSTM
    dummy_tokens_lstm = [f'word{i}' for i in range(500)] + ['<unk>', '<pad>'] # Меньше для скорости
    vocab_lstm_stub = StubVocab(dummy_tokens_lstm)
    vocab_lstm_stub.set_vectors(vocab_lstm_stub.stoi, None, 100) # Фиктивные векторы для .vectors

    PAD_IDX_LSTM = vocab_lstm_stub.pad_index
    MAX_LEN_LSTM = 50
    OPTIMAL_CONFIG_LSTM = { # стр. 26: BiLSTM, 2 слоя, hidden_dim=256
        'embedding_dim': 100, # Должно соответствовать vocab_lstm_stub.vectors.shape[1] если используются
        'hidden_dim': 128, # Уменьшено для скорости заглушки
        'n_layers': 2,
        'bidirectional': True,
        'dropout_lstm': 0.3,
        'pad_idx': PAD_IDX_LSTM
    }
    OPTIMAL_LR_LSTM = 0.001 # стр. 27

    # --- BERT: Настройка оптимальной конфигурации ---
    TOKENIZER_NAME_BERT = 'bert-base-multilingual-cased' # или ваш lite вариант
    try:
        tokenizer_bert_stub = BertTokenizer.from_pretrained(TOKENIZER_NAME_BERT)
    except OSError:
        print(f"ПРЕДУПРЕЖДЕНИЕ: Не удалось загрузить токенизатор {TOKENIZER_NAME_BERT}. Используется фиктивный.")
        # Создаем фиктивный токенизатор, если настоящий недоступен
        class DummyTokenizer:
            def __init__(self, vocab_file=None, name_or_path="dummy-tokenizer"):
                self.name_or_path = name_or_path
            def encode_plus(self, text, add_special_tokens, max_length, padding, truncation, return_tensors):
                return {'input_ids': torch.randint(0,1000,(1,max_length)), 'attention_mask': torch.ones((1,max_length))}
            @property
            def vocab_size(self): return 1000 # Фиктивный
        tokenizer_bert_stub = DummyTokenizer()


    MAX_LEN_BERT = 64
    OPTIMAL_CONFIG_BERT = { # стр. 27: 6 слоев, 8 голов, hidden_size=512
        'hidden_size': 256, # Уменьшено для скорости заглушки
        'num_hidden_layers': 3, # Уменьшено
        'num_attention_heads': 4, # Уменьшено
        'intermediate_size': 1024, # 4 * 256
        'dropout_rate_head': 0.3,
        'hidden_dropout_prob': 0.1,
        'attention_probs_dropout_prob': 0.1
    }
    OPTIMAL_LR_BERT = 2e-5 # стр. 27

    # --- Общие параметры для эксперимента ---
    BATCH_SIZE_FOR_EXP = 16 # Меньше для скорости
    N_EPOCHS_SUBSET = 1 # Очень мало, только для проверки работы скрипта
    DATA_FRACTIONS_TO_TEST = [0.25, 0.5, 1.0] # Уменьшеный набор для быстрого теста [0.1, 0.25, 0.5, 0.75, 1.0]

    # Запуск сравнения
    comparison_results_df = run_data_size_comparison(
        full_train_df_original=full_train_data,
        val_df_original=val_data,
        text_col=TEXT_COL,
        label_col=LABEL_COL,
        num_classes=NUM_CLASSES,
        device=DEVICE,
        vocab_lstm=vocab_lstm_stub,
        pad_idx_lstm=PAD_IDX_LSTM,
        max_len_lstm=MAX_LEN_LSTM,
        config_lstm_optimal=OPTIMAL_CONFIG_LSTM,
        lr_lstm_optimal=OPTIMAL_LR_LSTM,
        tokenizer_bert=tokenizer_bert_stub,
        max_len_bert=MAX_LEN_BERT,
        config_bert_optimal=OPTIMAL_CONFIG_BERT,
        lr_bert_optimal=OPTIMAL_LR_BERT,
        batch_size=BATCH_SIZE_FOR_EXP,
        n_epochs_per_subset=N_EPOCHS_SUBSET,
        data_fractions=DATA_FRACTIONS_TO_TEST
    )

    print("\n--- Результаты сравнения моделей на разных объемах данных ---")
    if not comparison_results_df.empty:
        print(comparison_results_df)
        plot_data_size_comparison_results(comparison_results_df)
    else:
        print("Нет результатов для отображения.")

    print("\nСкрипт сравнения на разных объемах данных завершен.")
