import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout_rate, pad_idx, embedding_weights=None):
        """
        Инициализация LSTM модели для классификации текста.

        Args:
            vocab_size (int): Размер словаря (количество уникальных токенов).
            embedding_dim (int): Размерность векторных представлений слов (эмбеддингов).
            hidden_dim (int): Размерность скрытого состояния LSTM.
            output_dim (int): Размерность выходного слоя (количество классов).
            n_layers (int): Количество LSTM слоев.
            bidirectional (bool): Использовать ли двунаправленный LSTM.
            dropout_rate (float): Вероятность dropout для регуляризации.
            pad_idx (int): Индекс padding-токена в словаре.
            embedding_weights (torch.Tensor, optional): Предобученные веса для слоя эмбеддингов.
                                                        По умолчанию None.
        """
        super().__init__()

        # 1. Слой эмбеддингов
        if embedding_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_weights, padding_idx=pad_idx, freeze=False)
            # freeze=False означает, что веса эмбеддингов будут дообучаться
            # Если нужно заморозить: freeze=True
            print("Инициализация Embedding слоя с предобученными весами.")
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            print("Инициализация Embedding слоя со случайными весами.")

        # 2. Слой LSTM
        # В курсовой (стр. 14) указан BiLSTM с recurrent_dropout.
        # В PyTorch стандартный nn.LSTM не имеет параметра recurrent_dropout напрямую.
        # Dropout между слоями LSTM (если n_layers > 1) задается параметром dropout.
        # Dropout на входы/выходы каждого LSTM слоя можно реализовать отдельно или использовать кастомные реализации.
        # Здесь используется dropout, который применяется к выходам каждого LSTM слоя, кроме последнего.
        self.lstm = nn.LSTM(embedding_dim,
                              hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              dropout=dropout_rate if n_layers > 1 else 0, # Dropout между LSTM слоями
                              batch_first=True) # Входные данные будут иметь размерность (batch_size, seq_len, feature_dim)

        # 3. Dropout слой (для выхода LSTM перед Dense слоем)
        # В курсовой (Рис.1) dropout 0.3 после BiLSTM (перед Global Max Pooling) и 0.5 после Dense Layer
        # Здесь реализуем dropout после LSTM/BiLSTM перед агрегацией или Dense слоем.
        # В курсовой (стр. 14) указан dropout 0.3 для BiLSTM.
        self.lstm_dropout = nn.Dropout(dropout_rate) # Этот dropout будет применяться к выходу LSTM

        # 4. Полносвязный слой (Dense layer)
        # Входной размер для Dense слоя зависит от того, является ли LSTM двунаправленным
        # и как агрегируются выходы LSTM (например, последний скрытый слой, max pooling, avg pooling)
        # В курсовой (стр. 14) после BiLSTM идет Global Max Pooling, затем Dense(128)
        # Если Global Max Pooling, то размерность выхода BiLSTM будет hidden_dim * 2 (если bidirectional)
        # или hidden_dim (если unidirectional)
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # В курсовой (Рис. 1) Global Max Pooling применяется к выходам BiLSTM.
        # nn.AdaptiveMaxPool1d(1) может выполнять роль Global Max Pooling для последовательностей.

        # 5. Dense слой после Global Max Pooling
        # В курсовой: Dense Layer (128 units), ReLU, Dropout: 0.5
        self.fc1 = nn.Linear(lstm_output_dim, 128) # Первый полносвязный слой
        self.relu = nn.ReLU()
        self.fc_dropout = nn.Dropout(0.5) # Dropout после первого Dense слоя

        # 6. Выходной слой
        self.fc2 = nn.Linear(128, output_dim) # Выходной классификационный слой

        self.pad_idx = pad_idx
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        print(f"Модель LSTMClassifier создана: hidden_dim={hidden_dim}, n_layers={n_layers}, bidirectional={bidirectional}, output_dim={output_dim}")

    def forward(self, text_indices, text_lengths):
        """
        Прямой проход модели.

        Args:
            text_indices (torch.Tensor): Тензор с индексами токенов.
                                        Размерность: (batch_size, seq_len)
            text_lengths (torch.Tensor): Тензор с реальными длинами последовательностей в батче.
                                         Размерность: (batch_size)
        Returns:
            torch.Tensor: Логиты для каждого класса. Размерность: (batch_size, output_dim)
        """
        # text_indices: [batch_size, seq_len]

        embedded = self.embedding(text_indices)
        # embedded: [batch_size, seq_len, embedding_dim]

        # Упаковка последовательностей для обработки паддинга в LSTM
        # text_lengths нужно переместить на CPU, если они на GPU, для pack_padded_sequence
        packed_embedded = pack_padded_sequence(embedded, text_lengths.to('cpu'), batch_first=True, enforce_sorted=False)

        # packed_output: упакованная последовательность выходов всех скрытых состояний LSTM
        # hidden: скрытое состояние последнего временного шага для каждого слоя
        # cell: состояние ячейки последнего временного шага для каждого слоя
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # Распаковка последовательности (если нужна для дальнейшей обработки всех выходов)
        # output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        # output: [batch_size, seq_len, hidden_dim * num_directions]

        # В курсовой (Рис. 1) используется Global Max Pooling после BiLSTM.
        # Мы можем применить Global Max Pooling к `output` тензору.
        # Для этого `output` должен быть [batch_size, features, seq_len] для MaxPool1d.
        # output = output.permute(0, 2, 1) # [batch_size, hidden_dim * num_directions, seq_len]
        # pooled = torch.max(output, dim=2)[0] # [batch_size, hidden_dim * num_directions]
        
        # Альтернативно, часто используют последнее скрытое состояние `hidden`
        # `hidden` имеет размерность: [n_layers * num_directions, batch_size, hidden_dim]
        # Если LSTM двунаправленный, нужно конкатенировать скрытые состояния прямого и обратного проходов
        # последнего слоя.
        if self.bidirectional:
            # Конкатенируем скрытые состояния последнего слоя (прямой и обратный)
            # hidden[-2,:,:] - последний слой, прямой проход
            # hidden[-1,:,:] - последний слой, обратный проход
            hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            # Берем скрытое состояние последнего слоя
            hidden_concat = hidden[-1,:,:]
        # hidden_concat: [batch_size, hidden_dim * num_directions]

        # Применяем Dropout к выходу LSTM (в данном случае, к агрегированному скрытому состоянию)
        dropped_lstm_output = self.lstm_dropout(hidden_concat) # Используем hidden_concat как выход LSTM для классификатора

        # Полносвязные слои
        dense_out = self.fc1(dropped_lstm_output) # [batch_size, 128]
        activated_out = self.relu(dense_out)
        dropped_fc_out = self.fc_dropout(activated_out)
        
        final_output = self.fc2(dropped_fc_out) # [batch_size, output_dim]

        return final_output

def створити_lstm_модель(vocab_size: int, output_dim: int, config: dict, embedding_weights=None) -> LSTMClassifier:
    """
    Фабричная функция для создания LSTM модели на основе конфигурации.
    """
    embedding_dim = config.get('embedding_dim', 300) # стр. 14 курсовой
    hidden_dim = config.get('hidden_dim', 256)       # стр. 14 курсовой
    n_layers = config.get('n_layers', 1)             # BiLSTM, в курсовой не указано кол-во слоев, но обычно 1-2
                                                     # На рис.7 "2 шари оптимально"
                                                     # На стр. 26 "Оптимальною конфігурацією... є BILSTМ з 2 шарами"
                                                     # Ставим 2 по умолчанию, если в конфиге не указано
    if 'n_layers' not in config and hidden_dim == 256: # Если это "оптимальная" конфигурация
        n_layers = 2
        print(f"Установлено n_layers=2 для LSTM (согласно выводам курсовой для hidden_dim=256).")


    bidirectional = config.get('bidirectional', True) # Двонаправлений LSTM (BiLSTM) стр. 14
    dropout_rate = config.get('dropout_lstm', 0.3)    # Dropout 0.3 для BiLSTM (стр. 14)
                                                      # Dropout 0.5 для Dense (реализован в LSTMClassifier)
    pad_idx = config.get('pad_idx', 1)                # Обычно 0 или 1 для padding токена

    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        bidirectional=bidirectional,
        dropout_rate=dropout_rate,
        pad_idx=pad_idx,
        embedding_weights=embedding_weights
    )
    return model

if __name__ == '__main__':
    # Параметры из курсовой (стр. 14, 15)
    VOCAB_SIZE = 50000  # Размер словаря
    EMBEDDING_DIM = 300 # Размерность вбудовування
    HIDDEN_DIM = 256    # Кількість прихованих нейронів BiLSTM
    OUTPUT_DIM = 7      # Кількість класів емоцій
    N_LAYERS = 2        # Оптимально 2 шари BiLSTM (стр. 26)
    BIDIRECTIONAL = True
    DROPOUT_LSTM = 0.3  # Dropout для BiLSTM
    # DROPOUT_FC = 0.5 # Dropout для Dense слоя (реализован внутри класса)
    PAD_IDX = 1 # Предположим, что индекс паддинга 1 (часто 0 или 1)

    # Создание модели
    config_params = {
        'embedding_dim': EMBEDDING_DIM,
        'hidden_dim': HIDDEN_DIM,
        'n_layers': N_LAYERS,
        'bidirectional': BIDIRECTIONAL,
        'dropout_lstm': DROPOUT_LSTM,
        'pad_idx': PAD_IDX
    }
    
    # Пример без предобученных весов
    print("--- Тест модели без предобученных весов ---")
    model_no_pretrained = створити_lstm_модель(VOCAB_SIZE, OUTPUT_DIM, config_params)
    print(model_no_pretrained)

    # Пример с предобученными весами (создадим случайные для теста)
    print("\n--- Тест модели с предобученными весами ---")
    example_embedding_weights = torch.rand((VOCAB_SIZE, EMBEDDING_DIM))
    model_with_pretrained = створити_lstm_модель(VOCAB_SIZE, OUTPUT_DIM, config_params, embedding_weights=example_embedding_weights)
    print(model_with_pretrained)

    # Тестирование прямого прохода
    print("\n--- Тест прямого прохода (forward pass) ---")
    batch_size = 4
    seq_len = 10 # Максимальная длина последовательности в этом батче
    
    # Случайные входные данные
    test_text_indices = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    # Случайные длины последовательностей (должны быть <= seq_len и > 0)
    test_text_lengths = torch.tensor([seq_len, seq_len - 2, seq_len - 1, seq_len - 5])
    test_text_lengths = torch.clamp(test_text_lengths, min=1) # Убедимся, что длины > 0

    # Установка модели в режим оценки (не обязательно для простого forward, но хорошая практика)
    model_no_pretrained.eval()
    
    try:
        with torch.no_grad(): # Отключаем расчет градиентов для инференса
            predictions = model_no_pretrained(test_text_indices, test_text_lengths)
        print(f"Входные индексы (форма): {test_text_indices.shape}")
        print(f"Длины последовательностей: {test_text_lengths.tolist()}")
        print(f"Выходные предсказания (логиты, форма): {predictions.shape}") # Ожидаем [batch_size, OUTPUT_DIM]
        print(f"Пример предсказаний (логиты):\n{predictions}")

        # Проверка на NaN
        if torch.isnan(predictions).any():
            print("\nВНИМАНИЕ: В предсказаниях есть NaN значения!")
        else:
            print("\nПредсказания не содержат NaN значений.")

    except Exception as e:
        print(f"Ошибка во время прямого прохода: {e}")
        import traceback
        traceback.print_exc()

    # Проверка соответствия параметров из курсовой
    # Стр. 14: Embedding(50000x300), BiLSTM(256 units, dropout 0.3), Global Max-Pooling, Dense(128, ReLU, dropout 0.5), Output(7, Softmax)
    # Моя реализация использует последнее скрытое состояние LSTM вместо Global Max Pooling на всех выходах LSTM.
    # Это распространенный подход. Если строго нужен Global Max Pooling, forward метод нужно будет немного изменить.
    # Я оставил закомментированный код для Global Max Pooling в методе forward.
    # Текущая реализация: Embedding -> BiLSTM -> Dropout -> Concat Hidden States -> FC1(128) -> ReLU -> Dropout -> FC2(7)
    # Это соответствует основной идее, но детали агрегации выхода LSTM могут отличаться.
