import torch
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np

# Предполагается, что модели и функции подготовки данных доступны для импорта
# Модели:
# from lstm_model import LSTMClassifier # или ваша функция для создания/загрузки
# from bert_lite_model import BertLiteClassifier # или ваша функция для создания/загрузки

# Функции препроцессинга:
# Из preprocessing.py для LSTM
# from preprocessing import препроцесинг_тексту as препроцесинг_тексту_lstm_базовий

# Из bert_data_preparation.py для BERT (или его части)
# from bert_data_preparation import підготовка_вхідних_даних_bert (может быть слишком специфично для DataLoader)

# Для LSTM нужен словарь (vocab) и обработка длин.
# Для BERT нужен токенизатор.

def get_device():
    """Возвращает устройство (GPU, если доступен, иначе CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def препроцесинг_тексту_для_lstm(текст: str, vocab, max_seq_len: int, препроцесор_базовий, stopwords_list=None):
    """
    Препроцессинг одного текста для LSTM модели.
    Возвращает тензор индексов и тензор длины.
    """
    # 1. Базовый препроцессинг (нормализация, токенизация, удаление стоп-слов)
    #    Функция препроцесор_базовий должна возвращать список токенов.
    токени = препроцесор_базовий(текст, stopwords_list=stopwords_list) # Используем импортированную функцию

    # 2. Преобразование токенов в индексы согласно словарю (vocab)
    #    Vocab должен иметь метод stoi (string-to-index) и unk_index
    indexed_tokens = [vocab.get(token, vocab.get(vocab.get('unk_token', '<unk>'))) for token in токени] # Используем get с default
    
    # 3. Обрезка/дополнение до max_seq_len
    if len(indexed_tokens) > max_seq_len:
        indexed_tokens = indexed_tokens[:max_seq_len]
        actual_length = max_seq_len
    else:
        actual_length = len(indexed_tokens)
        # Дополняем padding-токенами (pad_idx из модели LSTM)
        # Предположим, pad_token в словаре соответствует pad_idx модели (например, 1)
        pad_token_str = vocab.get('pad_token', '<pad>')
        pad_idx = vocab.get(pad_token_str, 1) # Получаем индекс паддинг токена
        indexed_tokens.extend([pad_idx] * (max_seq_len - len(indexed_tokens)))

    # 4. Преобразование в тензоры PyTorch
    tensor_indices = torch.LongTensor(indexed_tokens).unsqueeze(0) # [1, max_seq_len] (для batch_size=1)
    tensor_length = torch.LongTensor([actual_length]) # [1]

    return tensor_indices, tensor_length

def препроцесинг_тексту_для_bert(текст: str, токенізатор: BertTokenizer, макс_довжина: int):
    """
    Препроцессинг одного текста для BERT модели.
    Возвращает input_ids и attention_mask.
    """
    encoded_dict = токенізатор.encode_plus(
                        текст,
                        add_special_tokens=True,
                        max_length=макс_довжина,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt',
                   )
    return encoded_dict['input_ids'], encoded_dict['attention_mask']


def передбачити_емоцію(текст: str, model, model_type: str, device,
                        # Параметры для LSTM
                        vocab_lstm=None, max_len_lstm=None, препроцесор_базовий_lstm=None, stopwords_lstm=None,
                        # Параметры для BERT
                        токенізатор_bert=None, max_len_bert=None,
                        # Общие
                        label_encoder=None): # LabelEncoder для преобразования индекса в название класса
    """
    Предсказывает эмоцию для одного текста с использованием указанной модели.

    Args:
        текст (str): Входной текст для классификации.
        model (torch.nn.Module): Обученная модель (LSTM или BERT-lite).
        model_type (str): Тип модели ('lstm' или 'bert').
        device (torch.device): Устройство для вычислений.
        vocab_lstm (dict or torchtext.vocab.Vocab): Словарь для LSTM.
        max_len_lstm (int): Максимальная длина последовательности для LSTM.
        препроцесор_базовий_lstm (function): Функция базового препроцессинга для LSTM.
        stopwords_lstm (list): Список стоп-слов для LSTM.
        токенізатор_bert (BertTokenizer): Токенизатор для BERT.
        max_len_bert (int): Максимальная длина последовательности для BERT.
        label_encoder (sklearn.preprocessing.LabelEncoder): Обученный кодировщик меток.

    Returns:
        tuple: (предсказанная_эмоция_str, вероятность_float)
               или (None, None) в случае ошибки.
    """
    model.eval() # Перевод модели в режим оценки
    model.to(device)

    with torch.no_grad():
        if model_type == 'lstm':
            if not all([vocab_lstm, max_len_lstm, препроцесор_базовий_lstm]):
                print("Ошибка: Для LSTM модели не предоставлены vocab, max_len или препроцесор_базовий.")
                return None, None
            
            try:
                # Препроцессинг для LSTM
                tensor_indices, tensor_length = препроцесинг_тексту_для_lstm(
                    текст, vocab_lstm, max_len_lstm, препроцесор_базовий_lstm, stopwords_lstm
                )
                tensor_indices = tensor_indices.to(device)
                # tensor_length остается на CPU для pack_padded_sequence в модели LSTM

                # Предсказание
                logits = model(tensor_indices, tensor_length)
            except Exception as e:
                print(f"Ошибка при препроцессинге или предсказании LSTM: {e}")
                import traceback
                traceback.print_exc()
                return None, None

        elif model_type == 'bert':
            if not all([токенізатор_bert, max_len_bert]):
                print("Ошибка: Для BERT модели не предоставлены токенізатор или max_len.")
                return None, None
            
            try:
                # Препроцессинг для BERT
                input_ids, attention_mask = препроцесинг_тексту_для_bert(
                    текст, токенізатор_bert, max_len_bert
                )
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                # token_type_ids обычно не нужны для классификации одного предложения

                # Предсказание
                logits = model(input_ids, attention_mask) # token_type_ids=None
            except Exception as e:
                print(f"Ошибка при препроцессинге или предсказании BERT: {e}")
                import traceback
                traceback.print_exc()
                return None, None
        else:
            print(f"Ошибка: Неизвестный тип модели '{model_type}'. Допустимы 'lstm' или 'bert'.")
            return None, None

        # Получение вероятностей и предсказанного класса
        probabilities = F.softmax(logits, dim=1)
        predicted_idx_tensor = torch.argmax(probabilities, dim=1)
        predicted_idx = predicted_idx_tensor.item() # Получаем индекс как Python int
        
        predicted_probability = probabilities[0, predicted_idx].item() # Вероятность предсказанного класса

        if label_encoder:
            try:
                predicted_emotion_str = label_encoder.inverse_transform([predicted_idx])[0]
            except IndexError:
                 print(f"Ошибка: Индекс {predicted_idx} вне диапазона для LabelEncoder. Классы: {label_encoder.classes_}")
                 predicted_emotion_str = f"Класс_{predicted_idx}" # Запасной вариант
            except Exception as e:
                print(f"Ошибка при декодировании метки: {e}")
                predicted_emotion_str = f"Класс_{predicted_idx}"
        else:
            predicted_emotion_str = f"Класс_{predicted_idx}" # Если нет LabelEncoder

        return predicted_emotion_str, predicted_probability


if __name__ == '__main__':
    print("Файл emotion_classifier.py загружен.")
    # Для запуска этого блока нужно:
    # 1. Обученные модели (LSTM и BERT-lite) и их сохраненные веса.
    # 2. Соответствующие словари/токенізаторы и LabelEncoder.
    # 3. Функции для создания моделей (створити_lstm_модель, створити_bert_lite_модель).
    # 4. Функция базового препроцессинга для LSTM (например, из preprocessing.py).

    # --- Настройка для примера (замените на ваши реальные компоненты) ---
    device = get_device()
    print(f"Используется устройство: {device}")

    # --- Общие компоненты ---
    from sklearn.preprocessing import LabelEncoder
    emotion_classes = ["радість", "смуток", "гнів", "страх", "відраза", "здивування", "нейтральний"]
    le = LabelEncoder()
    le.fit(emotion_classes)
    NUM_CLASSES = len(emotion_classes)

    # --- Пример для LSTM ---
    # try:
    #     print("\n--- Пример предсказания с LSTM моделью (фиктивные компоненты) ---")
    #     from lstm_model import створити_lstm_модель # Предполагается, что файл доступен
    #     from preprocessing import препроцесинг_тексту as препроцесинг_тексту_lstm_базовий, ukrainian_stopwords
          # Создание фиктивного словаря для LSTM
    #     фиктивний_словник_lstm = {word: i for i, word in enumerate(['<unk>', '<pad>', 'це', 'тестовий', 'текст', 'для', 'lstm', 'приклад', 'дуже', 'добре'] + ukrainian_stopwords)}
    #     фиктивний_словник_lstm['unk_token'] = '<unk>' # Для функции препроцессинга
    #     фиктивний_словник_lstm['pad_token'] = '<pad>' # Для функции препроцессинга
    #     VOCAB_SIZE_LSTM = len(фиктивний_словник_lstm)
    #     PAD_IDX_LSTM = фиктивний_словник_lstm['<pad>']
    #     MAX_LEN_LSTM = 20

    #     config_lstm = {
    #         'embedding_dim': 100, 'hidden_dim': 128, 'n_layers': 1, 
    #         'bidirectional': True, 'dropout_lstm': 0.2, 'pad_idx': PAD_IDX_LSTM
    #     }
    #     # model_lstm = створити_lstm_модель(VOCAB_SIZE_LSTM, NUM_CLASSES, config_lstm).to(device)
    #     # model_lstm.eval() # Убедимся, что модель в режиме eval

    #     # текст_для_lstm = "Це тестовий текст для LSTM, дуже добре."
    #     # predicted_emotion_lstm, prob_lstm = передбачити_емоцію(
    #     #     текст_для_lstm, model_lstm, 'lstm', device,
    #     #     vocab_lstm=фиктивний_словник_lstm, 
    #     #     max_len_lstm=MAX_LEN_LSTM,
    #     #     препроцесор_базовий_lstm=препроцесинг_тексту_lstm_базовий,
    #     #     stopwords_lstm=ukrainian_stopwords,
    #     #     label_encoder=le
    #     # )
    #     # if predicted_emotion_lstm:
    #     #     print(f"LSTM предсказание для '{текст_для_lstm}': {predicted_emotion_lstm} (Вероятность: {prob_lstm:.4f})")

    # except ImportError as e:
    #     print(f"Пропуск примера LSTM: не удалось импортировать зависимости ({e})")
    # except Exception as e:
    #     print(f"Ошибка в примере LSTM: {e}")
    #     import traceback
    #     traceback.print_exc()


    # --- Пример для BERT-lite ---
    # try:
    #     print("\n--- Пример предсказания с BERT-lite моделью (фиктивные компоненты) ---")
    #     from bert_lite_model import створити_bert_lite_модель # Предполагается, что файл доступен
    #     TOKENIZER_NAME_BERT = 'bert-base-multilingual-cased' # Как в курсовой
    #     bert_tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME_BERT)
    #     MAX_LEN_BERT = 30 # Макс. длина для BERT

    #     config_bert_lite = {
    #         'hidden_size': 256, 'num_hidden_layers': 2, 'num_attention_heads': 4,
    #         'intermediate_size': 1024, 'dropout_rate_head': 0.1
    #     }
    #     # model_bert = створити_bert_lite_модель(NUM_CLASSES, config_bert_lite, TOKENIZER_NAME_BERT).to(device)
    #     # model_bert.eval()

    #     # текст_для_bert = "Це текст для BERT, він повинен працювати!"
    #     # predicted_emotion_bert, prob_bert = передбачити_емоцію(
    #     #     текст_для_bert, model_bert, 'bert', device,
    #     #     токенізатор_bert=bert_tokenizer,
    #     #     max_len_bert=MAX_LEN_BERT,
    #     #     label_encoder=le
    #     # )
    #     # if predicted_emotion_bert:
    #     #     print(f"BERT предсказание для '{текст_для_bert}': {predicted_emotion_bert} (Вероятность: {prob_bert:.4f})")
    # except ImportError as e:
    #     print(f"Пропуск примера BERT: не удалось импортировать зависимости ({e})")
    # except Exception as e:
    #     print(f"Ошибка в примере BERT: {e}")
    #     import traceback
    #     traceback.print_exc()
    pass
