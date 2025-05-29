import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Предполагается, что файл preprocessing.py находится в той же директории
# и мы можем импортировать из него функцию розділення_даних
# Если он в другом месте, нужно будет настроить PYTHONPATH или изменить импорт
try:
    from preprocessing import розділення_даних, препроцесинг_тексту, ukrainian_stopwords
except ImportError:
    print("Не удалось импортировать функции из preprocessing.py. Убедитесь, что файл находится в той же директории или доступен для импорта.")
    # Заглушки, если импорт не удался, чтобы остальная часть файла могла быть проанализирована
    def розділення_даних(df, text_column, label_column, test_size=0.15, val_size=0.15, random_state=42):
        print("Заглушка: розділення_даних")
        return df, df, df
    def препроцесинг_тексту(текст, stopwords_list=None):
        print("Заглушка: препроцесинг_тексту")
        return текст.split()
    ukrainian_stopwords = []


def завантажити_bert_токенізатор(model_name='bert-base-multilingual-cased'):
    """
    Загружает токенизатор BERT.
    В курсовой упоминается "bert-base-multilingual-cased" (стр. 43, vocab_size=119547)
    или специализированная украинская модель.
    """
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        return tokenizer
    except Exception as e:
        print(f"Ошибка при загрузке токенизатора {model_name}: {e}")
        print("Убедитесь, что у вас есть доступ в интернет и установлена библиотека transformers.")
        print("Попробуйте: pip install transformers")
        return None

def кодувати_мітки(мітки_список: list, fit_encoder: bool = True, encoder_instance: LabelEncoder = None) -> (np.ndarray, LabelEncoder):
    """
    Кодирует текстовые метки в числовые значения.
    """
    if fit_encoder or not encoder_instance:
        encoder = LabelEncoder()
        кодовані_мітки = encoder.fit_transform(мітки_список)
    else:
        encoder = encoder_instance
        try:
            кодовані_мітки = encoder.transform(мітки_список)
        except ValueError as e:
            print(f"Ошибка при кодировании меток: {e}")
            print("Возможно, в новых данных есть метки, которых не было при обучении кодировщика.")
            # Можно добавить обработку новых меток, например, присвоить им специальный код или проигнорировать
            # Для простоты, сейчас вернем None или вызовем исключение
            raise e
    return кодовані_мітки, encoder

def підготовка_вхідних_даних_bert(тексти: list[str], токенізатор: BertTokenizer, макс_довжина: int) -> (torch.Tensor, torch.Tensor):
    """
    Кодирует список текстов с помощью BERT-токенізатора.
    Возвращает input_ids и attention_masks.
    """
    input_ids_list = []
    attention_masks_list = []

    for текст in тексти:
        # Токенизация текста для BERT
        encoded_dict = токенізатор.encode_plus(
                            текст,
                            add_special_tokens=True, # Добавить '[CLS]' и '[SEP]'
                            max_length=макс_довжина,
                            padding='max_length',   # Дополнить до max_length
                            truncation=True,        # Обрезать до max_length
                            return_attention_mask=True,
                            return_tensors='pt',    # Вернуть PyTorch тензоры
                       )
        input_ids_list.append(encoded_dict['input_ids'])
        attention_masks_list.append(encoded_dict['attention_mask'])

    # Объединение списков тензоров в один тензор
    input_ids = torch.cat(input_ids_list, dim=0)
    attention_masks = torch.cat(attention_masks_list, dim=0)

    return input_ids, attention_masks


def створити_dataloaders_bert(input_ids: torch.Tensor, attention_masks: torch.Tensor, labels: torch.Tensor, розмір_батчу: int, is_train: bool = True) -> DataLoader:
    """
    Создает DataLoader для BERT модели.
    """
    dataset = TensorDataset(input_ids, attention_masks, labels)
    if is_train:
        # Для обучающего набора данных используется RandomSampler
        sampler = RandomSampler(dataset)
    else:
        # Для валидационного и тестового наборов данных используется SequentialSampler
        sampler = SequentialSampler(dataset)
    
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=розмір_батчу)
    return dataloader


def підготувати_дані_для_bert_з_df(df: pd.DataFrame,
                                 text_column: str,
                                 label_column: str,
                                 токенізатор: BertTokenizer,
                                 макс_довжина: int,
                                 розмір_батчу_train: int,
                                 розмір_батчу_val_test: int,
                                 label_encoder: LabelEncoder = None,
                                 do_split: bool = True,
                                 test_size_split: float = 0.15,
                                 val_size_split: float = 0.15,
                                 random_state_split: int = 42
                                 ) -> dict:
    """
    Полный конвейер подготовки данных из DataFrame для BERT:
    1. Разделение данных (опционально)
    2. Кодирование меток
    3. Токенизация текстов
    4. Создание DataLoaders
    """
    
    all_data_loaders = {}
    
    if do_split:
        train_df, val_df, test_df = розділення_даних(
            df, text_column, label_column, 
            test_size=test_size_split, val_size=val_size_split, random_state=random_state_split
        )
        
        datasets = {'train': train_df, 'val': val_df, 'test': test_df}
    else:
        # Если данные уже разделены, ожидаем DataFrame или словарь DataFrame'ов
        if isinstance(df, pd.DataFrame): # Предполагаем, что это один набор (например, только для предсказания)
             datasets = {'predict': df}
        elif isinstance(df, dict): # Ожидаем словарь типа {'train': df_train, 'val': df_val}
            datasets = df
        else:
            raise ValueError("Неверный формат df, если do_split=False. Ожидается DataFrame или dict of DataFrames.")

    fitted_label_encoder = label_encoder

    for name, current_df in datasets.items():
        print(f"Обработка набора данных: {name}")
        
        тексти = current_df[text_column].tolist()
        
        # Кодирование меток только если они есть (например, для 'predict' их может не быть)
        if label_column in current_df.columns:
            мітки = current_df[label_column].tolist()
            if name == 'train' and not fitted_label_encoder : # Обучаем кодировщик только на обучающих данных
                кодовані_мітки_np, le = кодувати_мітки(мітки, fit_encoder=True)
                fitted_label_encoder = le # Сохраняем обученный кодировщик
            elif fitted_label_encoder:
                 кодовані_мітки_np, _ = кодувати_мітки(мітки, fit_encoder=False, encoder_instance=fitted_label_encoder)
            else: # Случай, когда нет обучающего набора, но есть метки и нет кодировщика
                print(f"Предупреждение: Нет обученного LabelEncoder и это не 'train' набор. Метки для '{name}' будут обучены отдельно.")
                кодовані_мітки_np, le = кодувати_мітки(мітки, fit_encoder=True)
                # В этом случае, если есть другие наборы, их кодировщики могут не совпадать.
                # Лучше передавать один обученный LabelEncoder.
            
            labels_tensor = torch.tensor(кодовані_мітки_np, dtype=torch.long)
        else: # Если меток нет
            labels_tensor = torch.zeros(len(тексти), dtype=torch.long) # Заглушка для DataLoader
            print(f"В наборе '{name}' отсутствует колонка с метками '{label_column}'. Созданы фиктивные метки.")

        input_ids, attention_masks = підготовка_вхідних_даних_bert(тексти, токенізатор, макс_довжина)
        
        current_batch_size = розмір_батчу_train if name == 'train' else розмір_батчу_val_test
        dataloader = створити_dataloaders_bert(input_ids, attention_masks, labels_tensor, current_batch_size, is_train=(name == 'train'))
        all_data_loaders[name] = dataloader

    return {'dataloaders': all_data_loaders, 'label_encoder': fitted_label_encoder}


if __name__ == '__main__':
    # Загрузка токенизатора
    tokenizer_name = 'bert-base-multilingual-cased' # Как в курсовой
    bert_tokenizer = завантажити_bert_токенізатор(tokenizer_name)

    if bert_tokenizer:
        # Пример использования с DataFrame
        # Создадим пример DataFrame (аналогично preprocessing.py)
        data = {
            'text': [
                "Це позитивний відгук, все чудово!", "Дуже сподобалось, рекомендую.", "Жахливий сервіс, нікому не раджу.",
                "Нейтральний коментар про погоду.", "Я в захваті від цієї книги!", "Розчарований якістю товару.",
                "Фільм просто супер, емоції переповнюють.", "Звичайний день, нічого особливого.", "Це було страшно, але захоплююче.",
                "Відчуваю сум через цю новину.", "Я злий на таку несправедливість.", "Дивно, але факт."
            ] * 10,
            'emotion': [
                "радість", "радість", "гнів", "нейтральний", "радість", "смуток",
                "радість", "нейтральний", "страх", "смуток", "гнів", "здивування"
            ] * 10
        }
        sample_df = pd.DataFrame(data)
        print(f"Исходный DataFrame (первые 5 строк):\n{sample_df.head()}")

        МАКС_ДОВЖИНА_ПОСЛІДОВНОСТІ = 128 # Из курсовой (стр. 16 и 17)
        РОЗМІР_БАТЧУ_TRAIN = 32 # Из курсовой (стр. 18)
        РОЗМІР_БАТЧУ_VAL_TEST = 32 # Можно сделать другим, например 64

        # Полный конвейер
        try:
            prepared_data = підготувати_дані_для_bert_з_df(
                df=sample_df,
                text_column='text',
                label_column='emotion',
                токенізатор=bert_tokenizer,
                макс_довжина=МАКС_ДОВЖИНА_ПОСЛІДОВНОСТІ,
                розмір_батчу_train=РОЗМІР_БАТЧУ_TRAIN,
                розмір_батчу_val_test=РОЗМІР_БАТЧУ_VAL_TEST,
                do_split=True
            )

            train_dataloader = prepared_data['dataloaders'].get('train')
            val_dataloader = prepared_data['dataloaders'].get('val')
            test_dataloader = prepared_data['dataloaders'].get('test')
            label_enc = prepared_data['label_encoder']

            if train_dataloader:
                print(f"\nУспешно создан train_dataloader. Количество батчей: {len(train_dataloader)}")
                # Посмотрим на один батч
                for batch in train_dataloader:
                    b_input_ids, b_attention_mask, b_labels = batch
                    print(f"Размер input_ids в батче: {b_input_ids.shape}") # (batch_size, max_length)
                    print(f"Размер attention_mask в батче: {b_attention_mask.shape}") # (batch_size, max_length)
                    print(f"Размер labels в батче: {b_labels.shape}") # (batch_size)
                    if label_enc:
                         print(f"Пример меток (числовые): {b_labels[:5].tolist()}")
                         print(f"Пример меток (текстовые): {label_enc.inverse_transform(b_labels[:5].tolist())}")
                    break 
            if val_dataloader:
                 print(f"Успешно создан val_dataloader. Количество батчей: {len(val_dataloader)}")
            if test_dataloader:
                 print(f"Успешно создан test_dataloader. Количество батчей: {len(test_dataloader)}")
            if label_enc:
                print(f"Классы LabelEncoder: {label_enc.classes_}")


        except Exception as e:
            print(f"Ошибка при подготовке данных для BERT: {e}")
            import traceback
            traceback.print_exc()

