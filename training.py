import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup # For BERT scheduler

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import copy
import os

# Предполагается, что модели и функции подготовки данных доступны для импорта
# from lstm_model import LSTMClassifier, створити_lstm_модель
# from bert_lite_model import BertLiteClassifier, створити_bert_lite_модель
# from preprocessing import розділення_даних # Если нужно для примера
# from bert_data_preparation import підготувати_дані_для_bert_з_df, завантажити_bert_токенізатор # Если нужно для примера

def get_device():
    """Возвращает устройство (GPU, если доступен, иначе CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    # elif torch.backends.mps.is_available(): # Для M1/M2 Mac
    #     return torch.device('mps')
    else:
        return torch.device('cpu')

def save_model(model, path, model_name):
    """Сохраняет состояние модели."""
    if not os.path.exists(path):
        os.makedirs(path)
    full_path = os.path.join(path, f"{model_name}_best_model.pt")
    torch.save(model.state_dict(), full_path)
    print(f"Модель сохранена в {full_path}")

def load_model(model, path, model_name, device):
    """Загружает состояние модели."""
    full_path = os.path.join(path, f"{model_name}_best_model.pt")
    if os.path.exists(full_path):
        model.load_state_dict(torch.load(full_path, map_location=device))
        model.to(device)
        model.eval() # Перевод модели в режим оценки
        print(f"Модель загружена из {full_path}")
        return model
    else:
        print(f"Файл модели {full_path} не найден.")
        return None


def calculate_metrics(preds_flat, labels_flat, average_method='weighted'):
    """Рассчитывает точность и F1-меру."""
    acc = accuracy_score(labels_flat, preds_flat)
    f1 = f1_score(labels_flat, preds_flat, average=average_method, zero_division=0)
    # zero_division=0: если для какого-то класса нет предсказаний или истинных меток, F1 для него будет 0.
    return acc, f1

def train_epoch_lstm(model, iterator, optimizer, criterion, device):
    """Один проход обучения (эпоха) для LSTM модели."""
    model.train()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    for batch_idx, batch in enumerate(iterator):
        # В LSTM модели ожидается text_indices и text_lengths
        # Убедитесь, что ваш DataLoader для LSTM возвращает их в таком виде
        # и что batch.label содержит метки.
        # Пример структуры батча для torchtext Field:
        # text, text_lengths = batch.text 
        # labels = batch.label
        
        # Адаптируйте эту часть под вашу структуру данных из DataLoader'а
        # Для примера, если DataLoader возвращает кортеж (text_indices, text_lengths, labels)
        if len(batch) == 3: # Предполагаем (text_indices, text_lengths, labels)
            text_indices, text_lengths, labels = batch
        elif hasattr(batch, 'text') and hasattr(batch, 'label'): # Для torchtext Field
             text_data = batch.text
             if isinstance(text_data, tuple) and len(text_data) == 2: # (indices, lengths)
                 text_indices, text_lengths = text_data
             else: # Только indices, длины нужно будет вычислить или передать отдельно
                 text_indices = text_data
                 # Если text_lengths не передаются, LSTM может работать некорректно с padding
                 # Здесь предполагаем, что text_lengths передаются
                 raise ValueError("text_lengths не найдены в батче для LSTM. Убедитесь, что DataLoader их предоставляет.")
             labels = batch.label
        else:
            raise ValueError("Неожиданная структура батча для LSTM. Ожидается (text_indices, text_lengths, labels) или объект с атрибутами .text и .label.")

        text_indices = text_indices.to(device)
        # text_lengths остаются на CPU для pack_padded_sequence
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(text_indices, text_lengths) # predictions: [batch_size, output_dim]
        
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
        preds_classes = torch.argmax(predictions, dim=1)
        all_preds.extend(preds_classes.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_epoch_loss = epoch_loss / len(iterator)
    accuracy, f1 = calculate_metrics(np.array(all_preds), np.array(all_labels))
    return avg_epoch_loss, accuracy, f1

def evaluate_lstm(model, iterator, criterion, device):
    """Один проход оценки для LSTM модели."""
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            # Аналогично train_epoch_lstm, адаптируйте под структуру вашего батча
            if len(batch) == 3:
                text_indices, text_lengths, labels = batch
            elif hasattr(batch, 'text') and hasattr(batch, 'label'):
                 text_data = batch.text
                 if isinstance(text_data, tuple) and len(text_data) == 2:
                     text_indices, text_lengths = text_data
                 else:
                     text_indices = text_data
                     raise ValueError("text_lengths не найдены в батче для LSTM (eval).")
                 labels = batch.label
            else:
                raise ValueError("Неожиданная структура батча для LSTM (eval).")

            text_indices = text_indices.to(device)
            labels = labels.to(device)

            predictions = model(text_indices, text_lengths)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()

            preds_classes = torch.argmax(predictions, dim=1)
            all_preds.extend(preds_classes.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_epoch_loss = epoch_loss / len(iterator)
    accuracy, f1 = calculate_metrics(np.array(all_preds), np.array(all_labels))
    return avg_epoch_loss, accuracy, f1


def train_epoch_bert(model, iterator, optimizer, criterion, device, scheduler=None):
    """Один проход обучения (эпоха) для BERT модели."""
    model.train()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    for batch_idx, batch in enumerate(iterator):
        # DataLoader для BERT обычно возвращает input_ids, attention_mask, labels
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        token_type_ids = batch[3].to(device) if len(batch) > 3 else None # Опционально

        optimizer.zero_grad()
        
        if token_type_ids is not None:
            predictions = model(input_ids, attention_mask, token_type_ids)
        else:
            predictions = model(input_ids, attention_mask)
        
        loss = criterion(predictions, labels)
        loss.backward()
        
        # Gradient clipping (опционально, но часто используется для BERT)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step() # Для get_linear_schedule_with_warmup

        epoch_loss += loss.item()
        
        preds_classes = torch.argmax(predictions, dim=1)
        all_preds.extend(preds_classes.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_epoch_loss = epoch_loss / len(iterator)
    accuracy, f1 = calculate_metrics(np.array(all_preds), np.array(all_labels))
    return avg_epoch_loss, accuracy, f1

def evaluate_bert(model, iterator, criterion, device):
    """Один проход оценки для BERT модели."""
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            token_type_ids = batch[3].to(device) if len(batch) > 3 else None

            if token_type_ids is not None:
                predictions = model(input_ids, attention_mask, token_type_ids)
            else:
                predictions = model(input_ids, attention_mask)
                
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()

            preds_classes = torch.argmax(predictions, dim=1)
            all_preds.extend(preds_classes.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_epoch_loss = epoch_loss / len(iterator)
    accuracy, f1 = calculate_metrics(np.array(all_preds), np.array(all_labels))
    return avg_epoch_loss, accuracy, f1


def run_training(model, model_type, train_iterator, val_iterator, n_epochs, device,
                 model_save_path, model_name, 
                 lr_lstm=0.001, lr_bert=2e-5, 
                 patience_early_stopping=5, patience_lr_scheduler=3,
                 bert_warmup_steps=0, bert_total_steps=0): # bert_total_steps нужно вычислить заранее
    """
    Основная функция для запуска процесса обучения и валидации.
    model_type: 'lstm' или 'bert'
    """
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    if model_type == 'lstm':
        optimizer = optim.Adam(model.parameters(), lr=lr_lstm)
        # В курсовой для LSTM: ReduceLROnPlateau (стр. 15)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience_lr_scheduler, verbose=True)
        train_fn = train_epoch_lstm
        eval_fn = evaluate_lstm
    elif model_type == 'bert':
        # В курсовой для BERT: AdamW (стр. 17)
        optimizer = optim.AdamW(model.parameters(), lr=lr_bert, eps=1e-8)
        # В курсовой для BERT: линейный разогрев и спад (стр. 17)
        if bert_total_steps <= 0:
             bert_total_steps = len(train_iterator) * n_epochs # Оценка, если не передано
             print(f"Предупреждение: bert_total_steps не передан, оценен как {bert_total_steps}")
        if bert_warmup_steps <= 0 and bert_total_steps > 0: # 10% warmup steps (стр.17)
            bert_warmup_steps = int(0.1 * bert_total_steps)
            print(f"Установлено bert_warmup_steps = {bert_warmup_steps} (10% от total_steps)")

        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=bert_warmup_steps, 
                                                    num_training_steps=bert_total_steps)
        train_fn = train_epoch_bert
        eval_fn = evaluate_bert
    else:
        raise ValueError("model_type должен быть 'lstm' или 'bert'")

    model.to(device)

    best_val_f1 = -float('inf')
    epochs_without_improvement = 0
    
    history = {'train_loss': [], 'train_acc': [], 'train_f1': [],
               'val_loss': [], 'val_acc': [], 'val_f1': []}

    start_time_total = time.time()

    for epoch in range(n_epochs):
        start_time_epoch = time.time()

        if model_type == 'bert':
            train_loss, train_acc, train_f1 = train_fn(model, train_iterator, optimizer, criterion, device, scheduler)
        else: # LSTM
            train_loss, train_acc, train_f1 = train_fn(model, train_iterator, optimizer, criterion, device)
            
        val_loss, val_acc, val_f1 = eval_fn(model, val_iterator, criterion, device)
        
        end_time_epoch = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time_epoch, end_time_epoch)

        print(f'Эпоха: {epoch+1:02}/{n_epochs:02} | Время эпохи: {epoch_mins}м {epoch_secs}с')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1:.3f}')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}% |  Val. F1: {val_f1:.3f}')

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        # Обновление LR для ReduceLROnPlateau (для LSTM)
        if model_type == 'lstm' and isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_f1) # Метрика для ReduceLROnPlateau - F1 на валидации (стр. 15)

        # Сохранение лучшей модели и ранняя остановка
        # В курсовой для LSTM: метрика для ранней остановки - F1 на вал. выборке, patience 5 (стр. 15)
        # В курсовой для BERT: метрика для ранней остановки - F1 на вал. выборке, patience 3 (стр. 18)
        current_patience = patience_early_stopping
        if model_type == 'bert':
            current_patience = 3 # Как в курсовой для BERT

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_model(model, model_save_path, model_name)
            epochs_without_improvement = 0
            print(f"\tЛучшая F1-мера на валидации: {best_val_f1:.3f}. Модель сохранена.")
        else:
            epochs_without_improvement += 1
            print(f"\tF1-мера на валидации не улучшилась. Осталось попыток: {current_patience - epochs_without_improvement}")
            if epochs_without_improvement >= current_patience:
                print(f"Ранняя остановка на эпохе {epoch+1}. F1-мера не улучшалась {current_patience} эпох подряд.")
                break
    
    total_training_time_mins, total_training_time_secs = epoch_time(start_time_total, time.time())
    print(f"Общее время обучения: {total_training_time_mins}м {total_training_time_secs}с")
    
    # Загрузка лучшей модели для финальной оценки, если нужно
    # model = load_model(model, model_save_path, model_name, device)
    
    return model, history

def epoch_time(start_time, end_time):
    """Вычисляет время, затраченное на эпоху."""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def plot_training_history(history, model_name=""):
    """Визуализирует историю обучения (потери и F1-меру)."""
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Потери на обучении')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Потери на валидации')
    plt.title(f'Потери при обучении модели {model_name}')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_f1'], 'bo-', label='F1 на обучении')
    plt.plot(epochs, history['val_f1'], 'ro-', label='F1 на валидации')
    plt.title(f'F1-мера при обучении модели {model_name}')
    plt.xlabel('Эпохи')
    plt.ylabel('F1-мера')
    plt.legend()

    plt.tight_layout()
    plt.show()

def display_classification_report_and_confusion_matrix(model, iterator, device, label_encoder, model_type):
    """Отображает отчет о классификации и матрицу ошибок."""
    model.eval()
    all_preds = []
    all_labels = []

    eval_fn = evaluate_lstm if model_type == 'lstm' else evaluate_bert # Используем общую функцию оценки
                                                                       # или специфичные, если нужно передавать criterion

    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            if model_type == 'lstm':
                if len(batch) == 3:
                    text_indices, text_lengths, labels = batch
                elif hasattr(batch, 'text') and hasattr(batch, 'label'):
                     text_data = batch.text
                     if isinstance(text_data, tuple) and len(text_data) == 2:
                         text_indices, text_lengths = text_data
                     else:
                         text_indices = text_data; text_lengths = torch.tensor([text_indices.size(1)] * text_indices.size(0)) # Примерная длина
                     labels = batch.label
                else: raise ValueError("Batch structure error for LSTM in report.")
                text_indices = text_indices.to(device)
                predictions = model(text_indices, text_lengths)
            else: # bert
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                token_type_ids = batch[3].to(device) if len(batch) > 3 else None
                if token_type_ids is not None: predictions = model(input_ids, attention_mask, token_type_ids)
                else: predictions = model(input_ids, attention_mask)
            
            labels_cpu = labels.cpu().numpy()
            preds_classes = torch.argmax(predictions, dim=1).cpu().numpy()
            
            all_preds.extend(preds_classes)
            all_labels.extend(labels_cpu)

    class_names = label_encoder.classes_
    
    print("\nОтчет о классификации:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Матрица ошибок для модели {model_type.upper()}')
    plt.xlabel('Предсказанная эмоция')
    plt.ylabel('Истинная эмоция')
    plt.show()
    
    # Нормализованная матрица ошибок (Рис. 4, 5 в курсовой)
    cm_normalized = confusion_matrix(all_labels, all_preds, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Нормализованная матрица ошибок для модели {model_type.upper()}')
    plt.xlabel('Предсказанная эмоция')
    plt.ylabel('Истинная эмоция')
    plt.show()


if __name__ == '__main__':
    # Этот блок можно использовать для демонстрации или тестирования функций
    # Потребуется создать фиктивные модели, данные и итераторы.
    print("Файл training.py загружен.")
    print("Для запуска примера обучения разкомментируйте и настройте код в блоке if __name__ == '__main__'.")

    # Пример использования (нужно адаптировать под ваши данные и модели):
    # --------------------------------------------------------------------
    # 1. Подготовка данных (загрузка, токенизация, создание DataLoader'ов)
    #    - Используйте функции из preprocessing.py и bert_data_preparation.py
    #    - Пример:
    #      from sklearn.preprocessing import LabelEncoder
    #      import pandas as pd
    #      from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
    #
    #      device = get_device()
    #      NUM_CLASSES = 7 # Пример
    #      # Создание LabelEncoder (должен быть обучен на ваших данных)
    #      le = LabelEncoder()
    #      # Пример меток из курсовой (стр. 12)
    #      emotion_classes = ["радість", "смуток", "гнів", "страх", "відраза", "здивування", "нейтральний"]
    #      le.fit(emotion_classes)

    #      # --- Пример для LSTM ---
    #      print("\n--- Запуск примера обучения для LSTM (фиктивные данные) ---")
    #      VOCAB_SIZE_LSTM = 1000 # Фиктивный размер словаря
    #      PAD_IDX_LSTM = 1
    #      # model_lstm = LSTMClassifier(VOCAB_SIZE_LSTM, 100, 128, NUM_CLASSES, 2, True, 0.3, PAD_IDX_LSTM).to(device)
    #      # Фиктивные данные для LSTM
    #      # batch_size, seq_len, num_batches
    #      # train_texts_idx = torch.randint(0, VOCAB_SIZE_LSTM, (32*10, 50))
    #      # train_lengths = torch.randint(10, 50, (32*10,))
    #      # train_labels_lstm = torch.randint(0, NUM_CLASSES, (32*10,))
    #      # train_dataset_lstm = TensorDataset(train_texts_idx, train_lengths, train_labels_lstm)
    #      # train_iterator_lstm = TorchDataLoader(train_dataset_lstm, batch_size=32)
    #      # val_iterator_lstm = TorchDataLoader(train_dataset_lstm, batch_size=32) # Используем те же данные для примера

    #      # model_lstm_trained, history_lstm = run_training(
    #      #     model_lstm, 'lstm', train_iterator_lstm, val_iterator_lstm,
    #      #     n_epochs=3, device=device, model_save_path='./models', model_name='lstm_example',
    #      #     patience_early_stopping=2
    #      # )
    #      # if history_lstm: plot_training_history(history_lstm, "LSTM Example")
    #      # if model_lstm_trained: display_classification_report_and_confusion_matrix(model_lstm_trained, val_iterator_lstm, device, le, 'lstm')

    #      # --- Пример для BERT ---
    #      print("\n--- Запуск примера обучения для BERT (фиктивные данные) ---")
    #      # from transformers import BertTokenizer, BertConfig
    #      # TOKENIZER_NAME_BERT = 'bert-base-multilingual-cased' # или ваша lite версия
    #      # bert_tokenizer_example = BertTokenizer.from_pretrained(TOKENIZER_NAME_BERT)
    #      # bert_config_example = BertConfig.from_pretrained(TOKENIZER_NAME_BERT, num_labels=NUM_CLASSES) # Базовая конфиг
    #      # model_bert = BertLiteClassifier(bert_model_name_or_config=bert_config_example, num_classes=NUM_CLASSES).to(device)
         
    #      # Фиктивные данные для BERT
    #      # batch_size, seq_len, num_batches
    #      # train_input_ids_bert = torch.randint(0, bert_tokenizer_example.vocab_size, (32*10, 64))
    #      # train_attn_mask_bert = torch.ones((32*10, 64), dtype=torch.long)
    #      # train_labels_bert = torch.randint(0, NUM_CLASSES, (32*10,))
    #      # train_dataset_bert = TensorDataset(train_input_ids_bert, train_attn_mask_bert, train_labels_bert)
    #      # train_iterator_bert = TorchDataLoader(train_dataset_bert, batch_size=32)
    #      # val_iterator_bert = TorchDataLoader(train_dataset_bert, batch_size=32) # Используем те же данные

    #      # bert_total_training_steps = len(train_iterator_bert) * 3 # 3 эпохи
    #      # bert_warmup_s = int(0.1 * bert_total_training_steps)

    #      # model_bert_trained, history_bert = run_training(
    #      #     model_bert, 'bert', train_iterator_bert, val_iterator_bert,
    #      #     n_epochs=3, device=device, model_save_path='./models', model_name='bert_example',
    #      #     patience_early_stopping=2,
    #      #     bert_total_steps=bert_total_training_steps, bert_warmup_steps=bert_warmup_s
    #      # )
    #      # if history_bert: plot_training_history(history_bert, "BERT Example")
    #      # if model_bert_trained: display_classification_report_and_confusion_matrix(model_bert_trained, val_iterator_bert, device, le, 'bert')
    #      pass
