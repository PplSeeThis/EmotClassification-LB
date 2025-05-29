import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class BertLiteClassifier(nn.Module):
    def __init__(self, bert_model_name_or_config, num_classes, custom_bert_config_params=None, dropout_rate_head=0.3):
        """
        Инициализация BERT-lite модели для классификации текста.

        Args:
            bert_model_name_or_config (str or BertConfig): Имя предобученной модели BERT 
                                                          или объект BertConfig для кастомной архитектуры.
            num_classes (int): Количество классов для классификации.
            custom_bert_config_params (dict, optional): Параметры для создания кастомной BertConfig,
                                                       если bert_model_name_or_config это строка,
                                                       но мы хотим "lite" версию.
                                                       Используется, если мы хотим загрузить только токенизатор
                                                       от базовой модели, а саму модель создать с "lite" параметрами.
                                                       Пример: {'num_hidden_layers': 6, 'hidden_size': 512, ...}
            dropout_rate_head (float): Вероятность dropout для классификационной головы.
                                       В курсовой (Рис.2, стр.17) это 0.3.
        """
        super(BertLiteClassifier, self).__init__()

        if isinstance(bert_model_name_or_config, str) and custom_bert_config_params:
            # Загружаем конфигурацию по имени, затем изменяем её и создаем модель с этой конфигурацией
            # Это позволяет использовать vocab_size и другие параметры от базовой модели,
            # но с "lite" архитектурой.
            config = BertConfig.from_pretrained(bert_model_name_or_config)
            config.update(custom_bert_config_params)
            self.bert = BertModel(config)
            print(f"Создана BERT модель с кастомной 'lite' конфигурацией на основе {bert_model_name_or_config}:")
            print(f"  num_hidden_layers: {config.num_hidden_layers}")
            print(f"  hidden_size: {config.hidden_size}")
            print(f"  num_attention_heads: {config.num_attention_heads}")
        elif isinstance(bert_model_name_or_config, BertConfig):
            # Используем предоставленный объект BertConfig
            config = bert_model_name_or_config
            self.bert = BertModel(config)
            print("Создана BERT модель с предоставленным объектом BertConfig.")
        else: # isinstance(bert_model_name_or_config, str)
            # Загружаем предобученную модель BERT по имени
            # Это может быть полная модель, а не "lite", если имя не указывает на lite-версию.
            # Для курсовой, где определены параметры lite-версии, предпочтительнее первый вариант.
            self.bert = BertModel.from_pretrained(bert_model_name_or_config)
            config = self.bert.config # Получаем конфигурацию из загруженной модели
            print(f"Загружена предобученная BERT модель: {bert_model_name_or_config}")

        # Классификационная голова согласно Рис. 2 и стр. 17, 43
        # Pooler Layer (выход BERT для [CLS] токена) -> Dropout -> Dense (256) -> ReLU -> Dropout -> Output Layer (num_classes)
        
        # Dropout после Pooler Layer (выхода BERT)
        self.dropout1 = nn.Dropout(dropout_rate_head)
        
        # Полносвязный слой (Dense Layer)
        # config.hidden_size - это размерность выхода self.bert.pooler (если он есть и используется)
        # или выхода [CLS] токена из last_hidden_state.
        # self.bert.pooler это Linear(config.hidden_size, config.hidden_size) + Tanh
        self.fc1 = nn.Linear(config.hidden_size, 256)
        self.relu = nn.ReLU()
        
        # Второй Dropout слой
        self.dropout2 = nn.Dropout(dropout_rate_head)
        
        # Выходной слой
        self.fc2 = nn.Linear(256, num_classes)

        print(f"Классификационная голова: Dropout({dropout_rate_head}) -> FC({config.hidden_size}, 256) -> ReLU -> Dropout({dropout_rate_head}) -> FC(256, {num_classes})")

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Прямой проход модели.

        Args:
            input_ids (torch.Tensor): Тензор с индексами токенов.
                                      Размерность: (batch_size, seq_len)
            attention_mask (torch.Tensor): Тензор маски внимания.
                                           Размерность: (batch_size, seq_len)
            token_type_ids (torch.Tensor, optional): Тензор типов токенов (для задач с парами предложений).
                                                    Размерность: (batch_size, seq_len). По умолчанию None.

        Returns:
            torch.Tensor: Логиты для каждого класса. Размерность: (batch_size, num_classes)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Используем выход 'pooler_output', который представляет собой [CLS] токен,
        # пропущенный через дополнительный линейный слой и Tanh активацию.
        # Это стандартный способ получения представления всей последовательности для классификации.
        pooled_output = outputs.pooler_output 
        # pooled_output: [batch_size, hidden_size]

        x = self.dropout1(pooled_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.fc2(x) # Логиты (сырые выходы перед softmax)

        return logits

def створити_bert_lite_модель(num_classes: int, config_bert_lite: dict, tokenizer_name_for_vocab: str = 'bert-base-multilingual-cased') -> BertLiteClassifier:
    """
    Фабричная функция для создания BERT-lite модели.
    Использует параметры из config_bert_lite для создания BertConfig.
    Берет vocab_size из токенизатора tokenizer_name_for_vocab.

    Args:
        num_classes (int): Количество выходных классов.
        config_bert_lite (dict): Словарь с параметрами для BERT-lite.
            Пример: {'num_hidden_layers': 6, 'hidden_size': 512, 'num_attention_heads': 8, ...}
        tokenizer_name_for_vocab (str): Имя токенизатора для определения vocab_size.
    """
    
    # Загружаем "эталонную" конфигурацию, чтобы получить vocab_size и другие не переопределяемые параметры
    # Это важно, т.к. vocab_size должен соответствовать используемому токенизатору.
    base_config = BertConfig.from_pretrained(tokenizer_name_for_vocab)

    # Параметры BERT-lite из курсовой (стр. 10, 16, 43)
    # Если в config_bert_lite не указаны, берем из курсовой или base_config
    final_config_params = {
        'vocab_size': base_config.vocab_size, # Важно! Из токенизатора
        'hidden_size': config_bert_lite.get('hidden_size', 512),
        'num_hidden_layers': config_bert_lite.get('num_hidden_layers', 6),
        'num_attention_heads': config_bert_lite.get('num_attention_heads', 8),
        'intermediate_size': config_bert_lite.get('intermediate_size', 4 * config_bert_lite.get('hidden_size', 512)), # Обычно 4 * hidden_size
        'hidden_act': config_bert_lite.get('hidden_act', 'gelu'),
        'hidden_dropout_prob': config_bert_lite.get('hidden_dropout_prob', 0.1), # Стандартный BERT dropout
        'attention_probs_dropout_prob': config_bert_lite.get('attention_probs_dropout_prob', 0.1), # Стандартный BERT dropout
        'max_position_embeddings': config_bert_lite.get('max_position_embeddings', 512), # Стандартно для BERT
        'type_vocab_size': config_bert_lite.get('type_vocab_size', 2), # Для задач NSP
        'initializer_range': config_bert_lite.get('initializer_range', 0.02),
        # Добавляем другие параметры из base_config, если они не переопределены
    }
    
    # Обновляем base_config параметрами из final_config_params
    for key, value in final_config_params.items():
        setattr(base_config, key, value)

    # Dropout для классификационной головы
    dropout_head = config_bert_lite.get('dropout_rate_head', 0.3) # Стр. 17

    model = BertLiteClassifier(
        bert_model_name_or_config=base_config, # Передаем сконфигурированный объект BertConfig
        num_classes=num_classes,
        dropout_rate_head=dropout_head
    )
    return model

if __name__ == '__main__':
    # Параметры для BERT-lite из курсовой (стр. 10, 16, 17, 43)
    NUM_CLASSES = 7 # Количество классов эмоций
    
    # Конфигурация для "lite" версии BERT
    # vocab_size будет взят из токенизатора 'bert-base-multilingual-cased'
    bert_lite_params_from_thesis = {
        'hidden_size': 512,
        'num_hidden_layers': 6,
        'num_attention_heads': 8,
        'intermediate_size': 2048, # 4 * 512
        'hidden_dropout_prob': 0.1, # Пример, в курсовой не указан для BERT слоев, но стандартно
        'attention_probs_dropout_prob': 0.1, # Пример
        'dropout_rate_head': 0.3 # Для классификационной головы (стр. 17)
    }
    
    TOKENIZER_NAME = 'bert-base-multilingual-cased' # Для vocab_size и токенизации

    print(f"--- Создание модели BERT-lite с параметрами из курсовой ---")
    try:
        bert_lite_model = створити_bert_lite_модель(
            num_classes=NUM_CLASSES,
            config_bert_lite=bert_lite_params_from_thesis,
            tokenizer_name_for_vocab=TOKENIZER_NAME
        )
        print(bert_lite_model)
        
        # Подсчет количества параметров (примерный, т.к. некоторые могут быть заморожены/не обучаемы)
        total_params = sum(p.numel() for p in bert_lite_model.parameters())
        trainable_params = sum(p.numel() for p in bert_lite_model.parameters() if p.requires_grad)
        print(f"Общее количество параметров: {total_params:,}")
        print(f"Количество обучаемых параметров: {trainable_params:,}")
        # В курсовой (стр. 10) указано ~30 миллионов параметров.
        # bert-base-multilingual-cased имеет ~177M.
        # Модель с 6 слоями, hidden 512, attention heads 8 будет значительно меньше.
        # Например, distilbert-base-multilingual-cased (6 layers, 768 hidden, 12 heads) ~134M params.
        # Модель с (6 layers, 512 hidden, 8 heads) должна быть еще меньше.
        # Давайте проверим:
        # BertEmbeddings (512, vocab ~119547) ~61M
        # BertLayer (512 hidden, 8 heads, intermediate 2048) * 6 layers:
        #   - SelfAttention: Q,K,V (512*512)*3 + OutputDense(512*512) ~2M
        #   - Intermediate (512*2048) + Output (2048*512) ~2M
        #   - LayerNorms
        #   ~4-5M per layer * 6 = ~24-30M
        # BertPooler (512*512) ~0.26M
        # Classification head: (512*256) + (256*7) ~0.13M
        # Итого: ~61M (embeddings) + ~24-30M (encoder) + ~0.26M (pooler) + ~0.13M (head) = ~85-91M
        # Это больше 30М. Чтобы получить ~30М с vocab 119k, hidden_size должен быть меньше, или слоев меньше.
        # Возможно, в курсовой "30 миллионов" относится к параметрам *только энкодера* без эмбеддингов,
        # или используется модель с меньшим словарем/эмбеддингами.
        # Либо "BERT-lite" в курсовой - это более агрессивное урезание.
        # Я реализовал согласно указанным слоям/размерностям. Если нужно строго 30М, конфиг нужно будет сильно менять.

        # Тестирование прямого прохода
        print("\n--- Тест прямого прохода (forward pass) ---")
        batch_size = 2
        seq_len = bert_lite_model.bert.config.max_position_embeddings # Используем max_position_embeddings из конфига модели
                                                                    # но в курсовой макс. длина последовательности 128 (стр. 16)
        actual_seq_len = 128 

        # Случайные входные данные
        # Нужен токенизатор для получения правильных input_ids (включая спец. токены)
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
        dummy_texts = ["приклад тексту для bert", "інший приклад"]
        
        encoded_input = tokenizer(dummy_texts, padding='max_length', truncation=True, max_length=actual_seq_len, return_tensors='pt')
        test_input_ids = encoded_input['input_ids']
        test_attention_mask = encoded_input['attention_mask']
        
        bert_lite_model.eval() # Установка модели в режим оценки
        try:
            with torch.no_grad(): # Отключаем расчет градиентов
                predictions = bert_lite_model(test_input_ids, test_attention_mask)
            print(f"Входные input_ids (форма): {test_input_ids.shape}")
            print(f"Входная attention_mask (форма): {test_attention_mask.shape}")
            print(f"Выходные предсказания (логиты, форма): {predictions.shape}") # Ожидаем [batch_size, NUM_CLASSES]
            print(f"Пример предсказаний (логиты):\n{predictions}")

            if torch.isnan(predictions).any():
                print("\nВНИМАНИЕ: В предсказаниях есть NaN значения!")
            else:
                print("\nПредсказания не содержат NaN значений.")

        except Exception as e:
            print(f"Ошибка во время прямого прохода: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Ошибка при создании модели BERT-lite: {e}")
        import traceback
        traceback.print_exc()

