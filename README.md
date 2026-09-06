# War and Peace — Character-Level Text Generator

> **Генератор символьного текста на LSTM слоях, созданный на основе "Войны и мира" Льва Толстого. Реализует жадное декодирование, случайную выборку с учетом температуры и выборку с top-k. Создан с использованием PyTorch.**

---

## Навигация
- [Особенности](#особенности)
- [Архитектура модели](#архитектура-модели)
- [Результаты](#результаты)
- [Установка](#установка)
- [Использование](#использование)
  - [Генерация текста](#генерация-текста)
  - [Обучение модели](#обучение-модели)
- [Структура проекта](#структура-проекта)
- [Веса модели](#веса-модели)

---

## Особенности

- **Character-level LSTM** - модель обучается на уровне отдельных символов, что позволяет генерировать текст в стиле исходного произведения.
- **Три стратегии генерации**:
  - **Greedy Search** - выбор наиболее вероятного символа на каждом шагу (детерминировано).
  - **Random Sampling** - выбор символа с учётом вероятностей, регулируется температурой.
  - **Top-K Sampling** - выбор только из K наиболее вероятных символов, затем сэмплирование.
- **Обработка русского текста** - словарь из 135 символов (включая кириллицу, знаки препинания, цифры).
- **Обучение на "Войне и мире"** - 1.7 млн символов в обучающей выборке.
- **Сохранение чекпоинтов** - автоматическое сохранение лучшей модели по validation loss.

---

## Архитектура модели

Модель представляет собой двухслойную LSTM с механизмом dropout:

| Компонент | Описание |
|-----------|----------|
| **Embedding** | Векторное представление символов (размер 128) |
| **LSTM** | 2 слоя, скрытая размерность 256, dropout 0.3 |
| **Linear (FC)** | Проекция скрытого состояния на размер словаря (135) |
| **Функция потерь** | CrossEntropyLoss |
| **Оптимизатор** | AdamW с weight decay |
| **Планировщик** | ReduceLROnPlateau (patience=2, factor=0.5) |

**Количество параметров:** 973k

---

## Результаты

Модель обучалась 40 эпох с early stopping (patience=5). Лучшая эпоха - **28**, достигнутые метрики:

| Метрика | Train | Validation |
|---------|-------|------------|
| **Loss** | 1.2512 | **1.3402** |
| **Accuracy** | 60.86% | **59.18%** |
| **Perplexity** | 3.49 | **3.82** |

### Примеры генерации

**Prompt:** `"КНЯЗЬ АНДРЕЙ:\n"`

**Greedy Search:**
```
КНЯЗЬ АНДРЕЙ:
 -- Нет, не могу понять, -- сказал он, -- но не могу понять, что он не мог понять всего в этом состоянии своего собаки, и он не мог понять всего в этом состоянии своей страшное время, которое он не понимал и не понимал и не понимал и не понимал и не понимал и не понимал и не понимал и н
```

**Random Sampling (T=0.7):**
```
КНЯЗЬ АНДРЕЙ:
 Николай испугался в Сони, которому старой офицеров, разговаривавший головы на крайного колкового красного девушки.
 Так выражалось произвольно смотрели в антизманду, которой имел немцат. Он ничего не отвечал, но он не видал и особенно общество. В молодежь и даже она, никогда не убила с веселая пр
```

**Top-K Sampling (k=3, T=0.8):**
```
КНЯЗЬ АНДРЕЙ:
 -- Нет, не подождать, -- сказал он. -- Я не могу понимать, как всегда, как вы с тобой, как она на страх, чтобы от этого обращаются в ней императоры.
 -- Ах, мой милый, -- проговорил он, -- что вы представителисти себя, что я не могу не принимать составления и в своем стороне столь не сказать с ней,
```

> Полные примеры генерации можно найти в ноутбуке `notebooks/RNN_генерация_текста.ipynb`.

---

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/your-username/CharRNN-war-and-peace.git
cd CharRNN-war-and-peace
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

**Требования:**
- Python 3.8+
- PyTorch (с поддержкой CUDA, если есть GPU)
- torchvision, numpy, matplotlib, tqdm

## Использование

### Генерация текста

```python
from src import CharLSTMModel, generate_text, random_sample
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Загружаем информацию о словаре и параметрах
info = torch.load("checkpoints/char_rnn_war_and_peace_info.pt", map_location=device)
char_to_id = info["char_to_id"]
id_to_char = info["id_to_char"]
vocab_size = info["vocab_size"]

# 2. Создаём модель с теми же параметрами
model = CharLSTMModel(
    vocab_size=vocab_size,
    embedding_dim=128,
    hidden_dim=256,
    num_layers=2,
    dropout=0.3
)

# 3. Загружаем веса из чекпоинта
checkpoint = torch.load("checkpoints/best_model.pt", map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# 4. Генерация текста
prompt = "КНЯЗЬ АНДРЕЙ:\n"
generated = generate_text(
    model, prompt,
    sample_fn=lambda logits: random_sample(logits, temperature=0.8),
    gen_length=300,
    char_to_id=char_to_id,
    id_to_char=id_to_char,
    device=device
)
print(generated)
```

### Командная строка

```bash
# Greedy search
python src/generate.py --prompt "КНЯЗЬ АНДРЕЙ:" --method greedy

# Random sampling с температурой 0.8
python src/generate.py --prompt "КНЯЗЬ АНДРЕЙ:" --method random --temperature 0.8

# Top-k sampling (k=3, T=0.8)
python src/generate.py --prompt "КНЯЗЬ АНДРЕЙ:" --method top_k --k 3 --temperature 0.8 --length 300
```

> **Важно:** При использовании CLI-скрипта `src/generate.py` путь к файлам весов (`--weights`) и информации (`--info`) можно указать явно, либо они будут искаться в папке `checkpoints/` по умолчанию.

### Командная строка

```bash
python src/generate.py --prompt "КНЯЗЬ АНДРЕЙ:" --method top_k --k 3 --temperature 0.8 --length 300
```

### Обучение модели

```bash
python src/train.py --data_path warandpeace.txt --epochs 50 --batch_size 64 --hidden_dim 256 --num_layers 2
```

Все гиперпараметры можно настроить через аргументы командной строки или изменить в конфигурационном файле.

---

## Структура проекта

```
char-rnn-war-and-peace/
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── RNN_генератор_текста.ipynb       # Исходный ноутбук с экспериментами
│
├── src/
│   ├── model.py                   # CharLSTMModel
│   ├── train.py                   # Скрипт обучения
│   ├── generate.py                # Скрипт генерации
│   └── utils.py                   # Вспомогательные функции
│
├── configs/
│   └── default.yaml               # Конфигурация гиперпараметров
│
├── checkpoints/                   # (папка для весов, в .gitignore)
│   └── best_model.pt
│
└── data/
    └── warandpeace.txt            # Исходный текст (загрузится при выполнении кода)
```

---

## Веса модели

> Веса модели и информация об обучении сохранены в формате `.pt`. Для загрузки используйте код из раздела "Генерация текста".
> 
> Для работы необходимы два файла:
1. **`char_rnn_war_and_peace.pt`** — чекпоинт модели (содержит веса, состояние оптимизатора, историю обучения).
2. **`char_rnn_war_and_peace_info.pt`** — информация о словаре и гиперпараметрах.

**Ссылки для скачивания:**
- [char_rnn_war_and_peace.pt char_rnn_war_and_peace_info.pt](https://drive.google.com/drive/folders/1lpoLXtrQSFIgzQ8Cvc9UxRyVsGykcrLK?usp=sharing)


---

## Контакты

По вопросам сотрудничества или найденным ошибкам пишите: sokolovkirill489@gmail.com TG: @qqkiru
```
