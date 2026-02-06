# Text Keyword Extractor

Проект для извлечения ключевых слов из текстовых документов с использованием алгоритма TF-IDF, реализованного с нуля.

## Описание

Этот инструмент позволяет анализировать текстовые документы и выделять наиболее значимые ключевые слова с помощью метода TF-IDF (Term Frequency-Inverse Document Frequency). Проект включает собственную реализацию алгоритма TF-IDF и предоставляет интерактивный интерфейс для работы с текстами.

## Особенности

- Собственная реализация TF-IDF с нуля
- Обработка текстовых документов
- Извлечение ключевых слов
- Интерактивный интерфейс для поиска и анализа
- Сравнение с встроенной реализацией sklearn
- Статистика по данным

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/ваш-username/text-keyword-extractor.git
cd text-keyword-extractor
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Скачайте датасет с [Kaggle](https://www.kaggle.com/datasets/abdulraffayali/bbc-text-cls) и поместите файл `bbc-text-cls.csv` в папку `data/`

## Использование

### Как Python библиотека:

```python
from src.keyword_extractor import KeywordExtractor

# Загрузка данных и извлечение ключевых слов
extractor = KeywordExtractor()
extractor.load_data()
extractor.build_vocabulary()
extractor.tokenize_documents()
extractor.compute_tf()
extractor.compute_idf()
extractor.compute_tfidf()

# Извлечение ключевых слов для документа
keywords = extractor.extract_keywords(document_index=0, top_n=10)
print(keywords)
```

### Интерактивный режим:
```bash
python main.py
```

#### В интерактивном режиме доступны:
1. Поиск статей по ключевому слову
2. Анализ случайной статьи
3. Просмотр информации о данных
4. Сравнение с sklearn реализацией TF-IDF

## Структура проекта
```text
text-keyword-extractor/
├── src/
│   └── keyword_extractor.py   # Основной класс для извлечения ключевых слов
├── data/                      # Папка для данных
├── main.py                    # Интерактивная программа
├── requirements.txt           # Зависимости Python
├── .gitignore                 # Игнорируемые файлы Git
└── README.md                  # Документация
```

## Автор

Шапошников Илья Андреевич

ilia.a.shaposhnikov@gmail.com