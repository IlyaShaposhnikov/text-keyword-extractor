# Text Keyword Extractor

Проект для извлечения ключевых слов из текстовых документов с использованием алгоритма TF-IDF, реализованного с нуля.

## Описание

Этот инструмент позволяет анализировать текстовые документы и выделять наиболее значимые ключевые слова с помощью метода TF-IDF (Term Frequency-Inverse Document Frequency). Проект включает собственную реализацию алгоритма TF-IDF и предоставляет интерактивный интерфейс для работы с текстами.

## Особенности

- Собственная реализация TF-IDF с нуля
- Обработка текстовых документов
- Извлечение ключевых слов
- Интерактивный интерфейс

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

3. Скачайте данные с [Kaggle](https://www.kaggle.com/datasets/abdulraffayali/bbc-text-cls) и поместите файл `bbc-text-cls.csv` в папку `data/`

## Использование
```python
from src.keyword_extractor import KeywordExtractor

extractor = KeywordExtractor('data/bbc-text-cls.csv')
keywords = extractor.extract_keywords(document_index=0)
print(keywords)
```

## Структура проекта
```text
text-keyword-extractor/
├── src/                    # Исходный код
├── data/                   # Данные
├── requirements.txt        # Зависимости
└── README.md               # Документация
```

## Автор

Шапошников Илья Андреевич

ilia.a.shaposhnikov@gmail.com