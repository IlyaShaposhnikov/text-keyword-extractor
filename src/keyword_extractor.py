"""
Keyword Extractor - модуль для извлечения ключевых слов из текстовых документов
с использованием алгоритма TF-IDF.
"""

import pandas as pd
import numpy as np
from typing import List
import nltk
from nltk.tokenize import word_tokenize
import logging

# Путь к датасету
DEFAULT_DATA_PATH = 'data/bbc_text_cls.csv'

# Настройка логирования для отслеживания выполнения программы
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KeywordExtractor:
    """
    Класс для извлечения ключевых слов из текстовых документов.
    Реализует алгоритм TF-IDF с нуля.
    """

    def __init__(self):
        """Инициализация KeywordExtractor.        """
        # Используем константу для пути к данным
        self.data_path = DEFAULT_DATA_PATH
        # Инициализируем атрибуты класса значениями None
        # DataFrame с данными
        self.df = None
        # Словарь слово -> индекс
        self.word2idx = None
        # Словарь индекс -> слово
        self.idx2word = None
        # Матрица частоты терминов
        self.tf_matrix = None
        # Вектор обратной частоты документа
        self.idf_vector = None
        # Итоговая TF-IDF матрица
        self.tf_idf_matrix = None
        # Токенизированные документы
        self.tokenized_docs = None

        # Загружаем необходимые ресурсы NLTK при инициализации
        try:
            # Проверяем, установлен ли токенизатор punkt
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            # Если нет, скачиваем его
            logger.info("Загрузка токенизатора NLTK...")
            nltk.download('punkt', quiet=True)

    def load_data(self) -> None:
        """Загружает данные из CSV файла."""
        logger.info(f"Загрузка данных из {self.data_path}")

        # Читаем CSV файл в pandas DataFrame
        self.df = pd.read_csv(self.data_path)

        # Логируем успешную загрузку
        logger.info(f"Загружено {len(self.df)} документов")

    def preprocess_text(self, text: str) -> List[str]:
        """Предварительная обработка текста."""
        # Приводим текст к нижнему регистру и разбиваем на токены (слова)
        return word_tokenize(text.lower())

    def build_vocabulary(self) -> None:
        """
        Строит словарь уникальных слов из всех загруженных документов.
        Создает два взаимосвязанных словаря: word2idx и idx2word.
        """
        # Проверяем, загружены ли данные
        if self.df is None:
            raise ValueError(
                "Данные не загружены. Сначала вызовите load_data()"
            )

        logger.info("Построение словаря...")

        # Инициализируем словари и счетчик
        word2idx = {}
        idx2word = {}
        idx = 0

        # Проходим по всем текстам в DataFrame
        for doc in self.df['text']:
            # Токенизируем документ
            words = self.preprocess_text(doc)

            # Добавляем новые слова в словарь
            for word in words:
                if word not in word2idx:
                    word2idx[word] = idx
                    idx2word[idx] = word
                    idx += 1

        # Сохраняем словари как атрибуты класса
        self.word2idx = word2idx
        self.idx2word = idx2word

        logger.info(f"Словарь содержит {len(word2idx)} уникальных слов")

    def tokenize_documents(self) -> List[List[int]]:
        """Токенизирует документы, преобразуя слова в индексы."""
        # Проверяем, построен ли словарь
        if self.word2idx is None:
            raise ValueError(
                "Словарь не построен. Сначала вызовите build_vocabulary()"
            )

        tokenized = []

        # Проходим по всем текстам
        for doc in self.df['text']:
            # Токенизируем и преобразуем слова в индексы
            words = self.preprocess_text(doc)
            # Создает новый список, где каждое слово из words заменяется
            # его индексом из словаря self.word2idx.
            # Пропускает слова, которых нет в словаре
            # (например, слова, не вошедшие в словарь при обучении)
            doc_as_int = [
                self.word2idx[word] for word in words if word in self.word2idx
            ]
            tokenized.append(doc_as_int)

        # Сохраняем токенизированные документы
        self.tokenized_docs = tokenized
        return tokenized

    def __repr__(self) -> str:
        """Возвращает строковое представление объекта для отладки."""
        # Получаем текущие размеры данных
        num_docs = len(self.df) if self.df is not None else 0
        vocab_size = len(self.word2idx) if self.word2idx is not None else 0
        tf_shape = (
            self.tf_matrix.shape
            if self.tf_matrix is not None
            else 'Не вычислена'
        )
        idf_shape = (
            self.idf_vector.shape
            if self.idf_vector is not None
            else 'Не вычислен'
        )

        # Формируем информационную строку
        info = [
            "KeywordExtractor",
            f"Документов загружено: {num_docs}",
            f"Размер словаря: {vocab_size}",
            f"TF матрица: {tf_shape}",
            f"IDF вектор: {idf_shape}",
        ]
        return "\n".join(info)


if __name__ == "__main__":
    """
    Точка входа для тестирования класса.
    Создает экземпляр KeywordExtractor и выводит информацию о нем.
    """
    extractor = KeywordExtractor()  # создаем экземпляр класса
    print(extractor)  # выводим информацию об объекте
