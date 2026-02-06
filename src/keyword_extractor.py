"""
Keyword Extractor - модуль для извлечения ключевых слов из текстовых документов
с использованием алгоритма TF-IDF.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
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

    def compute_tf(self) -> np.ndarray:
        """Вычисляет матрицу частоты терминов (Term Frequency)."""
        # Проверяем, что документы токенизированы
        if self.tokenized_docs is None:
            raise ValueError(
                "Документы не токенизированы."
                "Сначала вызовите tokenize_documents()"
            )

        # Получаем количество документов и размер словаря
        num_docs = len(self.df)
        vocab_size = len(self.word2idx)

        logger.info(f"Вычисление TF матрицы ({num_docs}×{vocab_size})...")

        # Создаем нулевую матрицу TF
        tf_matrix = np.zeros((num_docs, vocab_size))

        # Заполняем матрицу TF
        for i, doc_indices in enumerate(self.tokenized_docs):
            for word_idx in doc_indices:
                tf_matrix[i, word_idx] += 1

        # Сохраняем матрицу как атрибут класса
        self.tf_matrix = tf_matrix
        logger.info("TF матрица вычислена")

        # Матрица TF размером (N документов × V слов в словаре)
        return tf_matrix

    def compute_idf(self) -> np.ndarray:
        """
        Вычисляет вектор обратной частоты документа
        (Inverse Document Frequency).
        """
        # Проверяем, вычислена ли TF матрица
        if self.tf_matrix is None:
            raise ValueError(
                "TF матрица не вычислена. Сначала вызовите compute_tf()"
            )

        num_docs = len(self.df)

        logger.info("Вычисление IDF вектора...")

        # Вычисляем document frequency
        # (сколько документов содержат каждое слово)
        # Используем порог > 0,
        # так как нас интересует наличие слова в документе
        document_freq = np.sum(self.tf_matrix > 0, axis=0)

        # Вычисляем IDF по формуле:
        # log(общее_число_документов / частота_документа)
        # Добавляем небольшое значение к
        # document_freq для избежания деления на ноль
        idf_vector = np.log(num_docs / (document_freq + 1e-12))

        # Сохраняем вектор как атрибут класса
        self.idf_vector = idf_vector
        logger.info("IDF вектор вычислен")

        # Вектор IDF размером V слов в словаре
        return idf_vector

    def compute_tfidf(self) -> np.ndarray:
        """Вычисляет матрицу TF-IDF."""
        # Проверяем, вычислены ли TF и IDF
        if self.tf_matrix is None:
            raise ValueError(
                "TF матрица не вычислена. Сначала вызовите compute_tf()"
            )
        if self.idf_vector is None:
            raise ValueError(
                "IDF вектор не вычислен. Сначала вызовите compute_idf()"
            )

        logger.info("Вычисление TF-IDF матрицы...")

        # Вычисляем TF-IDF: умножаем TF на IDF
        tfidf_matrix = self.tf_matrix * self.idf_vector

        # Сохраняем матрицу как атрибут класса
        self.tf_idf_matrix = tfidf_matrix
        logger.info("TF-IDF матрица вычислена")

        # Матрица TF-IDF размером (N документов × V слов в словаре)
        return tfidf_matrix

    def extract_keywords(
            self, doc_index: int, top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """Извлекает ключевые слова для указанного документа."""
        # Проверяем, вычислена ли TF-IDF матрица
        if self.tf_idf_matrix is None:
            raise ValueError(
                "TF-IDF матрица не вычислена. Сначала вызовите compute_tfidf()"
            )

        # Проверяем валидность индекса
        # doc_index: Индекс документа в DataFrame
        if doc_index < 0 or doc_index >= len(self.df):
            raise ValueError(
                f"Индекс документа должен быть от 0 до {len(self.df)-1}"
            )

        # Получаем TF-IDF веса для указанного документа
        doc_tfidf = self.tf_idf_matrix[doc_index]

        # Сортируем индексы по убыванию TF-IDF весов
        sorted_indices = np.argsort(doc_tfidf)[::-1]

        # Берем топ-N индексов
        # top_n: Количество ключевых слов для возврата (по умолчанию 5)
        top_indices = sorted_indices[:top_n]

        # Формируем результат
        keywords = []
        for idx in top_indices:
            word = self.idx2word[idx]
            weight = doc_tfidf[idx]
            keywords.append((word, weight))

        # Список кортежей (слово, tf-idf_вес) для топ-N ключевых слов
        return keywords

    def find_documents_with_word(self, word: str) -> List[int]:
        """Находит все документы, содержащие указанное слово."""
        # Проверяем, построен ли словарь
        if self.word2idx is None:
            raise ValueError(
                "Словарь не построен. Сначала вызовите build_vocabulary()"
            )

        # Проверяем, есть ли слово в словаре
        if word not in self.word2idx:
            return []  # Слово не найдено в словаре

        # Получаем индекс слова
        word_idx = self.word2idx[word]

        # Проверяем, токенизированы ли документы
        if self.tokenized_docs is None:
            raise ValueError(
                "Документы не токенизированы."
                "Сначала вызовите tokenize_documents()"
            )

        # Ищем документы, содержащие это слово
        matching_docs = []
        for doc_index, doc_indices in enumerate(self.tokenized_docs):
            if word_idx in doc_indices:
                matching_docs.append(doc_index)

        # Список индексов документов, содержащих это слово
        return matching_docs

    def get_document_preview(
            self, doc_index: int, max_length: int = 100
    ) -> str:
        """Возвращает краткий превью документа."""
        if self.df is None:
            raise ValueError(
                "Данные не загружены. Сначала вызовите load_data()"
            )

        # Получаем текст документа
        text = self.df.iloc[doc_index]['text']

        # Берем первые max_length символов
        preview = text[:max_length]

        # Добавляем многоточие, если текст был обрезан
        if len(text) > max_length:
            preview += "..."

        return preview

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
        tfidf_shape = (
            self.tf_idf_matrix.shape
            if self.tf_idf_matrix is not None
            else 'Не вычислена'
        )

        # Формируем информационную строку
        info = [
            "KeywordExtractor",
            f"Документов загружено: {num_docs}",
            f"Размер словаря: {vocab_size}",
            f"TF матрица: {tf_shape}",
            f"IDF вектор: {idf_shape}",
            f"TF-IDF матрица: {tfidf_shape}",
        ]
        return "\n".join(info)


if __name__ == "__main__":
    """
    Точка входа для тестирования класса.
    Создает экземпляр KeywordExtractor и выводит информацию о нем.
    """
    extractor = KeywordExtractor()  # создаем экземпляр класса
    print(extractor)  # выводим информацию об объекте
