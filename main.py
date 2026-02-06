#!/usr/bin/env python3
"""
Интерактивная программа для извлечения ключевых слов из текстовых документов.
Пользователь может искать статьи по слову и
получать ключевые слова для выбранной статьи.
"""

import sys
from src.keyword_extractor import KeywordExtractor


def display_welcome() -> None:
    """Выводит приветственное сообщение и инструкцию."""
    print("=" * 60)
    print("          КЛЮЧЕВЫЕ СЛОВА - ИЗВЛЕЧЕНИЕ ИЗ ТЕКСТОВ")
    print("=" * 60)
    print("\nЭта программа позволяет:")
    print("1. Найти статьи, содержащие определенное слово")
    print("2. Получить ключевые слова для выбранной статьи")
    print("3. Выйти из программы")
    print("\n" + "-" * 60)


def initialize_extractor() -> KeywordExtractor:
    """Инициализирует и загружает данные в KeywordExtractor."""
    print("Инициализация KeywordExtractor...")

    # Создаем экземпляр класса
    extractor = KeywordExtractor()

    try:
        # Загружаем данные
        extractor.load_data()

        # Строим словарь
        extractor.build_vocabulary()

        # Токенизируем документы
        extractor.tokenize_documents()

        # Вычисляем TF, IDF, TF-IDF
        print("Вычисление TF, IDF и TF-IDF матриц...")
        extractor.compute_tf()
        extractor.compute_idf()
        extractor.compute_tfidf()

        print("Данные успешно загружены и обработаны!")
        print(f"  Загружено документов: {len(extractor.df)}")
        print(f"  Размер словаря: {len(extractor.word2idx)} слов")

        return extractor

    except FileNotFoundError:
        print("\nОШИБКА: Файл данных не найден!")
        print("Пожалуйста, скачайте файл данных с Kaggle:")
        print("https://www.kaggle.com/datasets/abdulraffayali/bbc-text-cls")
        print("и поместите его в папку data/ под именем bbc_text_cls.csv")
        sys.exit(1)

    except Exception as e:
        print(f"\nОШИБКА при инициализации: {e}")
        sys.exit(1)


def compare_with_sklearn(
        extractor: KeywordExtractor, doc_index: int, extractor_keywords: list
) -> None:
    """Сравнивает ручную реализацию TF-IDF со встроенной из sklearn."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        print("\nСравнение с sklearn TF-IDF...")

        # Получаем все тексты для обучения векторайзера
        all_texts = extractor.df['text'].tolist()

        # Создаем и обучаем векторайзер
        vectorizer = TfidfVectorizer(
            lowercase=True,
            tokenizer=extractor.preprocess_text,
            # Отключаем стандартную токенизацию
            token_pattern=None
        )

        tfidf_matrix_sklearn = vectorizer.fit_transform(all_texts)

        # Получаем TF-IDF веса для выбранного документа
        doc_tfidf_sklearn = tfidf_matrix_sklearn[doc_index]

        # Получаем названия признаков (слов)
        feature_names = vectorizer.get_feature_names_out()

        # Сортируем слова по TF-IDF весу
        sorted_indices = doc_tfidf_sklearn.toarray()[0].argsort()[::-1]

        # Берем топ-10 слов из sklearn
        sklearn_keywords = []
        for idx in sorted_indices[:10]:
            word = feature_names[idx]
            weight = doc_tfidf_sklearn[0, idx]
            sklearn_keywords.append((word, weight))

        # Выводим сравнение
        print("\n" + "=" * 60)
        print("СРАВНЕНИЕ РЕАЛИЗАЦИЙ TF-IDF")
        print("=" * 60)

        print(f"\n{'Ручная реализация':<40} {'Sklearn реализация':<40}")
        print(f"{'-'*40:<40} {'-'*40:<40}")

        for i in range(10):
            extractor_word = (
                extractor_keywords[i][0]
                if i < len(extractor_keywords)
                else ""
            )
            extractor_weight = (
                extractor_keywords[i][1]
                if i < len(extractor_keywords)
                else 0
            )

            sklearn_word = (
                sklearn_keywords[i][0]
                if i < len(sklearn_keywords)
                else ""
            )
            sklearn_weight = (
                sklearn_keywords[i][1]
                if i < len(sklearn_keywords)
                else 0
            )

            print(f"{i+1:2}. {extractor_word:<20} {extractor_weight:<15.4f} | "
                  f"{sklearn_word:<20} {sklearn_weight:<15.4f}")

        # Считаем количество совпадающих слов в топ-10
        extractor_words_set = set(word for word, _ in extractor_keywords)
        sklearn_words_set = set(word for word, _ in sklearn_keywords)
        common_words = extractor_words_set.intersection(sklearn_words_set)

        print(f"\nСовпадающих слов в топ-10: {len(common_words)}")
        if common_words:
            print(f"Совпадающие слова: {', '.join(sorted(common_words))}")

    except ImportError:
        print("\n⚠️  Библиотека scikit-learn не установлена.")
        print("Установите её для сравнения: pip install scikit-learn")
    except Exception as e:
        print(f"\n⚠️  Ошибка при сравнении: {e}")


def analyze_document(extractor: KeywordExtractor, doc_index: int) -> None:
    """Анализирует документ и показывает ключевые слова."""
    print("\n" + "=" * 60)
    print(f"АНАЛИЗ ДОКУМЕНТА #{doc_index}")
    print("=" * 60)

    # Получаем информацию о документе
    doc_row = extractor.df.iloc[doc_index]
    label = doc_row['labels']
    text_preview = extractor.get_document_preview(doc_index, 200)

    print(f"\nТема: {label}")
    print("\nСодержание:")
    print(f"{text_preview}")

    # Извлекаем ключевые слова
    print("\nИзвлечение ключевых слов...")
    keywords = extractor.extract_keywords(doc_index, top_n=10)

    print("\nТОП-10 ключевых слов (с TF-IDF весами):")
    print("-" * 40)

    # Выводим ключевые слова в формате таблицы
    print(f"{'№':<3} {'Слово':<20} {'TF-IDF':<10}")
    print("-" * 40)

    for i, (word, weight) in enumerate(keywords, 1):
        print(f"{i:<3} {word:<20} {weight:<10.4f}")

    # Предлагаем сравнить со встроенной реализацией TF-IDF
    print("\n" + "-" * 60)
    print("Сравнить с встроенной реализацией TF-IDF? (да/нет)")
    compare_choice = input("> ").strip().lower()

    if compare_choice == 'да':
        compare_with_sklearn(extractor, doc_index, keywords)


def search_word_interactive(extractor: KeywordExtractor) -> None:
    """Интерактивный поиск документов по слову."""
    print("\n" + "=" * 60)
    print("ПОИСК СТАТЕЙ ПО СЛОВУ")
    print("=" * 60)

    while True:
        print("\nВведите слово для поиска (или 'назад' для возврата):")
        user_input = input("> ").strip().lower()

        # Проверяем команду возврата
        if user_input == 'назад':
            return

        # Проверяем, что введено не пустое слово
        if not user_input:
            print("Пожалуйста, введите слово для поиска.")
            continue

        # Ищем документы, содержащие это слово
        matching_docs = extractor.find_documents_with_word(user_input)

        if not matching_docs:
            print(f"\nСлово '{user_input}' не найдено в документах.")
            print("Попробуйте другое слово.")
            continue

        print(
            f"\nНайдено {len(matching_docs)}"
            f"документов со словом '{user_input}':"
        )

        # Показываем найденные документы
        # Ограничим показ 10 документами
        for i, doc_idx in enumerate(matching_docs[:10]):
            # Получаем информацию о документе
            label = extractor.df.iloc[doc_idx]['labels']
            preview = extractor.get_document_preview(doc_idx, 80)

            # Форматируем вывод
            print(f"\n[{i+1}] Документ #{doc_idx} (Тема: {label})")
            print(f"    {preview}")

        # Если документов больше 10, сообщаем об этом
        if len(matching_docs) > 10:
            print(f"\n... и еще {len(matching_docs) - 10} документов")

        # Предлагаем выбрать документ для анализа
        print(
            "\nВведите номер документа для анализа"
            "(1-10) или 'пропустить' для нового поиска:"
        )
        choice = input("> ").strip().lower()

        if choice == 'пропустить':
            continue

        try:
            choice_num = int(choice)
            if 1 <= choice_num <= min(10, len(matching_docs)):
                # Получаем индекс выбранного документа
                selected_doc_idx = matching_docs[choice_num - 1]

                # Анализируем документ
                analyze_document(extractor, selected_doc_idx)

                # После анализа возвращаемся к поиску
                print("\nНажмите Enter для продолжения поиска...")
                input()
            else:
                print("Неверный номер. Пожалуйста, выберите номер из списка.")
        except ValueError:
            print("Пожалуйста, введите номер или 'пропустить'.")


def main() -> None:
    """Главная функция программы."""
    display_welcome()

    # Инициализируем экстрактор
    extractor = initialize_extractor()

    # Главный цикл программы
    while True:
        print("\n" + "=" * 60)
        print("ГЛАВНОЕ МЕНЮ")
        print("=" * 60)
        print("\nВыберите действие:")
        print("1. Найти статьи по слову")
        print("2. Проанализировать случайную статью")
        print("3. Показать информацию о данных")
        print("4. Выйти из программы")

        choice = input("\nВаш выбор (1-4): ").strip()

        if choice == '1':
            search_word_interactive(extractor)

        elif choice == '2':
            # Выбираем случайный документ
            import numpy as np
            np.random.seed()
            random_doc_idx = np.random.choice(len(extractor.df))

            print(f"\nСлучайный документ #{random_doc_idx}:")
            analyze_document(extractor, random_doc_idx)
            print("\nНажмите Enter для возврата в меню...")
            input()

        elif choice == '3':
            print("\n" + "=" * 60)
            print("ИНФОРМАЦИЯ О ДАННЫХ")
            print("=" * 60)
            print(f"\n{extractor}")

            # Показываем распределение по категориям
            if extractor.df is not None:
                print("\nРаспределение документов по категориям:")
                category_counts = extractor.df['labels'].value_counts()
                for category, count in category_counts.items():
                    print(f"  {category}: {count} документов")

            print("\nНажмите Enter для возврата в меню...")
            input()

        elif choice == '4':
            print("\nСпасибо за использование программы! До свидания!")
            break

        else:
            print("Неверный выбор. Пожалуйста, выберите 1, 2, 3 или 4.")


if __name__ == "__main__":
    main()
