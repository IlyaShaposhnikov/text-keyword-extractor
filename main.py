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
    extractor = KeywordExtractor(remove_stopwords=True)

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

        # Получаем все тексты
        all_texts = extractor.df['text'].tolist()

        # Используем sklearn с параметрами, которые
        # максимально приближены к ручной реализации
        # Создаем векторайзер
        sklearn_vectorizer = TfidfVectorizer(
            lowercase=True,
            tokenizer=extractor.preprocess_text,
            # Отключаем стандартную токенизацию
            token_pattern=None,
            # Отключаем сглаживание IDF
            smooth_idf=False,
            # Отключаем нормализацию L2
            norm=None,
            # Используем обычную TF
            sublinear_tf=False,
            # Минимальная длина документа для DF
            min_df=1,
            # Максимальная DF (все документы)
            max_df=1.0
        )

        # Обучаем векторайзер
        sklearn_tfidf_matrix = sklearn_vectorizer.fit_transform(all_texts)
        # Получаем TF-IDF веса для выбранного документа
        sklearn_doc_tfidf = sklearn_tfidf_matrix[doc_index]
        # Получаем названия признаков (слов)
        sklearn_feature_names = sklearn_vectorizer.get_feature_names_out()

        # Сортируем слова по TF-IDF весу
        sklearn_sorted_indices = sklearn_doc_tfidf.toarray()[0].argsort()[::-1]

        # Берем топ-10 слов из sklearn
        sklearn_keywords = []
        for idx in sklearn_sorted_indices[:10]:
            word = sklearn_feature_names[idx]
            weight = sklearn_doc_tfidf[0, idx]
            sklearn_keywords.append((word, weight))

        # Сравнение с sklearn с настройками по умолчанию
        # Создаем векторайзер
        default_vectorizer = TfidfVectorizer(
            lowercase=True,
            tokenizer=extractor.preprocess_text,
            # Отключаем стандартную токенизацию
            token_pattern=None
        )

        # Обучаем векторайзер
        default_tfidf = default_vectorizer.fit_transform(all_texts)
        # Получаем TF-IDF веса для выбранного документа
        default_doc_tfidf = default_tfidf[doc_index]
        # Получаем названия признаков (слов)
        default_features = default_vectorizer.get_feature_names_out()
        # Сортируем слова по TF-IDF весу
        default_sorted_indices = default_doc_tfidf.toarray()[0].argsort()[::-1]

        # Берем топ-10 слов из sklearn с настройками по умолчанию
        default_keywords = []
        for idx in default_sorted_indices[:10]:
            word = default_features[idx]
            weight = default_doc_tfidf[0, idx]
            default_keywords.append((word, weight))

        # Выводим сравнение в одной таблице с тремя столбцами
        print("\n" + "=" * 80)
        print("СРАВНЕНИЕ ТРЕХ РЕАЛИЗАЦИЙ TF-IDF")
        print("=" * 80)

        # Заголовки столбцов
        extractor_header = "Ручная реализация"
        sklearn_header = "Sklearn (настроенный)"
        default_header = "Sklearn (по умолчанию)"

        # Определяем максимальную ширину для выравнивания
        column_width = 25

        print(f"\n{extractor_header:<{column_width}} "
              f"{sklearn_header:<{column_width}} "
              f"{default_header:<{column_width}}")

        # Разделители
        print(f"{'-'*column_width:<{column_width}} "
              f"{'-'*column_width:<{column_width}} "
              f"{'-'*column_width:<{column_width}}")

        # Выводим топ-10 ключевых слов для каждой реализации
        for i in range(10):
            # Ручная реализация
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
            extractor_str = f"{extractor_word} ({extractor_weight:.4f})"

            # Sklearn с настроенными параметрами
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
            sklearn_str = f"{sklearn_word} ({sklearn_weight:.4f})"

            # Sklearn по умолчанию
            default_word = (
                default_keywords[i][0]
                if i < len(default_keywords)
                else ""
            )
            default_weight = (
                default_keywords[i][1]
                if i < len(default_keywords)
                else 0
            )
            default_str = f"{default_word} ({default_weight:.4f})"

            print(f"{i+1:2}. {extractor_str:<{column_width}} "
                  f"{sklearn_str:<{column_width}} "
                  f"{default_str:<{column_width}}")

        # Объяснение различий
        print("\n" + "=" * 60)
        print("ПОЧЕМУ РЕЗУЛЬТАТЫ РАЗЛИЧАЮТСЯ:")
        print("=" * 60)
        print("1. Формула IDF:")
        print("   • Ручная реализация: idf = log(N / df)")
        print("   • Sklearn (по умолчанию): idf = log((1 + N) / (1 + df)) + 1")
        print("   • Sklearn (smooth_idf=False): idf = log(N / df) + 1")
        print("\n2. Нормализация:")
        print("   • Ручная реализация: нет нормализации")
        print("   • Sklearn (по умолчанию): L2 нормализация")
        print("\n3. Обработка TF:")
        print("   • Ручная реализация: raw counts")
        print("   • Sklearn (по умолчанию): raw counts")
        print("   • Sklearn (sublinear_tf=True): 1 + log(TF)")
        print("\n4. Удаление стоп-слов:")
        print("   • Все реализации используют одинаковый препроцессинг")
        print("   • Удаляются слова короче 3 символов")
        print("   • Удаляются стандартные английские стоп-слова")

        # Сравнение с исправленной формулой IDF (без +1)
        print("\n" + "=" * 60)
        print("ВАЖНОЕ ЗАМЕЧАНИЕ:")
        print("=" * 60)
        print("Sklearn добавляет +1 к IDF даже при smooth_idf=False")
        print("Это означает, что даже стоп-слова получают ненулевой вес!")
        print("В ручной реализации стоп-слова имеют вес близкий к 0")

    except ImportError:
        print("\nБиблиотека scikit-learn не установлена.")
        print("Установите её для сравнения: pip install scikit-learn")
    except Exception as e:
        print(f"\nОшибка при сравнении: {e}")


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
            f"\nНайдено {len(matching_docs)} "
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
