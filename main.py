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
        import numpy as np

        print("\n" + "=" * 60)
        print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ TF-IDF РЕАЛИЗАЦИЙ")
        print("=" * 60)

        # Получаем все тексты
        all_texts = extractor.df['text'].tolist()

        # 1. РУЧНАЯ РЕАЛИЗАЦИЯ
        manual_keywords = extractor_keywords

        # 2. SKLEARN ПО УМОЛЧАНИЮ
        sklearn_default = TfidfVectorizer(
            lowercase=True,
            tokenizer=extractor.preprocess_text,
            # Отключаем стандартную токенизацию
            token_pattern=None
        )

        # Обучаем векторайзер
        sklearn_matrix = sklearn_default.fit_transform(all_texts)
        # Получаем названия признаков (слов)
        sklearn_features = sklearn_default.get_feature_names_out()
        # Получаем TF-IDF веса для выбранного документа
        sklearn_doc_weights = sklearn_matrix[doc_index].toarray()[0]

        # Сортируем слова по TF-IDF весу
        sklearn_sorted = np.argsort(sklearn_doc_weights)[::-1][:10]
        sklearn_keywords = []
        for idx in sklearn_sorted:
            word = sklearn_features[idx]
            weight = sklearn_doc_weights[idx]
            sklearn_keywords.append((word, weight))

        print("\n1. Ручная реализация:")
        print("   IDF = log(N / df)")
        print("   Без нормализации")
        print("   Стоп-слова имеют вес ~0")

        print("\n2. Sklearn:")
        print("   IDF = log((1 + N) / (1 + df)) + 1")
        print("   L2 нормализация")
        print("   Сглаживание IDF")

        # Для наглядности выведем разницу в весах для одинаковых слов
        print("\n" + "=" * 60)
        print("СРАВНЕНИЕ ВЕСОВ ОДИНАКОВЫХ СЛОВ:")
        print("=" * 60)

        # Создаем словарь весов sklearn для быстрого доступа
        sklearn_weights = {word: weight for word, weight in sklearn_keywords}

        print(
            f"\n{'Слово':<20} {'Вес':<12} "
            f"{'Sklearn вес':<12} {'Отношение':<12}"
        )
        print("-" * 56)

        # Берем топ-5 слов из нашей реализации и ищем их в sklearn
        for manual_word, manual_weight in manual_keywords[:5]:
            sklearn_weight = sklearn_weights.get(manual_word, 0)
            if sklearn_weight > 0:
                ratio = (
                    manual_weight / sklearn_weight
                    if sklearn_weight != 0
                    else float('inf')
                )
                print(
                    f"{manual_word:<20} {manual_weight:<12.4f} "
                    f"{sklearn_weight:<12.4f} {ratio:<12.2f}"
                )

        # 4. ВЫВОДИМ ДВА СТОЛБЦА С ПРИМЕЧАНИЯМИ
        print("\n" + "=" * 60)
        print("ТОП-10 КЛЮЧЕВЫХ СЛОВ:")
        print("=" * 60)

        print(f"\n{'Ручная реализация':<30} {'Sklearn':<30}")
        print(f"{'-'*30:<30} {'-'*30:<30}")

        for i in range(10):
            # Ручная реализация
            manual_word = (
                manual_keywords[i][0]
                if i < len(manual_keywords)
                else "—"
            )
            manual_weight = (
                manual_keywords[i][1]
                if i < len(manual_keywords)
                else 0
            )
            manual_str = f"{manual_word} ({manual_weight:.4f})"

            # Sklearn
            sklearn_word = (
                sklearn_keywords[i][0]
                if i < len(sklearn_keywords)
                else "—"
            )
            sklearn_weight = (
                sklearn_keywords[i][1]
                if i < len(sklearn_keywords)
                else 0
            )
            sklearn_str = f"{sklearn_word} ({sklearn_weight:.4f})"

            # Подсветим различия
            if (
                manual_word != sklearn_word
                and manual_word != "—"
                and sklearn_word != "—"
            ):
                manual_str = f"*{manual_str}"
                sklearn_str = f"*{sklearn_str}"

            print(f"{i+1:2}. {manual_str:<30} {sklearn_str:<30}")

        # 5. ПОКАЗЫВАЕМ РАЗНИЦУ В ФОРМУЛАХ НА КОНКРЕТНОМ ПРИМЕРЕ
        print("\n" + "=" * 60)
        print("ПОЧЕМУ ВЕСА РАЗНЫЕ? РАЗБОР НА ПРИМЕРЕ:")
        print("=" * 60)

        # Найдем слово, которое есть в обоих списках
        common_words = set(
            word for word, _ in manual_keywords[:5]
        ).intersection(
            set(word for word, _ in sklearn_keywords[:5])
        )

        if common_words:
            example_word = next(iter(common_words))

            # Получаем TF из матрицы
            if example_word in extractor.word2idx:
                word_idx = extractor.word2idx[example_word]

                # Ручная реализация
                manual_tf = extractor.tf_matrix[doc_index, word_idx]
                manual_df = np.sum(extractor.tf_matrix[:, word_idx] > 0)
                manual_idf = extractor.idf_vector[word_idx]
                manual_tfidf = manual_tf * manual_idf

                # Sklearn
                print(f"\nСлово: '{example_word}'")
                print("\nРУЧНАЯ РЕАЛИЗАЦИЯ:")
                print(f"  TF: {manual_tf:.0f}")
                print(f"  DF: {manual_df:.0f}")
                print(
                    f"  IDF = log(N / df) = log({len(extractor.df)} "
                    f"/ {manual_df}) = {manual_idf:.4f}"
                )
                print(
                    f"  TF-IDF = TF × IDF = {manual_tf:.0f} × "
                    f"{manual_idf:.4f} = {manual_tfidf:.4f}"
                )

                print("\nSKLEARN:")
                print(f"  TF: {manual_tf:.0f}")
                print(f"  DF: {manual_df:.0f}")
                print(
                    "  IDF = log((1 + N) / (1 + df)) + 1 = "
                    f"log((1 + {len(extractor.df)}) / (1 + {manual_df})) + 1"
                )
                sklearn_idf_val = (
                    np.log((1 + len(extractor.df)) / (1 + manual_df)) + 1
                )
                print(f"  IDF = {sklearn_idf_val:.4f}")
                print(
                    f"  TF-IDF (до нормализации) = {manual_tf:.0f} × "
                    f"{sklearn_idf_val:.4f} = "
                    f"{manual_tf * sklearn_idf_val:.4f}"
                )
                print(
                    "  После L2 нормализации: ~"
                    f"{sklearn_weights.get(example_word, 0):.4f}"
                )

        # 6. ВЫВОДЫ
        print("=" * 60)
        print("\nРучная реализация лучше для:")
        print("   • Чистой математики TF-IDF")
        print("   • Нулевых весов у стоп-слов")
        print("   • Прозрачности расчетов")

        print("\nSklearn лучше для:")
        print("   • Устойчивости к нулевым значениям")
        print("   • Нормализованных векторов")
        print("   • Совместимости с другими алгоритмами sklearn")

        print("\nВывод: Обе реализации имеют свои преимущества.")
        print("Ручная реализация - 'чистый' TF-IDF.")
        print(
            "Sklearn - производственная версия "
            "с дополнительной стабильностью."
        )

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
    """Интерактивный поиск документов по слову с пагинацией."""
    print("\n" + "=" * 60)
    print("ПОИСК СТАТЕЙ ПО СЛОВУ")
    print("=" * 60)

    # Переменные для пагинации
    current_search_word = None
    current_matching_docs = []
    current_page = 0
    docs_per_page = 10

    while True:
        # Если у нас нет текущего поискового запроса, запрашиваем новое слово
        if not current_search_word:
            print(
                "\nВведите слово для поиска "
                "(или 'назад' для возврата в меню):"
            )
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

            # Сохраняем результаты поиска для пагинации
            current_search_word = user_input
            current_matching_docs = matching_docs
            current_page = 0

            print(f"\nНайдено {len(matching_docs)} "
                  f"документов со словом '{user_input}':")

        # Вычисляем диапазон документов для текущей страницы
        total_pages = (len(current_matching_docs) - 1) // docs_per_page + 1
        start_idx = current_page * docs_per_page
        end_idx = min(start_idx + docs_per_page, len(current_matching_docs))

        # Показываем информацию о текущей странице
        print(f"\nСтраница {current_page + 1} из {total_pages}:")
        print(
            f"Документы {start_idx + 1}-{end_idx} "
            f"из {len(current_matching_docs)}"
        )

        # Показываем найденные документы для текущей страницы
        for i, doc_idx in enumerate(
            current_matching_docs[start_idx:end_idx], 1
        ):
            # Получаем информацию о документе
            label = extractor.df.iloc[doc_idx]['labels']
            preview = extractor.get_document_preview(doc_idx, 80)

            # Форматируем вывод
            print(f"\n[{i}] Документ #{doc_idx} (Тема: {label})")
            print(f"    {preview}")

        # Показываем доступные команды
        print("\n" + "-" * 60)
        print("Доступные команды:")
        print("• [1-10] - выбрать документ для анализа")

        if current_page < total_pages - 1:
            print("• 'далее' - следующая страница")

        if current_page > 0:
            print("• 'предыдущая' - предыдущая страница")

        print("• 'новый' - новый поиск")
        print("• 'назад' - вернуться в меню")
        print("-" * 60)

        # Запрашиваем выбор пользователя
        choice = input("\nВаш выбор: ").strip().lower()

        # Обработка команды "далее"
        if choice == 'далее':
            if current_page < total_pages - 1:
                current_page += 1
            else:
                print("Это последняя страница.")
            continue

        # Обработка команды "предыдущая"
        elif choice == 'предыдущая':
            if current_page > 0:
                current_page -= 1
            else:
                print("Это первая страница.")
            continue

        # Обработка команды "новый"
        elif choice == 'новый':
            current_search_word = None
            current_matching_docs = []
            current_page = 0
            continue

        # Обработка команды "назад"
        elif choice == 'назад':
            return

        # Обработка выбора документа по номеру
        else:
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= docs_per_page:
                    # Вычисляем фактический индекс в списке matching_docs
                    actual_idx = start_idx + (choice_num - 1)

                    if actual_idx < len(current_matching_docs):
                        # Получаем индекс выбранного документа
                        selected_doc_idx = current_matching_docs[actual_idx]

                        # Анализируем документ
                        analyze_document(extractor, selected_doc_idx)

                        # После анализа возвращаемся к той же странице
                        print("\nНажмите Enter для продолжения...")
                        input()
                    else:
                        print(
                            f"Документ #{choice_num} "
                            "недоступен на этой странице."
                        )
                else:
                    print(
                        "Пожалуйста, введите число от "
                        f"1 до {docs_per_page} или команду."
                    )
            except ValueError:
                print(
                    "Неверный ввод. Пожалуйста, "
                    "используйте одну из доступных команд."
                )


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
