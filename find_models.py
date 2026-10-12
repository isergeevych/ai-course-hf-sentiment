"""
Шаг 2: поиск кандидатов на HuggingFace Hub для задачи "классификация
тональности" на русском языке, через HfApi.list_models().

Скрипт только ищет и печатает кандидатов — выбор 3 моделей, чтение их
Model Card и проверку лицензии нужно сделать самостоятельно, перейдя по
ссылкам (это явное требование шага 2 задания).

Запуск:
    python find_models.py
"""

from huggingface_hub import HfApi

# В версиях huggingface_hub>=1.0 параметр называется pipeline_tag (не task),
# а прямого фильтра по языку нет — язык проверяем по тегу "ru" в m.tags.
SEARCH_TERMS = ("sentiment", "тональность", "emotion")
PIPELINE_TAG = "text-classification"
LANGUAGE_TAG = "ru"
LIMIT = 100


def find_candidates():
    api = HfApi()
    seen_ids = set()
    candidates = []

    for term in SEARCH_TERMS:
        models = api.list_models(
            pipeline_tag=PIPELINE_TAG,
            search=term,
            sort="downloads",
            limit=LIMIT,
        )
        for m in models:
            if m.id in seen_ids:
                continue
            if LANGUAGE_TAG in (m.tags or []):
                seen_ids.add(m.id)
                candidates.append(m)

    candidates.sort(key=lambda m: getattr(m, "downloads", 0) or 0, reverse=True)
    return candidates


def main():
    candidates = find_candidates()
    print(f"Найдено {len(candidates)} кандидатов (pipeline_tag={PIPELINE_TAG}, язык={LANGUAGE_TAG}):\n")

    for m in candidates:
        # Лицензия публикуется тегом "license:<id>"; отсутствие тега = лицензия
        # не указана, т.е. по умолчанию «все права защищены» — для продукта риск.
        licenses = [t.removeprefix("license:") for t in (m.tags or []) if t.startswith("license:")]
        print(f"- {m.id}")
        print(f"    license: {', '.join(licenses) or 'НЕ УКАЗАНА'}")
        print(f"    downloads: {getattr(m, 'downloads', '?')}, likes: {getattr(m, 'likes', '?')}")
        print(f"    tags: {', '.join(m.tags or [])[:120]}")
        print(f"    Model Card: https://huggingface.co/{m.id}\n")

    print(
        "Дальше вручную: откройте карточки 3 моделей по ссылкам выше, "
        "прочитайте Model Card (архитектура, на чём обучена, метки классов) "
        "и проверьте лицензию (смотрите поле 'license' на странице модели или "
        "в файле README/model card). Впишите выбранные ID моделей и метки "
        "в MODELS в benchmark_local.py."
    )


if __name__ == "__main__":
    main()
