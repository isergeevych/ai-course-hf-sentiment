"""
Шаг 3: прогоняет 3 выбранные модели (найденные в find_models.py, после
чтения Model Card и проверки лицензии) через transformers.pipeline(),
измеряет время инференса и accuracy на eval_examples.py.

Выбор моделей — три разных компромисса (подробности в README):
- seara/rubert-base-cased-russian-sentiment — «большая» RuBERT-base (178M),
  MIT, обучена на отзывах, соцсетях и новостях;
- cointegrated/rubert-tiny-sentiment-balanced — крошечная (12M), быстрая;
  лицензия в карточке НЕ указана;
- mxlcw/rubert-tiny2-russian-financial-sentiment — tiny2 (29M), MIT,
  дообучена на финансовых Telegram-постах (ближе всего к нашему домену).

LABEL_MAP: сырая метка модели в нижнем регистре -> positive/negative/neutral.
У всех трёх моделей метки уже такие (проверено по id2label в config.json).

Запуск:
    python benchmark_local.py
"""

import json
import time

from transformers import pipeline

from eval_examples import EXAMPLES
from metrics import print_confusion, summarize

IDENTITY = {"positive": "positive", "negative": "negative", "neutral": "neutral"}

MODELS = {
    "seara/rubert-base-cased-russian-sentiment": IDENTITY,
    "cointegrated/rubert-tiny-sentiment-balanced": IDENTITY,
    "mxlcw/rubert-tiny2-russian-financial-sentiment": IDENTITY,
}


def run_model(model_id: str, label_map: dict) -> tuple[list[dict], int]:
    clf = pipeline("text-classification", model=model_id, device="cpu")
    n_params = clf.model.num_parameters()

    # Прогрев: первый вызов включает ленивую инициализацию и выделение памяти,
    # без него среднее время по 12 примерам заметно завышено.
    clf("прогрев")

    results = []
    for ex in EXAMPLES:
        start = time.perf_counter()
        raw = clf(ex["text"])[0]
        elapsed = time.perf_counter() - start

        predicted = label_map.get(raw["label"].lower(), raw["label"].lower())
        results.append(
            {
                "text": ex["text"],
                "expected": ex["expected"],
                "predicted": predicted,
                "raw_label": raw["label"],
                "confidence": round(raw["score"], 3),
                "correct": predicted == ex["expected"],
                "time_sec": round(elapsed, 4),
            }
        )
    return results, n_params


def main():
    all_summaries = {}
    all_raw = {}

    for model_id, label_map in MODELS.items():
        print(f"\n=== {model_id} ===")
        results, n_params = run_model(model_id, label_map)
        for r in results:
            mark = "OK " if r["correct"] else "ERR"
            print(
                f"  [{mark}] '{r['text'][:45]}...' -> {r['predicted']} "
                f"(ожидали {r['expected']}, conf={r['confidence']}, {r['time_sec']}s)"
            )
        summary = summarize(results)
        summary["params_millions"] = round(n_params / 1e6, 1)
        all_summaries[model_id] = summary
        all_raw[model_id] = results
        print(f"  Accuracy: {summary['accuracy_pct']}%, среднее время: {summary['avg_inference_sec']}с")
        print_confusion(summary["confusion"])

    with open("local_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    with open("local_benchmark_raw.json", "w", encoding="utf-8") as f:
        json.dump(all_raw, f, ensure_ascii=False, indent=2)

    print("\n\n=== ИТОГО ===")
    for model_id, s in all_summaries.items():
        print(
            f"{model_id:<50} params={s['params_millions']}M  "
            f"accuracy={s['accuracy_pct']}%  время={s['avg_inference_sec']}с  по классам={s['per_class_accuracy_pct']}"
        )


if __name__ == "__main__":
    main()
