"""
Шаг 4: тот же набор тестов (eval_examples.py) и те же метрики
(accuracy, время ответа) — но через облачные LLM на OpenRouter,
для сравнения с локальными HuggingFace-моделями.

Две модели (сверено с https://openrouter.ai/api/v1/models на 2026-10-12):
- google/gemma-4-31b-it:free — бесплатный тир (лимиты на число запросов в день);
- deepseek/deepseek-chat     — платная, но копеечная ($0.32 / $0.89 за 1M токенов).

Перед запуском: скопируйте .env.example в .env и впишите OPENROUTER_API_KEY.

Запуск:
    python benchmark_cloud.py
"""

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from eval_examples import EXAMPLES
from metrics import print_confusion, summarize

load_dotenv()

CLOUD_MODELS = [
    # :free-версия 12.10 стабильно отвечала 429 (общий лимит у Google AI Studio) — платная, $0.09/1M
    "google/gemma-4-31b-it",
    "deepseek/deepseek-chat",
]

SYSTEM_PROMPT = (
    "Определи тональность текста одним словом: positive, "
    "negative или neutral. Ответь только этим словом."
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    timeout=60,
)


def classify(model: str, text: str) -> tuple[str, float]:
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=10,
        extra_headers={"HTTP-Referer": "https://localhost", "X-Title": "hf-sentiment-homework"},
    )
    elapsed = time.perf_counter() - start

    raw = (response.choices[0].message.content or "").strip().lower()
    predicted = next((c for c in ("positive", "negative", "neutral") if c in raw), raw)
    return predicted, elapsed


def run_model(model: str) -> list[dict]:
    results = []
    for ex in EXAMPLES:
        predicted, elapsed = classify(model, ex["text"])
        correct = predicted == ex["expected"]
        results.append(
            {
                "text": ex["text"],
                "expected": ex["expected"],
                "predicted": predicted,
                "correct": correct,
                "time_sec": round(elapsed, 3),
            }
        )
        mark = "OK " if correct else "ERR"
        print(f"  [{mark}] '{ex['text'][:45]}...' -> {predicted} (ожидали {ex['expected']}, {elapsed:.2f}s)")
    return results


def main():
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("Нет OPENROUTER_API_KEY: скопируйте .env.example в .env и впишите ключ.")

    all_summaries = {}
    all_raw = {}
    for model in CLOUD_MODELS:
        print(f"\n=== {model} ===")
        try:
            results = run_model(model)
        except Exception as e:  # бесплатная модель может упереться в лимит — фиксируем и идём дальше
            print(f"  Модель недоступна: {type(e).__name__}: {e}")
            all_summaries[model] = {"error": f"{type(e).__name__}: {e}"}
            continue
        summary = summarize(results)
        all_summaries[model] = summary
        all_raw[model] = results
        print(f"  Accuracy: {summary['accuracy_pct']}%, среднее время: {summary['avg_inference_sec']}с")
        print_confusion(summary["confusion"])

    with open("cloud_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    with open("cloud_benchmark_raw.json", "w", encoding="utf-8") as f:
        json.dump(all_raw, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
