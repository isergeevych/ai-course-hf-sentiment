"""
Шаг 5: Gradio-демо для лучшей модели по результатам бенчмарка
(local_benchmark_summary.json / cloud_benchmark_summary.json).

Перед запуском впишите ID модели-победителя в BEST_MODEL.

Запуск:
    python gradio_demo.py

После запуска в консоли появится публичная ссылка вида
https://xxxxx.gradio.live (живёт 72 часа, share=True) — её и нужно
"расшарить" по заданию. Для постоянной ссылки — задеплойте на
HuggingFace Spaces (huggingface.co/new-space, Gradio SDK, скопировать
этот файл как app.py).
"""

import gradio as gr
from transformers import pipeline

# Лучшая по итогам benchmark_local.py: accuracy 83.3%, ~1.4 мс на пример (CPU), MIT
BEST_MODEL = "mxlcw/rubert-tiny2-russian-financial-sentiment"
MAX_CHARS = 2000  # демо публичное — ограничиваем размер ввода

clf = pipeline("text-classification", model=BEST_MODEL, device="cpu", top_k=None)


def predict(text: str) -> dict:
    """Возвращает вероятности всех трёх классов — gr.Label рисует их полосками."""
    text = text.strip()[:MAX_CHARS]
    if not text:
        return {}
    return {r["label"]: r["score"] for r in clf(text)[0]}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Текст обращения пользователя",
        placeholder="Например: Меня расстраивает, что расходы выросли в этом месяце...",
        lines=3,
    ),
    outputs=gr.Label(label="Тональность", num_top_classes=3),
    title="Анализ тональности обращений — финансовый AI-агент",
    description=(
        "Демо лучшей модели по результатам сравнения (см. отчёт проекта "
        "hf_sentiment_homework). Определяет эмоциональный тон обращения "
        "пользователя к финансовому боту."
    ),
    examples=[
        "Спасибо, теперь я наконец понимаю, куда уходят мои деньги!",
        "Приложение постоянно вылетает, это невыносимо.",
        "Добавил новый источник дохода в настройках.",
    ],
)

if __name__ == "__main__":
    demo.launch(share=True)
