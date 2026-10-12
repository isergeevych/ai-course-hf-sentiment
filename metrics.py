"""
Общие метрики для benchmark_local.py и benchmark_cloud.py — чтобы локальные
и облачные модели сравнивались строго одинаково.
"""

CLASSES = ("positive", "negative", "neutral")


def summarize(results: list[dict]) -> dict:
    accuracy = sum(r["correct"] for r in results) / len(results) * 100
    avg_time = sum(r["time_sec"] for r in results) / len(results)
    return {
        "accuracy_pct": round(accuracy, 1),
        "avg_inference_sec": round(avg_time, 4),
        "per_class_accuracy_pct": per_class_accuracy(results),
        "confusion": confusion(results),
    }


def per_class_accuracy(results: list[dict]) -> dict:
    """Доля верных ответов внутри каждого истинного класса — видно, что именно путает модель."""
    out = {}
    for cls in CLASSES:
        subset = [r for r in results if r["expected"] == cls]
        if subset:
            out[cls] = round(sum(r["correct"] for r in subset) / len(subset) * 100, 1)
    return out


def confusion(results: list[dict]) -> dict:
    """Матрица ошибок: {ожидали: {предсказали: сколько}}."""
    matrix = {exp: {pred: 0 for pred in CLASSES} for exp in CLASSES}
    for r in results:
        pred = r["predicted"] if r["predicted"] in CLASSES else "neutral"
        matrix[r["expected"]][pred] += 1
    return matrix


def print_confusion(matrix: dict) -> None:
    header = "ожидали / предсказали"
    print(f"    {header:<24}" + "".join(f"{c:>10}" for c in CLASSES))
    for exp in CLASSES:
        print(f"    {exp:<24}" + "".join(f"{matrix[exp][p]:>10}" for p in CLASSES))
