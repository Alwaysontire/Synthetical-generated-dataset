"""
Обработка готового batch_output.jsonl файла в JSON и CSV форматы.
Использует ту же логику, что и в batch_generation.py
"""

import json
import csv
import re
from collections import Counter
from build_anchors import INTENTS


def extract_from_batch(body: dict) -> str:
    """Извлекает content из batch response body"""
    choices = body.get("choices")
    if isinstance(choices, list) and len(choices) > 0:
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
    return None


def normalize_phrase(s: str) -> str:
    """Нормализация фразы для дедупликации"""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[«»""\"']", "", s)
    s = re.sub(r"[?!.,;:]+$", "", s)
    return s


def metrics(items: list[dict]) -> dict:
    """Вычисление метрик качества датасета"""
    phrases = [x["phrase"] for x in items]
    tokens = [w for p in phrases for w in re.findall(r"[а-яёa-z0-9]+", p.lower())]
    ttr = len(set(tokens)) / max(1, len(tokens))

    lens = [len(re.findall(r"[а-яёa-z0-9]+", p.lower())) for p in phrases]
    mean_len = sum(lens) / max(1, len(lens))
    var = sum((l - mean_len) ** 2 for l in lens) / max(1, len(lens))
    std_len = var ** 0.5

    starts = []
    for p in phrases:
        w = re.findall(r"[а-яёa-z0-9]+", p.lower())
        if w:
            starts.append(w[0])
    c = Counter(starts)
    top1_share = (c.most_common(1)[0][1] / max(1, len(starts))) if starts else 0.0

    polite = 0
    for p in phrases:
        pl = p.lower()
        if ("пожалуйста" in pl) or ("можно" in pl) or ("не мог" in pl) or ("не могли" in pl) or pl.strip().endswith("?"):
            polite += 1
    polite_share = polite / max(1, len(phrases))

    return {
        "n": len(items),
        "ttr": ttr,
        "len_std": std_len,
        "top1_start_share": top1_share,
        "polite_share": polite_share,
        "top_starts": c.most_common(10),
    }


def json_to_csv(json_path: str, csv_path: str = None) -> str:
    """Конвертирует JSON файл в CSV формат"""
    if csv_path is None:
        csv_path = json_path.replace('.json', '.csv')

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['phrase', 'intent', 'parameters'])
        writer.writeheader()
        writer.writerows(data)

    print(f"\n💾 Конвертировано: {json_path} → {csv_path}")
    print(f"   Записей: {len(data)}")

    return csv_path


def process_batch_output(
    input_path: str = "batch_output.jsonl",
    out_path: str = "batch_10k.json",
    raw_out_path: str = "raw_batch_10k.json",
    target_n: int = 10000
):
    """
    Обрабатывает batch_output.jsonl и создает финальные JSON и CSV файлы

    Args:
        input_path: Путь к batch_output.jsonl
        out_path: Путь к финальному JSON файлу
        raw_out_path: Путь к сырому JSON файлу (с intent_raw)
        target_n: Целевое количество записей
    """
    print(f"\n{'='*60}")
    print(f"ОБРАБОТКА BATCH OUTPUT")
    print(f"{'='*60}")
    print(f"Входной файл: {input_path}")
    print(f"Целевое количество: {target_n}")

    seen = set()
    collected = []
    failed_rows = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            row = json.loads(line)
            if row.get("error"):
                failed_rows += 1
                continue

            body = row["response"]["body"]
            text = extract_from_batch(body)
            if not text:
                failed_rows += 1
                continue

            try:
                response_obj = json.loads(text)
                chunk_items = response_obj.get("items", [])
            except Exception as e:
                failed_rows += 1
                print(f"⚠ Ошибка парсинга строки {line_num}: {e}")
                continue

            if not isinstance(chunk_items, list):
                failed_rows += 1
                continue

            for ex in chunk_items:
                # Защита: пропускаем если элемент не словарь
                if not isinstance(ex, dict):
                    continue

                phrase = ex.get("phrase")
                if not phrase or not isinstance(phrase, str):
                    continue

                phrase = phrase.strip()
                if not phrase:
                    continue

                key = normalize_phrase(phrase)
                if key in seen:
                    continue
                seen.add(key)

                intent = ex.get("intent")
                intent_raw = ex.get("intent_raw")
                params = ex.get("parameters") or {}

                if intent not in INTENTS:
                    continue
                if not isinstance(params, dict):
                    params = {}

                collected.append({
                    "phrase": phrase,
                    "intent_raw": intent_raw,
                    "intent": intent,
                    "parameters": params,
                })

    print(f"\n📊 Статистика обработки:")
    print(f"   Уникальных записей собрано: {len(collected)}")
    print(f"   Ошибок/пропусков: {failed_rows}")

    # Сохраняем сырой JSON с intent_raw
    with open(raw_out_path, "w", encoding="utf-8") as f:
        json.dump(collected[:target_n], f, ensure_ascii=False, indent=2)
    print(f"\n✓ Сырой JSON сохранен: {raw_out_path}")

    # Создаем финальный датасет (без intent_raw, parameters как строка)
    final = []
    for ex in collected:
        params_obj = ex["parameters"]
        params_str = "" if params_obj == {} else json.dumps(params_obj, ensure_ascii=False)
        final.append({
            "phrase": ex["phrase"],
            "intent": ex["intent"],
            "parameters": params_str
        })
    final = final[:target_n]

    # Сохраняем финальный JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(f"✓ Финальный JSON сохранен: {out_path}")
    print(f"   Итоговое количество записей: {len(final)}")

    # Вычисляем метрики
    metric = metrics(final)
    print(f"\n📈 Метрики датасета:")
    print(json.dumps(metric, ensure_ascii=False, indent=2))

    # Конвертируем в CSV
    csv_path = json_to_csv(out_path)
    print(f"\n✅ Готово! CSV файл: {csv_path}")

    return final


if __name__ == "__main__":
    # Обрабатываем batch_output.jsonl
    process_batch_output(
        input_path="batch_output.jsonl",
        out_path="batch_10k.json",
        raw_out_path="raw_batch_10k.json",
        target_n=10000
    )
