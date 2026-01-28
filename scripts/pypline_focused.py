#!/usr/bin/env python3
"""
Фокусированная генерация датасета по интентам.
Генерирует максимально разнообразные фразы для каждого интента отдельно.
"""

import argparse
import csv
import json
import random
import re
import time
import sys
from typing import List, Dict
from pathlib import Path

# Добавляем родительскую директорию в путь для импорта config
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from pydantic import BaseModel
from config.key_focused import OPENAI_API_KEY, SYSTEM_REQ_FOCUSED, INTENT_SPECS


class TextCarDataset(BaseModel):
    phrase: str
    intent: str
    parameters: str


def normalize_phrase_dedup(phrase: str) -> str:
    """Нормализация для детекции дубликатов"""
    normalized = phrase.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def extract_first_json(text: str):
    """Извлечение JSON из ответа модели"""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("` \n")
        if t.lower().startswith("json"):
            t = t[4:].lstrip()

    start_obj = t.find("{")
    start_arr = t.find("[")

    if start_arr == -1 and start_obj == -1:
        raise ValueError("No JSON found")

    if start_arr != -1 and (start_obj == -1 or start_arr < start_obj):
        end = t.rfind("]")
        if end == -1:
            raise ValueError("Unclosed array")
        json_str = t[start_arr:end+1]
    else:
        end = t.rfind("}")
        if end == -1:
            raise ValueError("Unclosed object")
        json_str = t[start_obj:end+1]

    try:
        result = json.loads(json_str)
        return result
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")


def build_prompt_for_intent(intent_name: str, intent_spec: Dict, n_samples: int, seed: int) -> str:
    """Создание промта для конкретного интента"""
    hints_text = "\n".join([f"   - {hint}" for hint in intent_spec.get("hints", [])])

    prompt = SYSTEM_REQ_FOCUSED + f"""

═══════════════════════════════════════════════════════════════
ТЕКУЩЕЕ ЗАДАНИЕ
═══════════════════════════════════════════════════════════════

Интент: {intent_name}
Описание: {intent_spec.get("description", "")}
Параметры: {intent_spec.get("parameters", "")}

Количество фраз: {n_samples}

ПОДСКАЗКИ ДЛЯ РАЗНООБРАЗИЯ:
{hints_text}

Сгенерируй {n_samples} МАКСИМАЛЬНО РАЗНООБРАЗНЫХ фраз для интента "{intent_name}".
Используй ВСЕ техники: короткие, длинные, вопросы, команды, сленг, эмоции.

# seed:{seed}
"""
    return prompt


def call_model_for_intent(
    client: OpenAI,
    intent_name: str,
    intent_spec: Dict,
    n_samples: int,
    model: str,
    temperature: float,
    top_p: float,
    seed: int
) -> List[Dict]:
    """Вызов модели для генерации фраз одного интента"""
    prompt = build_prompt_for_intent(intent_name, intent_spec, n_samples, seed)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Ты креативный эксперт по генерации разнообразных фраз."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=top_p,
        seed=seed
    )

    text = response.choices[0].message.content.strip()
    result = extract_first_json(text)

    if isinstance(result, dict):
        return [result]
    return result


def generate_diverse_dataset(
    intents: Dict[str, Dict],
    samples_per_intent: int,
    model: str = "gpt-4o-mini",
    temperature: float = 1.2,
    top_p: float = 0.95
) -> List[TextCarDataset]:
    """Генерация датасета с фокусом на разнообразие внутри каждого интента"""

    client = OpenAI(api_key=OPENAI_API_KEY)
    all_results = []
    seen_phrases = set()

    print(f"\n🎯 Фокусированная генерация: {samples_per_intent} фраз × {len(intents)} интентов")
    print(f"   Модель: {model}, Temperature: {temperature}, Top-p: {top_p}\n")

    for intent_name, intent_spec in intents.items():
        print(f"📝 Генерация интента: {intent_name}")
        intent_results = []
        attempts = 0
        max_attempts = samples_per_intent * 3  # Максимум попыток

        while len(intent_results) < samples_per_intent and attempts < max_attempts:
            attempts += 1
            seed = random.randint(1, 1_000_000)

            # Генерируем батч (запрашиваем больше, чем нужно)
            batch_size = min(10, samples_per_intent - len(intent_results) + 5)

            try:
                raw_list = call_model_for_intent(
                    client, intent_name, intent_spec,
                    batch_size, model, temperature, top_p, seed
                )

                for raw in raw_list:
                    if len(intent_results) >= samples_per_intent:
                        break

                    try:
                        item = TextCarDataset(**raw)
                        phrase_norm = normalize_phrase_dedup(item.phrase)

                        # Проверка на дубликаты
                        if phrase_norm in seen_phrases:
                            print(f"   ⚠ Дубликат: '{item.phrase}'")
                            continue

                        # Проверка, что intent совпадает
                        if item.intent.lower() != intent_name.lower():
                            print(f"   ⚠ Неверный интент: {item.intent} (ожидался {intent_name})")
                            continue

                        seen_phrases.add(phrase_norm)
                        intent_results.append(item)
                        print(f"   ✓ [{len(intent_results)}/{samples_per_intent}] {item.phrase}")

                    except Exception as e:
                        print(f"   ⚠ Ошибка парсинга: {e}")
                        continue

                time.sleep(0.5)  # Пауза между запросами

            except Exception as e:
                print(f"   ⚠ Ошибка API: {e}")
                time.sleep(2)
                continue

        if len(intent_results) < samples_per_intent:
            print(f"   ⚠ Получено только {len(intent_results)}/{samples_per_intent} уникальных фраз")

        all_results.extend(intent_results)
        print(f"   ✅ Интент {intent_name}: {len(intent_results)} фраз\n")

    return all_results


def save_csv(path: str, items: List[TextCarDataset]):
    """Сохранение в CSV"""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phrase", "intent", "parameters"])
        for item in items:
            writer.writerow([item.phrase, item.intent, item.parameters])
    print(f"\n💾 Сохранено {len(items)} записей в {path}")


def main():
    parser = argparse.ArgumentParser(description="Фокусированная генерация датасета по интентам")
    parser.add_argument("--csv", type=str, default="dataset_focused.csv", help="Путь к выходному CSV")
    parser.add_argument("--samples", type=int, default=50, help="Количество фраз на каждый интент")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Модель OpenAI")
    parser.add_argument("--temperature", type=float, default=1.2, help="Temperature (чем выше, тем разнообразнее)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--intents", type=str, nargs="+", help="Список интентов (по умолчанию все)")

    args = parser.parse_args()

    # Фильтрация интентов, если указано
    if args.intents:
        intents = {k: v for k, v in INTENT_SPECS.items() if k in args.intents}
    else:
        intents = INTENT_SPECS

    if not intents:
        print("❌ Не найдено интентов для генерации")
        return

    print(f"\n🚀 Старт фокусированной генерации")
    print(f"   Интенты: {', '.join(intents.keys())}")
    print(f"   Фраз на интент: {args.samples}")
    print(f"   Всего фраз: {len(intents) * args.samples}")

    items = generate_diverse_dataset(
        intents=intents,
        samples_per_intent=args.samples,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p
    )

    save_csv(args.csv, items)

    # Статистика
    unique_phrases = len(set(normalize_phrase_dedup(item.phrase) for item in items))
    duplicate_rate = (1 - unique_phrases / len(items)) * 100 if items else 0

    print(f"\n📊 Статистика:")
    print(f"   Всего фраз: {len(items)}")
    print(f"   Уникальных: {unique_phrases}")
    print(f"   Дубликатов: {duplicate_rate:.1f}%")
    print(f"\n✅ Готово!")


if __name__ == "__main__":
    main()
