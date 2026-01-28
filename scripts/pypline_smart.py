#!/usr/bin/env python3
"""
Умная генерация с контролем разнообразия параметров.
Отслеживает использованные значения параметров и запрещает повторы.
"""

import argparse
import csv
import json
import random
import re
import time
import sys
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple
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


def extract_parameters_values(parameters_str: str) -> Dict:
    """Извлечение значений параметров из JSON-строки"""
    if not parameters_str or parameters_str.strip() == "":
        return {}
    try:
        return json.loads(parameters_str)
    except:
        return {}


def get_parameter_key(intent: str, params: Dict) -> Tuple:
    """Создание ключа для отслеживания использованных комбинаций параметров"""
    if not params:
        return (intent, "NO_PARAMS")

    # Сортируем для консистентности
    key_parts = [intent]
    for k in sorted(params.keys()):
        key_parts.append(f"{k}={params[k]}")
    return tuple(key_parts)


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


def build_smart_prompt(
    intent_name: str,
    intent_spec: Dict,
    n_samples: int,
    used_param_keys: Set[Tuple],
    seed: int
) -> str:
    """Создание промта с информацией об уже использованных параметрах"""
    hints_text = "\n".join([f"   - {hint}" for hint in intent_spec.get("hints", [])])

    # Формируем список уже использованных комбинаций
    used_combinations = []
    for key in used_param_keys:
        if key[0] == intent_name:
            params_str = ", ".join(key[1:])
            used_combinations.append(params_str)

    used_text = ""
    if used_combinations:
        used_text = f"""
⚠️ УЖЕ ИСПОЛЬЗОВАННЫЕ КОМБИНАЦИИ ПАРАМЕТРОВ (НЕ ПОВТОРЯЙ!):
{chr(10).join([f"   - {combo}" for combo in used_combinations[:20]])}

ИСПОЛЬЗУЙ ДРУГИЕ ЗНАЧЕНИЯ ПАРАМЕТРОВ!
"""

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

{used_text}

⚠️ КРИТИЧЕСКИ ВАЖНО:
- КАЖДАЯ фраза должна иметь РАЗНЫЕ значения параметров
- НЕ ПОВТОРЯЙ комбинации из списка выше
- Варьируй ВСЕ параметры: числа, позиции, действия

Сгенерируй {n_samples} МАКСИМАЛЬНО РАЗНООБРАЗНЫХ фраз с РАЗНЫМИ параметрами.

# seed:{seed}
"""
    return prompt


def call_model_smart(
    client: OpenAI,
    intent_name: str,
    intent_spec: Dict,
    n_samples: int,
    used_param_keys: Set[Tuple],
    model: str,
    temperature: float,
    top_p: float,
    seed: int
) -> List[Dict]:
    """Вызов модели с учётом использованных параметров"""
    prompt = build_smart_prompt(intent_name, intent_spec, n_samples, used_param_keys, seed)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Ты креативный эксперт по генерации разнообразных фраз с разными параметрами."},
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


def generate_smart_dataset(
    intents: Dict[str, Dict],
    samples_per_intent: int,
    model: str = "gpt-4o-mini",
    temperature: float = 1.3,
    top_p: float = 0.95,
    max_param_repeats: int = 3
) -> List[TextCarDataset]:
    """
    Умная генерация с контролем разнообразия параметров.

    Args:
        max_param_repeats: Максимум повторений одной комбинации параметров
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    all_results = []
    seen_phrases = set()
    used_param_keys = set()  # Отслеживание использованных комбинаций параметров
    param_counter = defaultdict(int)  # Счётчик повторений параметров

    print(f"\n🧠 Умная генерация: {samples_per_intent} фраз × {len(intents)} интентов")
    print(f"   Модель: {model}, Temperature: {temperature}, Max повторов параметров: {max_param_repeats}\n")

    for intent_name, intent_spec in intents.items():
        print(f"📝 Генерация интента: {intent_name}")
        intent_results = []
        attempts = 0
        max_attempts = samples_per_intent * 5

        # Локальный счётчик параметров для этого интента
        local_param_counter = Counter()

        while len(intent_results) < samples_per_intent and attempts < max_attempts:
            attempts += 1
            seed = random.randint(1, 1_000_000)

            batch_size = min(15, samples_per_intent - len(intent_results) + 10)

            try:
                raw_list = call_model_smart(
                    client, intent_name, intent_spec,
                    batch_size, used_param_keys,
                    model, temperature, top_p, seed
                )

                for raw in raw_list:
                    if len(intent_results) >= samples_per_intent:
                        break

                    try:
                        item = TextCarDataset(**raw)
                        phrase_norm = normalize_phrase_dedup(item.phrase)

                        # Проверка intent
                        if item.intent.lower() != intent_name.lower():
                            print(f"   ⚠ Неверный интент: {item.intent}")
                            continue

                        # Проверка на дубликаты фраз
                        if phrase_norm in seen_phrases:
                            print(f"   ⚠ Дубликат фразы: '{item.phrase}'")
                            continue

                        # Извлекаем параметры
                        params = extract_parameters_values(item.parameters)
                        param_key = get_parameter_key(intent_name, params)

                        # Проверка на слишком частые параметры
                        if local_param_counter[param_key] >= max_param_repeats:
                            params_display = ", ".join([f"{k}={v}" for k, v in params.items()]) if params else "NO_PARAMS"
                            print(f"   ⚠ Слишком часто: {params_display} ({local_param_counter[param_key]} раз)")
                            continue

                        # Добавляем
                        seen_phrases.add(phrase_norm)
                        used_param_keys.add(param_key)
                        local_param_counter[param_key] += 1
                        param_counter[param_key] += 1
                        intent_results.append(item)

                        # Отображение параметров в выводе
                        params_display = json.loads(item.parameters) if item.parameters else {}
                        params_str = ", ".join([f"{k}={v}" for k, v in params_display.items()]) if params_display else ""
                        print(f"   ✓ [{len(intent_results)}/{samples_per_intent}] {item.phrase[:50]}... [{params_str}]")

                    except Exception as e:
                        print(f"   ⚠ Ошибка парсинга: {e}")
                        continue

                time.sleep(0.5)

            except Exception as e:
                print(f"   ⚠ Ошибка API: {e}")
                time.sleep(2)
                continue

        if len(intent_results) < samples_per_intent:
            print(f"   ⚠ Получено только {len(intent_results)}/{samples_per_intent} уникальных фраз")

        # Статистика по параметрам для этого интента
        unique_params = len(set(get_parameter_key(intent_name, extract_parameters_values(item.parameters))
                                 for item in intent_results))
        print(f"   📊 Уникальных комбинаций параметров: {unique_params}")

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
    parser = argparse.ArgumentParser(description="Умная генерация с контролем разнообразия параметров")
    parser.add_argument("--csv", type=str, default="dataset_smart.csv", help="Путь к выходному CSV")
    parser.add_argument("--samples", type=int, default=50, help="Количество фраз на каждый интент")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Модель OpenAI")
    parser.add_argument("--temperature", type=float, default=1.3, help="Temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--max-repeats", type=int, default=3, help="Максимум повторений одной комбинации параметров")
    parser.add_argument("--intents", type=str, nargs="+", help="Список интентов")

    args = parser.parse_args()

    if args.intents:
        intents = {k: v for k, v in INTENT_SPECS.items() if k in args.intents}
    else:
        intents = INTENT_SPECS

    if not intents:
        print("❌ Не найдено интентов для генерации")
        return

    print(f"\n🚀 Старт умной генерации")
    print(f"   Интенты: {', '.join(intents.keys())}")
    print(f"   Фраз на интент: {args.samples}")
    print(f"   Всего фраз: {len(intents) * args.samples}")

    items = generate_smart_dataset(
        intents=intents,
        samples_per_intent=args.samples,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_param_repeats=args.max_repeats
    )

    save_csv(args.csv, items)

    # Детальная статистика
    unique_phrases = len(set(normalize_phrase_dedup(item.phrase) for item in items))
    duplicate_rate = (1 - unique_phrases / len(items)) * 100 if items else 0

    # Статистика по параметрам
    param_diversity = defaultdict(set)
    for item in items:
        params = extract_parameters_values(item.parameters)
        for key, value in params.items():
            param_diversity[key].add(str(value))

    print(f"\n📊 Статистика:")
    print(f"   Всего фраз: {len(items)}")
    print(f"   Уникальных фраз: {unique_phrases}")
    print(f"   Дубликатов: {duplicate_rate:.1f}%")

    if param_diversity:
        print(f"\n📊 Разнообразие параметров:")
        for param_name, values in sorted(param_diversity.items()):
            print(f"   {param_name}: {len(values)} уникальных значений")
            if len(values) <= 15:
                print(f"      → {', '.join(sorted(values, key=lambda x: (not x.replace('-','').isdigit(), x)))}")

    print(f"\n✅ Готово!")


if __name__ == "__main__":
    main()
