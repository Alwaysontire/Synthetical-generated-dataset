from openai import OpenAI
import math, random, re, json, os, time, sys, csv
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from build_anchors import build_anchor

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Редкие классы (< ~125 примеров в датасете) — генерируем именно их
RARE_INTENTS = [
    "phone_unpair",
    "phone_disconnect",
    "phone_pair",
    "phone_connect",
    "suspension_mode_set",
    "steering_wheel_memory_recall",
    "cruise_control_resume",
    "powertrain_mode_set",
    "language_set",
    "reading_light_set",
    "regen_level_set",
    "hud_set",
    "trunk_close",
    "start_stop_set",
    "voice_assistant_set",
    "mirror_fold_set",
    "auto_hold_set",
    "lane_keep_assist_set",
    "adaptive_cruise_distance_set",
    "message_reply",
]


@dataclass
class RunConfig:
    model: str = "gpt-4o-mini"
    chunk: int = 60   # 60 / 20 классов = 3 примера на класс за запрос
    overgen: float = 1.3
    target_n: int = 8000  # ~400 примеров на каждый из 20 классов
    temperature: float = 1.0
    out_path: str = "samples/rare_classes.json"
    batch_input_path: str = "samples/batch_input_rare.jsonl"


def build_system_rq_rare() -> str:
    intents_desc = """
- phone_pair                  — сопряжение нового телефона по Bluetooth (первичная настройка)
- phone_unpair                — удаление / отвязка телефона из памяти системы
- phone_connect               — подключение уже сопряжённого телефона
- phone_disconnect            — отключение активного телефона (без удаления)
- suspension_mode_set         — изменение режима подвески (спортивная / комфорт / авто)
- steering_wheel_memory_recall — возврат рулевой колонки в сохранённое положение
- cruise_control_resume       — возобновление круиз-контроля после паузы (не включение с нуля)
- powertrain_mode_set         — смена режима трансмиссии/двигателя (спорт / эко / авто / норм)
- language_set                — смена языка интерфейса мультимедиа / навигации
- reading_light_set           — управление светом для чтения (для пассажиров)
- regen_level_set             — уровень рекуперации энергии (только для электромобилей)
- hud_set                     — управление проекционным дисплеем (HUD) — включить/выключить/настроить
- trunk_close                 — закрытие багажника (не открытие!)
- start_stop_set              — управление системой авто-старт/стоп двигателя
- voice_assistant_set         — настройка голосового ассистента (язык, чувствительность, вкл/выкл)
- mirror_fold_set             — складывание / раскладывание боковых зеркал
- auto_hold_set               — включение / выключение функции авто-удержания на тормозе
- lane_keep_assist_set        — управление системой удержания в полосе
- adaptive_cruise_distance_set — изменение дистанции адаптивного круиз-контроля
- message_reply               — ответить на входящее сообщение голосом
""".strip()

    return f"""
Ты эксперт по генерации синтетических данных для обучения голосовых ассистентов в автомобилях.
Твоя задача — создать МАКСИМАЛЬНО РАЗНООБРАЗНЫЙ датасет для КОНКРЕТНЫХ редких намерений водителя.

ЦЕЛЕВЫЕ ИНТЕНТЫ (только они!):
{intents_desc}

ВАЖНЫЕ РАЗЛИЧИЯ МЕЖДУ ПОХОЖИМИ ИНТЕНТАМИ:
- phone_pair vs phone_connect: pair = первый раз добавляем новый телефон; connect = снова подключаем уже добавленный
- phone_unpair vs phone_disconnect: unpair = удаляем из памяти; disconnect = просто разрываем текущее соединение
- cruise_control_resume vs cruise_control_set: resume = возобновить после паузы; set = включить с нуля
- trunk_close (только закрытие) vs trunk_open (только открытие) — не путать

КРИТИЧЕСКИ ВАЖНО:
- Генерируй ТОЛЬКО фразы для этих 20 интентов, не уходи в другие темы
- Каждая phrase должна быть уникальной по формулировке
- Меняй длину, стиль (коротко/длинно/вопрос/вежливо/разговорно)
- Не зацикливайся на одних и тех же глаголах
- Добавляй синонимы, перифразы, разговорные выражения
- Иногда добавляй контекст ("у меня новый телефон", "слишком яркий экран")

ФОРМАТ ВЫВОДА: вернуть JSON-объект с полем "items". Без markdown и комментариев.
Формат: {{"items": [...]}}

Каждый элемент:
1) "phrase"      — команда на русском
2) "intent"      — СТРОГО один из: {", ".join(RARE_INTENTS)}
3) "parameters"  — JSON-объект параметров или {{}} если нет

Параметры (если реально присутствуют в фразе):
- mode: режим (sport / eco / comfort / auto / normal)
- level: уровень (числом или словом)
- device_name: имя устройства ("iPhone Маши")
- value: вкл/выкл или числовое значение (on / off / число)
- только значения на английском языке
"""


def build_schema_rare(chunk: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": chunk,
                "maxItems": chunk,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["phrase", "intent", "parameters"],
                    "properties": {
                        "phrase": {"type": "string", "minLength": 1},
                        "intent": {"type": "string", "enum": RARE_INTENTS},
                        "parameters": {"type": "object"}
                    }
                }
            }
        },
        "required": ["items"],
        "additionalProperties": False
    }


def make_user_prompt_rare(chunk: int, idx: int) -> str:
    rng = random.Random(idx * 1009 + 7)

    # Равномерно распределяем по редким классам
    per_intent = chunk // len(RARE_INTENTS)
    remainder = chunk % len(RARE_INTENTS)
    counts = [per_intent + (1 if i < remainder else 0) for i in range(len(RARE_INTENTS))]
    rng.shuffle(counts)

    distribution = "\n".join(
        f"  - {intent}: {count} примеров"
        for intent, count in zip(RARE_INTENTS, counts)
    )

    anchor = f'Якорь сцены: "{build_anchor(rng)}".\n' if rng.random() < 0.4 else ""

    return (
        f"Сгенерируй ровно {chunk} примеров.\n"
        f"{anchor}"
        f"Распределение по интентам:\n{distribution}\n\n"
        f"Требования:\n"
        f"- максимально разнообразные формулировки, без дублей\n"
        f"- имитируй реального водителя, который хочет управлять телефоном через Bluetooth\n"
        f"- вариативность длины: 1–3 слова, 4–7 слов, 8+ слов\n"
        f"- избегай однообразных стартовых слов в серии\n"
        f"Верни только JSON-объект в формате: {{\"items\": [...]}}"
    )


def create_batch_input_rare(cfg: RunConfig) -> int:
    total = int(cfg.target_n * cfg.overgen)
    num_requests = math.ceil(total / cfg.chunk)

    schema = build_schema_rare(cfg.chunk)
    prompt = build_system_rq_rare()

    os.makedirs(os.path.dirname(cfg.batch_input_path), exist_ok=True)

    with open(cfg.batch_input_path, "w", encoding="utf-8") as f:
        for i in range(num_requests):
            body = {
                "model": cfg.model,
                "temperature": cfg.temperature,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": make_user_prompt_rare(cfg.chunk, i)}
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "rare_intents_chunk",
                        "strict": False,
                        "schema": schema
                    }
                }
            }
            line = {
                "custom_id": f"rare-{i:06d}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Создано {num_requests} запросов → {cfg.batch_input_path}, целевых примеров: {total}")
    return num_requests


def normalize_phrase(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[«»""\"']", "", s)
    s = re.sub(r"[?!.,;:]+$", "", s)
    return s


def extract_from_batch(body: dict) -> str:
    choices = body.get("choices")
    if isinstance(choices, list) and len(choices) > 0:
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
    return None


def run_batch_generation_rare(cfg: RunConfig):
    client = OpenAI(api_key=OPENAI_API_KEY)

    print("Создание запроса для редких классов...")
    num_requests = create_batch_input_rare(cfg)

    print(f"\nЗагрузка файла: {cfg.batch_input_path}")
    upload = client.files.create(file=open(cfg.batch_input_path, "rb"), purpose="batch")
    print(f"  File ID: {upload.id}")

    print(f"\nСоздание batch...")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"  Batch ID: {batch.id}")
    print(f"  Status: {batch.status}")

    print(f"\nОжидание завершения...")
    while batch.status not in ("completed", "failed", "cancelled", "expired"):
        time.sleep(30)
        batch = client.batches.retrieve(batch.id)
        print(f"  status: {batch.status}, request_counts: {batch.request_counts}")

    print(f"\nBatch завершён: {batch.status}")

    if batch.status != "completed":
        raise RuntimeError(f"Batch ended with status={batch.status}")

    content = client.files.content(batch.output_file_id)
    out_path = "samples/batch_output_rare.jsonl"
    with open(out_path, "wb") as f:
        f.write(content.read())

    seen = set()
    collected = []
    intent_counts = {intent: 0 for intent in RARE_INTENTS}
    failed_rows = 0

    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
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
            except Exception:
                failed_rows += 1
                continue

            for ex in chunk_items:
                if not isinstance(ex, dict):
                    continue
                phrase = ex.get("phrase", "").strip()
                if not phrase:
                    continue
                key = normalize_phrase(phrase)
                if key in seen:
                    continue
                seen.add(key)

                intent = ex.get("intent")
                if intent not in RARE_INTENTS:
                    continue

                params = ex.get("parameters") or {}
                if not isinstance(params, dict):
                    params = {}

                collected.append({
                    "phrase": phrase,
                    "intent": intent,
                    "parameters": params,
                })
                intent_counts[intent] += 1

    # Сохраняем JSON
    final = []
    for ex in collected[:cfg.target_n]:
        params_str = "" if ex["parameters"] == {} else json.dumps(ex["parameters"], ensure_ascii=False)
        final.append({
            "phrase": ex["phrase"],
            "intent": ex["intent"],
            "parameters": params_str
        })

    os.makedirs(os.path.dirname(cfg.out_path), exist_ok=True)
    with open(cfg.out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    # Сохраняем CSV
    csv_path = cfg.out_path.replace(".json", ".csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phrase", "intent", "parameters"])
        writer.writeheader()
        writer.writerows(final)

    print(f"\nОшибок: {failed_rows}")
    print(f"Уникальных собрано: {len(collected)} → сохранено: {len(final)}")
    print(f"Распределение по интентам: {json.dumps(intent_counts, ensure_ascii=False)}")
    print(f"JSON: {cfg.out_path}")
    print(f"CSV:  {csv_path}")


cfg = RunConfig(
    model="gpt-4o-mini",
    target_n=8000,
    chunk=60,
    overgen=1.3,
    temperature=1.0,
    out_path="samples/rare_classes.json",
    batch_input_path="samples/batch_input_rare.jsonl"
)

run_batch_generation_rare(cfg)
