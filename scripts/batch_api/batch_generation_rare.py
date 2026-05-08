from openai import OpenAI
import math, random, re, json, os, time, sys, csv
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from build_anchors import build_anchor

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


RARE_INTENTS = [
    "regen_level_set",
    "phone_unpair",
    "language_set",
    "suspension_mode_set",
    "phone_disconnect",
    "odometer_query",
    "hud_set",
    "powertrain_mode_set",
    "mirror_heat_set",
    "auto_hold_set",
    "window_child_lock_set",
    "trip_reset",
    "cruise_control_resume",
    "park_distance_alert_set",
    "cruise_control_cancel",
    "warning_explain",
    "start_stop_set",
    "phone_pair",
    "fuel_cap_open",
    "steering_wheel_memory_recall",
    "camera_view_set",
    "voice_assistant_set",
    "system_volume_set",
]


@dataclass
class RunConfig:
    model: str = "gpt-4o-mini"
    chunk: int = 60   
    overgen: float = 1.3
    target_n: int = 8000  
    temperature: float = 1.0
    out_path: str = "samples/rare_classes.json"
    batch_input_path: str = "samples/batch_input_rare.jsonl"


def build_system_rq_rare() -> str:
    intents_desc = """
- regen_level_set             — уровень рекуперации энергии (только для электромобилей: высокий/низкий/авто)
- phone_unpair                — удаление телефона из памяти системы навсегда (не просто отключение)
- language_set                — смена языка интерфейса / навигации / голосового ассистента
- suspension_mode_set         — изменение режима подвески (спорт / комфорт / авто / внедорожный)
- phone_disconnect            — отключить активный телефон от системы (без удаления из памяти)
- odometer_query              — узнать показания одометра / пробег автомобиля
- hud_set                     — управление проекционным дисплеем HUD (включить / выключить / яркость)
- powertrain_mode_set         — сменить режим трансмиссии или двигателя (спорт / эко / авто / нормальный)
- mirror_heat_set             — включить / выключить обогрев боковых зеркал
- auto_hold_set               — включить / выключить авто-удержание на тормозе на светофоре
- window_child_lock_set       — детская блокировка окон (заблокировать / разблокировать кнопки окон)
- trip_reset                  — сбросить данные поездки (расход, пробег поездки, среднюю скорость)
- cruise_control_resume       — ВОЗОБНОВИТЬ круиз-контроль после паузы (не включать с нуля!)
- park_distance_alert_set     — датчики парковки / парктроник (включить / выключить / настроить звук)
- cruise_control_cancel       — полностью ОТКЛЮЧИТЬ / выключить круиз-контроль
- warning_explain             — попросить объяснить значение горящего предупреждающего индикатора
- start_stop_set              — система авто-старт/стоп двигателя (включить / выключить)
- phone_pair                  — сопряжение нового телефона по Bluetooth (первый раз, новое устройство)
- fuel_cap_open               — открыть крышку топливного бака
- steering_wheel_memory_recall — вернуть рулевую колонку в сохранённое положение
- camera_view_set             — выбрать вид камеры (задняя / передняя / 360° / боковые)
- voice_assistant_set         — настройка голосового ассистента (язык / чувствительность / вкл/выкл)
- system_volume_set           — установить конкретное числовое значение общей громкости системы
""".strip()

    return f"""Ты эксперт по генерации синтетических данных для обучения голосовых ассистентов в автомобилях.
Твоя задача — создать МАКСИМАЛЬНО РАЗНООБРАЗНЫЙ датасет для редких намерений водителя.

ЦЕЛЕВЫЕ ИНТЕНТЫ (только они!):
{intents_desc}

ВАЖНЫЕ РАЗЛИЧИЯ:
- phone_pair vs phone_disconnect vs phone_unpair: pair = добавить новый; disconnect = отключить текущий; unpair = удалить из памяти
- cruise_control_resume vs cruise_control_cancel: resume = возобновить после паузы; cancel = выключить совсем
- trip_reset vs odometer_query: reset = сбросить счётчик поездки; query = просто спросить пробег

ТРЕБОВАНИЯ К ФРАЗАМ:
- Каждая phrase уникальна по формулировке, на русском или английском
- Меняй длину: 1–3 слова, 4–7 слов, 8+ слов
- Меняй стиль: коротко / развёрнуто / вопросом / вежливо / разговорно
- Для system_volume_set: обязательно числовое значение (1–20 или проценты)
- Для steering_wheel_memory_recall: упоминание возврата/памяти позиции

ФОРМАТ ВЫВОДА: JSON-объект с полем "items". Без markdown и комментариев.
Формат: {{"items": [...]}}

Каждый элемент:
1) "phrase"     — команда на русском или английском
2) "intent"     — СТРОГО один из: {", ".join(RARE_INTENTS)}
3) "parameters" — JSON-объект параметров или {{}} если нет

Параметры (если явно присутствуют в фразе):
- mode: режим (sport / eco / comfort / auto / normal / off-road)
- level: уровень (high / medium / low / числом)
- value: числовое значение или on / off
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
        f"- имитируй реального водителя в автомобиле\n"
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


    csv_path = cfg.out_path.replace(".json", ".csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phrase", "intent", "parameters"])
        writer.writeheader()
        writer.writerows(final)

    print(f"Уникальных собрано: {len(collected)} и сохранено: {len(final)}")
    print(f"Распределение по интентам: {json.dumps(intent_counts, ensure_ascii=False)}")
    print(f"JSON: {cfg.out_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    cfg = RunConfig(
        model="gpt-4o-mini",
        target_n=9200,   
        chunk=69,        
        overgen=1.3,
        temperature=1.0,
        out_path="samples/rare_classes_v2.json",
        batch_input_path="samples/batch_input_rare_v2.jsonl"
    )
    run_batch_generation_rare(cfg)
