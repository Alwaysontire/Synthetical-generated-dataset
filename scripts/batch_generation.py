from openai import OpenAI
import math, random, re, json, os, time, sys, csv
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from build_anchors import INTENTS, build_anchor

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@dataclass
class RunConfig:
    model: str = "gpt-5-nano"
    chunk: int = 60
    overgen: float = 1.35
    target_n: int = 20000
    out_path: str = "dataset_final.json"
    raw_out_path: str = "raw_intent.json"
    batch_input_path: str = "batch_input.jsonl"

def build_system_rq() -> str:
    return f"""
Ты эксперт по генерации синтетических данных для обучения голосовых ассистентов в автомобилях.
Твоя задача — создать МАКСИМАЛЬНО РАЗНООБРАЗНЫЙ датасет для распознавания намерений водителя.

КРИТИЧЕСКИ ВАЖНО: избегай повторений и шаблонов.
- Каждая phrase должна быть уникальной по формулировке.
- Не зацикливайся на одних и тех же глаголах и первых словах.
- Меняй длину, порядок слов, стиль (коротко/длинно/вопрос/вежливо/разговорно), добавляй эмоции и контекст.
- Не копируй примеры из инструкции.

ФОРМАТ ВЫВОДА: вернуть JSON-объект с полем "items" (массив объектов). Без markdown и комментариев.
Формат: {{"items": [...]}}

Каждый элемент массива items:
1) "phrase"      — команда на русском
2) "intent_raw"  — свободный интент в snake_case (латиница), может быть более детальным
3) "intent"      — канонический интент СТРОГО из списка
4) "parameters"  — JSON-ОБЪЕКТ параметров или {{}} если параметров нет

Правила для intent_raw:
- snake_case, латиница, 2–5 слов максимум
- intent_raw может уточнять смысл, но НЕ должен менять смысл phrase и выдумывать параметры

КАНОНИЧЕСКИЕ intent (enum):
{", ".join(INTENTS)}

Параметры:
- извлекай ТОЛЬКО если реально присутствуют в phrase
- если параметров нет, parameters = {{}}
- числа могут быть словами (пять, пятёрку, наполовину) — извлекай корректно

АНТИПАТТЕРНЫ:
- серии фраз с одинаковым началом
- копипаст с микроправками
- шаблон "сделай X на Y" много раз подряд
- одни и те же глаголы механически

САМОПРОВЕРКА перед ответом:
- внутри массива нет дублей phrase
- много разных первых слов
- есть короткие/средние/длинные, вопросы и просьбы
"""

def build_schema(chunk: int) -> dict:
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
                    "required": ["phrase", "intent_raw", "intent", "parameters"],
                    "properties": {
                        "phrase": {"type": "string", "minLength": 1},
                        "intent_raw": {"type": "string", "minLength": 2},
                        "intent": {"type": "string", "enum": INTENTS},
                        "parameters": {
                            "type": "object"
                        }
                    }
                }
            }
        },
        "required": ["items"],
        "additionalProperties": False
    }

def make_user_prompt(chunk: int, idx: int) -> str:
    rng = random.Random(idx * 1009 + 7)
    use_anchor = rng.random() < 0.5
    anchor = f'Якорь сцены: "{build_anchor(rng)}".\n' if use_anchor else ""

    
    return (
        f"Сгенерируй ровно {chunk} примеров.\n"
        f"{anchor}"
        f"Требования:\n"
        f"- максимально разнообразные формулировки, без дублей\n"
        f"- пытайся максимально близко съимитировать человека за рулем"
        f"- вариативность длины: 1–2 слова, 3–6 слов, 10+ слов\n"
        f"- избегай однообразных стартовых слов в серии\n"
        f"Верни только JSON-объект в формате: {{\"items\": [...]}}"
    )

def create_batch_input(cfg: RunConfig) -> int:
    total = int(cfg.target_n * cfg.overgen)
    num_requests = math.ceil(total / cfg.chunk)

    schema = build_schema(cfg.chunk)
    prompt = build_system_rq()

    with open(cfg.batch_input_path, "w", encoding="utf-8") as f:
        for i in range(num_requests):
            body = {
                "model": cfg.model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": make_user_prompt(cfg.chunk, i)}
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "car_intents_chunk",
                        "strict": False,  
                        "schema": schema
                    }
                }
            }
            line = {
                "custom_id": f"chunk-{i:06d}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"Сделано {num_requests} запросов к {cfg.batch_input_path}, всего сделано {total}.")
    return num_requests

def extract_from_batch(body: dict) -> str:
    choices = body.get("choices")
    if isinstance(choices, list) and len(choices) > 0:
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
    return None


def normalize_phrase(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[«»“”\"']", "", s)
    s = re.sub(r"[?!.,;:]+$", "", s)
    return s

def metrics(items: list[dict]) -> dict:
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
    if csv_path is None:
        csv_path = json_path.replace('.json', '.csv')

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)


    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['phrase', 'intent', 'parameters'])
        writer.writeheader()
        writer.writerows(data)

    print(f"\n Конвертировано: {json_path} → {csv_path}")
    print(f"   Записей: {len(data)}")

    return csv_path


def run_batch_generation(cfg: RunConfig):
    client = OpenAI(api_key=OPENAI_API_KEY)

    print(f"Создание запроса")

    num_requests = create_batch_input(cfg)

    print(f"\n Загрузка файла: {cfg.batch_input_path}")
    upload = client.files.create(file=open(cfg.batch_input_path, "rb"), purpose="batch")
    print(f"   File ID: {upload.id}")

    print(f"\n Создание batch...")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"   Batch ID: {batch.id}")
    print(f"   Status: {batch.status}")
    print(f"   Model: {cfg.model}")
    print(f"   Total requests: {num_requests}")
    print(f"\n Ожидание завершения...")

    while batch.status not in ("completed", "failed", "cancelled", "expired"):
        time.sleep(30)
        batch = client.batches.retrieve(batch.id)
        print(f"status: {batch.status}, request_counts: {batch.request_counts}")

    print(f"\n Batch завершён: {batch.status}")
    print(f"  Request counts: {batch.request_counts}")
    print(f"  Output file ID: {batch.output_file_id}")
    print(f"  Error file ID: {batch.error_file_id}")

    if batch.error_file_id:
        error_content = client.files.content(batch.error_file_id)
        with open("batch_errors.jsonl", "wb") as f:
            f.write(error_content.read())
        print(f"\n Файл ошибок сохранён: batch_errors.jsonl")

        with open("batch_errors.jsonl", "r", encoding="utf-8") as f:
            first_line = f.readline()
            if first_line:
                error_obj = json.loads(first_line)
                print(f"\n Пример ошибки:")
                print(json.dumps(error_obj.get("error", {}), ensure_ascii=False, indent=2))

    if batch.status != "completed":
        raise RuntimeError(f"Batch ended with status={batch.status}")

    if not batch.output_file_id:
        raise RuntimeError(f"Batch completed but output_file_id is None. Request counts: {batch.request_counts}")

    content = client.files.content(batch.output_file_id)
    out_path = "batch_output.jsonl"
    with open(out_path, "wb") as f:
        f.write(content.read())
    
    seen = set()
    collected = []
    failed_rows = 0

    with open(out_path,"r", encoding="utf-8") as f:
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

            if not isinstance(chunk_items, list):
                failed_rows += 1
                continue
            for ex in chunk_items:
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
                


    with open(cfg.raw_out_path, "w", encoding="utf-8") as f:
        json.dump(collected[:cfg.target_n], f, ensure_ascii=False, indent=2)

    final = []
    for ex in collected:
        params_obj = ex["parameters"]
        params_str = "" if params_obj == {} else json.dumps(params_obj, ensure_ascii=False)
        final.append({
            "phrase": ex["phrase"],
            "intent": ex["intent"],
            "parameters": params_str
        })
    final = final[:cfg.target_n]

    with open(cfg.out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    
    print(f"Количество ошибок - {failed_rows}")
    print(f"Уникальных собрано: {len(collected)}; Всего сохранено: {len(final)} -> {cfg.out_path}")

    metric = metrics(final)
    print("Метрики:", json.dumps(metric, ensure_ascii=False, indent=2))

    csv_path = json_to_csv(cfg.out_path)
    print(f"\n Готово! CSV файл: {csv_path}")


cfg = RunConfig(
    target_n=10000, 
    overgen=1.35, 
    out_path="batch_10k.json",
    raw_out_path="raw_batch_10k.json",
    batch_input_path="batch_input.jsonl"
    )

run_batch_generation(cfg)