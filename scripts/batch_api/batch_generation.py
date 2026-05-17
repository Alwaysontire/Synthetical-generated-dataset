from openai import OpenAI
import math, random, re, json, os, time, sys, csv
import argparse
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(ROOT))

from build_anchors import INTENTS, build_anchor

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


LANG_PROMPTS = {
    "ru": "prompts.ru_promt",
    "en": "prompts.en_promt",
}


@dataclass
class RunConfig:
    model: str = "gpt-4o-mini"
    chunk: int = 40
    overgen: float = 1.35
    target_n: int = 20000
    temperature: float = 1.15
    out_path: str = "dataset_final.json"
    batch_input_path: str = "batch_input.jsonl"
    language: str = "ru"


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
                    "required": ["phrase", "intent", "parameters"],
                    "properties": {
                        "phrase": {"type": "string", "minLength": 1},
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

def make_user_prompt(chunk: int, idx: int, lang: str = "ru") -> str:
    rng = random.Random(idx * 1009 + 7)
    use_anchor = rng.random() < 0.5

    if lang == "en":
        anchor = f'Scene anchor: "{build_anchor(rng, lang="en")}".\n' if use_anchor else ""
        return (
            f"Generate exactly {chunk} examples.\n"
            f"{anchor}"
            f"Requirements:\n"
            f"- maximally diverse phrasings, no duplicates\n"
            f"- imitate a real driver as closely as possible\n"
            f"- vary length: 1–2 words, 3–6 words, 10+ words\n"
            f"- avoid repetitive opening words in a series\n"
            f'Return only a JSON object in the format: {{"items": [...]}}'
        )
    else:
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
    import importlib
    module = importlib.import_module(LANG_PROMPTS[cfg.language])
    build_system_rq = module.build_system_rq

    total = int(cfg.target_n * cfg.overgen)
    num_requests = math.ceil(total / cfg.chunk)

    schema = build_schema(cfg.chunk)
    prompt = build_system_rq(INTENTS)

    with open(cfg.batch_input_path, "w", encoding="utf-8") as f:
        for i in range(num_requests):
            body = {
                "model": cfg.model,
                "temperature": cfg.temperature,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": make_user_prompt(cfg.chunk, i, cfg.language)}
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
    print(f"{num_requests} запросов к {cfg.batch_input_path}, всего {total}.")
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

    return csv_path


def run_batch_generation(cfg: RunConfig):
    client = OpenAI(api_key=OPENAI_API_KEY)


    num_requests = create_batch_input(cfg)

    upload = client.files.create(file=open(cfg.batch_input_path, "rb"), purpose="batch")

    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    while batch.status not in ("completed", "failed", "cancelled", "expired"):
        time.sleep(30)
        batch = client.batches.retrieve(batch.id)
        print(f"status: {batch.status}, request_counts: {batch.request_counts}")

    if batch.error_file_id:
        error_content = client.files.content(batch.error_file_id)
        with open("batch_errors.jsonl", "wb") as f:
            f.write(error_content.read())

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

                params = ex.get("parameters") or {}

                if intent not in INTENTS:
                    continue
                if not isinstance(params, dict):
                    params = {}

                collected.append({
                    "phrase": phrase,
                    "intent": intent,
                    "parameters": params,
                })
                


    # with open(cfg.raw_out_path, "w", encoding="utf-8") as f:
        # json.dump(collected[:cfg.target_n], f, ensure_ascii=False, indent=2)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch generation of car intent dataset")
    parser.add_argument("--lang", choices=["ru", "en"], default="ru", help="Language for generated phrases (default: ru)")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--target-n", type=int, default=20000, help="Target number of examples")
    parser.add_argument("--chunk", type=int, default=50, help="Examples per request")
    parser.add_argument("--overgen", type=float, default=1.2, help="Overgeneration factor")
    parser.add_argument("--out", default=None, help="Output JSON path (default: dataset_{lang}.json)")
    parser.add_argument("--batch-input", default=None, help="Batch input JSONL path")
    args = parser.parse_args()

    out_path = args.out or f"dataset_{args.lang}.json"
    batch_input_path = args.batch_input or f"batch_input_{args.lang}.jsonl"

    cfg = RunConfig(
        model=args.model,
        target_n=args.target_n,
        chunk=args.chunk,
        overgen=args.overgen,
        out_path=out_path,
        batch_input_path=batch_input_path,
        language=args.lang,
    )

    run_batch_generation(cfg)