from openai import OpenAI
import json, re, csv, os
from dotenv import load_dotenv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.batch_generation_rare import RARE_INTENTS

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BATCH_ID = "batch_699ee5273ef88190b7d529bdfeace281"
OUT_JSONL = "samples/batch_output_rare.jsonl"
OUT_JSON  = "samples/rare_classes.json"
OUT_CSV   = "samples/rare_classes.csv"


def normalize_phrase(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[«»""\"']", "", s)
    s = re.sub(r"[?!.,;:]+$", "", s)
    return s


def extract_content(body: dict) -> str:
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message", {})
        return msg.get("content")
    return None


def main():
    batch = client.batches.retrieve(BATCH_ID)
    print(f"Status: {batch.status}")
    print(f"Output file ID: {batch.output_file_id}")

    if batch.status != "completed":
        print("Батч ещё не завершён!")
        return

    content = client.files.content(batch.output_file_id)
    with open(OUT_JSONL, "wb") as f:
        f.write(content.read())
    print(f"Скачано → {OUT_JSONL}")

    seen = set()
    collected = []
    intent_counts = {intent: 0 for intent in RARE_INTENTS}
    failed = 0

    with open(OUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("error"):
                failed += 1
                continue
            text = extract_content(row["response"]["body"])
            if not text:
                failed += 1
                continue
            try:
                items = json.loads(text).get("items", [])
            except Exception:
                failed += 1
                continue

            for ex in items:
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

                collected.append({"phrase": phrase, "intent": intent, "parameters": params})
                intent_counts[intent] += 1

    final = []
    for ex in collected:
        params_str = "" if ex["parameters"] == {} else json.dumps(ex["parameters"], ensure_ascii=False)
        final.append({"phrase": ex["phrase"], "intent": ex["intent"], "parameters": params_str})

    os.makedirs("samples", exist_ok=True)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phrase", "intent", "parameters"])
        writer.writeheader()
        writer.writerows(final)

    print(f"\nОшибок: {failed}")
    print(f"Уникальных: {len(collected)}")
    print(f"Распределение: {json.dumps(intent_counts, ensure_ascii=False, indent=2)}")
    print(f"\nJSON → {OUT_JSON}")
    print(f"CSV  → {OUT_CSV}")


if __name__ == "__main__":
    main()
