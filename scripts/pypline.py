import json, time, csv, os, re
from typing import List, Dict, Tuple
# import torch
from openai import OpenAI
from pydantic import BaseModel
from key import OPENAI_API_KEY, SYSTEM_REQ
import random
import argparse


client = OpenAI(api_key=OPENAI_API_KEY)


class TextCarDataset(BaseModel):
    phrase: str
    intent: str
    parameters: str  
    # phrase_embedding:List[int]
    # intent_embedding:List[int]

    @classmethod
    def norm_phrase(s : str) -> str:
        s = s.strip()
        s = s.strip('"\n` ').replace("  ", " ")
        s = s.lower()
        return s
    
    @classmethod 
    def str_snakecase(s : str) -> str:
        s = s.strip()
        s = re.sub(r"[\s\-]+", "_", s)
        s = re.sub(r"[^a-zA-Z0-9_]", "", s)
        s = re.sub(r"_+", "_", s).strip("_").lower()
        return s


def build_promt(seed:int) ->str:
    return SYSTEM_REQ + f"\n# seed:{seed}\n"


def extract_first_json(text: str):
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("` \n")
        if t.lower().startswith("json"):
            t = t[4:].lstrip()

    start_obj = t.find("{")
    start_arr = t.find("[")
    starts = [i for i in [start_obj, start_arr] if i != -1]
    if not starts:
        raise ValueError("No JSON start found")
    i = min(starts)

    stack = []
    in_str = False
    esc = False
    end_index = None

    opening = t[i]
    closing = "}" if opening == "{" else "]"
    stack.append(closing)

    j = i + 1
    while j < len(t):
        ch = t[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == opening:
                stack.append(closing)
            elif ch == stack[-1]:
                stack.pop()
                if not stack:
                    end_index = j
                    break
        j += 1

    if end_index is None:
        raise ValueError("Unbalanced JSON")

    candidate = t[i:end_index + 1].strip()
    return json.loads(candidate)



def call_model(client: OpenAI, model: str, temperature: float, top_p: float, seed: int) -> List[Dict]:
    prompt = build_promt(seed=seed)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Ты эксперт по генерации синтетических данных."},
            {"role": "user", "content": prompt}
        ],
        seed=seed
    )
    text = response.choices[0].message.content.strip()
    result = extract_first_json(text)
    # Ensure we return a list
    if isinstance(result, dict):
        return [result]
    return result

def normalize_phrase_dedup(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def dedup(item : TextCarDataset) -> Tuple:
    item1 = item.intent
    item2 = item.parameters 
    return (item1, item2)


def generate_dataset(n: int,  model: str ="gpt-5-mini", temperature: float = 1.3, top_p: float = 0.9, max_tries : int = 3) -> List[TextCarDataset]:
    client = OpenAI(api_key=OPENAI_API_KEY)
    results: List[TextCarDataset] = []
    seen_keys = set()
    seen_phrases = set()
    attempt = 0

    while len(results) < n:
        attempt += 1
        seed = random.randint(0, 100000)

        try:
            raw_list = call_model(client, model, temperature, top_p, seed)

            for raw in raw_list:
                if len(results) >= n:
                    break

                try:
                    item = TextCarDataset(**raw)


                    k = dedup(item)
                    phrase_norm = normalize_phrase_dedup(item.phrase)

                    # if not item.intent:
                    #     print(f"⚠ Skipping: empty intent")
                    #     continue

                    # if k in seen_keys:
                    #     print(f"⚠ Duplicate intent+params: {item.intent}")
                    #     continue

                    # if phrase_norm in seen_phrases:
                    #     print(f"⚠ Duplicate phrase: {item.phrase}")
                    #     continue

                    # Add to results
                    seen_keys.add(k)
                    seen_phrases.add(phrase_norm)
                    results.append(item)
                    if len(results) % 100 == 0:
                        print(f"✓ Added [{len(results)}/{n}]: {item.intent} - {item.phrase[:50]}")

                except Exception as e:
                    print(f"⚠ Error parsing item: {e}")
                    continue

        except Exception as e:
            print(f"⚠ API call failed (attempt {attempt}): {e}")
            if attempt >= max_tries:
                print(f"❌ Max retries reached, continuing with {len(results)} items")
            time.sleep(1)  # Brief pause before retry
            continue

    return results


def save_json(path: str, items: List[TextCarDataset]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([it.model_dump() for it in items], f, ensure_ascii=False, indent=2)
    

def save_csv(path: str, items: List[TextCarDataset]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phrase", "intent", "parameters"])
        for it in items:
            w.writerow([it.phrase, it.intent, it.parameters])  
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="size of dataset")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="model name")
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--out", type=str, default="dataset_car.json")
    parser.add_argument("--csv", type=str, default="")
    args = parser.parse_args()
    print(f"Generating {args.n} items with {args.model} ...")

    items = generate_dataset(
        n=args.n,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p
    )
    save_json(args.out, items)
    if args.csv:
        save_csv(args.csv, items)  

    print(f"Dataset saved: {os.path.abspath(args.out)}")
    if args.csv:
        print(f"Dataset saved as csv: {os.path.abspath(args.csv)}")

if __name__ == "__main__":
    main()
