# Car Voice Assistant — Intent Classification

Multilingual (RU + EN) intent classification for automotive voice commands.  
**64 intent classes · 153k phrases · 4 model backbones**

---

## What this is

The idea is simple: a driver says something like "turn down the fan" or "сделай потише музыку", and the system should figure out what they mean and what parameters to pass. The tricky part is that the same intent can be expressed in dozens of ways across two languages, with different formality levels, fillers, and phrasing.

To handle that, I built a synthetic data generation pipeline using the OpenAI Batch API, then trained several encoder-based classifiers on top of it.


The dataset is available on [HuggingFace](https://huggingface.co/datasets/INFINITY1023/MultilingualDriverCommands).
The models are available on [HuggingFace](https://huggingface.co/INFINITY1023/multilingual-driver-command-models)

---

## Project Structure

```
synt_gen/
│
├── config.py                      # Central path config
│
├── models/
│   ├── e5-multilingual/           # intfloat/multilingual-e5-base + TextCNN head
│   ├── bge-m3/                    # BAAI/bge-m3 + CLS/Mean/Max pooling head
│   ├── Qwen2/                     # Alibaba-NLP/gte-Qwen2-7B-instruct + LoRA (r=16)
│   └── mmBERT-base/               # jhu-clsp/mmBERT-base + classification head
│       ├── model.py
│       ├── train.py
│       ├── preprocess.py
│       ├── dataloader.py
│       └── metrics.py
│
├── scripts/
│   ├── api/
│   │   ├── pypline.py             # Simple OpenAI API generation
│   │   └── retrieve_rare_batch.py # Download completed batch by ID
│   └── batch_api/
│       ├── batch_generation.py    # Main Batch API generator (--lang ru/en)
│       ├── batch_generation_rare.py
│       ├── build_anchors.py       # Random context anchors (RU + EN)
│       └── control_batch.py       # Deduplication + intent balancing
│
├── prompts/
│   ├── ru_promt.py
│   └── en_promt.py
│
├── quality/
│   └── quality_control.py         # Quality metrics: duplicates, TTR, Self-BLEU
│
├── samples/
│   ├── data.csv                   # Current training dataset
│   └── intent_mapping.py          # Maps old intent names → new consolidated names
│
├── hf/
│   └── upload_datasset.py         # Push dataset to HuggingFace Hub
│
├── app_demo/                      # Gradio demo app
│   ├── app.py
│   ├── model.py
│   └── label_map.json
│
├── requirements.txt
└── README.md
```

---

## Models

| Model | Backbone | Head | Notes |
|-------|----------|------|-------|
| `e5-multilingual` | `intfloat/multilingual-e5-base` | Mean + TextCNN | Input prefixed with `query: ` |
| `bge-m3` | `BAAI/bge-m3` | CLS + Mean + Max pooling | Strong multilingual encoder |
| `Qwen2` | `Alibaba-NLP/gte-Qwen2-7B-instruct` | Last token + LoRA r=16 | Decoder-based, largest model |
| `mmBERT-base` | `jhu-clsp/mmBERT-base` | CLS + Mean + Max pooling | Compact multilingual BERT |

---

## Quick Start

```bash
pip install -r requirements.txt
```

Run from the repo root:

```bash
# Preprocess (tokenize + split)
python models/bge-m3/preprocess.py

# Train
python models/bge-m3/train.py

# Evaluate
python models/bge-m3/metrics.py
```

Each model has its own `preprocess.py` / `train.py` / `metrics.py`.

---

## Dataset

| Property | Value |
|----------|-------|
| Languages | Russian + English |
| Total phrases | 153,062 |
| Unique phrases | 153,059 |
| Intent classes | 64 |
| Domain groups | 10 |
| Language split | 50% RU / 50% EN |
| Train / Val / Test | 80 / 10 / 10 |


---

## Data Generation

Phrases are generated via OpenAI Batch API with structured prompts that vary phrasing style, politeness, and context. Generation runs separately for RU and EN.

```bash
# Generate for all intents
python scripts/batch_api/batch_generation.py --lang ru --target-n 50000
python scripts/batch_api/batch_generation.py --lang en --target-n 50000

# Top up underrepresented classes
python scripts/batch_api/batch_generation_rare.py

# Deduplicate and balance
python scripts/batch_api/control_batch.py --input samples/new_batch.json
```

Quality check:

```bash
python quality/quality_control.py
```


---

## Requirements

- Python 3.9+
- PyTorch
- Transformers + PEFT
- OpenAI API key (for data generation)
