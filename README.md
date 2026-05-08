# Car Voice Assistant — Intent Classification

Multilingual (RU + EN) intent classification system for automotive voice commands.  
**90 intent classes · 4 model backbones · OpenAI Batch API data generation**

---

## Overview

The pipeline has two independent parts:

| Part | What it does |
|------|-------------|
| **Data generation** | Generates synthetic phrases via OpenAI Batch API, controls quality, deduplicates |
| **Model training** | Fine-tunes a classification head (+ optional LoRA) on top of a backbone for 90-class classification |

---

## Project Structure

```
synt_gen/
│
├── config.py                      # Centralized path config (ROOT, RAW_DATA, processed_path)
│
├── models/
│   ├── e5-multilingual/           # intfloat/multilingual-e5-base + TextCNN head
│   ├── bge-m3/                    # BAAI/bge-m3 + CLS/Mean/Max pooling head
│   ├── Qwen2/                     # Alibaba-NLP/gte-Qwen2-7B-instruct + LoRA (r=16)
│   └── mmBERT-base/               # jhu-clsp/mmBERT-base + classification head
│       ├── model.py               # Model architecture
│       ├── train.py               # Training loop with early stopping
│       ├── preprocess.py          # Tokenization + stratified split → .pt tensors
│       ├── dataloader.py          # Dataset / DataLoader wrappers
│       └── metrics.py             # Evaluation: macro/weighted P/R/F1
│
├── scripts/
│   ├── api/
│   │   ├── pypline.py             # Simple OpenAI API generation pipeline
│   │   └── retrieve_rare_batch.py # Download completed batch by ID
│   └── batch_api/
│       ├── batch_generation.py    # Main Batch API generator (--lang ru/en)
│       ├── batch_generation_rare.py # Generation focused on rare classes
│       ├── build_anchors.py       # Random context anchors (RU + EN)
│       └── control_batch.py       # Deduplication + intent balancing
│
├── prompts/
│   ├── ru_promt.py                # Russian system prompt builder
│   └── en_promt.py                # English system prompt builder
│
├── quality/
│   ├── quality_control.py         # Dataset quality metrics (duplicates, Self-BLEU, etc.)
│   ├── generate_statistics.py     # Per-class statistics report
│   └── export_problem_intents.py  # Export underrepresented intents
│
├── samples/
│   └── data.csv      # Current training dataset (RU + EN, ~150k phrases)
│
├── data/
│   ├── processed/                 # Tokenized .pt tensors per model
│   │   ├── e5-multilingual/
│   │   ├── bge-m3/
│   │   ├── qwen2/
│   │   └── mmBERT/
│   └── best_models/               # Saved checkpoints per model
│
├── requirements.txt
└── README.md
```

---

## Models

| Model | Backbone | Pooling | Params | Notes |
|-------|----------|---------|--------|-------|
| `e5-multilingual` | `intfloat/multilingual-e5-base` | Mean + TextCNN | 278M | Input prefixed with `query: ` |
| `bge-m3` | `BAAI/bge-m3` | CLS + Mean + Max | 570M | Bidirectional encoder |
| `Qwen2` | `Alibaba-NLP/gte-Qwen2-7B-instruct` | Last token | 7B | LoRA r=16, decoder-based |
| `mmBERT-base` | `jhu-clsp/mmBERT-base` | CLS + Mean + Max | 178M | Multilingual BERT |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess dataset

Run from the model directory you want to use:

```bash
cd models/e5-multilingual
python preprocess.py
```

Reads `samples/multilingual_data.csv` via `config.py`, tokenizes, saves stratified train/val/test splits to `data/processed/<model>/`.

### 3. Train

```bash
python train.py
```

Saves best checkpoint to `data/best_models/<model>/`.

### 4. Evaluate

```bash
python metrics.py
```

Prints per-class and overall macro/weighted Precision, Recall, F1.

---

## Data Generation

### Generate phrases (Batch API)

```bash
# Russian
python scripts/batch_api/batch_generation.py --lang ru --target-n 50000

# English
python scripts/batch_api/batch_generation.py --lang en --target-n 50000
```

### Generate for rare classes only

```bash
python scripts/batch_api/batch_generation_rare.py
```

Targets underrepresented intents, generates ~8000 examples evenly distributed.

### Retrieve completed batch

```bash
python scripts/api/retrieve_rare_batch.py --batch-id <batch_id>
```

### Quality control

```bash
python scripts/batch_api/control_batch.py --input samples/new_batch.json
python quality/quality_control.py
```

---

## Dataset

| Property | Value |
|----------|-------|
| Languages | Russian + English |
| Classes | 90 intents |
| Size | ~150,000 phrases |
| Split | 80 / 10 / 10 (train / val / test) |

Semantic deduplication uses cosine similarity on `multilingual-e5-small` embeddings.  
Phrases with similarity ≥ 0.97 are treated as near-duplicates (second occurrence removed).

---

## Training Config

| Parameter | e5-multilingual | bge-m3 | Qwen2 | mmBERT |
|-----------|----------------|--------|-------|--------|
| LR (backbone) | 5e-6 | 1e-5 | 1e-5 | 1e-5 |
| LR (head) | 3e-4 | 4e-4 | 4e-4 | 2e-4 |
| Batch size | 256 | 256 | 16 | 256 |
| Max epochs | 20 | 15 | 15 | 15 |
| Early stopping | patience=2 | patience=4 | patience=4 | patience=4 |
| Scheduler | CosineAnnealing | Cosine warmup | Cosine warmup | Cosine warmup |
| Label smoothing | 0.05 | 0.09 | 0.09 | 0.05 |
| Gradient clipping | 1.0 | 1.0 | 1.0 | 1.0 |

---

## Requirements

- Python 3.9+
- PyTorch
- Transformers (HuggingFace)
- PEFT (for Qwen2 LoRA)
- OpenAI API key (for data generation)
