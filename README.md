# Car Voice Assistant — Intent Classification

Multilingual (RU + EN) intent classification system for automotive voice commands.
**90 intent classes · TextCNN + multilingual-e5-base backbone · OpenAI Batch API data generation**

---

## Overview

The pipeline has two independent parts:

| Part | What it does |
|------|-------------|
| **Data generation** | Generates synthetic phrases via OpenAI Batch API, controls quality, deduplicates |
| **Model training** | Fine-tunes TextCNN on top of a frozen/unfrozen E5 backbone for 90-class classification |

---

## Project Structure

```
synt_gen/
│
├── model/                         # Classification model
│   ├── model.py                   # TextCNN architecture (E5 backbone + CNN head)
│   ├── train.py                   # Training loop with early stopping
│   ├── preprocess.py              # Tokenization + stratified split → .pt tensors
│   ├── dataloader.py              # Dataset / DataLoader wrappers
│   └── metrics.py                 # Evaluation: macro/weighted P/R/F1
│
├── scripts/
│   ├── api/
│   │   ├── pypline.py             # Simple OpenAI API generation pipeline
│   │   └── retrieve_rare_batch.py # Download completed batch by ID
│   └── batch_api/
│       ├── batch_generation.py    # Main Batch API generator (--lang ru/en)
│       ├── batch_generation_rare.py # Generation focused on rare classes
│       ├── build_anchors.py       # Random context anchors (RU + EN)
│       └── control_batch.py       # Quality check for generated batch output
│
├── prompts/
│   ├── ru_promt.py                # Russian system prompt builder
│   └── en_promt.py                # English system prompt builder
│
├── quality/
│   ├── quality_control.py         # Dataset quality metrics (duplicates, Self-BLEU, etc.)
│   ├── generate_statistics.py     # Per-class statistics
│   └── export_problem_intents.py  # Export underrepresented intents
│
│
├── samples/                       # Generated datasets
│   ├── multilingual_data.csv      # Current training dataset (RU + EN)
│   └── car_voice_assistant_multilingual_200k.csv  # Raw 200k dataset (pre-dedup)
│
├── data/                          # Preprocessed tensors + saved model weights
│   └── best_model_multilingual.pt
│
├── server_finetuning/             # Server-side training scripts (remote GPU)
│
├── requirements.txt
└── README.md
```

---

## Model Architecture

```
Input text
    │
    ▼
multilingual-e5-base (backbone, unfrozen)
    │
    ├── last_hidden_state ──► masked mean pool ──────────────────┐
    │                                                             │
    └── transpose ──► Conv1d(k=2,3,4,5) ──► GELU ──► MaxPool ──► concat
                                                                  │
                                                                  ▼
                                                        Linear → LayerNorm → GELU → Dropout
                                                                  │
                                                                  ▼
                                                        Linear(256 → 90 classes)

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess dataset

```bash
cd model
python preprocess.py
```

Reads `samples/multilingual_data.csv`, tokenizes with `multilingual-e5-base`, saves stratified train/val/test splits to `data/processed/`.

### 3. Train

```bash
python train.py
```

Saves best checkpoint to `data/best_model_multilingual.pt`.

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

Targets underrepresented intents (<125 samples), generates ~8000 examples evenly distributed.

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
| Raw size | ~200,000 phrases |


Semantic deduplication uses cosine similarity on `multilingual-e5-small` embeddings.
Phrases with similarity ≥ 0.97 are treated as near-duplicates (second occurrence removed).

---

## Training Config

| Parameter | Value |
|-----------|-------|
| Backbone | `intfloat/multilingual-e5-large` |
| Optimizer | AdamW |
| LR (head) | 2e-4 |
| LR (backbone) | 5e-6 |
| Weight decay (head) | 0.1 |
| Weight decay (backbone) | 0.01 |
| Scheduler | CosineAnnealingLR |
| Max epochs | 20 |
| Early stopping patience | 2 |
| Gradient clipping | max_norm=1.0 |
| Dropout | 0.3 |

---

## Requirements

- Python 3.9+
- PyTorch
- Transformers (HuggingFace)
- OpenAI API key (for data generation)
