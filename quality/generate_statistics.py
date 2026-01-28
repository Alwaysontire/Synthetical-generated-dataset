#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генерация статистики в JSON формате
"""

import pandas as pd
import numpy as np
import json
from collections import Counter
import re

# Загрузка данных
df = pd.read_csv('/Users/ilanizankovskij/Documents/vs_code/synt_gen/sample_10000.csv')

# Подготовка данных
df['word_count'] = df['phrase'].str.split().str.len()
df['char_count'] = df['phrase'].str.len()
df['first_word'] = df['phrase'].str.lower().str.split().str[0]
df['first_2_words'] = df['phrase'].str.lower().str.split().str[:2].str.join(' ')

# Токенизация
all_text = ' '.join(df['phrase'].astype(str))
tokens = re.findall(r'\b\w+\b', all_text.lower())
word_freq = Counter(tokens)

# Базовая статистика
stats = {
    "dataset_info": {
        "total_phrases": len(df),
        "unique_phrases": df['phrase'].nunique(),
        "duplicates": len(df) - df['phrase'].nunique(),
        "duplicate_percentage": ((len(df) - df['phrase'].nunique()) / len(df)) * 100
    },

    "lexical_diversity": {
        "total_tokens": len(tokens),
        "unique_tokens": len(set(tokens)),
        "ttr": len(set(tokens)) / len(tokens),
        "top_30_words": [
            {"word": word, "count": count, "percentage": (count / len(tokens)) * 100}
            for word, count in word_freq.most_common(30)
        ]
    },

    "phrase_length": {
        "words": {
            "min": int(df['word_count'].min()),
            "max": int(df['word_count'].max()),
            "mean": float(df['word_count'].mean()),
            "median": float(df['word_count'].median()),
            "std": float(df['word_count'].std())
        },
        "characters": {
            "min": int(df['char_count'].min()),
            "max": int(df['char_count'].max()),
            "mean": float(df['char_count'].mean()),
            "median": float(df['char_count'].median()),
            "std": float(df['char_count'].std())
        }
    },

    "intent_distribution": {
        "top_10": [
            {"intent": intent, "count": int(count), "percentage": (count / len(df)) * 100}
            for intent, count in df['intent'].value_counts().head(10).items()
        ],
        "total_intents": int(df['intent'].nunique())
    },

    "top_duplicates": [
        {"phrase": phrase, "count": int(count)}
        for phrase, count in df['phrase'].value_counts().head(20).items()
        if count > 1
    ],

    "first_word_patterns": {
        "top_20": [
            {"word": word, "count": int(count), "percentage": (count / len(df)) * 100}
            for word, count in df['first_word'].value_counts().head(20).items()
        ]
    },

    "bigram_patterns": {
        "top_20": [
            {"pattern": pattern, "count": int(count), "percentage": (count / len(df)) * 100}
            for pattern, count in df['first_2_words'].value_counts().head(20).items()
        ]
    },

    "intent_statistics": []
}

# Статистика по интентам
for intent, group in df.groupby('intent'):
    unique = group['phrase'].nunique()
    total = len(group)
    dup_pct = ((total - unique) / total * 100) if total > 0 else 0

    intent_stat = {
        "intent": intent,
        "total": total,
        "unique": unique,
        "duplicates": total - unique,
        "duplicate_percentage": dup_pct,
        "avg_word_length": float(group['word_count'].mean()),
        "std_word_length": float(group['word_count'].std()) if len(group) > 1 else 0,
        "top_phrases": [
            {"phrase": phrase, "count": int(count)}
            for phrase, count in group['phrase'].value_counts().head(5).items()
        ]
    }
    stats["intent_statistics"].append(intent_stat)

# Сортировка по проценту дубликатов
stats["intent_statistics"] = sorted(
    stats["intent_statistics"],
    key=lambda x: x['duplicate_percentage'],
    reverse=True
)

# Проблемы
problems = []

if stats["dataset_info"]["duplicate_percentage"] > 5:
    problems.append({
        "level": "CRITICAL",
        "type": "High duplicate rate",
        "value": stats["dataset_info"]["duplicate_percentage"],
        "description": f"Duplicate percentage is {stats['dataset_info']['duplicate_percentage']:.2f}%, should be < 5%"
    })

if stats["lexical_diversity"]["ttr"] < 0.3:
    problems.append({
        "level": "CRITICAL",
        "type": "Low lexical diversity",
        "value": stats["lexical_diversity"]["ttr"],
        "description": f"TTR is {stats['lexical_diversity']['ttr']:.4f}, should be > 0.3"
    })

if df['word_count'].std() < 2:
    problems.append({
        "level": "WARNING",
        "type": "Low phrase length diversity",
        "value": float(df['word_count'].std()),
        "description": f"Phrase length std is {df['word_count'].std():.2f}, should be > 2"
    })

top_first_word_pct = (df['first_word'].value_counts().iloc[0] / len(df)) * 100
if top_first_word_pct > 20:
    problems.append({
        "level": "CRITICAL",
        "type": "High concentration of first word",
        "value": top_first_word_pct,
        "description": f"Top first word '{df['first_word'].value_counts().index[0]}' appears in {top_first_word_pct:.2f}% of phrases"
    })

stats["problems"] = problems

# Рекомендации
recommendations = []

if stats["dataset_info"]["duplicate_percentage"] > 50:
    recommendations.append({
        "priority": "HIGH",
        "action": "Remove all duplicate phrases",
        "expected_impact": f"Will reduce dataset from {stats['dataset_info']['total_phrases']} to {stats['dataset_info']['unique_phrases']} phrases"
    })

if stats["lexical_diversity"]["ttr"] < 0.3:
    recommendations.append({
        "priority": "HIGH",
        "action": "Generate synonym variations for each phrase",
        "expected_impact": "Should increase TTR from {:.4f} to > 0.3".format(stats['lexical_diversity']['ttr'])
    })

recommendations.append({
    "priority": "MEDIUM",
    "action": "Add short commands (1-2 words)",
    "expected_impact": "Will increase phrase length diversity"
})

recommendations.append({
    "priority": "MEDIUM",
    "action": "Add polite long phrases (6-10 words)",
    "expected_impact": "Will increase phrase length diversity and politeness"
})

stats["recommendations"] = recommendations

# Сохранение в JSON
output_file = '/Users/ilanizankovskij/Documents/vs_code/synt_gen/diversity_statistics.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print(f"Statistics saved to: {output_file}")
print(f"\nKey findings:")
print(f"  - Total phrases: {stats['dataset_info']['total_phrases']}")
print(f"  - Unique phrases: {stats['dataset_info']['unique_phrases']}")
print(f"  - Duplicate rate: {stats['dataset_info']['duplicate_percentage']:.2f}%")
print(f"  - TTR: {stats['lexical_diversity']['ttr']:.4f}")
print(f"  - Total problems found: {len(problems)}")
print(f"  - Recommendations: {len(recommendations)}")
