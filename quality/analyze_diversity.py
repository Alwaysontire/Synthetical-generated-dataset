import pandas as pd
import numpy as np
from collections import Counter
import re
import json

df = pd.read_csv('/Users/ilanizankovskij/Documents/vs_code/synt_gen/sample_10000.csv')


unique_phrases = df['phrase'].nunique()
total_phrases = len(df)
duplicates = total_phrases - unique_phrases
duplication_rate = (duplicates / total_phrases) * 100

print(f"\nУникальные фразы: {unique_phrases}")
print(f"Дубликаты (полные совпадения): {duplicates}")
print(f"Процент дубликатов: {duplication_rate:.2f}%")


phrase_counts = df['phrase'].value_counts()
top_duplicates = phrase_counts[phrase_counts > 1].head(20)
print(f"\nТОП-20 наиболее повторяющихся фраз:")
for phrase, count in top_duplicates.items():
    print(f"  {count:4d}x: {phrase}")

print(f"\nТОП-10 самых частых интентов:")
intent_counts = df['intent'].value_counts().head(10)
for intent, count in intent_counts.items():
    percent = (count / total_phrases) * 100
    print(f"  {intent:25s}: {count:5d} ({percent:5.2f}%)")

# ============================================================================
# 2. ЛЕКСИЧЕСКОЕ РАЗНООБРАЗИЕ (TTR)
# ============================================================================
print("\n" + "=" * 80)
print("2. ЛЕКСИЧЕСКОЕ РАЗНООБРАЗИЕ")
print("=" * 80)

# Токенизация всех фраз
all_text = ' '.join(df['phrase'].astype(str))
# Простая токенизация по словам (убираем знаки препинания)
tokens = re.findall(r'\b\w+\b', all_text.lower())

unique_tokens = len(set(tokens))
total_tokens = len(tokens)
ttr = unique_tokens / total_tokens

print(f"\nВсего токенов (слов): {total_tokens}")
print(f"Уникальных токенов: {unique_tokens}")
print(f"TTR (Type-Token Ratio): {ttr:.4f}")
print(f"  (Чем ближе к 1.0, тем больше разнообразие)")
print(f"  (Для хорошего разнообразия TTR должен быть > 0.3)")

# Самые частые слова
word_freq = Counter(tokens)
print(f"\nТОП-30 самых частых слов:")
for word, count in word_freq.most_common(30):
    percent = (count / total_tokens) * 100
    print(f"  {word:20s}: {count:5d} ({percent:5.2f}%)")

# ============================================================================
# 3. ПАТТЕРНЫ ПОВТОРЯЕМОСТИ
# ============================================================================
print("\n" + "=" * 80)
print("3. ПАТТЕРНЫ ПОВТОРЯЕМОСТИ")
print("=" * 80)

# Начальные слова фраз
first_words = df['phrase'].str.lower().str.split().str[0]
first_word_counts = first_words.value_counts().head(20)
print(f"\nТОП-20 начальных слов:")
for word, count in first_word_counts.items():
    percent = (count / total_phrases) * 100
    print(f"  {word:20s}: {count:5d} ({percent:5.2f}%)")

# Глаголы (приблизительный анализ по окончаниям и распространенным глаголам)
common_verbs = ['включи', 'выключи', 'открой', 'закрой', 'установи', 'сделай',
                'увеличь', 'уменьши', 'переключи', 'запусти', 'останови',
                'поставь', 'позвони', 'ответь', 'завершить', 'отмени',
                'приоткрой', 'покажи', 'найди', 'проложи', 'включить',
                'выключить', 'поменяй', 'измени', 'добавь', 'убери']

verb_usage = {}
for verb in common_verbs:
    count = df['phrase'].str.lower().str.contains(verb, na=False).sum()
    if count > 0:
        verb_usage[verb] = count

verb_usage_sorted = sorted(verb_usage.items(), key=lambda x: x[1], reverse=True)
print(f"\nЧастота использования глаголов:")
for verb, count in verb_usage_sorted[:20]:
    percent = (count / total_phrases) * 100
    print(f"  {verb:20s}: {count:5d} ({percent:5.2f}%)")

# Поиск шаблонных конструкций (n-граммы начала фраз)
print(f"\nШаблонные конструкции (начало фраз, первые 2-3 слова):")
df['first_2_words'] = df['phrase'].str.lower().str.split().str[:2].str.join(' ')
df['first_3_words'] = df['phrase'].str.lower().str.split().str[:3].str.join(' ')

two_word_patterns = df['first_2_words'].value_counts().head(20)
print(f"\nТОП-20 биграмм (2 первых слова):")
for pattern, count in two_word_patterns.items():
    percent = (count / total_phrases) * 100
    print(f"  {count:4d}x ({percent:5.2f}%): '{pattern}'")

three_word_patterns = df['first_3_words'].value_counts().head(20)
print(f"\nТОП-20 триграмм (3 первых слова):")
for pattern, count in three_word_patterns.items():
    percent = (count / total_phrases) * 100
    print(f"  {count:4d}x ({percent:5.2f}%): '{pattern}'")

# ============================================================================
# 4. РАЗНООБРАЗИЕ СИНТАКСИСА
# ============================================================================
print("\n" + "=" * 80)
print("4. РАЗНООБРАЗИЕ СИНТАКСИСА")
print("=" * 80)

# Длина фраз (в словах)
df['word_count'] = df['phrase'].str.split().str.len()

print(f"\nСтатистика длины фраз (в словах):")
print(f"  Минимум:              {df['word_count'].min()}")
print(f"  Максимум:             {df['word_count'].max()}")
print(f"  Среднее:              {df['word_count'].mean():.2f}")
print(f"  Медиана:              {df['word_count'].median():.1f}")
print(f"  Стандартное откл.:    {df['word_count'].std():.2f}")

# Распределение по длине
print(f"\nРаспределение по длине фраз:")
length_dist = df['word_count'].value_counts().sort_index()
for length, count in length_dist.items():
    percent = (count / total_phrases) * 100
    bar = '#' * int(percent / 2)
    print(f"  {length:2d} слов: {count:5d} ({percent:5.2f}%) {bar}")

# Длина фраз (в символах)
df['char_count'] = df['phrase'].str.len()

print(f"\nСтатистика длины фраз (в символах):")
print(f"  Минимум:              {df['char_count'].min()}")
print(f"  Максимум:             {df['char_count'].max()}")
print(f"  Среднее:              {df['char_count'].mean():.2f}")
print(f"  Медиана:              {df['char_count'].median():.1f}")
print(f"  Стандартное откл.:    {df['char_count'].std():.2f}")

# Знаки препинания
df['has_comma'] = df['phrase'].str.contains(',', na=False)
df['has_question'] = df['phrase'].str.contains(r'\?', na=False)
df['has_exclamation'] = df['phrase'].str.contains('!', na=False)

print(f"\nИспользование знаков препинания:")
print(f"  Фразы с запятыми:           {df['has_comma'].sum():5d} ({df['has_comma'].mean()*100:5.2f}%)")
print(f"  Фразы с вопросом:           {df['has_question'].sum():5d} ({df['has_question'].mean()*100:5.2f}%)")
print(f"  Фразы с восклицанием:       {df['has_exclamation'].sum():5d} ({df['has_exclamation'].mean()*100:5.2f}%)")

# Вежливые слова
polite_words = ['пожалуйста', 'спасибо', 'можно', 'будь добр']
polite_count = 0
for word in polite_words:
    polite_count += df['phrase'].str.lower().str.contains(word, na=False).sum()

print(f"  Фразы с вежливыми словами:  {polite_count:5d} ({polite_count/total_phrases*100:5.2f}%)")

# ============================================================================
# 5. ПРОБЛЕМЫ И РЕКОМЕНДАЦИИ
# ============================================================================
print("\n" + "=" * 80)
print("5. ВЫЯВЛЕННЫЕ ПРОБЛЕМЫ И РЕКОМЕНДАЦИИ")
print("=" * 80)

problems = []

# Проверка дубликатов
if duplication_rate > 5:
    problems.append(f"КРИТИЧНО: Высокий процент дубликатов ({duplication_rate:.2f}%). Рекомендуется удалить повторяющиеся фразы.")

# Проверка TTR
if ttr < 0.3:
    problems.append(f"КРИТИЧНО: Низкое лексическое разнообразие (TTR={ttr:.4f}). Необходимо добавить больше синонимов и вариаций.")
elif ttr < 0.4:
    problems.append(f"ВНИМАНИЕ: Среднее лексическое разнообразие (TTR={ttr:.4f}). Рекомендуется улучшить.")

# Проверка концентрации начальных слов
top_first_word_percent = (first_word_counts.iloc[0] / total_phrases) * 100
if top_first_word_percent > 20:
    problems.append(f"КРИТИЧНО: Слишком высокая концентрация начального слова '{first_word_counts.index[0]}' ({top_first_word_percent:.2f}%). Нужно разнообразие структур предложений.")

# Проверка разнообразия длины фраз
if df['word_count'].std() < 2:
    problems.append(f"ВНИМАНИЕ: Низкое разнообразие длины фраз (std={df['word_count'].std():.2f}). Добавьте короткие и длинные варианты.")

# Проверка шаблонности
top_pattern_percent = (two_word_patterns.iloc[0] / total_phrases) * 100
if top_pattern_percent > 15:
    problems.append(f"ВНИМАНИЕ: Высокая шаблонность начала фраз ('{two_word_patterns.index[0]}' - {top_pattern_percent:.2f}%). Используйте разные структуры.")

# Проверка вежливости
if polite_count / total_phrases < 0.1:
    problems.append(f"РЕКОМЕНДАЦИЯ: Мало вежливых формулировок ({polite_count/total_phrases*100:.2f}%). Добавьте 'пожалуйста', 'можно' и т.д.")

# Проверка вопросов
if df['has_question'].mean() < 0.05:
    problems.append(f"РЕКОМЕНДАЦИЯ: Мало вопросительных конструкций ({df['has_question'].mean()*100:.2f}%). Добавьте вопросы типа 'Как...?', 'Можно...?'")

if problems:
    for i, problem in enumerate(problems, 1):
        print(f"\n{i}. {problem}")
else:
    print("\nПроблем не обнаружено! Датасет имеет хорошее разнообразие.")

# ============================================================================
# 6. ПРИМЕРЫ ДЛЯ УЛУЧШЕНИЯ
# ============================================================================
print("\n" + "=" * 80)
print("6. КОНКРЕТНЫЕ ПРИМЕРЫ ПРОБЛЕМ")
print("=" * 80)

# Показать примеры самых частых дубликатов
if duplicates > 0:
    print(f"\nПримеры дубликатов (показаны первые 5):")
    for i, (phrase, count) in enumerate(top_duplicates.head(5).items(), 1):
        print(f"\n{i}. Повторяется {count} раз:")
        print(f"   '{phrase}'")
        # Показать интенты для этой фразы
        intent_for_phrase = df[df['phrase'] == phrase]['intent'].value_counts()
        print(f"   Интенты: {', '.join([f'{intent} ({cnt})' for intent, cnt in intent_for_phrase.items()])}")

# Показать примеры шаблонных конструкций
print(f"\n\nПримеры шаблонных конструкций (одинаковые начала):")
for i, (pattern, count) in enumerate(two_word_patterns.head(5).items(), 1):
    print(f"\n{i}. Начинается с '{pattern}' ({count} раз, {count/total_phrases*100:.2f}%):")
    examples = df[df['first_2_words'] == pattern]['phrase'].head(5).tolist()
    for example in examples:
        print(f"   - {example}")

print("\n" + "=" * 80)
print("АНАЛИЗ ЗАВЕРШЕН")
print("=" * 80)
