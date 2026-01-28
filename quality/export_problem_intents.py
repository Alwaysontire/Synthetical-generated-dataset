#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Экспорт проблемных интентов для ручной доработки
"""

import pandas as pd
import os

# Загрузка данных
df = pd.read_csv('/Users/ilanizankovskij/Documents/vs_code/synt_gen/sample_10000.csv')

# Создание директории для экспорта
output_dir = '/Users/ilanizankovskij/Documents/vs_code/synt_gen/problem_intents'
os.makedirs(output_dir, exist_ok=True)

# Вычисление статистики по интентам
intent_stats = []
for intent, group in df.groupby('intent'):
    total = len(group)
    unique = group['phrase'].nunique()
    dup_pct = ((total - unique) / total * 100) if total > 0 else 0

    intent_stats.append({
        'intent': intent,
        'total': total,
        'unique': unique,
        'dup_pct': dup_pct
    })

intent_stats = sorted(intent_stats, key=lambda x: x['dup_pct'], reverse=True)

# Топ-10 проблемных интентов (> 70% дубликатов)
problem_intents = [stat for stat in intent_stats if stat['dup_pct'] > 70]

print("=" * 80)
print("ЭКСПОРТ ПРОБЛЕМНЫХ ИНТЕНТОВ")
print("=" * 80)
print(f"\nНайдено проблемных интентов (> 70% дубликатов): {len(problem_intents)}")

for idx, stat in enumerate(problem_intents, 1):
    intent = stat['intent']
    print(f"\n{idx}. {intent}: {stat['dup_pct']:.1f}% дубликатов ({stat['total']} фраз, {stat['unique']} уникальных)")

    # Получить все уникальные фразы для этого интента
    group = df[df['intent'] == intent]
    unique_phrases = group.drop_duplicates(subset=['phrase'])

    # Сохранить в отдельный файл
    output_file = os.path.join(output_dir, f"{intent}.txt")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Интент: {intent}\n")
        f.write(f"# Статистика:\n")
        f.write(f"#   Всего фраз в датасете: {stat['total']}\n")
        f.write(f"#   Уникальных фраз: {stat['unique']}\n")
        f.write(f"#   Дубликатов: {stat['dup_pct']:.1f}%\n")
        f.write(f"#\n")
        f.write(f"# ЗАДАЧА: Создать минимум {stat['unique'] * 3} разнообразных вариантов\n")
        f.write(f"#\n")
        f.write(f"# Текущие уникальные фразы:\n")
        f.write("#" + "=" * 79 + "\n\n")

        # Записать все уникальные фразы с количеством их повторений
        phrase_counts = group['phrase'].value_counts()

        for phrase in unique_phrases['phrase']:
            count = phrase_counts[phrase]
            f.write(f"# [{count:3d}x] {phrase}\n")

        f.write("\n" + "#" + "=" * 79 + "\n")
        f.write("# ДОБАВЬТЕ НОВЫЕ ВАРИАНТЫ НИЖЕ:\n")
        f.write("#" + "=" * 79 + "\n\n")

        # Примеры для вдохновения
        f.write("# Примеры разнообразия:\n")
        f.write("#\n")
        f.write("# 1. Короткие (1-2 слова)\n")
        f.write("#    - Да, Стоп, Громче, Тише\n")
        f.write("#\n")
        f.write("# 2. Средние (3-5 слов)\n")
        f.write("#    - Включи музыку, Открой окно водителя\n")
        f.write("#\n")
        f.write("# 3. Длинные/вежливые (6-10 слов)\n")
        f.write("#    - Пожалуйста, сделай чуть погромче музыку\n")
        f.write("#    - Не могла бы ты открыть окно?\n")
        f.write("#\n")
        f.write("# 4. Вопросительные\n")
        f.write("#    - Можно громче?, Открыть окно?\n")
        f.write("#\n")
        f.write("# 5. Разговорные\n")
        f.write("#    - Врубь музыку, Давай громче\n")
        f.write("#\n")
        f.write("#" + "=" * 79 + "\n\n")

    print(f"   Экспортировано в: {output_file}")

# Создать сводный файл
summary_file = os.path.join(output_dir, "_SUMMARY.txt")
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("СВОДКА ПО ПРОБЛЕМНЫМ ИНТЕНТАМ\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Всего проблемных интентов: {len(problem_intents)}\n\n")

    f.write("Приоритет исправления (по убыванию критичности):\n\n")

    for idx, stat in enumerate(problem_intents, 1):
        f.write(f"{idx:2d}. {stat['intent']:<30} | "
                f"Дубликаты: {stat['dup_pct']:5.1f}% | "
                f"Всего: {stat['total']:4d} | "
                f"Уникальных: {stat['unique']:3d}\n")

        target_phrases = max(50, stat['unique'] * 3)
        f.write(f"    Целевое количество вариантов: {target_phrases}\n")
        f.write(f"    Нужно добавить: {target_phrases - stat['unique']} новых фраз\n\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("РЕКОМЕНДАЦИИ\n")
    f.write("=" * 80 + "\n\n")

    f.write("1. Начните с интентов с наибольшим процентом дубликатов\n")
    f.write("2. Для каждого интента создайте минимум в 3 раза больше уникальных фраз\n")
    f.write("3. Используйте разные:\n")
    f.write("   - Глаголы (включи, запусти, поставь, активируй)\n")
    f.write("   - Длины фраз (короткие, средние, длинные)\n")
    f.write("   - Структуры (императив, вопрос, вежливая форма)\n")
    f.write("   - Стили (формальный, разговорный)\n\n")

    f.write("4. Примеры улучшения:\n\n")

    f.write("   call_answer - 'Ответь на звонок' (245x повторений)\n")
    f.write("   Варианты:\n")
    f.write("   - Короткие: Да, Прими, Подними, Алло\n")
    f.write("   - Средние: Возьми трубку, Прими звонок, Кто звонит?\n")
    f.write("   - Длинные: Можно ответить на звонок?, Пожалуйста, ответь\n")
    f.write("   - Разговорные: Давай ответим, Беру трубку\n\n")

    f.write("   ac_off - 'Выключи кондиционер' (217x повторений)\n")
    f.write("   Варианты:\n")
    f.write("   - Короткие: AC офф, Вырубь кондер\n")
    f.write("   - Средние: Отключи кондиционер, Останови климат\n")
    f.write("   - Длинные: Холодно стало, выключи кондиционер\n")
    f.write("   - Вопросительные: Можно выключить климат?\n\n")

print(f"\nСводный файл создан: {summary_file}")

# Создать файл с инструкциями
instructions_file = os.path.join(output_dir, "_INSTRUCTIONS.txt")
with open(instructions_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("ИНСТРУКЦИИ ПО УЛУЧШЕНИЮ ДАТАСЕТА\n")
    f.write("=" * 80 + "\n\n")

    f.write("ЦЕЛЬ:\n")
    f.write("Увеличить разнообразие фраз для каждого интента\n\n")

    f.write("ТЕКУЩЕЕ СОСТОЯНИЕ:\n")
    f.write("- 69.24% дубликатов в датасете\n")
    f.write("- TTR (лексическое разнообразие) = 0.0233 (критически низкое)\n")
    f.write("- Шаблонность конструкций\n\n")

    f.write("ЦЕЛЕВОЕ СОСТОЯНИЕ:\n")
    f.write("- < 5% дубликатов\n")
    f.write("- TTR > 0.3\n")
    f.write("- Разнообразие длины фраз\n")
    f.write("- > 15% вежливых формулировок\n")
    f.write("- > 15% вопросительных конструкций\n\n")

    f.write("ПРОЦЕСС РАБОТЫ:\n\n")

    f.write("ШАГ 1: Изучите текущие фразы\n")
    f.write("- Откройте файл для каждого интента (например, call_answer.txt)\n")
    f.write("- Посмотрите на существующие уникальные фразы\n")
    f.write("- Обратите внимание на паттерны и повторения\n\n")

    f.write("ШАГ 2: Создайте новые варианты\n")
    f.write("- Для каждой существующей фразы придумайте 5-10 синонимичных вариантов\n")
    f.write("- Используйте разные глаголы\n")
    f.write("- Варьируйте длину фраз\n")
    f.write("- Добавляйте вежливые слова\n")
    f.write("- Создавайте вопросительные конструкции\n\n")

    f.write("ШАГ 3: Соблюдайте разнообразие\n\n")

    f.write("A. РАЗНООБРАЗИЕ ГЛАГОЛОВ:\n")
    f.write("   Вместо 'включи': запусти, поставь, активируй, врубь, дай, зажги\n")
    f.write("   Вместо 'выключи': отключи, останови, вырубь, убери, деактивируй\n")
    f.write("   Вместо 'сделай': поставь, установи, настрой, выстави, измени\n\n")

    f.write("B. РАЗНООБРАЗИЕ ДЛИНЫ:\n")
    f.write("   Короткие (1-2 слова): Громче, Тише, Стоп, Пауза, Далее\n")
    f.write("   Средние (3-5 слов): Включи кондиционер, Открой окно водителя\n")
    f.write("   Длинные (6-10 слов): Можно немного прибавить температуру?\n\n")

    f.write("C. РАЗНООБРАЗИЕ СТРУКТУР:\n")
    f.write("   Императивные: Включи музыку\n")
    f.write("   Вопросительные: Можно включить музыку?\n")
    f.write("   Вежливые: Пожалуйста, включи музыку\n")
    f.write("   Разговорные: Врубь музыку, Давай музыку\n\n")

    f.write("ШАГ 4: Запишите новые фразы\n")
    f.write("- Добавьте каждую новую фразу в соответствующий файл интента\n")
    f.write("- Записывайте по одной фразе на строку\n")
    f.write("- Не дублируйте существующие фразы\n\n")

    f.write("ШАГ 5: Проверка\n")
    f.write("- Убедитесь, что создали минимум в 3 раза больше фраз\n")
    f.write("- Проверьте разнообразие глаголов, длины и структур\n")
    f.write("- Прочитайте фразы вслух - звучат ли они естественно?\n\n")

    f.write("ПРИМЕРЫ ХОРОШЕГО РАЗНООБРАЗИЯ:\n\n")

    f.write("Интент: call_answer\n")
    f.write("Плохо (1 вариант, повторяется 245 раз):\n")
    f.write("  - Ответь на звонок\n\n")

    f.write("Хорошо (50+ вариантов):\n")
    f.write("  Короткие:\n")
    f.write("  - Да\n")
    f.write("  - Прими\n")
    f.write("  - Подними\n")
    f.write("  - Алло\n")
    f.write("  - Отвечай\n\n")

    f.write("  Средние:\n")
    f.write("  - Ответь на звонок\n")
    f.write("  - Прими звонок\n")
    f.write("  - Возьми трубку\n")
    f.write("  - Подними трубку\n")
    f.write("  - Кто звонит?\n\n")

    f.write("  Длинные:\n")
    f.write("  - Можно ответить на звонок?\n")
    f.write("  - Пожалуйста, ответь на входящий\n")
    f.write("  - Не могла бы ты ответить?\n")
    f.write("  - Давай возьмем трубку\n\n")

    f.write("  Вопросительные:\n")
    f.write("  - Ответить?\n")
    f.write("  - Взять?\n")
    f.write("  - Можно ответить?\n\n")

    f.write("  Разговорные:\n")
    f.write("  - Давай ответим\n")
    f.write("  - Возьмем трубку?\n")
    f.write("  - Бери трубку\n\n")

    f.write("=" * 80 + "\n")
    f.write("УСПЕХОВ В УЛУЧШЕНИИ ДАТАСЕТА!\n")
    f.write("=" * 80 + "\n")

print(f"Инструкции созданы: {instructions_file}")

print("\n" + "=" * 80)
print("ЭКСПОРТ ЗАВЕРШЕН")
print("=" * 80)
print(f"\nВсе файлы находятся в: {output_dir}")
print(f"\nСледующие шаги:")
print(f"1. Прочитайте {instructions_file}")
print(f"2. Изучите {summary_file}")
print(f"3. Начните доработку с файлов с наибольшим процентом дубликатов")
