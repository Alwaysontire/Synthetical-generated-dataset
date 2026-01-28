# 🚀 Быстрый старт: Решение проблемы разнообразия

## Проблема

**Дубликатов мало, но разнообразия нет** — модель генерирует шаблонные фразы.

## Решение

Используйте **фокусированную генерацию** — каждый интент отдельно с максимальным разнообразием.

---

## ⚡ Быстрый тест (30 секунд)

Протестируйте новый подход на 1 интенте:

```bash
cd /Users/ilanizankovskij/Documents/vs_code/synt_gen

# Генерация 30 разных способов ответить на звонок
python pypline_focused.py --csv test_call_answer.csv --samples 30 --intents call_answer

# Проверка качества
python quality_control.py test_call_answer.csv
```

**Ожидаемый результат:**
- 30 уникальных фраз
- 0 дубликатов
- TTR > 0.50

**Примеры фраз:**
- "Алло"
- "Да"
- "Слушаю"
- "Прими вызов"
- "Возьми трубку"
- "Кто звонит?"
- "Давай ответим"
- "Бери трубку"
- ... (30 разных способов!)

---

## 📊 Полная генерация (рекомендуемая)

### Вариант 1: Топ-5 интентов (500 фраз, ~5 минут)

```bash
python pypline_focused.py --csv dataset_top5_500.csv --samples 100 \
  --intents call_answer volume_change ac_set window_set ac_off
```

**Результат:**
- 500 уникальных фраз
- TTR > 0.40
- Дубликаты < 2%
- По 100 разных способов для каждого действия

---

### Вариант 2: Все доступные интенты (250 фраз, ~3 минуты)

```bash
python pypline_focused.py --csv dataset_all_250.csv --samples 50
```

**Результат:**
- 250 уникальных фраз (50 × 5 интентов)
- По 50 разных способов для каждого действия
- Максимальное разнообразие

---

### Вариант 3: С повышенной креативностью

```bash
python pypline_focused.py --csv dataset_creative.csv --samples 50 --temperature 1.4
```

**Параметры:**
- `--temperature 1.4` — больше креативности (возможны странные фразы)
- `--temperature 1.0` — меньше креативности (более естественно)

---

## 🔄 Сравнение подходов

### Старый подход (pypline.py)

```bash
python pypline.py --csv test_old.csv --n 100
```

**Проблема:**
- GPT пытается распределить 100 фраз между 30+ интентами
- По 2-3 фразы на интент
- Шаблонные фразы без разнообразия

**Результат:**
- TTR = 0.02-0.05 (очень низкое)
- 50-70% дубликатов
- "Ответь на звонок" × 245 раз

---

### Новый подход (pypline_focused.py)

```bash
python pypline_focused.py --csv test_new.csv --samples 30 --intents call_answer
```

**Преимущества:**
- GPT фокусируется на 1 интенте
- Генерирует 30 разных способов сказать одно и то же
- Максимальное разнообразие

**Результат:**
- TTR > 0.40 (в 20 раз выше!)
- 0-2% дубликатов
- 30 уникальных способов ответить на звонок

---

## 📁 Какой файл использовать?

| Файл | Описание | Когда использовать |
|------|----------|-------------------|
| **pypline_focused.py** | Фокусированная генерация по интентам | **РЕКОМЕНДУЕТСЯ** для максимального разнообразия |
| pypline.py | Оригинальный скрипт (обновлён промт) | Для быстрой генерации без контроля |
| key_focused.py | Промт для фокусированной генерации | Используется в pypline_focused.py |
| key.py | Обновлённый промт без балансировки | Используется в pypline.py |

---

## 🎯 Рекомендуемая команда

Для начала используйте:

```bash
python pypline_focused.py --csv my_dataset.csv --samples 50
```

Это сгенерирует 250 уникальных фраз (50 × 5 интентов) с максимальным разнообразием.

---

## 📊 Как проверить качество?

После генерации всегда проверяйте:

```bash
python quality_control.py my_dataset.csv
```

**Хорошие метрики:**
- ✅ Duplicates: < 5%
- ✅ TTR (Lexical Diversity): > 0.30
- ✅ Top Starting Word: < 10%
- ✅ Standard Deviation (Length): > 2.0

**Плохие метрики:**
- ❌ Duplicates: > 20%
- ❌ TTR: < 0.10
- ❌ Top Starting Word: > 15%
- ❌ Standard Deviation: < 1.5

---

## 🔧 Параметры pypline_focused.py

```bash
python pypline_focused.py --help

Параметры:
  --csv ПУТЬ              Путь к выходному CSV (по умолчанию: dataset_focused.csv)
  --samples N             Количество фраз на каждый интент (по умолчанию: 50)
  --model ИМЯ             Модель OpenAI (по умолчанию: gpt-4o-mini)
  --temperature ЧИСЛО     Креативность 0.0-2.0 (по умолчанию: 1.2)
  --top-p ЧИСЛО           Top-p sampling 0.0-1.0 (по умолчанию: 0.95)
  --intents СПИСОК        Список интентов (по умолчанию: все)
```

**Примеры:**

```bash
# Только call_answer, 100 фраз
python pypline_focused.py --samples 100 --intents call_answer

# Топ-3 интента с высокой креативностью
python pypline_focused.py --samples 80 --temperature 1.5 \
  --intents call_answer volume_change ac_set

# Использовать GPT-4o (дороже, но качественнее)
python pypline_focused.py --samples 50 --model gpt-4o
```

---

## 🆘 Что делать, если все равно низкое качество?

1. **Увеличьте temperature:**
   ```bash
   python pypline_focused.py --samples 50 --temperature 1.5
   ```

2. **Генерируйте меньшими порциями:**
   ```bash
   python pypline_focused.py --samples 20 --intents call_answer
   python pypline_focused.py --samples 20 --intents volume_change
   # и т.д.
   ```

3. **Используйте более мощную модель:**
   ```bash
   python pypline_focused.py --samples 50 --model gpt-4o
   ```

---

## ✅ Следующие шаги

1. Протестируйте на 1 интенте (30 секунд)
2. Если результат хороший — генерируйте полный датасет
3. Проверьте качество через quality_control.py
4. Объедините с другими данными и балансируйте через batch_control.py

---

Удачи! 🚀

**Подробная инструкция:** [DIVERSITY_STRATEGY.md](DIVERSITY_STRATEGY.md)
