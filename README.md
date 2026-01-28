# Генератор датасета для голосового ассистента автомобиля

Система для генерации разнообразных фраз на русском языке для обучения голосового ассистента в автомобиле.

## Структура проекта

```
synt_gen/
├── scripts/          # Основные скрипты генерации
│   ├── pypline.py           # Оригинальный генератор
│   ├── pypline_focused.py   # Фокусированная генерация по интентам
│   ├── pypline_smart.py     # Умная генерация с контролем параметров
│   └── batch_control.py     # Дедупликация и балансировка
│
├── config/           # Конфигурация промтов
│   ├── key.py              # Основной промт
│   ├── key_focused.py      # Фокусированный промт
│   └── key_minimal.py      # Минималистичный промт без примеров
│
├── quality/          # Инструменты контроля качества
│   ├── quality_control.py         # Анализ качества датасета
│   ├── analyze_diversity.py       # Анализ разнообразия
│   ├── generate_statistics.py     # Генерация статистики
│   └── export_problem_intents.py  # Экспорт проблемных интентов
│
├── data/             # Датасеты
│   ├── raw/         # Исходные датасеты
│   ├── generated/   # Сгенерированные датасеты
│   └── backups/     # Резервные копии
│
├── analysis/         # Результаты анализа
│   ├── problem_intents/  # Проблемные интенты
│   └── *.json           # JSON отчеты
│
├── docs/             # Документация
│   ├── SOLUTION.md                    # Решение проблемы разнообразия
│   ├── DIVERSITY_STRATEGY.md          # Стратегия улучшения разнообразия
│   ├── QUICK_START.md                 # Быстрый старт
│   └── DIVERSITY_ANALYSIS_REPORT.md   # Детальный отчет
│
└── samples/          # Примеры и тесты

```

## 🚀 Быстрый старт

### 1. Тест на одном интенте (30 секунд)

```bash
python scripts/pypline_smart.py \
  --csv data/generated/test.csv \
  --samples 30 \
  --intents ac_set \
  --max-repeats 2
```

### 2. Проверка качества

```bash
python quality/quality_control.py data/generated/test.csv
```

### 3. Полная генерация (рекомендуется)

```bash
python scripts/pypline_smart.py \
  --csv data/generated/dataset_500.csv \
  --samples 100 \
  --max-repeats 2
```

## 📊 Метрики качества

**Хорошие показатели:**
- ✅ Duplicates: < 5%
- ✅ TTR (Type-Token Ratio): > 0.30
- ✅ Уникальных параметров: > 15 для каждого интента

**Плохие показатели:**
- ❌ Duplicates: > 20%
- ❌ TTR: < 0.10
- ❌ Повторение одного параметра > 5 раз

## 🛠 Основные команды

### Генерация с контролем параметров

```bash
# Топ-5 интентов, 500 фраз
python scripts/pypline_smart.py \
  --csv data/generated/dataset.csv \
  --samples 100 \
  --intents call_answer volume_change ac_set window_set ac_off \
  --max-repeats 2 \
  --temperature 1.4
```

### Дедупликация и балансировка

```bash
# Очистка датасета с резервной копией
python scripts/batch_control.py data/generated/dataset.csv
```

### Анализ разнообразия

```bash
# Детальная статистика
python quality/analyze_diversity.py data/generated/dataset.csv

# Экспорт проблемных интентов
python quality/export_problem_intents.py data/generated/dataset.csv
```

## 📝 Доступные интенты

- **call_answer** - Ответить на входящий звонок
- **volume_change** - Изменить громкость (delta: -20 до +20)
- **ac_set** - Включить кондиционер (температура: 16-30)
- **window_set** - Управление окнами (open/close/set, driver/passenger/all)
- **ac_off** - Выключить кондиционер

## 🔧 Параметры генерации

| Параметр | Значение | Описание |
|----------|----------|----------|
| `--samples` | 50-100 | Количество фраз на интент |
| `--max-repeats` | 1-3 | Макс повторов одной комбинации параметров |
| `--temperature` | 1.2-1.5 | Креативность модели |
| `--model` | gpt-4o-mini | Модель OpenAI |

## 💡 Стратегии улучшения качества

1. **Используйте pypline_smart.py** вместо pypline.py для контроля параметров
2. **Уменьшите max-repeats** до 1-2 для максимального разнообразия
3. **Повысьте temperature** до 1.4-1.5 для более креативных фраз
4. **Генерируйте меньшими порциями** (30-50 фраз) для лучшего контроля

## 📖 Документация

- [SOLUTION.md](docs/SOLUTION.md) - Решение проблемы разнообразия параметров
- [QUICK_START.md](docs/QUICK_START.md) - Подробный гайд по быстрому старту
- [DIVERSITY_STRATEGY.md](docs/DIVERSITY_STRATEGY.md) - Стратегия улучшения разнообразия

## ⚠️ Известные проблемы

1. GPT копирует конкретные числа из примеров → используйте `key_minimal.py`
2. Низкое разнообразие при больших батчах → генерируйте по 30-50 фраз
3. Дисбаланс интентов → используйте `batch_control.py` для балансировки

## 🔑 Настройка API

Добавьте ваш OpenAI API ключ в `config/key.py`:

```python
OPENAI_API_KEY = "sk-..."
```

## 💰 Стоимость генерации

**GPT-4o-mini** ($0.15 / $0.60 за 1M токенов):
- 100,000 строк ≈ $8-9

**GPT-4o** ($0.25 / $2.00 за 1M токенов):
- 100,000 строк ≈ $23-24

---

Создано для генерации высококачественного датасета голосового ассистента 🚗
