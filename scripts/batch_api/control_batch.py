"""
Система дедупликации и балансировки датасета для голосового ассистента

Этапы обработки:
1. Удаление дубликатов фраз (одинаковые тексты)
2. Балансировка интентов (удаление избыточных примеров)
"""
import csv
import json
import os
import shutil
import random
import re
import argparse
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
from datetime import datetime


def load_dataset(path: str) -> List[Dict]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))




def normalize_phrase(phrase: str) -> str:
    """
    Нормализация фразы для сравнения
    - Приводит к нижнему регистру
    - Удаляет лишние пробелы
    - Удаляет пробелы в начале/конце
    """
    normalized = phrase.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def deduplicate_phrases(dataset: List[Dict]) -> Tuple[List[Dict], int]:
    """
    Этап 1: Удаление дубликатов фраз

    Args:
        dataset: Список записей датасета

    Returns:
        Tuple[List[Dict], int]: (очищенный датасет, количество удаленных)
    """
    seen_phrases = set()
    unique_dataset = []
    duplicates_removed = 0

    for item in dataset:
        phrase = item.get('phrase', '')

        # Нормализуем фразу для проверки
        normalized = normalize_phrase(phrase)

        if normalized not in seen_phrases:
            # Уникальная фраза - добавляем
            seen_phrases.add(normalized)
            unique_dataset.append(item)
        else:
            # Дубликат - пропускаем
            duplicates_removed += 1
            if duplicates_removed <= 10:  # Показываем только первые 10
                print(f"  ⚠ Removed duplicate: '{phrase[:50]}'")

    if duplicates_removed > 10:
        print(f"  ... и еще {duplicates_removed - 10} дубликатов")

    return unique_dataset, duplicates_removed


def balance_intents(dataset: List[Dict], target_deviation: float = 0.3) -> Tuple[List[Dict], int]:
    """
    Этап 2: Балансировка интентов

    Удаляет избыточные записи из переполненных классов интентов.
    Максимальное количество записей на интент = avg * (1 + target_deviation)

    Args:
        dataset: Список записей датасета
        target_deviation: Максимальное отклонение от среднего (default: 0.3 = 30%)

    Returns:
        Tuple[List[Dict], int]: (сбалансированный датасет, количество удаленных)
    """
    # Подсчет записей по интентам
    intent_counts = Counter([item['intent'] for item in dataset])

    if len(intent_counts) == 0:
        return dataset, 0

    # Вычисляем среднее и максимум
    avg_count = sum(intent_counts.values()) / len(intent_counts)
    max_allowed = int(avg_count * (1 + target_deviation))

    print(f"\n  Среднее количество на интент: {avg_count:.1f}")
    print(f"  Максимально допустимое: {max_allowed}")

    # Группируем датасет по интентам
    grouped = defaultdict(list)
    for item in dataset:
        grouped[item['intent']].append(item)

    # Балансируем
    balanced_dataset = []
    removed_count = 0

    for intent, items in grouped.items():
        current_count = len(items)

        if current_count <= max_allowed:
            # Интент в пределах нормы - оставляем все
            balanced_dataset.extend(items)
        else:
            # Интент переполнен - случайно отбираем max_allowed записей
            selected = random.sample(items, max_allowed)
            balanced_dataset.extend(selected)
            removed = current_count - len(selected)
            removed_count += removed

            print(f"  ⚠ Intent '{intent}' reduced: {current_count} -> {len(selected)} (-{removed})")

    return balanced_dataset, removed_count


class Batch_Norm:
    """
    Класс для дедупликации и балансировки датасета

    Использование:
        processor = Batch_Norm('dataset_car.csv')
        processor.run_deduplication()
        processor.run_balancing(target_deviation=0.3)
        processor.save_with_backup()
        processor.print_summary()
    """

    def __init__(self, dataset_path: str):
        """
        Инициализация обработчика

        Args:
            dataset_path: Путь к файлу датасета (CSV или JSON)
        """
        self.dataset_path = dataset_path
        self.dataset = load_dataset(dataset_path)
        self.original_size = len(self.dataset)
        self.stats = {
            'original_size': self.original_size,
            'duplicates_removed': 0,
            'balance_removed': 0,
            'final_size': 0
        }

    def run_deduplication(self) -> None:
        """Этап 1: Удаление дубликатов фраз"""
        print("="*60)
        print("ЭТАП 1: Удаление дубликатов фраз")
        print("="*60)

        self.dataset, removed = deduplicate_phrases(self.dataset)
        self.stats['duplicates_removed'] = removed

        print(f"\n✓ Удалено дубликатов: {removed}")
        print(f"✓ Осталось записей: {len(self.dataset)}")

    def run_balancing(self, target_deviation: float = 0.3) -> None:
        """
        Этап 2: Балансировка интентов

        Args:
            target_deviation: Максимальное отклонение от среднего (default: 0.3)
        """
        print("\n" + "="*60)
        print("ЭТАП 2: Балансировка интентов")
        print("="*60)

        # Показать текущее состояние
        intent_counts = Counter([item['intent'] for item in self.dataset])
        print(f"\nТекущее распределение интентов (топ-10):")
        for intent, count in intent_counts.most_common(10):
            print(f"  {intent}: {count}")

        self.dataset, removed = balance_intents(self.dataset, target_deviation)
        self.stats['balance_removed'] = removed

        print(f"\n✓ Удалено для балансировки: {removed}")
        print(f"✓ Осталось записей: {len(self.dataset)}")

    def save_with_backup(self) -> None:
        """Сохранение датасета с созданием backup оригинала"""
        # Создать backup оригинального файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.dataset_path.endswith('.csv'):
            backup_path = self.dataset_path.replace('.csv', f'_backup_{timestamp}.csv')
        elif self.dataset_path.endswith('.json'):
            backup_path = self.dataset_path.replace('.json', f'_backup_{timestamp}.json')
        else:
            backup_path = f"{self.dataset_path}_backup_{timestamp}"

        if os.path.exists(self.dataset_path):
            shutil.copy2(self.dataset_path, backup_path)
            print(f"\n📦 Backup создан: {backup_path}")

        # Сохранить очищенный датасет (перезаписать оригинал)
        if self.dataset_path.endswith('.csv'):
            self._save_dataset_csv(self.dataset_path)
            # Также сохранить JSON версию
            json_path = self.dataset_path.replace('.csv', '.json')
            self._save_dataset_json(json_path)
            print(f"✓ Датасет сохранен: {self.dataset_path}")
            print(f"✓ JSON версия: {json_path}")
        else:
            self._save_dataset_json(self.dataset_path)
            # Также сохранить CSV версию
            csv_path = self.dataset_path.replace('.json', '.csv')
            self._save_dataset_csv(csv_path)
            print(f"✓ Датасет сохранен: {self.dataset_path}")
            print(f"✓ CSV версия: {csv_path}")

        self.stats['final_size'] = len(self.dataset)

    def _save_dataset_csv(self, path: str) -> None:
        """Сохранение датасета в CSV формате"""
        with open(path, 'w', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['phrase', 'intent', 'parameters'])
            writer.writeheader()
            writer.writerows(self.dataset)

    def _save_dataset_json(self, path: str) -> None:
        """Сохранение датасета в JSON формате"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=2)

    def print_summary(self) -> None:
        """Вывести итоговую статистику обработки"""
        print("\n" + "="*60)
        print("ИТОГОВАЯ СТАТИСТИКА")
        print("="*60)
        print(f"Исходный размер:          {self.stats['original_size']}")
        print(f"Удалено дубликатов:       {self.stats['duplicates_removed']}")
        print(f"Удалено при балансировке: {self.stats['balance_removed']}")
        print(f"Итоговый размер:          {self.stats['final_size']}")

        total_removed = self.stats['original_size'] - self.stats['final_size']
        if self.stats['original_size'] > 0:
            compression = (total_removed / self.stats['original_size']) * 100
            print(f"Общее сжатие:             {total_removed} ({compression:.1f}%)")

        print("="*60)


def main():
    """Главная функция с парсингом аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='Дедупликация и балансировка датасета для голосового ассистента',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  Базовый вариант (дедупликация + балансировка):
    python batch_control.py dataset_car.csv

  Только дедупликация:
    python batch_control.py dataset_car.csv --skip-balance

  С настройкой порога балансировки:
    python batch_control.py dataset_car.csv --deviation 0.2
        """
    )

    parser.add_argument(
        'dataset',
        type=str,
        help='Путь к датасету (CSV или JSON)'
    )

    parser.add_argument(
        '--deviation',
        type=float,
        default=0.3,
        help='Максимальное отклонение от среднего для балансировки (default: 0.3 = 30%%)'
    )

    parser.add_argument(
        '--skip-balance',
        action='store_true',
        help='Пропустить этап балансировки (только дедупликация)'
    )

    args = parser.parse_args()

    # Проверка существования файла
    if not os.path.exists(args.dataset):
        print(f"❌ Ошибка: Файл '{args.dataset}' не найден!")
        return

    print(f"\n📊 Обработка датасета: {args.dataset}")
    print(f"⚙️  Режим: {'Только дедупликация' if args.skip_balance else 'Дедупликация + балансировка'}")
    if not args.skip_balance:
        print(f"⚙️  Допустимое отклонение: {args.deviation * 100:.0f}%\n")

    # Создать обработчик
    processor = Batch_Norm(args.dataset)

    # Этап 1: Дедупликация
    processor.run_deduplication()

    # Этап 2: Балансировка (опционально)
    if not args.skip_balance:
        processor.run_balancing(target_deviation=args.deviation)

    # Сохранение с backup
    processor.save_with_backup()

    # Итоговая статистика
    processor.print_summary()

    print(f"\n✅ Обработка завершена успешно!")


if __name__ == "__main__":
    main()
