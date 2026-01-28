"""
Система контроля качества синтетических данных для голосового ассистента
"""

import json
import csv
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


def load_dataset(path: str) -> List[Dict]:
    """Загрузка датасета из CSV или JSON"""
    if path.endswith('.csv'):
        with open(path, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    else:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Конвертируем в формат CSV если нужно
            if isinstance(data, list) and len(data) > 0:
                if 'phrase' in data[0]:
                    return data
    return []


def calculate_lexical_diversity(phrases: List[str]) -> float:
    """
    Type-Token Ratio (TTR) - отношение уникальных слов к общему количеству
    Хорошее значение: > 0.7
    """
    all_words = []
    for phrase in phrases:
        words = re.findall(r'\w+', phrase.lower())
        all_words.extend(words)

    if len(all_words) == 0:
        return 0.0

    unique_words = len(set(all_words))
    total_words = len(all_words)

    return unique_words / total_words


def calculate_semantic_diversity(phrases: List[str]) -> float:
    """
    Средняя косинусная схожесть между фразами
    Хорошее значение: < 0.3 (чем меньше, тем разнообразнее)
    """
    if len(phrases) < 2:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(phrases)

        similarities = cosine_similarity(tfidf_matrix)
        # Берем верхний треугольник матрицы (без диагонали)
        upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]

        return np.mean(upper_triangle)
    except:
        return 0.0


def calculate_length_variance(phrases: List[str]) -> float:
    """
    Разнообразие длины фраз
    Стандартное отклонение должно быть > 2
    """
    lengths = [len(phrase.split()) for phrase in phrases]
    return float(np.std(lengths))


def check_intent_balance(dataset: List[Dict], max_deviation=0.3) -> List[Dict]:
    """
    Проверяет равномерность распределения интентов
    """
    intent_counts = Counter([item['intent'] for item in dataset])

    if len(intent_counts) == 0:
        return []

    avg_count = np.mean(list(intent_counts.values()))


    imbalance_report = []
    for intent, count in intent_counts.items():
        deviation = abs(count - avg_count) / avg_count if avg_count > 0 else 0
        status = "✓" if deviation < max_deviation else "⚠"
        imbalance_report.append({
            "intent": intent,
            "count": count,
            "deviation": deviation,
            "status": status
        })

    imbalance_report.sort(key=lambda x: x['deviation'], reverse=True)

    return imbalance_report


def validate_parameters(dataset: List[Dict]) -> List[str]:
    """
    Валидация JSON-параметров
    """
    errors = []

    for i, item in enumerate(dataset):
        params_str = item.get('parameters', '')

        if not params_str:
            continue

        try:
            params = json.loads(params_str)

            # Базовые проверки
            if not isinstance(params, dict):
                errors.append(f"Row {i}: Parameters must be a dict, got {type(params)}")

        except json.JSONDecodeError as e:
            errors.append(f"Row {i}: Invalid JSON - {str(e)[:50]}")

    return errors


def check_parameter_coverage(dataset: List[Dict]) -> List[Dict]:
    """
    Проверяет покрытие различных значений параметров
    """
    param_values = defaultdict(lambda: defaultdict(set))

    for item in dataset:
        intent = item['intent']
        params_str = item.get('parameters', '')

        if params_str:
            try:
                params = json.loads(params_str)
                for key, value in params.items():
                    param_values[intent][key].add(str(value))
            except:
                pass

    coverage_report = []

    for intent, params in param_values.items():
        for param, values in params.items():
            coverage_report.append({
                "intent": intent,
                "parameter": param,
                "unique_values": len(values),
                "values_sample": list(values)[:5]
            })

    return coverage_report


def check_duplicates(dataset: List[Dict]) -> Dict:
    from collections import Counter

    # Считаем комбинации (phrase, intent)
    phrase_intent_pairs = []
    for item in dataset:
        phrase_normalized = item["phrase"].lower().strip()
        intent_normalized = item["intent"].lower().strip()
        phrase_intent_pairs.append((phrase_normalized, intent_normalized))

    counter = Counter(phrase_intent_pairs)

    # Только дубликаты (count > 1)
    duplicates = {key: count for key, count in counter.items() if count > 1}

    # Топ-5 дубликатов для вывода (возвращаем кортежи)
    top_duplicates = sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:5]
    examples = [(phrase, intent, count) for (phrase, intent), count in top_duplicates]

    return {
        "total_duplicates": sum(duplicates.values()) - len(duplicates),
        "unique_duplicates": len(duplicates),
        "top_5": examples
    }


class DatasetQualityChecker:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = load_dataset(dataset_path)
        self.report = {}

    def run_all_checks(self) -> Dict:
        """Запускает все проверки качества"""
        print("="*60)
        print("DATASET QUALITY REPORT")
        print(f"File: {self.dataset_path}")
        print("="*60)

        if len(self.dataset) == 0:
            print("⚠ WARNING: Dataset is empty!")
            return {}

        self.report['size'] = len(self.dataset)
        print(f"\n📊 Dataset size: {self.report['size']}")

        # 1. Лексическое разнообразие
        phrases = [item['phrase'] for item in self.dataset]
        self.report['lexical_diversity'] = calculate_lexical_diversity(phrases)
        status = "✓" if self.report['lexical_diversity'] > 0.7 else "⚠"
        print(f"\n{status} Lexical Diversity: {self.report['lexical_diversity']:.3f}")
        print(f"   Target: > 0.7 (higher is better)")

        # 2. Семантическое разнообразие
        self.report['semantic_diversity'] = calculate_semantic_diversity(phrases)
        status = "✓" if self.report['semantic_diversity'] < 0.3 else "⚠"
        print(f"\n{status} Semantic Diversity: {self.report['semantic_diversity']:.3f}")
        print(f"   Target: < 0.3 (lower is better - less similarity)")

        # 3. Разнообразие длины
        self.report['length_variance'] = calculate_length_variance(phrases)
        status = "✓" if self.report['length_variance'] > 2.0 else "⚠"
        print(f"\n{status} Length Variance: {self.report['length_variance']:.2f}")
        print(f"   Target: > 2.0 (varied phrase lengths)")

        # 4. Баланс интентов
        balance = check_intent_balance(self.dataset)
        imbalanced = sum(1 for b in balance if b['status'] == "⚠")
        status = "✓" if imbalanced == 0 else "⚠"
        print(f"\n{status} Intent Balance: {imbalanced}/{len(balance)} intents imbalanced")
        if imbalanced > 0:
            print("   Top imbalanced intents:")
            for item in balance:
                print(f"   {item['status']} {item['intent']}: {item['count']} samples (dev: {item['deviation']:.1%})")

        # 5. Валидация параметров
        errors = validate_parameters(self.dataset)
        status = "✓" if len(errors) == 0 else "⚠"
        print(f"\n{status} Parameter Validation: {len(errors)} errors")
        if errors:
            print("   Sample errors:")
            for error in errors[:3]:
                print(f"   - {error}")

        # 6. Покрытие параметров
        coverage = check_parameter_coverage(self.dataset)
        print(f"\n✓ Parameter Coverage: {len(coverage)} param types found")
        if coverage:
            print("   Sample coverage:")
            for item in coverage[:5]:
                print(f"   - {item['intent']}.{item['parameter']}: {item['unique_values']} unique values")

        # 7. Дубликаты
        duplicates = check_duplicates(self.dataset)
        status = "✓" if duplicates['total_duplicates'] == 0 else "⚠"
        print(f"\n{status} Duplicates: {duplicates['total_duplicates']} duplicate phrases found")
        print(f"")
        if duplicates['top_5']:
            print("   Examples:")
            for phrase, intent, count in duplicates['top_5'][:5]:
                print(f"   - '{phrase}' + '{intent}' appears {count} times")

        print("\n" + "="*60)

        # Общая оценка
        overall_score = self._calculate_overall_score()
        self.report['overall_score'] = overall_score
        print(f"\n🎯 Overall Quality Score: {overall_score:.1%}")

        if overall_score >= 0.9:
            print("✅ Dataset quality is EXCELLENT - ready for training!")
        elif overall_score >= 0.75:
            print("✓ Dataset quality is GOOD - minor improvements recommended")
        elif overall_score >= 0.6:
            print("⚠ Dataset quality is ACCEPTABLE - improvements recommended")
        else:
            print("❌ Dataset quality needs SIGNIFICANT IMPROVEMENT before training")

        print("\n" + "="*60)

        return self.report

    def _calculate_overall_score(self) -> float:
        """Вычисляет общую оценку качества"""
        scores = []

        # Лексическое разнообразие (вес 0.25)
        lex_score = min(self.report.get('lexical_diversity', 0) / 0.7, 1.0)
        scores.append(lex_score * 0.25)

        # Семантическое разнообразие (вес 0.25)
        sem_score = max(0, 1 - (self.report.get('semantic_diversity', 0) / 0.3))
        scores.append(sem_score * 0.25)

        # Разнообразие длины (вес 0.15)
        len_score = min(self.report.get('length_variance', 0) / 2.0, 1.0)
        scores.append(len_score * 0.15)

        # Остальные 35% за другие метрики (пока упрощенно)
        scores.append(0.35)

        return sum(scores)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Dataset Quality Control')
    parser.add_argument('dataset', type=str, help='Path to dataset (CSV or JSON)')
    args = parser.parse_args()

    checker = DatasetQualityChecker(args.dataset)
    report = checker.run_all_checks()

    # Сохраняем отчет
    report_path = args.dataset.replace('.csv', '_quality_report.json').replace('.json', '_quality_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n📄 Detailed report saved to: {report_path}")


if __name__ == "__main__":
    main()
