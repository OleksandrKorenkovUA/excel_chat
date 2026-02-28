import pytest

from pipelines.spreadsheet_analyst_pipeline import (
    _detect_metrics,
    _enforce_count_code,
    _enforce_entity_nunique_code,
    _has_edit_triggers,
)


def test_detect_metrics_ukr_mean_min_max_order():
    q = "Надай середнє, мінімальне та максимальне значення ціни товарів"
    assert _detect_metrics(q) == ["mean", "min", "max"]


def test_detect_metrics_en_mean_sum():
    q = "Please provide average price and sum of sales"
    assert _detect_metrics(q) == ["mean", "sum"]


def test_metrics_no_substring_collisions():
    q = "summary for administrator"
    assert _detect_metrics(q) == []

def test_detect_metrics_median_count_ukr():
    q = "Порахуй медіану та кількість"
    assert _detect_metrics(q) == ["median", "count"]


def test_detect_metrics_median_count_en_synonyms():
    q = "Need median and number of items (qty)"
    assert _detect_metrics(q) == ["median", "count"]


def test_detect_metrics_min_max_variants():
    q = "потрібне мін та макс значення"
    assert _detect_metrics(q) == ["min", "max"]


def test_detect_metrics_combined_short():
    q = "середнє, мін, макс"
    assert _detect_metrics(q) == ["mean", "min", "max"]


def test_detect_metrics_negative_no_metric():
    q = "покажи ціну"
    assert _detect_metrics(q) == []


def test_metrics_negative_false_positives_english():
    assert _detect_metrics("maximum effort") == []
    assert _detect_metrics("number of years") == []
    assert _detect_metrics("qty of time") == []


def test_metrics_log_sorted_stable():
    q = "max and min and mean"
    metrics = _detect_metrics(q)
    metrics_log = sorted(metrics)
    assert metrics_log == sorted(["mean", "min", "max"])


def test_has_edit_triggers():
    assert _has_edit_triggers("зміни значення в комірці") is True
    assert _has_edit_triggers("порахуй кількість рядків") is False


def test_enforce_entity_nunique_prefers_model_column() -> None:
    profile = {"columns": ["ID", "Марка машин", "Модель"]}
    code = "result = len(df['Марка машин'].dropna().unique())\n"
    guarded = _enforce_entity_nunique_code("Скільки унікальних моделей у таблиці?", code, profile)
    assert "df['Модель']" in guarded
    assert ".nunique()" in guarded


def test_enforce_entity_nunique_matches_semantic_column() -> None:
    profile = {"columns": ["ID", "Марка машин", "Модель"]}
    code = "result = 0\n"
    guarded = _enforce_entity_nunique_code("Скільки унікальних марок авто?", code, profile)
    assert "df['Марка машин']" in guarded


def test_enforce_entity_nunique_skips_grouped_requests() -> None:
    profile = {"columns": ["Brand", "Model"]}
    code = "result = df['Brand'].value_counts().to_dict()\n"
    guarded = _enforce_entity_nunique_code("Покажи кількість для кожного бренду", code, profile)
    assert guarded == code


def test_enforce_count_code_keeps_revenue_product_sum() -> None:
    question = "Виручка по категоріях (Ціна × Кількість)"
    code = (
        "_work['_metric'] = _left * _right\n"
        "result = _work.groupby(_group_col)['_metric'].sum().reset_index(name='revenue_sum')\n"
    )
    guarded, err = _enforce_count_code(question, code)
    assert err is None
    assert "['_metric'].sum()" in guarded
    assert "['_metric'].size()" not in guarded


def test_enforce_count_code_rewrites_sum_for_count_intent() -> None:
    question = "Скільки товарів по категоріях?"
    code = "result = df.groupby('Категорія')['ID'].sum().reset_index(name='cnt')\n"
    guarded, err = _enforce_count_code(question, code)
    assert err is None
    assert ".size()" in guarded
