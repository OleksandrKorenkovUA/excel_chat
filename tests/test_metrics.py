import pytest

from pipelines.spreadsheet_analyst_pipeline import _detect_metrics, _has_edit_triggers


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
