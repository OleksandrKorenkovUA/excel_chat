import re
import numpy as np
import pandas as pd

from pipelines.spreadsheet_analyst_pipeline import (
    _code_has_subset_filter_ops,
    _code_has_groupby_aggregation,
    _extract_numeric_threshold_condition,
    _finalize_code_for_sandbox,
    _is_groupby_without_subset_question,
    _missing_subset_filter_guard_applies,
    _question_has_filter_context_for_router_guard,
    _subset_term_pattern,
    _status_message,
    _subset_keyword_metric_shortcut_code,
    _question_requires_subset_filter,
)


def test_question_requires_subset_filter_for_metric_subset() -> None:
    assert _question_requires_subset_filter("Яка максимальна ціна серед мишей?") is True


def test_groupby_without_subset_question_detects_each_status() -> None:
    assert _is_groupby_without_subset_question("Тепер порахуй кількість по кожному Статусу") is True
    assert _is_groupby_without_subset_question("Порахуй по кожному статусу серед бренду Apple") is False


def test_code_has_groupby_aggregation_detects_groupby() -> None:
    assert _code_has_groupby_aggregation("result = df.groupby('Статус').size().reset_index(name='count')\n") is True
    assert _code_has_groupby_aggregation("result = df['Кількість'].sum()\n") is False


def test_missing_subset_filter_guard_applies_only_for_real_subset_queries() -> None:
    profile = {
        "columns": ["Категорія", "Кількість", "Статус"],
        "dtypes": {"Категорія": "str", "Кількість": "int64", "Статус": "str"},
        "preview": [
            {"Категорія": "Миша", "Кількість": 2, "Статус": "В наявності"},
            {"Категорія": "Ноутбук", "Кількість": 5, "Статус": "Резерв"},
        ],
    }
    err = "missing_subset_filter: Generated code does not filter data for requested subset."
    assert _missing_subset_filter_guard_applies(err, "Яка максимальна ціна серед мишей?", profile) is True
    assert _missing_subset_filter_guard_applies(err, "Порахуй кількість по кожному Статусу", profile) is False


def test_status_message_for_non_subset_guard_conflict_retry() -> None:
    msg = _status_message("codegen_retry", {"reason": "subset_guard_conflict_non_subset"})
    assert "subset-фільтр не потрібен" in msg


def test_finalize_rejects_metric_subset_without_filter() -> None:
    code = "result = df['Ціна_UAH'].max()\n"
    _, _, err = _finalize_code_for_sandbox(
        question="Яка максимальна ціна серед мишей?",
        analysis_code=code,
        op="read",
        commit_df=False,
        df_profile={"columns": ["Категорія", "Ціна_UAH"], "dtypes": {"Ціна_UAH": "float64"}},
    )
    assert err is not None
    assert "missing_subset_filter" in err


def test_finalize_allows_metric_subset_with_contains_filter() -> None:
    code = (
        "result = df[df['Категорія'].astype(str).str.contains('миш', case=False, na=False)]['Ціна_UAH'].max()\n"
    )
    _, _, err = _finalize_code_for_sandbox(
        question="Яка максимальна ціна серед мишей?",
        analysis_code=code,
        op="read",
        commit_df=False,
        df_profile={"columns": ["Категорія", "Ціна_UAH"], "dtypes": {"Ціна_UAH": "float64"}},
    )
    assert err is None
    assert _code_has_subset_filter_ops(code) is True


def test_question_requires_subset_filter_for_entity_count_without_explicit_where() -> None:
    assert _question_requires_subset_filter("Порахуй кількість відеокарт nvidia") is True


def test_question_requires_subset_filter_for_cyrillic_entity_with_profile_preview() -> None:
    profile = {
        "columns": ["Категорія", "Ціна_UAH"],
        "dtypes": {"Категорія": "str", "Ціна_UAH": "float64"},
        "preview": [
            {"Категорія": "Ноутбук", "Ціна_UAH": 1000.0},
            {"Категорія": "Миша", "Ціна_UAH": 200.0},
        ],
    }
    assert _question_requires_subset_filter("Середня ціна ноутбуків", profile) is True


def test_question_has_filter_context_for_availability_phrase() -> None:
    assert _question_has_filter_context_for_router_guard("Загальна кількість мишей на складі") is True


def test_question_requires_subset_filter_false_for_grouped_totals_query_without_entity() -> None:
    profile = {
        "columns": ["Категорія", "Кількість", "Статус"],
        "dtypes": {"Категорія": "str", "Кількість": "int64", "Статус": "str"},
        "preview": [
            {"Категорія": "Ноутбук", "Кількість": 5, "Статус": "В наявності"},
            {"Категорія": "Миша", "Кількість": 2, "Статус": "В наявності"},
        ],
    }
    assert _question_requires_subset_filter("Загальна кількість штук по категоріях", profile) is False


def test_question_requires_subset_filter_false_for_grouped_totals_generic_columns() -> None:
    profile = {
        "columns": ["segment_name", "units_on_hand", "state_flag"],
        "dtypes": {"segment_name": "str", "units_on_hand": "int64", "state_flag": "str"},
        "preview": [
            {"segment_name": "A", "units_on_hand": 11, "state_flag": "ok"},
            {"segment_name": "B", "units_on_hand": 7, "state_flag": "ok"},
        ],
    }
    assert _question_requires_subset_filter("Total units by segment", profile) is False


def test_question_requires_subset_filter_false_for_grouped_count_each_status() -> None:
    profile = {
        "columns": ["Категорія", "Кількість", "Статус"],
        "dtypes": {"Категорія": "str", "Кількість": "int64", "Статус": "str"},
        "preview": [
            {"Категорія": "Ноутбук", "Кількість": 5, "Статус": "В наявності"},
            {"Категорія": "Миша", "Кількість": 2, "Статус": "Резерв"},
        ],
    }
    assert _question_requires_subset_filter("Count total for each status", profile) is False


def test_question_requires_subset_filter_false_for_grouped_revenue_all_items_query() -> None:
    profile = {
        "columns": ["Категорія", "Ціна_UAH", "Кількість"],
        "dtypes": {"Категорія": "str", "Ціна_UAH": "float64", "Кількість": "int64"},
        "preview": [
            {"Категорія": "Ноутбук", "Ціна_UAH": 35000.0, "Кількість": 2},
            {"Категорія": "Миша", "Ціна_UAH": 1200.0, "Кількість": 10},
        ],
    }
    q = "Яка загальна сума грошей за продаж усіх товарів по категоріях?"
    assert _question_requires_subset_filter(q, profile) is False


def test_finalize_allows_grouped_revenue_without_subset_filter() -> None:
    question = "Яка загальна сума грошей за продаж усіх товарів по категоріях?"
    code = (
        "df = df.copy()\n"
        "result = df.groupby('Категорія').apply(lambda x: (x['Ціна_UAH'] * x['Кількість']).sum()).reset_index(name='Виручка_UAH')\n"
    )
    _, _, err = _finalize_code_for_sandbox(
        question=question,
        analysis_code=code,
        op="read",
        commit_df=False,
        df_profile={
            "columns": ["Категорія", "Ціна_UAH", "Кількість"],
            "dtypes": {"Категорія": "str", "Ціна_UAH": "float64", "Кількість": "int64"},
        },
    )
    assert err is None


def test_finalize_allows_groupby_count_for_each_status_without_subset_filter() -> None:
    question = "Count total for each status"
    code = "result = df.groupby('Статус').size().reset_index(name='count')\n"
    _, _, err = _finalize_code_for_sandbox(
        question=question,
        analysis_code=code,
        op="read",
        commit_df=False,
        df_profile={
            "columns": ["Категорія", "Кількість", "Статус"],
            "dtypes": {"Категорія": "str", "Кількість": "int64", "Статус": "str"},
        },
    )
    assert err is None


def test_question_requires_subset_filter_true_for_grouped_query_with_explicit_filter_words() -> None:
    profile = {
        "columns": ["Категорія", "Бренд", "Ціна_UAH", "Кількість"],
        "dtypes": {"Категорія": "str", "Бренд": "str", "Ціна_UAH": "float64", "Кількість": "int64"},
        "preview": [
            {"Категорія": "Миша", "Бренд": "Logitech", "Ціна_UAH": 1400.0, "Кількість": 3},
            {"Категорія": "Клавіатура", "Бренд": "Keychron", "Ціна_UAH": 3200.0, "Кількість": 2},
        ],
    }
    q = "Яка загальна сума по категоріях серед товарів бренду Logitech?"
    assert _question_requires_subset_filter(q, profile) is True


def test_subset_keyword_shortcut_builds_filtered_count_code() -> None:
    profile = {
        "columns": ["c_text", "c_desc", "c_num"],
        "dtypes": {"c_text": "str", "c_desc": "str", "c_num": "float64"},
        "preview": [
            {"c_text": "Відеокарта", "c_desc": "NVIDIA RTX 4070", "c_num": 100.0},
            {"c_text": "Ноутбук", "c_desc": "Intel i7", "c_num": 50.0},
        ],
    }
    out = _subset_keyword_metric_shortcut_code("Порахуй кількість відеокарт NVIDIA", profile)
    assert out is not None
    code, _plan = out
    assert ("_text_cols =" in code) or ("_structured_filters =" in code)
    assert ("_work = _work.loc[_mask].copy()" in code) or ("_work = _work.loc[_mask_col].copy()" in code)
    assert "result = int(len(_work))" in code


def test_subset_keyword_shortcut_builds_filtered_metric_code() -> None:
    profile = {
        "columns": ["group_text", "item_desc", "value_num"],
        "dtypes": {"group_text": "str", "item_desc": "str", "value_num": "float64"},
        "preview": [
            {"group_text": "миша", "item_desc": "wireless", "value_num": 1200.0},
            {"group_text": "клавіатура", "item_desc": "mechanical", "value_num": 2200.0},
        ],
    }
    out = _subset_keyword_metric_shortcut_code("Яка максимальна ціна серед мишей?", profile)
    assert out is not None
    code, _plan = out
    assert ("_work = _work.loc[_mask].copy()" in code) or ("_work = _work.loc[_mask_col].copy()" in code)
    assert "_metric = 'max'" in code
    assert "result = None if pd.isna(_v) else float(_v)" in code


def test_subset_keyword_shortcut_avoids_hard_column_lock_on_weak_preview_signal() -> None:
    profile = {
        "columns": ["Категорія", "Бренд", "Модель", "Кількість"],
        "dtypes": {"Категорія": "str", "Бренд": "str", "Модель": "str", "Кількість": "int64"},
        "preview": [
            {"Категорія": "Відеокарта", "Бренд": "NVIDIA", "Модель": "RTX 4070", "Кількість": 1},
            {"Категорія": "Монітор", "Бренд": "Dell", "Модель": "U2720Q", "Кількість": 1},
        ],
    }
    out = _subset_keyword_metric_shortcut_code("Порахуй кількість відеокарт NVIDIA", profile)
    assert out is not None
    code, _plan = out
    assert "_structured_filters =" not in code
    assert "_patterns =" in code
    assert "result = int(len(_work))" in code


def test_subset_keyword_shortcut_falls_back_to_question_terms_when_preview_misses_entity() -> None:
    profile = {
        "columns": ["Категорія", "Опис", "Кількість"],
        "dtypes": {"Категорія": "str", "Опис": "str", "Кількість": "int64"},
        "preview": [
            {"Категорія": "Відеокарта", "Опис": "RTX 4070", "Кількість": 7},
            {"Категорія": "Відеокарта", "Опис": "RX 7800", "Кількість": 4},
        ],
    }
    out = _subset_keyword_metric_shortcut_code("Загальна кількість мишей на складі", profile)
    assert out is not None
    code, _plan = out
    assert "_metric = 'sum'" in code
    assert "_avail_mode = 'in'" in code
    assert "_work = _work.loc[_in & ~_out].copy()" in code
    assert "_patterns =" in code


def test_subset_keyword_shortcut_prefers_quantity_sum_for_total_quantity_intent() -> None:
    profile = {
        "columns": ["Категорія", "Кількість", "Статус"],
        "dtypes": {"Категорія": "str", "Кількість": "int64", "Статус": "str"},
        "preview": [
            {"Категорія": "Ноутбук", "Кількість": 5, "Статус": "В наявності"},
            {"Категорія": "Миша", "Кількість": 2, "Статус": "В наявності"},
        ],
    }
    out = _subset_keyword_metric_shortcut_code("Загальна кількість мишей", profile)
    assert out is not None
    code, _plan = out
    assert "_metric_col = 'Кількість'" in code
    assert "_metric = 'sum'" in code
    assert "result = None if pd.isna(_v) else float(_v)" in code


def test_subset_keyword_shortcut_accepts_llm_slots_fallback() -> None:
    profile = {
        "columns": ["Категорія", "Кількість", "Статус"],
        "dtypes": {"Категорія": "str", "Кількість": "int64", "Статус": "str"},
        "preview": [
            {"Категорія": "Миша", "Кількість": 3, "Статус": "В наявності"},
            {"Категорія": "Миша", "Кількість": 0, "Статус": "Немає"},
        ],
    }
    out = _subset_keyword_metric_shortcut_code(
        "Загальна кількість мишей на складі",
        profile,
        terms_hint=["миша"],
        slots_hint={
            "agg": "sum",
            "metric_col": "Кількість",
            "availability_mode": "in",
            "availability_col": "Статус",
        },
    )
    assert out is not None
    code, _plan = out
    assert "_metric = 'sum'" in code
    assert "_metric_col = 'Кількість'" in code
    assert "_avail_mode = 'in'" in code
    assert "_avail_col = 'Статус'" in code


def test_subset_term_pattern_is_universal_for_inflections() -> None:
    pat = _subset_term_pattern("миша")
    assert pat.startswith(r"\b")
    assert pat.endswith(r"\w*")
    assert re.search(pat, "миші", re.I)
    assert re.search(pat, "мышь", re.I)


def test_extract_numeric_threshold_condition_parses_less_than_phrase() -> None:
    cond = _extract_numeric_threshold_condition("товари які закінчуються (менше 5 штук)")
    assert cond == ("<", 5.0)


def test_subset_keyword_shortcut_uses_numeric_threshold_and_count_for_running_out_query() -> None:
    profile = {
        "columns": ["ID", "Категорія", "Бренд", "Ціна_UAH", "Кількість", "Статус"],
        "dtypes": {
            "ID": "float64",
            "Категорія": "str",
            "Бренд": "str",
            "Ціна_UAH": "float64",
            "Кількість": "int64",
            "Статус": "str",
        },
        "preview": [
            {"ID": 1, "Категорія": "Відеокарта", "Бренд": "NVIDIA", "Ціна_UAH": 85000, "Кількість": 1, "Статус": "Резерв"},
            {"ID": 2, "Категорія": "Відеокарта", "Бренд": "AMD", "Ціна_UAH": 24000, "Кількість": 7, "Статус": "В наявності"},
        ],
    }
    out = _subset_keyword_metric_shortcut_code(
        "Загальна кількість товарів які закінчуються (менше 5 штук)",
        profile,
        terms_hint=["менше 5", "закінчуються"],
        slots_hint={"agg": "count", "metric_col": "Кількість"},
    )
    assert out is not None
    code, _plan = out
    assert "_q = pd.to_numeric(_work[_qty_col], errors='coerce')" in code
    assert "_work = _work.loc[_q < 5.0].copy()" in code
    assert "result = int(len(_work))" in code
    assert "_metric = 'sum'" not in code


def test_subset_keyword_shortcut_handles_brand_and_series_terms_in_generic_domain() -> None:
    profile = {
        "columns": ["ID", "Тип", "Марка", "Модель", "Опис", "Характеристика", "Кількість"],
        "dtypes": {
            "ID": "int64",
            "Тип": "str",
            "Марка": "str",
            "Модель": "str",
            "Опис": "str",
            "Характеристика": "str",
            "Кількість": "int64",
        },
        "preview": [
            {"ID": 1, "Тип": "Авто", "Марка": "BMW", "Модель": "X5 M", "Опис": "", "Характеристика": "SUV", "Кількість": 1},
            {"ID": 2, "Тип": "Авто", "Марка": "BMW", "Модель": "X3 M", "Опис": "", "Характеристика": "SUV", "Кількість": 1},
            {"ID": 3, "Тип": "Авто", "Марка": "Audi", "Модель": "Q5", "Опис": "", "Характеристика": "SUV", "Кількість": 1},
        ],
    }
    out = _subset_keyword_metric_shortcut_code(
        "Порахуй кількість авто BMW (серії X5/X3)",
        profile,
        terms_hint=["авто", "BMW", "X5", "X3"],
        slots_hint={"agg": "count", "availability_mode": "none"},
    )
    assert out is not None
    code, _plan = out

    assert "_structured_filters =" in code
    assert "_all_patterns =" in code
    assert "if len(_work) == 0 and _all_patterns:" in code
    assert "_work = _work_fb.loc[_mask_fb].copy()" in code

    scope = {"df": pd.DataFrame(profile["preview"]), "pd": pd, "np": np}
    exec(code, scope, scope)
    assert scope.get("result") == 2
