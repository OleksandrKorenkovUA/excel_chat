import re

from pipelines.spreadsheet_analyst_pipeline import (
    _code_has_subset_filter_ops,
    _finalize_code_for_sandbox,
    _question_has_filter_context_for_router_guard,
    _subset_term_pattern,
    _subset_keyword_metric_shortcut_code,
    _question_requires_subset_filter,
)


def test_question_requires_subset_filter_for_metric_subset() -> None:
    assert _question_requires_subset_filter("Яка максимальна ціна серед мишей?") is True


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
    assert "_text_cols =" in code
    assert "_work = _work.loc[_mask].copy()" in code
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
    assert "_work = _work.loc[_mask].copy()" in code
    assert "_metric = 'max'" in code
    assert "result = None if pd.isna(_v) else float(_v)" in code


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
