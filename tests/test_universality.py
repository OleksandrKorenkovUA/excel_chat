import json

from pipelines.spreadsheet_analyst_pipeline import (
    AGG_COL_PLACEHOLDER,
    GROUP_COL_PLACEHOLDER,
    SUM_COL_PLACEHOLDER,
    TOP_N_PLACEHOLDER,
    Pipeline,
    _choose_column_from_question,
    _classify_columns_by_role,
    _find_column_in_text,
    _find_columns_in_text,
    _is_top_expensive_available_intent,
    _looks_like_value_filter_query,
    _rewrite_top_expensive_available_code,
    _stats_shortcut_code,
    _template_shortcut_code,
    _total_inventory_value_shortcut_code,
)


def test_choose_column_from_question_semantic_not_price_only() -> None:
    profile = {
        "columns": ["ID", "Кількість", "Вага_кг", "Стан"],
        "dtypes": {"ID": "int64", "Кількість": "int64", "Вага_кг": "float64", "Стан": "object"},
    }
    col = _choose_column_from_question("Яка вага в рядку 7?", profile)
    assert col == "Вага_кг"


def test_template_shortcut_generic_row_value_lookup() -> None:
    profile = {"columns": ["ID", "Кількість", "Вага_кг"]}
    q = "Яке значення ваги в рядку 7?"
    shortcut = _template_shortcut_code(q, profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "_row_num = 7" in code
    assert "_col = 'Вага_кг'" in code


def test_template_shortcut_availability_count() -> None:
    profile = {"columns": ["ID", "Назва", "Статус наявності"]}
    q = "Перерахуй уважно кількість товарів зі статусом в наявності"
    shortcut = _template_shortcut_code(q, profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "_col = '_SHORTCUT_COL_'" in code
    assert "_status = df[_col].astype(str).str.strip().str.lower()" in code
    assert "result = int(df.loc[_in & ~_out].shape[0])" in code


def test_template_shortcut_availability_count_without_status_column_name() -> None:
    profile = {
        "columns": ["ID", "Назва", "Склад"],
        "preview": [
            {"ID": 1, "Назва": "A", "Склад": "в наявності"},
            {"ID": 2, "Назва": "B", "Склад": "немає"},
        ],
    }
    q = "Скільки товарів доступно на складі?"
    shortcut = _template_shortcut_code(q, profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "_col = '_SHORTCUT_COL_'" in code
    assert "_mode = 'in'" in code
    assert "result = int(df.loc[_in & ~_out].shape[0])" in code


def test_template_shortcut_availability_out_of_stock_count() -> None:
    profile = {"columns": ["ID", "Наявність"]}
    q = "Скільки товарів немає в наявності?"
    shortcut = _template_shortcut_code(q, profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "_mode = 'out'" in code
    assert "result = int(df.loc[_out & ~_in].shape[0])" in code


def test_template_shortcut_availability_has_no_lambda() -> None:
    profile = {"columns": ["ID", "Статус"]}
    q = "Порахуй кількість товарів, які є в наявності"
    shortcut = _template_shortcut_code(q, profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "lambda" not in code


def test_template_shortcut_lowercase_columns_has_no_lambda() -> None:
    profile = {"columns": ["ID", "Назва"]}
    q = "Зроби назви колонок у нижній регістр"
    shortcut = _template_shortcut_code(q, profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "rename(columns=lambda" not in code
    assert "df.columns = [str(x).lower() for x in df.columns]" in code


def test_template_shortcut_fill_numeric_mean_has_no_lambda() -> None:
    profile = {"columns": ["ID", "Ціна", "Кількість"]}
    q = "замін усіх числових на mean"
    shortcut = _template_shortcut_code(q, profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "apply(lambda" not in code
    assert "fillna(df[num_cols].mean())" in code


def test_find_column_preserves_original_spacing() -> None:
    cols = ["import ", "status"]
    picked = _find_column_in_text("покажи import", cols)
    assert picked == "import "


def test_find_columns_in_text_semantic_multiple() -> None:
    cols = ["Категорія товару", "Кількість_шт", "Ціна_UAH"]
    found = _find_columns_in_text("Покажи топ категорій за загальною кількістю", cols)
    assert "Категорія товару" in found
    assert "Кількість_шт" in found


def test_find_column_does_not_match_id_inside_nvidia() -> None:
    cols = ["ID", "Категорія", "Бренд", "Статус"]
    picked = _find_column_in_text("Скільки в нас відеокарт NVIDIA у наявності?", cols)
    assert picked is None


def test_find_columns_does_not_match_id_inside_nvidia() -> None:
    cols = ["ID", "Категорія", "Бренд", "Статус"]
    found = _find_columns_in_text("Скільки в нас відеокарт NVIDIA у наявності?", cols)
    assert "ID" not in found


def test_classify_columns_by_role_prefers_categorical_plus_numeric() -> None:
    roles = _classify_columns_by_role(
        "Які топ-3 категорії мають найбільшу загальну кількість товарів?",
        ["Категорія", "Кількість"],
        {"dtypes": {"Категорія": "object", "Кількість": "int64"}},
    )
    assert roles["group_by"] == "Категорія"
    assert roles["aggregate"] == "Кількість"


def test_resolve_shortcut_placeholders_groupby_sum_topn_llm_first() -> None:
    pipeline = Pipeline()
    pipeline._llm_pick_columns_by_role_for_shortcut = lambda _q, _p: {  # type: ignore[method-assign]
        "group_by": "Категорія",
        "aggregate": "Кількість",
        "top_n": 3,
    }
    pipeline._llm_pick_column_for_shortcut = lambda _q, _p: None  # type: ignore[method-assign]
    profile = {"columns": ["ID", "Категорія", "Кількість"], "dtypes": {"Кількість": "int64"}}
    code = (
        f"result = df.groupby('{GROUP_COL_PLACEHOLDER}')['{SUM_COL_PLACEHOLDER}'].sum()"
        f".sort_values(ascending=False).head({TOP_N_PLACEHOLDER})\n"
    )
    plan = f"Топ по {GROUP_COL_PLACEHOLDER} і {AGG_COL_PLACEHOLDER}"
    resolved_code, resolved_plan = pipeline._resolve_shortcut_placeholders(
        code, plan, "Які топ-3 категорії за кількістю?", profile
    )
    assert "groupby('Категорія')" in resolved_code
    assert "['Кількість']" in resolved_code
    assert ".head(3)" in resolved_code
    assert "Категорія" in resolved_plan
    assert "Кількість" in resolved_plan


def test_availability_shortcut_scope_filtered_for_entity_token() -> None:
    pipeline = Pipeline()
    profile = {"columns": ["ID", "Категорія", "Бренд", "Статус"], "dtypes": {}, "preview": []}
    allowed = pipeline._should_use_availability_shortcut(
        "Скільки в нас відеокарт NVIDIA у наявності?",
        profile,
    )
    assert allowed is False


def test_total_inventory_value_shortcut_builds_product_sum() -> None:
    profile = {
        "columns": ["ID", "Ціна_UAH", "Кількість", "Статус"],
        "dtypes": {"ID": "float64", "Ціна_UAH": "float64", "Кількість": "int64", "Статус": "str"},
    }
    q = "Яка сума виручки, якщо продати весь наявний товар?"
    shortcut = _total_inventory_value_shortcut_code(q, profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "result = (df['Ціна_UAH'] * df['Кількість']).sum()" in code


def test_top_expensive_available_intent_detected() -> None:
    q = "Які 5 найдорожчих товарів у нас є у наявності?"
    assert _is_top_expensive_available_intent(q) is True


def test_top_expensive_available_rewrite_uses_inventory_value_mode_for_total_value_query() -> None:
    profile = {
        "columns": ["ID", "Модель", "Ціна_UAH", "Кількість", "Статус"],
        "dtypes": {
            "ID": "float64",
            "Модель": "str",
            "Ціна_UAH": "float64",
            "Кількість": "int64",
            "Статус": "str",
        },
    }
    rewritten = _rewrite_top_expensive_available_code(
        "result = df.head(5)\n",
        "Топ-5 товарів за сумарною вартістю на складі",
        profile,
    )
    assert "_rank_mode = 'inventory_value'" in rewritten
    assert "_top_df['_inventory_value'] = _top_df[_price_col] * _top_df[_qty_col]" in rewritten
    assert "_top_df = _top_df.nlargest(_top_n, '_inventory_value')" in rewritten


def test_top_expensive_available_rewrite_skips_when_code_already_has_inventory_metric() -> None:
    profile = {
        "columns": ["Модель", "Ціна_UAH", "Кількість", "Статус"],
        "dtypes": {"Модель": "str", "Ціна_UAH": "float64", "Кількість": "int64", "Статус": "str"},
    }
    code = (
        "_work['_metric'] = _left * _right\n"
        "result = _work.groupby(_group_col)['_metric'].sum().reset_index(name='metric_sum')\n"
        "result = result.sort_values('metric_sum', ascending=False).head(5)\n"
    )
    rewritten = _rewrite_top_expensive_available_code(
        code,
        "Топ-5 товарів за сумарною вартістю на складі",
        profile,
    )
    assert rewritten == code


def test_top_expensive_available_rewrite_skips_grouping_queries() -> None:
    profile = {
        "columns": ["Категорія", "Ціна_UAH", "Кількість", "Статус"],
        "dtypes": {"Категорія": "str", "Ціна_UAH": "float64", "Кількість": "int64", "Статус": "str"},
    }
    code = "result = df.head(5)\n"
    rewritten = _rewrite_top_expensive_available_code(
        code,
        "Топ-5 категорій за сумарною вартістю на складі",
        profile,
    )
    assert rewritten == code


def test_stats_shortcut_prefers_llm_metric_column_hint() -> None:
    profile = {
        "columns": ["ID", "Ціна_UAH", "Кількість"],
        "dtypes": {"ID": "float64", "Ціна_UAH": "float64", "Кількість": "int64"},
    }
    shortcut = _stats_shortcut_code("Яке середнє значення вартості?", profile, preferred_col="Ціна_UAH")
    assert shortcut is not None
    code, _plan = shortcut
    assert "_col = df['Ціна_UAH']" in code


def test_deterministic_scalar_does_not_guess_column_without_explicit_mention() -> None:
    pipeline = Pipeline()
    profile = {"columns": ["ID", "Категорія", "Бренд", "Статус"]}
    out = pipeline._deterministic_answer("Скільки в нас відеокарт NVIDIA у наявності?", "2", profile)
    assert out == "2"


def test_deterministic_scalar_uses_explicit_column_mention() -> None:
    pipeline = Pipeline()
    profile = {"columns": ["ID", "Категорія", "Бренд", "Статус"]}
    out = pipeline._deterministic_answer("Скільки значень у колонці ID?", "2", profile)
    assert out == "ID — 2."


def test_ranking_shortcut_code_uses_llm_slots_row_ranking() -> None:
    pipeline = Pipeline()
    pipeline._llm_pick_ranking_slots = lambda _q, _p: {  # type: ignore[method-assign]
        "query_mode": "row_ranking",
        "metric_col": "Ціна_UAH",
        "top_n": 5,
        "order": "desc",
        "require_available": True,
        "availability_col": "Статус",
        "entity_cols": ["Модель", "Бренд"],
    }
    profile = {
        "columns": ["ID", "Модель", "Бренд", "Ціна_UAH", "Статус"],
        "dtypes": {"ID": "float64", "Модель": "str", "Бренд": "str", "Ціна_UAH": "float64", "Статус": "str"},
    }
    shortcut = pipeline._ranking_shortcut_code("Які 5 найдорожчих товарів у наявності?", profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "_metric_col = 'Ціна_UAH'" in code
    assert "_top_n = 5" in code
    assert "_avail_col = 'Статус'" in code
    assert "_work = _work.nlargest(_top_n, _metric_col)" in code


def test_ranking_shortcut_code_skips_group_ranking() -> None:
    pipeline = Pipeline()
    pipeline._llm_pick_ranking_slots = lambda _q, _p: {  # type: ignore[method-assign]
        "query_mode": "group_ranking",
        "metric_col": "Кількість",
        "top_n": 3,
    }
    profile = {"columns": ["Категорія", "Кількість"], "dtypes": {"Кількість": "int64"}}
    shortcut = pipeline._ranking_shortcut_code("Топ-3 категорії за кількістю", profile)
    assert shortcut is None


def test_ranking_shortcut_code_group_ranking_sum_from_llm_slots() -> None:
    pipeline = Pipeline()
    pipeline._llm_pick_ranking_slots = lambda _q, _p: {  # type: ignore[method-assign]
        "query_mode": "group_ranking",
        "group_col": "Категорія",
        "target_col": "Кількість",
        "agg": "sum",
        "top_n": 3,
        "order": "desc",
        "require_available": False,
    }
    profile = {
        "columns": ["Категорія", "Кількість", "Статус"],
        "dtypes": {"Категорія": "str", "Кількість": "int64", "Статус": "str"},
    }
    shortcut = pipeline._ranking_shortcut_code("Топ-3 категорії за сумою кількості", profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "_group_col = 'Категорія'" in code
    assert "_agg = 'sum'" in code
    assert "_target_col = 'Кількість'" in code
    assert "groupby(_group_col)[_target_col].sum()" in code
    assert "result = _res.head(_top_n)" in code


def test_ranking_shortcut_code_group_ranking_sum_uses_metric_col_as_target_fallback() -> None:
    pipeline = Pipeline()
    pipeline._llm_pick_ranking_slots = lambda _q, _p: {  # type: ignore[method-assign]
        "query_mode": "group_ranking",
        "group_col": "Категорія",
        "metric_col": "Кількість",
        "target_col": "",
        "agg": "sum",
        "top_n": 10,
        "order": "desc",
        "require_available": False,
    }
    profile = {
        "columns": ["Категорія", "Кількість", "Статус"],
        "dtypes": {"Категорія": "str", "Кількість": "int64", "Статус": "str"},
    }
    shortcut = pipeline._ranking_shortcut_code("Загальна кількість штук по категоріях", profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "_target_col = 'Кількість'" in code
    assert "groupby(_group_col)[_target_col].sum()" in code


def test_ranking_shortcut_code_group_ranking_sum_metric_col_fallback_generic_names() -> None:
    pipeline = Pipeline()
    pipeline._llm_pick_ranking_slots = lambda _q, _p: {  # type: ignore[method-assign]
        "query_mode": "group_ranking",
        "group_col": "segment_name",
        "metric_col": "units_on_hand",
        "target_col": "",
        "agg": "sum",
        "top_n": 10,
        "order": "desc",
        "require_available": False,
    }
    profile = {
        "columns": ["segment_name", "units_on_hand", "state_flag"],
        "dtypes": {"segment_name": "str", "units_on_hand": "int64", "state_flag": "str"},
    }
    shortcut = pipeline._ranking_shortcut_code("Total units by segment", profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "_target_col = 'units_on_hand'" in code
    assert "groupby(_group_col)[_target_col].sum()" in code


def test_ranking_shortcut_code_group_ranking_count_from_llm_slots() -> None:
    pipeline = Pipeline()
    pipeline._llm_pick_ranking_slots = lambda _q, _p: {  # type: ignore[method-assign]
        "query_mode": "group_ranking",
        "group_col": "Бренд",
        "agg": "count",
        "top_n": 5,
        "order": "desc",
    }
    profile = {
        "columns": ["Бренд", "Модель", "Ціна_UAH"],
        "dtypes": {"Бренд": "str", "Модель": "str", "Ціна_UAH": "float64"},
    }
    shortcut = pipeline._ranking_shortcut_code("Топ-5 брендів за кількістю моделей", profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "_agg = 'count'" in code
    assert "groupby(_group_col).size().reset_index(name='count')" in code


def test_ranking_shortcut_code_group_ranking_count_with_target_uses_nunique() -> None:
    pipeline = Pipeline()
    pipeline._llm_pick_ranking_slots = lambda _q, _p: {  # type: ignore[method-assign]
        "query_mode": "group_ranking",
        "group_col": "Бренд",
        "target_col": "Модель",
        "agg": "count",
        "top_n": 5,
        "order": "desc",
    }
    profile = {
        "columns": ["Бренд", "Модель", "Ціна_UAH"],
        "dtypes": {"Бренд": "str", "Модель": "str", "Ціна_UAH": "float64"},
    }
    shortcut = pipeline._ranking_shortcut_code("Топ-5 брендів за кількістю моделей", profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "_agg == 'count'" in code
    assert "groupby(_group_col)[_target_col].nunique(dropna=True).reset_index(name='count')" in code


def test_ranking_shortcut_skips_when_top_n_is_zero() -> None:
    pipeline = Pipeline()
    pipeline._llm_pick_ranking_slots = lambda _q, _p: {  # type: ignore[method-assign]
        "query_mode": "group_ranking",
        "group_col": "Бренд",
        "target_col": "Модель",
        "agg": "count",
        "top_n": 0,
        "order": "desc",
    }
    profile = {
        "columns": ["Бренд", "Модель"],
        "dtypes": {"Бренд": "str", "Модель": "str"},
    }
    shortcut = pipeline._ranking_shortcut_code("Топ брендів за кількістю моделей", profile)
    assert shortcut is None


def test_ranking_shortcut_skips_filter_like_query_without_topn() -> None:
    pipeline = Pipeline()
    pipeline._llm_pick_ranking_slots = lambda _q, _p: {  # type: ignore[method-assign]
        "query_mode": "row_ranking",
        "metric_col": "Ціна_UAH",
        "order": "asc",
        "require_available": False,
        "entity_cols": ["Модель"],
    }
    profile = {
        "columns": ["ID", "Модель", "Ціна_UAH"],
        "dtypes": {"ID": "float64", "Модель": "str", "Ціна_UAH": "float64"},
    }
    shortcut = pipeline._ranking_shortcut_code("Поверни всі моделі, де ціна 5200", profile)
    assert shortcut is None


def test_ranking_shortcut_generated_code_has_no_pd_to_numeric_calls() -> None:
    pipeline = Pipeline()
    pipeline._llm_pick_ranking_slots = lambda _q, _p: {  # type: ignore[method-assign]
        "query_mode": "row_ranking",
        "metric_col": "Ціна_UAH",
        "top_n": 5,
        "order": "desc",
        "require_available": False,
        "entity_cols": ["Модель"],
    }
    profile = {
        "columns": ["ID", "Модель", "Ціна_UAH"],
        "dtypes": {"ID": "float64", "Модель": "str", "Ціна_UAH": "float64"},
    }
    shortcut = pipeline._ranking_shortcut_code("Топ-5 найдорожчих моделей", profile)
    assert shortcut is not None
    code, _ = shortcut
    assert "pd.to_numeric(" not in code


def test_ranking_shortcut_code_includes_generic_id_like_column_in_output() -> None:
    pipeline = Pipeline()
    pipeline._llm_pick_ranking_slots = lambda _q, _p: {  # type: ignore[method-assign]
        "query_mode": "row_ranking",
        "metric_col": "unit_price",
        "top_n": 3,
        "order": "desc",
        "require_available": False,
        "entity_cols": ["item_name"],
    }
    profile = {
        "columns": ["record_id", "item_name", "unit_price"],
        "dtypes": {"record_id": "int64", "item_name": "str", "unit_price": "float64"},
    }
    shortcut = pipeline._ranking_shortcut_code("Top 3 most expensive items", profile)
    assert shortcut is not None
    code, _ = shortcut
    assert "_out_cols = ['record_id', 'item_name', 'unit_price']" in code


def test_filter_like_detector_hits_price_equality_query() -> None:
    assert _looks_like_value_filter_query("Поверни всі моделі мишей які мають ціну 5200") is True


def test_lookup_shortcut_code_generates_filter_path_from_llm_slots() -> None:
    pipeline = Pipeline()
    pipeline._llm_pick_lookup_slots = lambda _q, _p: {  # type: ignore[method-assign]
        "mode": "lookup",
        "filters": [
            {"column": "Категорія", "op": "eq", "value": "Миша"},
            {"column": "Ціна_UAH", "op": "eq", "value": 5200},
        ],
        "output_columns": ["Модель"],
        "limit": None,
    }
    profile = {
        "columns": ["ID", "Категорія", "Бренд", "Модель", "Ціна_UAH"],
        "dtypes": {
            "ID": "float64",
            "Категорія": "str",
            "Бренд": "str",
            "Модель": "str",
            "Ціна_UAH": "float64",
        },
    }
    shortcut = pipeline._lookup_shortcut_code("Поверни всі моделі мишей які мають ціну 5200", profile)
    assert shortcut is not None
    code, _plan = shortcut
    assert "_c0 = 'Категорія'" in code
    assert "_c1 = 'Ціна_UAH'" in code
    assert "_out_cols = ['Модель']" in code
    assert "pd.to_numeric(" not in code


def test_lookup_shortcut_code_deduplicates_text_projection_by_default() -> None:
    pipeline = Pipeline()
    slots = {
        "mode": "lookup",
        "filters": [{"column": "Ціна_UAH", "op": "lt", "value": 3000}],
        "output_columns": ["Категорія"],
        "limit": None,
    }
    profile = {
        "columns": ["ID", "Категорія", "Ціна_UAH"],
        "dtypes": {"ID": "float64", "Категорія": "str", "Ціна_UAH": "float64"},
    }
    shortcut = pipeline._lookup_shortcut_code_from_slots("Товари дешевші за 3000, категорії", profile, slots)
    assert shortcut is not None
    code, _plan = shortcut
    assert "_out = _out.drop_duplicates()" in code


def test_lookup_shortcut_code_keeps_duplicates_for_detail_rows_query() -> None:
    pipeline = Pipeline()
    slots = {
        "mode": "lookup",
        "filters": [{"column": "Ціна_UAH", "op": "lt", "value": 3000}],
        "output_columns": ["Категорія"],
        "limit": None,
    }
    profile = {
        "columns": ["ID", "Категорія", "Ціна_UAH"],
        "dtypes": {"ID": "float64", "Категорія": "str", "Ціна_UAH": "float64"},
    }
    shortcut = pipeline._lookup_shortcut_code_from_slots("Покажи всі рядки товарів дешевших за 3000, категорії", profile, slots)
    assert shortcut is not None
    code, _plan = shortcut
    assert "_out = _out.drop_duplicates()" not in code


def test_lookup_shortcut_code_returns_none_for_non_lookup_mode() -> None:
    pipeline = Pipeline()
    pipeline._llm_pick_lookup_slots = lambda _q, _p: {  # type: ignore[method-assign]
        "mode": "other",
        "filters": [],
    }
    profile = {"columns": ["ID", "Модель"], "dtypes": {"ID": "float64", "Модель": "str"}}
    assert pipeline._lookup_shortcut_code("Які 5 найдорожчих товарів?", profile) is None


def test_format_table_from_result_unlimited_when_top_n_zero() -> None:
    pipeline = Pipeline()
    data = [{"ID": i, "Категорія": "Тест"} for i in range(1, 26)]
    out = pipeline._format_table_from_result(json.dumps(data, ensure_ascii=False), top_n=0)
    assert out is not None
    lines = out.splitlines()
    assert len(lines) == 27


def test_format_top_pairs_from_result_unlimited_when_top_n_zero() -> None:
    pipeline = Pipeline()
    data = {f"k{i}": i for i in range(1, 31)}
    out = pipeline._format_top_pairs_from_result(json.dumps(data, ensure_ascii=False), top_n=0)
    assert out is not None
    lines = out.splitlines()
    assert len(lines) == 32
