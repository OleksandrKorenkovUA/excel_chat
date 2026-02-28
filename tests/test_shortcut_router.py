import json

from pipelines.shortcut_router.shortcut_router import ShortcutRouter, ShortcutRouterConfig


def test_match_enum_handles_ukrainian_inflections() -> None:
    router = ShortcutRouter()
    values = ["mean", "sum", "min", "max", "median"]
    assert router._match_enum(values, "яка середня вартість?") == "mean"
    assert router._match_enum(values, "який мінімальний чек?") == "min"
    assert router._match_enum(values, "який максимальний чек?") == "max"


def test_fill_slots_stats_aggregation_forces_numeric_column() -> None:
    router = ShortcutRouter()
    intent = {
        "id": "stats_aggregation",
        "slots": {
            "column": {"type": "column", "required": True},
            "metric": {"type": "enum", "required": True, "values": ["mean", "sum", "min", "max", "median"]},
        },
    }
    query = "Яка середня вартість ноутбука в нашому каталогу?"
    columns = ["ID", "Категорія", "Бренд", "Ціна_UAH", "Кількість"]
    profile = {
        "dtypes": {
            "ID": "float64",
            "Категорія": "str",
            "Бренд": "str",
            "Ціна_UAH": "float64",
            "Кількість": "int64",
        }
    }
    slots = router._fill_slots(intent, query, columns, profile)
    assert slots is not None
    assert slots["metric"] == "mean"
    assert slots["column"] == "Ціна_UAH"


def test_has_filter_context_detects_metric_subset_query() -> None:
    router = ShortcutRouter()
    profile = {
        "columns": ["Категорія", "Ціна_UAH", "Бренд"],
        "preview": [
            {"Категорія": "Миша", "Ціна_UAH": 1200, "Бренд": "Logitech"},
            {"Категорія": "Клавіатура", "Ціна_UAH": 2200, "Бренд": "Keychron"},
        ],
    }
    assert router._has_filter_context("Яка максимальна ціна серед мишей?", profile) is True


def test_has_filter_context_false_for_plain_metric_query() -> None:
    router = ShortcutRouter()
    profile = {
        "columns": ["Категорія", "Ціна_UAH"],
        "preview": [{"Категорія": "Миша", "Ціна_UAH": 1200}],
    }
    assert router._has_filter_context("Яка максимальна ціна?", profile) is False


def test_assess_query_complexity_high_for_filtered_metric_query() -> None:
    router = ShortcutRouter()
    profile = {
        "columns": ["Категорія", "Ціна_UAH", "Бренд"],
        "preview": [
            {"Категорія": "Миша", "Ціна_UAH": 1200, "Бренд": "Logitech"},
            {"Категорія": "Клавіатура", "Ціна_UAH": 2200, "Бренд": "Keychron"},
        ],
    }
    score = router._assess_query_complexity("Яка максимальна ціна серед мишей та клавіатур?", profile)
    assert score >= 0.7


def test_fill_slots_filter_contains_uses_value_to_pick_filter_column() -> None:
    def fake_llm(_system: str, user: str) -> dict:
        payload = json.loads(user)
        if payload.get("slot_name") == "value":
            return {"value": "Миша"}
        return {"value": None}

    router = ShortcutRouter(llm_json=fake_llm)
    intent = {
        "id": "filter_contains",
        "slots": {
            "column": {"type": "column", "required": True},
            "value": {"type": "str", "required": True},
        },
    }
    query = "Загальна кількість товарів категорії Миша зі статусом на складі"
    columns = ["ID", "Категорія", "Кількість", "Статус"]
    profile = {
        "dtypes": {"ID": "int64", "Категорія": "object", "Кількість": "int64", "Статус": "object"},
        "preview": [
            {"ID": 1, "Категорія": "Миша", "Кількість": 12, "Статус": "В наявності"},
            {"ID": 2, "Категорія": "Клавіатура", "Кількість": 8, "Статус": "В наявності"},
        ],
    }
    slots = router._fill_slots(intent, query, columns, profile)
    assert slots is not None
    assert slots["value"] == "Миша"
    assert slots["column"] == "Категорія"


def test_fill_slots_filtered_metric_aggregation_enforces_filter_and_numeric_target() -> None:
    def fake_llm(_system: str, user: str) -> dict:
        payload = json.loads(user)
        if payload.get("slot_name") == "filter_value":
            return {"value": "Миша"}
        if payload.get("slot_name") == "target_col":
            return {"value": "Кількість"}
        return {"value": None}

    router = ShortcutRouter(llm_json=fake_llm)
    intent = {
        "id": "filtered_metric_aggregation",
        "slots": {
            "filter_col": {"type": "column", "required": True},
            "filter_value": {"type": "str", "required": True},
            "target_col": {"type": "column", "required": True},
            "metric": {"type": "enum", "required": True, "values": ["mean", "sum", "min", "max", "median"]},
        },
    }
    query = "Яка максимальна кількість серед товарів категорії Миша?"
    columns = ["ID", "Категорія", "Кількість", "Статус"]
    profile = {
        "dtypes": {"ID": "int64", "Категорія": "object", "Кількість": "int64", "Статус": "object"},
        "preview": [
            {"ID": 1, "Категорія": "Миша", "Кількість": 12, "Статус": "В наявності"},
            {"ID": 2, "Категорія": "Клавіатура", "Кількість": 8, "Статус": "В наявності"},
        ],
    }
    slots = router._fill_slots(intent, query, columns, profile)
    assert slots is not None
    assert slots["filter_col"] == "Категорія"
    assert slots["filter_value"] == "Миша"
    assert slots["target_col"] == "Кількість"


def test_fill_slots_filter_contains_is_column_name_agnostic() -> None:
    router = ShortcutRouter()
    intent = {
        "id": "filter_contains",
        "slots": {
            "column": {"type": "column", "required": True},
            "value": {"type": "str", "required": True},
        },
    }
    query = "Загальна кількість товарів миша зі статусом на складі"
    columns = ["col_1", "col_2", "col_3"]
    profile = {
        "dtypes": {"col_1": "int64", "col_2": "object", "col_3": "object"},
        "preview": [
            {"col_1": 12, "col_2": "Миша", "col_3": "В наявності"},
            {"col_1": 8, "col_2": "Клавіатура", "col_3": "В наявності"},
        ],
    }
    slots = router._fill_slots(intent, query, columns, profile)
    assert slots is not None
    assert slots["value"] == "миша"
    assert slots["column"] == "col_2"


def test_resolve_intent_prefers_groupby_sum_for_grouped_total_quantity_query() -> None:
    def fake_llm(_system: str, _user: str) -> dict:
        return {
            "group_col": "Категорія",
            "target_col": "ID",
            "agg": "count",
            "top_n": 0,
        }

    router = ShortcutRouter(llm_json=fake_llm)
    router._intents = {
        "groupby_count": {"id": "groupby_count"},
        "groupby_agg": {"id": "groupby_agg"},
    }
    base_intent = {"id": "groupby_count"}
    columns = ["ID", "Категорія", "Кількість", "Статус"]
    profile = {
        "dtypes": {"ID": "float64", "Категорія": "str", "Кількість": "int64", "Статус": "str"},
        "rows": 100,
        "preview": [
            {"ID": 1.0, "Категорія": "Миша", "Кількість": 25, "Статус": "В наявності"},
            {"ID": 2.0, "Категорія": "SSD", "Кількість": 12, "Статус": "В наявності"},
        ],
    }
    resolved_intent, preset = router._resolve_intent_and_slots(
        base_intent,
        "Загальна кількість товарів по категоріях",
        columns,
        profile,
    )
    assert resolved_intent.get("id") == "groupby_agg"
    assert preset.get("group_col") == "Категорія"
    assert preset.get("agg") == "sum"
    assert preset.get("target_col") == "Кількість"


def test_resolve_intent_prefers_groupby_sum_for_grouped_total_quantity_query_generic_names() -> None:
    def fake_llm(_system: str, _user: str) -> dict:
        return {
            "group_col": "segment_name",
            "target_col": "record_id",
            "agg": "count",
            "top_n": 0,
        }

    router = ShortcutRouter(llm_json=fake_llm)
    router._intents = {
        "groupby_count": {"id": "groupby_count"},
        "groupby_agg": {"id": "groupby_agg"},
    }
    base_intent = {"id": "groupby_count"}
    columns = ["record_id", "segment_name", "units_on_hand", "state_flag"]
    profile = {
        "dtypes": {
            "record_id": "int64",
            "segment_name": "str",
            "units_on_hand": "int64",
            "state_flag": "str",
        },
        "rows": 100,
        "preview": [
            {"record_id": 1, "segment_name": "A", "units_on_hand": 10, "state_flag": "in"},
            {"record_id": 2, "segment_name": "B", "units_on_hand": 8, "state_flag": "in"},
        ],
    }
    resolved_intent, preset = router._resolve_intent_and_slots(
        base_intent,
        "Total units by segment",
        columns,
        profile,
    )
    assert resolved_intent.get("id") == "groupby_agg"
    assert preset.get("group_col") == "segment_name"
    assert preset.get("agg") == "sum"
    assert preset.get("target_col") == "units_on_hand"


def test_resolve_intent_groupby_adds_startswith_filter_for_group_col() -> None:
    def fake_llm(_system: str, _user: str) -> dict:
        return {
            "group_col": "Бренд",
            "target_col": "Модель",
            "agg": "count",
            "top_n": 0,
        }

    router = ShortcutRouter(llm_json=fake_llm)
    router._intents = {
        "groupby_count": {"id": "groupby_count"},
        "groupby_agg": {"id": "groupby_agg"},
    }
    base_intent = {"id": "groupby_count"}
    columns = ["ID", "Бренд", "Модель", "Ціна_UAH"]
    profile = {
        "dtypes": {"ID": "float64", "Бренд": "str", "Модель": "str", "Ціна_UAH": "float64"},
        "rows": 100,
        "preview": [],
    }
    resolved_intent, preset = router._resolve_intent_and_slots(
        base_intent,
        'Бренди які починаються на літеру "A" та кількість моделей',
        columns,
        profile,
    )
    assert resolved_intent.get("id") == "groupby_count"
    assert preset.get("group_col") == "Бренд"
    assert preset.get("filter_col") == "Бренд"
    assert preset.get("filter_op") == "startswith"
    assert preset.get("filter_value") == "A"


def test_resolve_intent_does_not_override_price_target_with_quantity_heuristic() -> None:
    def fake_llm(_system: str, _user: str) -> dict:
        return {
            "group_col": "Категорія",
            "target_col": "Ціна_UAH",
            "agg": "sum",
            "top_n": 0,
        }

    router = ShortcutRouter(llm_json=fake_llm)
    router._intents = {
        "groupby_count": {"id": "groupby_count"},
        "groupby_agg": {"id": "groupby_agg"},
    }
    base_intent = {"id": "groupby_agg"}
    columns = ["ID", "Категорія", "Ціна_UAH", "Кількість", "Статус"]
    profile = {
        "dtypes": {
            "ID": "float64",
            "Категорія": "str",
            "Ціна_UAH": "float64",
            "Кількість": "int64",
            "Статус": "str",
        },
        "rows": 100,
        "preview": [
            {"ID": 1.0, "Категорія": "Ноутбук", "Ціна_UAH": 89999.0, "Кількість": 5, "Статус": "В наявності"},
            {"ID": 2.0, "Категорія": "Планшет", "Ціна_UAH": 21999.0, "Кількість": 8, "Статус": "В наявності"},
        ],
    }
    resolved_intent, preset = router._resolve_intent_and_slots(
        base_intent,
        "Порахуй загальну суму по цінам для всіх товарів кожної категорії",
        columns,
        profile,
    )
    assert resolved_intent.get("id") == "groupby_agg"
    assert preset.get("group_col") == "Категорія"
    assert preset.get("agg") == "sum"
    assert preset.get("target_col") == "Ціна_UAH"


def test_pick_group_like_column_prefers_generic_dimension_column() -> None:
    router = ShortcutRouter()
    columns = ["record_id", "segment_name", "unit_price", "units_on_hand"]
    profile = {
        "dtypes": {
            "record_id": "int64",
            "segment_name": "str",
            "unit_price": "float64",
            "units_on_hand": "int64",
        }
    }
    picked = router._pick_group_like_column("Total revenue by segment", columns, profile)
    assert picked == "segment_name"


def test_build_group_revenue_slots_uses_generic_output_column_name() -> None:
    router = ShortcutRouter()
    columns = ["segment_name", "unit_price", "units_on_hand", "revenue_sum"]
    profile = {
        "dtypes": {
            "segment_name": "str",
            "unit_price": "float64",
            "units_on_hand": "int64",
            "revenue_sum": "float64",
        }
    }
    slots = router._build_group_revenue_slots(
        query="Total revenue by segment",
        columns=columns,
        profile=profile,
        llm_slots={"group_col": "segment_name"},
    )
    assert slots
    assert slots.get("group_col") == "segment_name"
    assert slots.get("target_col") == "unit_price"
    assert slots.get("mul_right_col") == "units_on_hand"
    assert str(slots.get("out_col")).startswith("revenue_sum")
    assert slots.get("out_col") != "revenue_sum"


def test_build_group_revenue_slots_preserves_top_n_from_llm_slots() -> None:
    router = ShortcutRouter()
    columns = ["segment_name", "unit_price", "units_on_hand"]
    profile = {
        "dtypes": {
            "segment_name": "str",
            "unit_price": "float64",
            "units_on_hand": "int64",
        }
    }
    slots = router._build_group_revenue_slots(
        query="Top 5 items by total value in stock",
        columns=columns,
        profile=profile,
        llm_slots={"group_col": "segment_name", "top_n": 5},
    )
    assert slots
    assert slots.get("top_n") == 5


def test_resolve_intent_group_revenue_redirect_keeps_top_n() -> None:
    def fake_llm(_system: str, _user: str) -> dict:
        return {
            "group_col": "Модель",
            "target_col": "Ціна_UAH",
            "agg": "sum",
            "top_n": 5,
        }

    router = ShortcutRouter(llm_json=fake_llm)
    router._intents = {
        "groupby_agg": {"id": "groupby_agg"},
        "groupby_count": {"id": "groupby_count"},
    }
    base_intent = {"id": "groupby_agg"}
    columns = ["ID", "Модель", "Ціна_UAH", "Кількість", "Статус"]
    profile = {
        "dtypes": {
            "ID": "float64",
            "Модель": "str",
            "Ціна_UAH": "float64",
            "Кількість": "int64",
            "Статус": "str",
        },
        "rows": 100,
        "preview": [
            {"ID": 1.0, "Модель": "A", "Ціна_UAH": 100.0, "Кількість": 2, "Статус": "В наявності"},
            {"ID": 2.0, "Модель": "B", "Ціна_UAH": 200.0, "Кількість": 3, "Статус": "В наявності"},
        ],
    }
    resolved_intent, preset = router._resolve_intent_and_slots(
        base_intent,
        "Топ-5 товарів за сумарною вартістю на складі",
        columns,
        profile,
    )
    assert resolved_intent.get("id") == "groupby_agg"
    assert preset.get("top_n") == 5


def test_group_revenue_detector_handles_total_price_for_all_items_phrase() -> None:
    router = ShortcutRouter()
    assert router._is_group_revenue_product_query(
        "Порахуй загальну суму по цінам для всіх товарів кожної категорії"
    )


def test_compile_groupby_revenue_uses_sum_not_size() -> None:
    router = ShortcutRouter()
    intent = {
        "id": "groupby_agg",
        "plan": [{"op": "groupby_agg"}],
        "slots": {},
    }
    slots = {
        "group_col": "Категорія",
        "target_col": "Ціна_UAH",
        "agg": "sum",
        "mul_left_col": "Ціна_UAH",
        "mul_right_col": "Кількість",
        "out_col": "Виручка_UAH",
    }
    code = router._compile_plan(intent, slots, profile={})
    assert code is not None
    assert "_work.groupby(_group_col)['_metric'].sum().reset_index(name='Виручка_UAH')" in code
    assert "['_metric'].size()" not in code


def test_compile_groupby_count_uses_nunique_when_target_col_provided() -> None:
    router = ShortcutRouter()
    intent = {
        "id": "groupby_count",
        "plan": [{"op": "groupby_count"}],
        "slots": {},
    }
    slots = {
        "group_col": "Бренд",
        "target_col": "Модель",
        "out_col": "count",
    }
    code = router._compile_plan(intent, slots, profile={})
    assert code is not None
    assert "groupby('Бренд')[_target_col].nunique(dropna=True).reset_index(name='count')" in code
    assert "groupby('Бренд').size().reset_index(name='count')" in code


def test_compile_groupby_count_applies_optional_startswith_filter() -> None:
    router = ShortcutRouter()
    intent = {
        "id": "groupby_count",
        "plan": [{"op": "groupby_count"}],
        "slots": {},
    }
    slots = {
        "group_col": "Бренд",
        "target_col": "Модель",
        "out_col": "count",
        "filter_col": "Бренд",
        "filter_op": "startswith",
        "filter_value": "A",
    }
    code = router._compile_plan(intent, slots, profile={})
    assert code is not None
    assert "_src = df" in code
    assert ".str.lower().str.startswith(str('A').lower(), na=False)" in code
    assert "_src.groupby('Бренд')[_target_col].nunique(dropna=True).reset_index(name='count')" in code


def test_validate_slots_does_not_require_out_col_to_exist_in_df_columns() -> None:
    router = ShortcutRouter()
    intent = {
        "id": "groupby_count",
        "slots": {
            "group_col": {"type": "column", "required": True},
            "out_col": {"type": "str", "required": False},
        },
        "plan": [{"op": "groupby_count"}],
    }
    columns = ["Бренд", "Модель"]
    profile = {"dtypes": {"Бренд": "str", "Модель": "str"}}
    issues = router._validate_slots(
        intent=intent,
        slots={"group_col": "Бренд", "out_col": "custom_count"},
        columns=columns,
        profile=profile,
    )
    assert "unknown_column_ref:out_col" not in issues


def test_llm_groupby_slots_normalizes_non_positive_top_n_to_none() -> None:
    def fake_llm(_system: str, _user: str) -> dict:
        return {
            "group_col": "Бренд",
            "target_col": "Кількість",
            "agg": "sum",
            "top_n": 0,
        }

    router = ShortcutRouter(llm_json=fake_llm)
    out = router._llm_groupby_slots(
        query="Сума кількості по брендах",
        columns=["Бренд", "Кількість"],
        profile={"dtypes": {"Бренд": "str", "Кількість": "int64"}},
    )
    assert out is not None
    assert out.get("group_col") == "Бренд"
    assert out.get("agg") == "sum"
    assert "top_n" not in out


def test_compile_filter_equals_and_comparison_are_read_only_results() -> None:
    router = ShortcutRouter()
    intent = {
        "id": "filter_ops",
        "plan": [{"op": "filter_equals"}, {"op": "filter_comparison"}],
        "slots": {},
    }
    slots = {
        "column": "Статус",
        "value": "Закінчується",
        "case_insensitive": True,
        "operator": ">",
    }
    code = router._compile_plan(intent, slots, profile={})
    assert code is not None
    assert "_work = df[df['Статус'].astype(str).str.lower() == str('Закінчується').lower()]" in code
    assert "_work = df[df['Статус'] > 'Закінчується']" in code
    assert "result = _work" in code
    assert "COMMIT_DF = True" not in code


def test_compile_filter_equals_with_preview_uses_filtered_workset() -> None:
    router = ShortcutRouter()
    intent = {
        "id": "filter_equals_preview",
        "plan": [
            {"op": "filter_equals"},
            {"op": "return_df_preview", "args": {"rows": 20}},
        ],
        "slots": {},
    }
    slots = {
        "column": "Статус",
        "value": "Закінчується",
        "case_insensitive": True,
    }
    code = router._compile_plan(intent, slots, profile={})
    assert code is not None
    assert "df = _work" not in code
    assert "result = _work\n" in code
    assert "result = _work.head(20)" not in code


def test_compile_filter_equals_with_preview_can_respect_catalog_rows_when_enabled() -> None:
    router = ShortcutRouter(ShortcutRouterConfig(respect_catalog_preview_rows=True))
    intent = {
        "id": "filter_equals_preview",
        "plan": [
            {"op": "filter_equals"},
            {"op": "return_df_preview", "args": {"rows": 20}},
        ],
        "slots": {},
    }
    slots = {
        "column": "Статус",
        "value": "Закінчується",
        "case_insensitive": True,
    }
    code = router._compile_plan(intent, slots, profile={})
    assert code is not None
    assert "result = _work.head(20)" in code


def test_fill_slots_filter_equals_top_n_ignores_plain_id_numbers() -> None:
    router = ShortcutRouter()
    intent = {
        "id": "filter_equals",
        "slots": {
            "column": {"type": "column", "required": True},
            "value": {"type": "str", "required": True},
            "case_insensitive": {"type": "bool", "required": False, "default": True},
            "top_n": {"type": "int", "required": False, "default": 0},
        },
    }
    query = "Покажи записи де id = '12345'"
    columns = ["id", "name"]
    profile = {"dtypes": {"id": "int64", "name": "str"}}

    slots = router._fill_slots(intent, query, columns, profile)
    assert slots is not None
    assert slots.get("top_n") == 0


def test_fill_slots_filter_equals_top_n_uses_top_cue() -> None:
    router = ShortcutRouter()
    intent = {
        "id": "filter_equals",
        "slots": {
            "column": {"type": "column", "required": True},
            "value": {"type": "str", "required": True},
            "case_insensitive": {"type": "bool", "required": False, "default": True},
            "top_n": {"type": "int", "required": False, "default": 0},
        },
    }
    query = "Покажи топ 3 записи де бренд = 'Apple'"
    columns = ["бренд", "модель"]
    profile = {"dtypes": {"бренд": "str", "модель": "str"}}

    slots = router._fill_slots(intent, query, columns, profile)
    assert slots is not None
    assert slots.get("top_n") == 3


def test_fill_slots_groupby_count_out_col_not_taken_from_unrelated_quotes() -> None:
    router = ShortcutRouter()
    intent = {
        "id": "groupby_count",
        "slots": {
            "group_col": {"type": "column", "required": True},
            "target_col": {"type": "column", "required": False},
            "sort": {"type": "enum", "required": False, "values": ["asc", "desc"], "default": "desc"},
            "top_n": {"type": "int", "required": False, "default": 0},
            "out_col": {"type": "str", "required": False, "default": "count"},
        },
    }
    query = 'Бренди які починаються на літеру "A" та кількість моделей'
    columns = ["Бренд", "Модель", "Кількість"]
    profile = {"dtypes": {"Бренд": "str", "Модель": "str", "Кількість": "int64"}}
    slots = router._fill_slots(intent, query, columns, profile)
    assert slots is not None
    assert slots.get("out_col") == "count"


def test_compile_filter_equals_applies_optional_top_n() -> None:
    router = ShortcutRouter()
    intent = {
        "id": "filter_equals_top_n",
        "plan": [{"op": "filter_equals"}],
        "slots": {},
    }
    slots = {
        "column": "Бренд",
        "value": "Apple",
        "case_insensitive": True,
        "top_n": 3,
    }
    code = router._compile_plan(intent, slots, profile={})
    assert code is not None
    assert "_work = df[df['Бренд'].astype(str).str.lower() == str('Apple').lower()]" in code
    assert "result = _work.head(3)" in code
    assert "result = _work\n" not in code


def test_extract_multi_conditions_llm_returns_valid_conditions() -> None:
    def fake_llm(_system: str, _user: str) -> dict:
        return {
            "conditions": [
                {"column": "Категорія", "operator": "equals", "value": "Ноутбук", "case_insensitive": True},
                {"column": "Бренд", "operator": "equals", "value": "Apple", "case_insensitive": True},
                {"column": "Missing", "operator": "equals", "value": "x"},
            ]
        }

    router = ShortcutRouter(llm_json=fake_llm)
    out = router._extract_multi_conditions_llm(
        query="Знайди всі ноутбуки Apple",
        columns=["Категорія", "Бренд", "Статус"],
        profile={"dtypes": {"Категорія": "str", "Бренд": "str", "Статус": "str"}},
    )
    assert out is not None
    assert len(out) == 2
    assert out[0]["column"] == "Категорія"
    assert out[1]["column"] == "Бренд"


def test_extract_multi_conditions_llm_requires_at_least_two_conditions() -> None:
    def fake_llm(_system: str, _user: str) -> dict:
        return {
            "conditions": [
                {"column": "Бренд", "operator": "equals", "value": "Apple", "case_insensitive": True},
            ]
        }

    router = ShortcutRouter(llm_json=fake_llm)
    out = router._extract_multi_conditions_llm(
        query="Покажи товари бренду Apple",
        columns=["Категорія", "Бренд", "Статус"],
        profile={"dtypes": {"Категорія": "str", "Бренд": "str", "Статус": "str"}},
    )
    assert out is None


def test_extract_multi_conditions_llm_ignores_template_placeholders() -> None:
    def fake_llm(_system: str, _user: str) -> dict:
        return {
            "conditions": [
                {
                    "column": "Категорія",
                    "operator": "equals|contains|gt|lt|gte|lte",
                    "value": "<value>",
                    "case_insensitive": True,
                }
            ]
        }

    router = ShortcutRouter(llm_json=fake_llm)
    out = router._extract_multi_conditions_llm(
        query="Знайди всі ноутбуки Apple",
        columns=["Категорія", "Бренд", "Статус"],
        profile={"dtypes": {"Категорія": "str", "Бренд": "str", "Статус": "str"}},
        min_conditions=1,
    )
    assert out is None


def test_safe_float_parses_thousands_and_decimal_separators() -> None:
    router = ShortcutRouter()
    assert router._safe_float("3,000") == 3000.0
    assert router._safe_float("12 500,75") == 12500.75


def test_fill_slots_filter_comparison_prefers_numeric_llm_condition() -> None:
    def fake_llm(system: str, user: str) -> dict:
        if "Extract ALL filter conditions from the query." in system:
            return {
                "conditions": [
                    {"column": "Ціна_UAH", "operator": "lt", "value": "3000"},
                ]
            }
        if "Fill the requested slot value based on the query and columns." in system:
            payload = json.loads(user)
            if payload.get("slot_name") == "operator":
                return {"value": "<"}
        return {"value": None}

    router = ShortcutRouter(llm_json=fake_llm)
    intent = {
        "id": "filter_comparison",
        "slots": {
            "column": {"type": "column", "required": True},
            "operator": {"type": "enum", "required": True, "values": [">", "<", ">=", "<=", "="]},
            "value": {"type": "float", "required": True},
        },
        "plan": [{"op": "filter_comparison"}],
    }
    query = "**Товари дешевші за 3,000 UAH (категорії)**"
    columns = ["ID", "Категорія", "Бренд", "Ціна_UAH", "Кількість", "Статус"]
    profile = {
        "dtypes": {
            "ID": "float64",
            "Категорія": "str",
            "Бренд": "str",
            "Ціна_UAH": "float64",
            "Кількість": "int64",
            "Статус": "str",
        },
        "preview": [
            {"ID": 1.0, "Категорія": "Ноутбук", "Бренд": "Apple", "Ціна_UAH": 89999.0, "Кількість": 5, "Статус": "В наявності"},
            {"ID": 2.0, "Категорія": "Мишка", "Бренд": "Logitech", "Ціна_UAH": 1999.0, "Кількість": 12, "Статус": "В наявності"},
        ],
    }
    slots = router._fill_slots(intent, query, columns, profile)
    assert slots is not None
    assert slots["column"] == "Ціна_UAH"
    assert slots["operator"] == "<"
    assert slots["value"] == 3000.0


def test_fill_slots_filter_comparison_infers_lt_operator_without_llm_condition() -> None:
    router = ShortcutRouter(llm_json=lambda _system, _user: {"value": None})
    intent = {
        "id": "filter_comparison",
        "slots": {
            "column": {"type": "column", "required": True},
            "operator": {"type": "enum", "required": True, "values": [">", "<", ">=", "<=", "="]},
            "value": {"type": "float", "required": True},
        },
        "plan": [{"op": "filter_comparison"}],
    }
    query = "Товари дешевші за 3,000 UAH (категорії)"
    columns = ["Категорія", "Ціна_UAH", "Кількість"]
    profile = {
        "dtypes": {"Категорія": "str", "Ціна_UAH": "float64", "Кількість": "int64"},
        "preview": [
            {"Категорія": "Ноутбук", "Ціна_UAH": 89999.0, "Кількість": 5},
            {"Категорія": "Мишка", "Ціна_UAH": 1999.0, "Кількість": 12},
        ],
    }
    slots = router._fill_slots(intent, query, columns, profile)
    assert slots is not None
    assert slots["column"] == "Ціна_UAH"
    assert slots["operator"] == "<"
    assert slots["value"] == 3000.0


def test_pick_intent_prefers_strong_top_retrieval_over_weaker_llm_override() -> None:
    router = ShortcutRouter()
    candidates = [
        {"intent_id": "filter_multi_conditions", "score": 0.98, "example": "multi"},
        {"intent_id": "filter_contains", "score": 0.43, "example": "contains"},
        {"intent_id": "keyword_search_rows", "score": 0.40, "example": "keyword"},
    ]

    def fake_select(_query: str, _profile: dict, _candidates: list[dict]) -> dict:
        return {
            "intent_id": "filter_contains",
            "confidence": 0.95,
            "retrieval_score": 0.43,
            "example": "contains",
        }

    router._select_intent_with_llm = fake_select  # type: ignore[method-assign]
    picked = router._pick_intent_from_candidates("query", {}, candidates)
    assert picked is not None
    assert picked["intent_id"] == "filter_multi_conditions"
    assert picked["selector_mode"] in {"retrieval_guarded", "retrieval_confident"}


def test_pick_intent_skips_llm_when_retrieval_is_confident() -> None:
    router = ShortcutRouter()
    candidates = [
        {"intent_id": "filter_multi_conditions", "score": 0.986, "example": "multi"},
        {"intent_id": "keyword_search_rows", "score": 0.39, "example": "keyword"},
    ]

    def fail_select(_query: str, _profile: dict, _candidates: list[dict]) -> dict:
        raise AssertionError("LLM selector must be skipped for confident retrieval")

    router._select_intent_with_llm = fail_select  # type: ignore[method-assign]
    picked = router._pick_intent_from_candidates("query", {}, candidates)
    assert picked is not None
    assert picked["intent_id"] == "filter_multi_conditions"
    assert picked["selector_mode"] == "retrieval_confident"


def test_pick_intent_prefers_clear_leader_even_when_top_score_below_075() -> None:
    router = ShortcutRouter()
    candidates = [
        {"intent_id": "filter_multi_conditions", "score": 0.7216, "example": "multi"},
        {"intent_id": "keyword_search_rows", "score": 0.3979, "example": "keyword"},
        {"intent_id": "filtered_metric_aggregation", "score": 0.3819, "example": "agg"},
    ]

    def fake_select(_query: str, _profile: dict, _candidates: list[dict]) -> dict:
        return {
            "intent_id": "keyword_search_rows",
            "confidence": 0.95,
            "retrieval_score": 0.3979,
            "example": "keyword",
        }

    router._select_intent_with_llm = fake_select  # type: ignore[method-assign]
    picked = router._pick_intent_from_candidates("query", {}, candidates)
    assert picked is not None
    assert picked["intent_id"] == "filter_multi_conditions"
    assert picked["selector_mode"] == "retrieval_guarded"


def test_resolve_intent_redirects_filter_equals_to_multi_conditions() -> None:
    def fake_llm(_system: str, _user: str) -> dict:
        return {
            "conditions": [
                {"column": "Категорія", "operator": "equals", "value": "Ноутбук", "case_insensitive": True},
                {"column": "Бренд", "operator": "equals", "value": "Apple", "case_insensitive": True},
            ]
        }

    router = ShortcutRouter(llm_json=fake_llm)
    router._intents = {
        "filter_equals": {"id": "filter_equals"},
        "filter_multi_conditions": {"id": "filter_multi_conditions"},
    }
    resolved, preset = router._resolve_intent_and_slots(
        intent=router._intents["filter_equals"],
        query="Знайди всі ноутбуки Apple",
        columns=["Категорія", "Бренд", "Статус"],
        profile={"dtypes": {"Категорія": "str", "Бренд": "str", "Статус": "str"}},
    )
    assert resolved.get("id") == "filter_multi_conditions"
    assert isinstance(preset.get("conditions"), list)
    assert len(preset["conditions"]) == 2


def test_fill_slots_filter_multi_conditions_augments_single_llm_condition_from_preview() -> None:
    def fake_llm(system: str, _user: str) -> dict:
        if "Extract ALL filter conditions from the query." in system:
            return {
                "conditions": [
                    {"column": "Бренд", "operator": "equals", "value": "Apple", "case_insensitive": True},
                ]
            }
        return {}

    router = ShortcutRouter(llm_json=fake_llm)
    intent = {
        "id": "filter_multi_conditions",
        "slots": {
            "conditions": {"type": "json", "required": True},
        },
    }
    columns = ["Категорія", "Бренд", "Модель", "Ціна_UAH"]
    profile = {
        "dtypes": {"Категорія": "str", "Бренд": "str", "Модель": "str", "Ціна_UAH": "float64"},
        "preview": [
            {"Категорія": "Ноутбук", "Бренд": "Apple", "Модель": "MacBook Pro 14", "Ціна_UAH": 89999.0},
            {"Категорія": "Ноутбук", "Бренд": "ASUS", "Модель": "ROG Strix G16", "Ціна_UAH": 65000.0},
            {"Категорія": "Планшет", "Бренд": "Apple", "Модель": "iPad Pro 11", "Ціна_UAH": 45999.0},
        ],
    }
    slots = router._fill_slots(intent, "Знайди всі ноутбуки Apple", columns, profile)
    assert slots is not None
    conditions = slots.get("conditions") or []
    assert len(conditions) >= 2
    cols = {str(c.get("column") or "") for c in conditions if isinstance(c, dict)}
    assert "Бренд" in cols
    assert "Категорія" in cols


def test_resolve_intent_redirects_to_multi_conditions_with_preview_augmentation() -> None:
    def fake_llm(_system: str, _user: str) -> dict:
        return {
            "conditions": [
                {"column": "Бренд", "operator": "equals", "value": "Apple", "case_insensitive": True},
            ]
        }

    router = ShortcutRouter(llm_json=fake_llm)
    router._intents = {
        "filter_equals": {"id": "filter_equals"},
        "filter_multi_conditions": {"id": "filter_multi_conditions"},
    }
    resolved, preset = router._resolve_intent_and_slots(
        intent=router._intents["filter_equals"],
        query="Знайди всі ноутбуки Apple",
        columns=["Категорія", "Бренд", "Модель", "Ціна_UAH"],
        profile={
            "dtypes": {"Категорія": "str", "Бренд": "str", "Модель": "str", "Ціна_UAH": "float64"},
            "preview": [
                {"Категорія": "Ноутбук", "Бренд": "Apple", "Модель": "MacBook Pro 14", "Ціна_UAH": 89999.0},
                {"Категорія": "Ноутбук", "Бренд": "ASUS", "Модель": "ROG Strix G16", "Ціна_UAH": 65000.0},
                {"Категорія": "Планшет", "Бренд": "Apple", "Модель": "iPad Pro 11", "Ціна_UAH": 45999.0},
            ],
        },
    )
    assert resolved.get("id") == "filter_multi_conditions"
    conditions = preset.get("conditions") or []
    assert len(conditions) >= 2
    cols = {str(c.get("column") or "") for c in conditions if isinstance(c, dict)}
    assert "Бренд" in cols
    assert "Категорія" in cols


def test_should_augment_multi_conditions_skips_single_numeric_filter_without_multi_cue() -> None:
    router = ShortcutRouter()
    conditions = [{"column": "Кількість", "operator": "lt", "value": "5"}]
    assert router._should_augment_multi_conditions("Які товари закінчуються (менше 5 штук)?", conditions) is False


def test_resolve_intent_does_not_redirect_numeric_single_filter_to_multi_conditions() -> None:
    def fake_llm(_system: str, _user: str) -> dict:
        return {"conditions": [{"column": "Кількість", "operator": "lt", "value": "5"}]}

    router = ShortcutRouter(llm_json=fake_llm)
    router._intents = {
        "filter_comparison": {"id": "filter_comparison"},
        "filter_multi_conditions": {"id": "filter_multi_conditions"},
    }
    resolved, preset = router._resolve_intent_and_slots(
        intent=router._intents["filter_comparison"],
        query="Які товари закінчуються (менше 5 штук)?",
        columns=["ID", "Категорія", "Модель", "Кількість", "Статус"],
        profile={
            "dtypes": {"ID": "float64", "Категорія": "str", "Модель": "str", "Кількість": "int64", "Статус": "str"},
            "preview": [
                {"ID": 1.0, "Категорія": "Ноутбук", "Модель": "ROG Strix G16", "Кількість": 3, "Статус": "Закінчується"},
                {"ID": 2.0, "Категорія": "Ноутбук", "Модель": "MacBook Air 15", "Кількість": 12, "Статус": "В наявності"},
            ],
        },
    )
    assert resolved.get("id") == "filter_comparison"
    assert preset == {}


def test_extract_query_ir_builds_hard_and_soft_constraints() -> None:
    def fake_llm(system: str, _user: str) -> dict:
        if "Extract canonical spreadsheet QueryIR." in system:
            return {
                "hard_conditions": [{"column": "Кількість", "operator": "lt", "value": "5"}],
                "soft_conditions": [{"column": "Статус", "operator": "equals", "value": "Закінчується"}],
                "explicit_multi_filter": False,
            }
        if "Extract ALL filter conditions from the query." in system:
            return {"conditions": [{"column": "Кількість", "operator": "lt", "value": "5"}]}
        return {}

    router = ShortcutRouter(llm_json=fake_llm)
    ir = router._extract_query_ir(
        "Які товари закінчуються (менше 5 штук)?",
        "",
        {
            "columns": ["Категорія", "Модель", "Кількість", "Статус"],
            "dtypes": {"Категорія": "str", "Модель": "str", "Кількість": "int64", "Статус": "str"},
            "preview": [],
        },
    )
    hard = ir.get("hard_conditions") or []
    soft = ir.get("soft_conditions") or []
    assert any(str(c.get("operator")) == "lt" for c in hard if isinstance(c, dict))
    assert any(str(c.get("column")) == "Статус" for c in soft if isinstance(c, dict))


def test_extract_query_ir_drops_blank_soft_condition_values() -> None:
    def fake_llm(system: str, _user: str) -> dict:
        if "Extract canonical spreadsheet QueryIR." in system:
            return {
                "hard_conditions": [],
                "soft_conditions": [{"column": "Статус", "operator": "equals", "value": ""}],
                "explicit_multi_filter": False,
            }
        if "Extract ALL filter conditions from the query." in system:
            return {"conditions": []}
        return {}

    router = ShortcutRouter(llm_json=fake_llm)
    ir = router._extract_query_ir(
        "Порахуй кількість по кожному Статусу",
        "",
        {
            "columns": ["Категорія", "Кількість", "Статус"],
            "dtypes": {"Категорія": "str", "Кількість": "int64", "Статус": "str"},
            "preview": [],
        },
    )
    assert (ir.get("soft_conditions") or []) == []


def test_extract_query_ir_resolves_missing_hard_condition_column_from_preview() -> None:
    def fake_llm(system: str, _user: str) -> dict:
        if "Extract canonical spreadsheet QueryIR." in system:
            return {
                "hard_conditions": [{"column": "", "operator": "contains", "value": "Apple"}],
                "soft_conditions": [],
                "explicit_multi_filter": False,
            }
        if "Extract ALL filter conditions from the query." in system:
            return {"conditions": []}
        return {}

    router = ShortcutRouter(llm_json=fake_llm)
    ir = router._extract_query_ir(
        "Яка найбільша ціна в Apple ноутбуках?",
        "",
        {
            "columns": ["Категорія", "Бренд", "Ціна_UAH"],
            "dtypes": {"Категорія": "str", "Бренд": "str", "Ціна_UAH": "float64"},
            "preview": [
                {"Категорія": "Ноутбук", "Бренд": "Apple", "Ціна_UAH": 5200},
                {"Категорія": "Ноутбук", "Бренд": "ASUS", "Ціна_UAH": 4800},
            ],
        },
    )
    hard = ir.get("hard_conditions") or []
    assert hard
    assert str(hard[0].get("column")) == "Бренд"


def test_intent_slots_compatible_rejects_promoted_soft_condition() -> None:
    router = ShortcutRouter()
    query_ir = {
        "hard_conditions": [{"column": "Кількість", "operator": "lt", "value": 5}],
        "soft_conditions": [{"column": "Статус", "operator": "equals", "value": "Закінчується"}],
        "explicit_multi_filter": False,
    }
    slots = {
        "conditions": [
            {"column": "Кількість", "operator": "lt", "value": 5, "source": "llm_extract"},
            {"column": "Статус", "operator": "equals", "value": "Закінчується", "source": "augmented_preview"},
        ]
    }
    ok, reason = router._intent_slots_compatible_with_query(
        intent_id="filter_multi_conditions",
        slots=slots,
        query="Які товари закінчуються (менше 5 штук)?",
        query_ir=query_ir,
    )
    assert ok is False
    assert reason == "constraint_soft_conditions_promoted_to_hard"


def test_intent_slots_compatible_allows_soft_condition_when_explicitly_requested() -> None:
    router = ShortcutRouter()
    query_ir = {
        "hard_conditions": [{"column": "Кількість", "operator": "lt", "value": 5}],
        "soft_conditions": [{"column": "Статус", "operator": "equals", "value": "Закінчується"}],
        "explicit_multi_filter": True,
    }
    slots = {
        "conditions": [
            {"column": "Кількість", "operator": "lt", "value": 5, "source": "llm_extract"},
            {"column": "Статус", "operator": "equals", "value": "Закінчується", "source": "augmented_preview"},
        ]
    }
    ok, reason = router._intent_slots_compatible_with_query(
        intent_id="filter_multi_conditions",
        slots=slots,
        query="Покажи товари менше 5 штук і статус Закінчується",
        query_ir=query_ir,
    )
    assert ok is True
    assert reason == "ok"


def test_intent_slots_compatible_rejects_unresolved_hard_condition_column() -> None:
    router = ShortcutRouter()
    query_ir = {
        "hard_conditions": [{"column": "", "operator": "contains", "value": "Apple"}],
        "soft_conditions": [],
        "explicit_multi_filter": False,
    }
    slots = {
        "conditions": [
            {"column": "Бренд", "operator": "contains", "value": "Apple", "source": "llm_extract"},
        ]
    }
    ok, reason = router._intent_slots_compatible_with_query(
        intent_id="filter_multi_conditions",
        slots=slots,
        query="Покажи товари бренду Apple",
        query_ir=query_ir,
    )
    assert ok is False
    assert reason == "constraint_hard_conditions_not_preserved"


def test_intent_slots_compatible_accepts_filtered_metric_hard_contains_via_filter_slots() -> None:
    router = ShortcutRouter()
    query_ir = {
        "hard_conditions": [{"column": "Категорія", "operator": "contains", "value": "Миша"}],
        "soft_conditions": [],
        "explicit_multi_filter": False,
    }
    slots = {
        "filter_col": "Категорія",
        "filter_value": "Миша",
        "target_col": "Ціна_UAH",
        "metric": "max",
    }
    ok, reason = router._intent_slots_compatible_with_query(
        intent_id="filtered_metric_aggregation",
        slots=slots,
        query="Яка найбільша ціна на товар категорії Миша",
        query_ir=query_ir,
    )
    assert ok is True
    assert reason == "ok"


def test_intent_slots_compatible_accepts_hard_equals_when_slots_use_contains_for_text_entity() -> None:
    router = ShortcutRouter()
    query_ir = {
        "hard_conditions": [{"column": "Категорія", "operator": "equals", "value": "Миша"}],
        "soft_conditions": [],
        "explicit_multi_filter": False,
    }
    slots = {
        "filter_col": "Категорія",
        "filter_value": "Миша",
        "filter_op": "contains",
        "target_col": "Ціна_UAH",
        "metric": "max",
    }
    ok, reason = router._intent_slots_compatible_with_query(
        intent_id="filtered_metric_aggregation",
        slots=slots,
        query="Яка найбільша ціна на товар категорії Миша",
        query_ir=query_ir,
    )
    assert ok is True
    assert reason == "ok"


def test_compile_filter_multi_conditions_builds_and_mask() -> None:
    router = ShortcutRouter()
    intent = {
        "id": "filter_multi_conditions",
        "plan": [{"op": "filter_multi_conditions"}],
        "slots": {},
    }
    slots = {
        "conditions": [
            {"column": "Категорія", "operator": "equals", "value": "Ноутбук", "case_insensitive": True},
            {"column": "Бренд", "operator": "equals", "value": "Apple", "case_insensitive": True},
        ]
    }
    code = router._compile_plan(
        intent=intent,
        slots=slots,
        profile={"columns": ["Категорія", "Бренд", "Статус"]},
    )
    assert code is not None
    assert "_work = df[" in code
    assert "&" in code
    assert "df['Категорія'].astype(str).str.lower() == str('Ноутбук').lower()" in code
    assert "df['Бренд'].astype(str).str.lower() == str('Apple').lower()" in code
    assert "result = _work" in code


def test_shortcut_router_retries_retrieval_with_normalized_query_on_miss() -> None:
    normalized_query = "Виручка по категоріях ціна помножити на кількість"

    def fake_llm(system: str, _user: str) -> dict:
        if "Normalize spreadsheet user query for intent retrieval." in system:
            return {"normalized_query": normalized_query, "confidence": 0.9}
        return {"intent_id": "NONE", "confidence": 0.0}

    router = ShortcutRouter(llm_json=fake_llm)
    router._intents = {
        "stats_shape": {
            "id": "stats_shape",
            "plan": [{"op": "stats_shape"}],
            "slots": {},
        }
    }

    seen_queries: list[str] = []

    def fake_retrieve(query: str) -> list[dict]:
        seen_queries.append(query)
        if query == normalized_query:
            return [{"intent_id": "stats_shape", "score": 0.95, "example": "shape"}]
        return []

    router._ensure_loaded = lambda: True  # type: ignore[method-assign]
    router._assess_query_complexity = lambda _q, _p: 0.0  # type: ignore[method-assign]
    router._has_filter_context = lambda _q, _p: False  # type: ignore[method-assign]
    router._retrieve_candidates = fake_retrieve  # type: ignore[method-assign]

    profile = {"columns": ["Категорія", "Ціна", "Кількість"], "dtypes": {}}
    out = router.shortcut_to_sandbox_code("Виручка по категоріях (Ціна × Кількість)", profile)

    assert out is not None
    _code, meta = out
    assert seen_queries == ["Виручка по категоріях (Ціна × Кількість)", normalized_query]
    assert meta.get("retrieval_query_used") == normalized_query
    assert meta.get("normalized_query") == normalized_query


def test_shortcut_router_retries_normalized_query_when_primary_below_threshold() -> None:
    normalized_query = "Порахуй виручку по категоріях як ціна помножена на кількість"

    def fake_llm(system: str, _user: str) -> dict:
        if "Normalize spreadsheet user query for intent retrieval." in system:
            return {"normalized_query": normalized_query, "confidence": 0.92}
        return {"intent_id": "NONE", "confidence": 0.0}

    router = ShortcutRouter(llm_json=fake_llm)
    router._intents = {
        "stats_shape": {
            "id": "stats_shape",
            "plan": [{"op": "stats_shape"}],
            "slots": {},
        }
    }

    seen_queries: list[str] = []

    def fake_retrieve(query: str) -> list[dict]:
        seen_queries.append(query)
        if query == normalized_query:
            return [{"intent_id": "stats_shape", "score": 0.9, "example": "shape"}]
        return [{"intent_id": "stats_shape", "score": 0.12, "example": "weak"}]

    router._ensure_loaded = lambda: True  # type: ignore[method-assign]
    router._assess_query_complexity = lambda _q, _p: 0.0  # type: ignore[method-assign]
    router._has_filter_context = lambda _q, _p: False  # type: ignore[method-assign]
    router._retrieve_candidates = fake_retrieve  # type: ignore[method-assign]

    profile = {"columns": ["Категорія", "Ціна", "Кількість"], "dtypes": {}}
    out = router.shortcut_to_sandbox_code("Виручка по категоріях (Ціна × Кількість)", profile)

    assert out is not None
    _code, meta = out
    assert seen_queries == ["Виручка по категоріях (Ціна × Кількість)", normalized_query]
    assert meta.get("retrieval_query_used") == normalized_query


def test_shortcut_router_falls_back_to_next_candidate_when_primary_cannot_compile() -> None:
    router = ShortcutRouter()
    router._intents = {
        "filter_contains": {
            "id": "filter_contains",
            "slots": {
                "column": {"type": "column", "required": True},
                "value": {"type": "str", "required": True},
            },
            "plan": [{"op": "filter_contains"}],
        },
        "filter_comparison": {
            "id": "filter_comparison",
            "slots": {
                "column": {"type": "column", "required": True},
                "operator": {"type": "enum", "required": True, "values": [">", "<", ">=", "<=", "="]},
                "value": {"type": "float", "required": True},
            },
            "plan": [{"op": "filter_comparison"}],
        },
    }
    router._ensure_loaded = lambda: True  # type: ignore[method-assign]
    router._assess_query_complexity = lambda _q, _p: 0.0  # type: ignore[method-assign]
    router._has_filter_context = lambda _q, _p: False  # type: ignore[method-assign]
    router._llm_normalize_query_for_retrieval = lambda _q, _p: ""  # type: ignore[method-assign]
    router._retrieve_candidates = (  # type: ignore[method-assign]
        lambda _q: [
            {"intent_id": "filter_contains", "score": 0.83, "example": "contains"},
            {"intent_id": "filter_comparison", "score": 0.79, "example": "comparison"},
        ]
    )
    router._pick_intent_from_candidates = (  # type: ignore[method-assign]
        lambda _q, _p, _c: {
            "intent_id": "filter_contains",
            "confidence": 0.95,
            "retrieval_score": 0.83,
            "example": "contains",
            "selector_mode": "llm",
        }
    )

    profile = {
        "columns": ["Назва", "Ціна_UAH"],
        "dtypes": {"Назва": "str", "Ціна_UAH": "float64"},
        "preview": [
            {"Назва": "Ноутбук", "Ціна_UAH": 89999.0},
            {"Назва": "Миша", "Ціна_UAH": 1999.0},
        ],
    }
    out = router.shortcut_to_sandbox_code("Товари дешевші за 3,000 UAH", profile)
    assert out is not None
    code, meta = out
    assert "_work = df[df['Ціна_UAH'] < 3000.0]" in code
    assert meta.get("intent_id") == "filter_comparison"
    assert meta.get("initial_intent_id") == "filter_contains"
    assert meta.get("fallback_attempts") == 1


def test_intent_slots_compatible_with_query_rejects_missing_numeric_comparison_constraint() -> None:
    router = ShortcutRouter()
    ok, reason = router._intent_slots_compatible_with_query(
        intent_id="keyword_search_rows",
        slots={"keyword": "дешевші"},
        query="Товари дешевші за 3,000 UAH",
    )
    assert ok is False
    assert reason == "constraint_numeric_comparison_unsupported_by_capability"


def test_intent_slots_compatible_with_query_accepts_revenue_product_with_declared_capability() -> None:
    router = ShortcutRouter()
    router._intents = {
        "groupby_agg": {
            "id": "groupby_agg",
            "capabilities": {
                "metric_types": ["numeric", "product"],
                "filters": ["equals", "contains", "startswith", "endswith"],
                "prefilter_before_groupby": True,
            },
        }
    }
    ok, reason = router._intent_slots_compatible_with_query(
        intent_id="groupby_agg",
        slots={
            "group_col": "Категорія",
            "agg": "sum",
            "mul_left_col": "Ціна_UAH",
            "mul_right_col": "Кількість",
        },
        query="Виручка по категоріях (Ціна × Кількість)",
    )
    assert ok is True
    assert reason == "ok"
