import json

from pipelines.shortcut_router.shortcut_router import ShortcutRouter


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
