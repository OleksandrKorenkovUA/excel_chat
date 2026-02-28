import json

from pipelines.spreadsheet_analyst_pipeline import Pipeline


def test_lookup_slots_maps_semantic_column_aliases_via_llm() -> None:
    pipeline = Pipeline()

    def fake_llm_json(system: str, user: str) -> dict:
        if system.startswith("Map the query to table lookup slots."):
            return {
                "mode": "lookup",
                "filters": [{"column": "Запаси", "op": "eq", "value": "в запасах"}],
                "output_columns": ["К-сть"],
                "limit": None,
            }
        if system.startswith("Resolve a semantic alias from a user request"):
            payload = json.loads(user)
            alias = str(payload.get("alias") or "")
            if alias == "Запаси":
                return {"column": "Складські запаси", "confidence": 0.91}
            if alias == "К-сть":
                return {"column": "Кількість", "confidence": 0.87}
        return {}

    pipeline._llm_json = fake_llm_json  # type: ignore[method-assign]
    profile = {
        "columns": ["Складські запаси", "Кількість", "Товар"],
        "dtypes": {"Складські запаси": "str", "Кількість": "int64", "Товар": "str"},
        "preview": [{"Складські запаси": "в наявності", "Кількість": 3, "Товар": "A"}],
    }

    slots = pipeline._llm_pick_lookup_slots("Скільки товарів в запасах?", profile)

    assert slots.get("mode") == "lookup"
    filters = slots.get("filters") or []
    assert filters and filters[0]["column"] == "Складські запаси"
    assert filters[0]["op"] == "contains"
    assert slots.get("output_columns") == ["Кількість"]


def test_lookup_shortcut_status_pattern_handles_inventory_synonym() -> None:
    pipeline = Pipeline()
    profile = {
        "columns": ["Склад", "Кількість"],
        "dtypes": {"Склад": "str", "Кількість": "int64"},
    }
    slots = {
        "mode": "lookup",
        "filters": [{"column": "Склад", "op": "eq", "value": "в запасах"}],
        "output_columns": ["Кількість"],
    }

    shortcut = pipeline._lookup_shortcut_code_from_slots("Скільки товарів в запасах?", profile, slots)
    assert shortcut is not None
    code, _ = shortcut
    assert "_pat0" in code
    assert "запас" in code
    assert ".str.contains(_pat0" in code


def test_lookup_semantic_alias_rejects_low_confidence_mapping() -> None:
    pipeline = Pipeline()

    def fake_llm_json(system: str, user: str) -> dict:
        if system.startswith("Map the query to table lookup slots."):
            return {
                "mode": "lookup",
                "filters": [{"column": "Запаси", "op": "contains", "value": "в запасах"}],
                "output_columns": [],
            }
        if system.startswith("Resolve a semantic alias from a user request"):
            return {"column": "Склад", "confidence": 0.1}
        return {}

    pipeline._llm_json = fake_llm_json  # type: ignore[method-assign]
    profile = {
        "columns": ["Склад", "Кількість"],
        "dtypes": {"Склад": "str", "Кількість": "int64"},
        "preview": [{"Склад": "в наявності", "Кількість": 2}],
    }

    slots = pipeline._llm_pick_lookup_slots("Скільки товарів в запасах?", profile)
    assert slots.get("filters") == []


def test_lookup_status_eq_kept_exact_when_value_is_quoted_in_question() -> None:
    pipeline = Pipeline()
    profile = {
        "columns": ["Статус", "Модель"],
        "dtypes": {"Статус": "str", "Модель": "str"},
    }
    slots = {
        "mode": "lookup",
        "filters": [{"column": "Статус", "op": "eq", "value": "Закінчується"}],
        "output_columns": ["Модель", "Статус"],
    }
    shortcut = pipeline._lookup_shortcut_code_from_slots(
        'Товари зі статусом "Закінчується"',
        profile,
        slots,
    )
    assert shortcut is not None
    code, _ = shortcut
    assert "_pat0" not in code
    assert ".str.contains(" not in code
    assert "== 'Закінчується'.strip().lower()" in code


def test_lookup_status_semantic_low_stock_pattern_is_not_broad_instock_pattern() -> None:
    pipeline = Pipeline()
    profile = {
        "columns": ["Статус", "Модель"],
        "dtypes": {"Статус": "str", "Модель": "str"},
    }
    slots = {
        "mode": "lookup",
        "filters": [{"column": "Статус", "op": "contains", "value": "Закінчується"}],
        "output_columns": ["Модель", "Статус"],
    }
    shortcut = pipeline._lookup_shortcut_code_from_slots(
        "Товари зі статусом закінчується",
        profile,
        slots,
    )
    assert shortcut is not None
    code, _ = shortcut
    assert "_pat0" in code
    assert "закінч" in code
    assert "наявн" not in code


def test_lookup_slots_prefix_cue_converts_contains_anchor_to_startswith() -> None:
    pipeline = Pipeline()

    def fake_llm_json(system: str, _user: str) -> dict:
        if system.startswith("Map the query to table lookup slots."):
            return {
                "mode": "lookup",
                "filters": [{"column": "Бренд", "op": "contains", "value": "^A"}],
                "output_columns": ["Бренд"],
                "limit": None,
            }
        return {}

    pipeline._llm_json = fake_llm_json  # type: ignore[method-assign]
    profile = {
        "columns": ["Бренд", "Модель"],
        "dtypes": {"Бренд": "str", "Модель": "str"},
        "preview": [{"Бренд": "ASUS", "Модель": "ROG"}],
    }

    slots = pipeline._llm_pick_lookup_slots('Бренди, що починаються на літеру "A"', profile)
    filters = slots.get("filters") or []
    assert filters
    assert filters[0]["op"] == "startswith"
    assert filters[0]["value"] == "A"


def test_lookup_shortcut_startswith_operator_generates_prefix_match() -> None:
    pipeline = Pipeline()
    profile = {
        "columns": ["Бренд", "Модель"],
        "dtypes": {"Бренд": "str", "Модель": "str"},
    }
    slots = {
        "mode": "lookup",
        "filters": [{"column": "Бренд", "op": "startswith", "value": "A"}],
        "output_columns": ["Бренд", "Модель"],
    }

    shortcut = pipeline._lookup_shortcut_code_from_slots('Бренди на літеру "A"', profile, slots)
    assert shortcut is not None
    code, _ = shortcut
    assert ".str.startswith(" in code
    assert ".str.contains('^A'" not in code


def test_lookup_slots_infers_max_aggregation_and_clears_limit() -> None:
    pipeline = Pipeline()

    def fake_llm_json(system: str, _user: str) -> dict:
        if system.startswith("Map the query to table lookup slots."):
            return {
                "mode": "lookup",
                "filters": [{"column": "Категорія", "op": "contains", "value": "Миша"}],
                "output_columns": ["Ціна_UAH"],
                "limit": 1,
                "aggregation": "none",
            }
        return {}

    pipeline._llm_json = fake_llm_json  # type: ignore[method-assign]
    profile = {
        "columns": ["Категорія", "Ціна_UAH", "Модель"],
        "dtypes": {"Категорія": "str", "Ціна_UAH": "float64", "Модель": "str"},
        "preview": [{"Категорія": "Миша", "Ціна_UAH": 5200, "Модель": "M1"}],
    }

    slots = pipeline._llm_pick_lookup_slots("Яка найбліьша ціна на товар категорії Миша", profile)
    assert slots.get("aggregation") == "max"
    assert "limit" not in slots


def test_lookup_shortcut_aggregation_max_uses_aggregate_not_head_limit() -> None:
    pipeline = Pipeline()
    profile = {
        "columns": ["Категорія", "Ціна_UAH", "Модель"],
        "dtypes": {"Категорія": "str", "Ціна_UAH": "float64", "Модель": "str"},
    }
    slots = {
        "mode": "lookup",
        "filters": [{"column": "Категорія", "op": "contains", "value": "Миша"}],
        "output_columns": ["Ціна_UAH"],
        "aggregation": "max",
        "limit": 1,
    }

    shortcut = pipeline._lookup_shortcut_code_from_slots("Яка найбільша ціна на товар категорії Миша", profile, slots)
    assert shortcut is not None
    code, _ = shortcut
    assert "_agg_value = _num_non_na.max()" in code
    assert ".head(" not in code


def test_lookup_slots_aggregate_query_keeps_lookup_mode_for_ambiguous_entity_filter() -> None:
    pipeline = Pipeline()

    def fake_llm_json(system: str, _user: str) -> dict:
        if system.startswith("Map the query to table lookup slots."):
            return {
                "mode": "lookup",
                "filters": [{"column": "Модель", "op": "eq", "value": "Миша"}],
                "output_columns": ["Ціна_UAH"],
                "aggregation": "max",
                "limit": None,
            }
        return {}

    pipeline._llm_json = fake_llm_json  # type: ignore[method-assign]
    profile = {
        "columns": ["Категорія", "Модель", "Опис", "Ціна_UAH"],
        "dtypes": {"Категорія": "str", "Модель": "str", "Опис": "str", "Ціна_UAH": "float64"},
        "preview": [{"Категорія": "Миша", "Модель": "M110", "Опис": "Офісна миша", "Ціна_UAH": 5200.0}],
    }

    slots = pipeline._llm_pick_lookup_slots("Яка максимальна ціна на товар Миша?", profile)
    assert slots.get("mode") == "lookup"
    assert slots.get("aggregation") == "max"


def test_lookup_shortcut_aggregation_adds_fallback_columns_and_not_found_message() -> None:
    pipeline = Pipeline()
    profile = {
        "columns": ["Категорія", "Модель", "Опис", "Ціна_UAH"],
        "dtypes": {"Категорія": "str", "Модель": "str", "Опис": "str", "Ціна_UAH": "float64"},
    }
    slots = {
        "mode": "lookup",
        "filters": [{"column": "Модель", "op": "eq", "value": "Миша"}],
        "output_columns": ["Ціна_UAH"],
        "aggregation": "max",
    }

    shortcut = pipeline._lookup_shortcut_code_from_slots("Яка максимальна ціна на товар Миша?", profile, slots)
    assert shortcut is not None
    code, _ = shortcut
    assert "_alt_cols0" in code
    assert "Категорія" in code
    assert "Не знайдено рядків для фільтра" in code
