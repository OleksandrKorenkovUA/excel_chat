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
