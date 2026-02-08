import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
PIPELINES_DIR = os.path.join(ROOT, "pipelines")
if PIPELINES_DIR not in sys.path:
    sys.path.append(PIPELINES_DIR)

from pipelines.shortcut_router import ShortcutRouter, ShortcutRouterConfig


def _router() -> ShortcutRouter:
    cfg = ShortcutRouterConfig(enabled=False)
    return ShortcutRouter(cfg, llm_json=None)


def test_fill_slots_groupby_count_rule_based() -> None:
    router = _router()
    intent = {
        "id": "groupby_count",
        "slots": {"group_col": {"type": "column", "required": True}, "top_n": {"type": "int", "required": False, "default": 0}},
        "plan": [{"op": "groupby_count"}],
    }
    profile = {"rows": 100, "dtypes": {"Марка машин": "object"}}
    columns = ["Марка машин", "Кількість"]
    slots = router._fill_slots(intent, "скільки для кожної марки машин", columns, profile)  # type: ignore[attr-defined]
    assert slots is not None
    assert slots.get("group_col") == "Марка машин"


def test_compile_drop_rows_by_position_contains_commit() -> None:
    router = _router()
    intent = {
        "id": "drop_rows_by_position",
        "slots": {"row_indices": {"type": "row_indices", "required": True}},
        "plan": [{"op": "drop_rows_by_position"}],
    }
    profile = {"rows": 10, "columns": ["A"]}
    slots = {"row_indices": [1, 3, 5]}
    code = router._compile_plan(intent, slots, profile)  # type: ignore[attr-defined]
    assert code
    assert "drop_rows_by_position" not in code
    assert "COMMIT_DF = True" in code
    assert "_idx_1b" in code


def test_resolve_groupby_count_to_groupby_agg_with_llm() -> None:
    def _fake_llm_json(_system: str, _payload: str) -> dict:
        return {"group_col": "Категорія", "target_col": "Кількість", "agg": "sum", "top_n": 3}

    cfg = ShortcutRouterConfig(enabled=False)
    router = ShortcutRouter(cfg, llm_json=_fake_llm_json)
    groupby_count_intent = {
        "id": "groupby_count",
        "slots": {"group_col": {"type": "column", "required": True}},
        "plan": [{"op": "groupby_count"}],
    }
    groupby_agg_intent = {
        "id": "groupby_agg",
        "slots": {
            "group_col": {"type": "column", "required": True},
            "target_col": {"type": "column", "required": True},
            "agg": {"type": "enum", "required": False, "default": "sum", "values": ["sum", "mean", "min", "max"]},
        },
        "plan": [{"op": "groupby_agg"}],
    }
    router._intents = {"groupby_count": groupby_count_intent, "groupby_agg": groupby_agg_intent}  # type: ignore[attr-defined]
    profile = {"rows": 100, "columns": ["Категорія", "Кількість"], "dtypes": {"Кількість": "int64"}}
    resolved, preset = router._resolve_intent_and_slots(  # type: ignore[attr-defined]
        groupby_count_intent,
        "Які ТОП-3 категорії мають найбільшу загальну кількість?",
        ["Категорія", "Кількість"],
        profile,
    )
    assert resolved.get("id") == "groupby_agg"
    assert preset.get("group_col") == "Категорія"
    assert preset.get("target_col") == "Кількість"
    assert preset.get("agg") == "sum"
    assert preset.get("top_n") == 3


def test_compile_groupby_agg_applies_sort_and_top_n() -> None:
    router = _router()
    intent = {"id": "groupby_agg", "slots": {}, "plan": [{"op": "groupby_agg"}]}
    profile = {"rows": 100, "columns": ["Категорія", "Кількість"]}
    slots = {"group_col": "Категорія", "target_col": "Кількість", "agg": "sum", "top_n": 3}
    code = router._compile_plan(intent, slots, profile)  # type: ignore[attr-defined]
    assert code
    assert ".agg('sum').reset_index(name='Кількість')" in code
    assert "sort_values('Кількість', ascending=False)" in code
    assert "result = result.head(3)" in code


def test_compile_stats_nulls_has_no_lambda() -> None:
    router = _router()
    intent = {"id": "stats_nulls", "slots": {}, "plan": [{"op": "stats_nulls"}]}
    code = router._compile_plan(intent, {"top_n": 5}, {"rows": 10})  # type: ignore[attr-defined]
    assert code
    assert "lambda" not in code
    assert "sort_values(ascending=False).head(5)" in code
