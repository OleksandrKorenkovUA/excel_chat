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
