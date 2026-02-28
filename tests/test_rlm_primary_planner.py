from pipelines.spreadsheet_analyst_pipeline import Pipeline


def test_prepare_prefers_rlm_primary_over_shortcut() -> None:
    pipeline = Pipeline()
    pipeline.valves.rlm_codegen_enabled = True
    pipeline.valves.rlm_primary_planner_enabled = True

    pipeline._select_router_or_shortcut = lambda _q, _p, _h: (  # type: ignore[method-assign]
        None,
        ("result = 999\n", "shortcut_plan"),
        {},
        {},
    )
    pipeline._plan_code_with_rlm_tool = lambda **_kwargs: (  # type: ignore[method-assign]
        "result = 1\n",
        "rlm_primary_plan",
        "read",
        False,
    )
    pipeline._plan_code = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("plan_code should not be called"))  # type: ignore[method-assign]

    out = pipeline._prepare_analysis_code_for_question(
        question="Скільки рядків?",
        profile={"columns": ["A"], "dtypes": {"A": "int64"}},
        has_edit=False,
    )

    assert out["ok"] is True
    assert "result = 1" in str(out.get("analysis_code") or "")
    events = list(out.get("events") or [])
    assert any(e[0] == "codegen_rlm_tool" and (e[1] or {}).get("phase") == "primary_planner" for e in events)
    assert not any(e[0] == "codegen_shortcut" and e[1] == {} for e in events)


def test_prepare_uses_shortcut_when_rlm_primary_returns_empty() -> None:
    pipeline = Pipeline()
    pipeline.valves.rlm_codegen_enabled = True
    pipeline.valves.rlm_primary_planner_enabled = True

    calls = []

    def _fake_rlm(**kwargs):
        calls.append(str(kwargs.get("retry_reason") or ""))
        return "", "", "read", False

    pipeline._select_router_or_shortcut = lambda _q, _p, _h: (  # type: ignore[method-assign]
        None,
        ("result = 999\n", "shortcut_plan"),
        {},
        {},
    )
    pipeline._plan_code_with_rlm_tool = _fake_rlm  # type: ignore[method-assign]
    pipeline._plan_code = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("plan_code should not be called"))  # type: ignore[method-assign]

    out = pipeline._prepare_analysis_code_for_question(
        question="Скільки рядків?",
        profile={"columns": ["A"], "dtypes": {"A": "int64"}},
        has_edit=False,
    )

    assert out["ok"] is True
    assert "result = 999" in str(out.get("analysis_code") or "")
    events = list(out.get("events") or [])
    assert any(e[0] == "codegen_shortcut" and e[1] == {} for e in events)
    assert calls == ["primary_planner"]
