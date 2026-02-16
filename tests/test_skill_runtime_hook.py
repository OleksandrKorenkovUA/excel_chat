from pipelines.spreadsheet_analyst_pipeline import Pipeline


def _write_file(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_plan_codegen_injects_spreadsheet_skill_prompt(tmp_path) -> None:
    skill_dir = tmp_path / "skills" / "spreadsheet-guardrails"
    _write_file(
        skill_dir / "SKILL.md",
        "---\nname: spreadsheet-guardrails\ndescription: test\n---\n\n# Rules\nNever guess columns.\n",
    )
    _write_file(skill_dir / "references" / "column-matching.md", "Column matching rules.")
    _write_file(skill_dir / "references" / "table-mutation-playbooks.md", "Mutation playbook.")
    _write_file(skill_dir / "references" / "forbidden-code-patterns.md", "Forbidden patterns.")

    pipeline = Pipeline()
    pipeline.valves.spreadsheet_skill_dir = str(skill_dir)
    pipeline.valves.spreadsheet_skill_runtime_enabled = True
    pipeline.valves.spreadsheet_skill_force_on_all_queries = True

    captured = {}

    def fake_llm_json(system: str, user: str) -> dict:
        captured["system"] = system
        return {"analysis_code": "result = 1", "short_plan": "ok", "op": "read", "commit_df": False}

    pipeline._llm_json = fake_llm_json  # type: ignore[method-assign]

    _, _, _, _ = pipeline._plan_code("Порахуй рядки", {"columns": ["A"], "dtypes": {"A": "int64"}})

    system = captured.get("system", "")
    assert "RUNTIME SKILL HOOK: spreadsheet-guardrails" in system
    assert "[SKILL.md]" in system
    assert "[references/table-mutation-playbooks.md]" in system


def test_column_picker_uses_column_focused_skill_context(tmp_path) -> None:
    skill_dir = tmp_path / "skills" / "spreadsheet-guardrails"
    _write_file(skill_dir / "SKILL.md", "---\nname: s\ndescription: d\n---\n\n# Base\n")
    _write_file(skill_dir / "references" / "column-matching.md", "Column matching rules.")
    _write_file(skill_dir / "references" / "table-mutation-playbooks.md", "Mutation playbook.")
    _write_file(skill_dir / "references" / "forbidden-code-patterns.md", "Forbidden patterns.")

    pipeline = Pipeline()
    pipeline.valves.spreadsheet_skill_dir = str(skill_dir)
    pipeline.valves.spreadsheet_skill_runtime_enabled = True
    pipeline.valves.spreadsheet_skill_force_on_all_queries = True

    captured = {}

    def fake_llm_json(system: str, user: str) -> dict:
        captured["system"] = system
        return {"column": "A"}

    pipeline._llm_json = fake_llm_json  # type: ignore[method-assign]

    col = pipeline._llm_pick_column_for_shortcut("Знайди колонку ціни", {"columns": ["A"], "dtypes": {"A": "int64"}})

    system = captured.get("system", "")
    assert col == "A"
    assert "RUNTIME SKILL HOOK: spreadsheet-guardrails" in system
    assert "[references/column-matching.md]" in system
    assert "[references/table-mutation-playbooks.md]" not in system
