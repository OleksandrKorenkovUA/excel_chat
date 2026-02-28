import json
from pathlib import Path

from pipelines.spreadsheet_analyst_pipeline import Pipeline


def test_success_learning_promotes_after_min_support(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    pending_path = tmp_path / "pending.jsonl"
    index_path = tmp_path / "index.faiss"
    meta_path = tmp_path / "meta.json"

    catalog = {
        "catalog_version": "0.2.0",
        "language": "uk",
        "intents": [
            {
                "id": "filter_multi_conditions",
                "name": "multi",
                "description": "d",
                "examples": [],
                "slots": {"conditions": {"type": "json", "required": True}},
                "plan": [{"op": "filter_multi_conditions", "out": "result"}],
            }
        ],
    }
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pipe = Pipeline()
    pipe.valves.shortcut_learning_enabled = True
    pipe.valves.shortcut_learning_pending_path = str(pending_path)
    pipe.valves.shortcut_learning_min_support = 2
    pipe.valves.shortcut_learning_consistency_ratio = 0.8
    pipe.valves.shortcut_learning_promote_every = 1
    pipe.valves.shortcut_catalog_path = str(catalog_path)
    pipe.valves.shortcut_index_path = str(index_path)
    pipe.valves.shortcut_meta_path = str(meta_path)

    q = "Знайди всі записи з двома умовами"
    code = "result = df.head(2)\n"
    plan = "retrieval_intent:filter_multi_conditions"

    pipe._maybe_record_success_learning(
        question=q,
        plan=plan,
        analysis_code=code,
        run_status="ok",
        edit_expected=False,
        result_text="ok",
        result_meta={},
    )
    pipe._maybe_record_success_learning(
        question=q,
        plan=plan,
        analysis_code=code,
        run_status="ok",
        edit_expected=False,
        result_text="ok",
        result_meta={},
    )

    updated = json.loads(catalog_path.read_text(encoding="utf-8"))
    intent = updated["intents"][0]
    assert q in intent.get("examples", [])
    learned_cases = intent.get("learned_cases") or []
    assert len(learned_cases) == 1
    assert learned_cases[0].get("query") == q

    pending_lines = [ln for ln in pending_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert pending_lines == []


def test_success_learning_llm_judge_blocks_low_score(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    pending_path = tmp_path / "pending.jsonl"
    index_path = tmp_path / "index.faiss"
    meta_path = tmp_path / "meta.json"

    catalog = {
        "catalog_version": "0.2.0",
        "language": "uk",
        "intents": [
            {
                "id": "filter_multi_conditions",
                "name": "multi",
                "description": "d",
                "examples": [],
                "slots": {"conditions": {"type": "json", "required": True}},
                "plan": [{"op": "filter_multi_conditions", "out": "result"}],
            }
        ],
    }
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pipe = Pipeline()
    pipe.valves.shortcut_learning_enabled = True
    pipe.valves.shortcut_learning_pending_path = str(pending_path)
    pipe.valves.shortcut_learning_min_support = 2
    pipe.valves.shortcut_learning_consistency_ratio = 0.8
    pipe.valves.shortcut_learning_promote_every = 1
    pipe.valves.shortcut_catalog_path = str(catalog_path)
    pipe.valves.shortcut_index_path = str(index_path)
    pipe.valves.shortcut_meta_path = str(meta_path)
    pipe.valves.shortcut_learning_llm_judge_enabled = True
    pipe.valves.shortcut_learning_llm_judge_min_score = 0.90
    pipe.valves.shortcut_learning_llm_judge_fail_open = False
    pipe._llm_judge_learning_candidate = lambda **kwargs: (0.62, "low_alignment", "ok")

    pipe._maybe_record_success_learning(
        question="Знайди всі записи з двома умовами",
        plan="retrieval_intent:filter_multi_conditions",
        analysis_code="result = df.head(2)\n",
        run_status="ok",
        edit_expected=False,
        result_text="ok",
        result_meta={},
    )

    if pending_path.exists():
        pending_lines = [ln for ln in pending_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert pending_lines == []
    else:
        assert True


def test_success_learning_llm_judge_promote_threshold(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    pending_path = tmp_path / "pending.jsonl"
    index_path = tmp_path / "index.faiss"
    meta_path = tmp_path / "meta.json"

    catalog = {
        "catalog_version": "0.2.0",
        "language": "uk",
        "intents": [
            {
                "id": "filter_multi_conditions",
                "name": "multi",
                "description": "d",
                "examples": [],
                "slots": {"conditions": {"type": "json", "required": True}},
                "plan": [{"op": "filter_multi_conditions", "out": "result"}],
            }
        ],
    }
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pipe = Pipeline()
    pipe.valves.shortcut_learning_enabled = True
    pipe.valves.shortcut_learning_pending_path = str(pending_path)
    pipe.valves.shortcut_learning_min_support = 2
    pipe.valves.shortcut_learning_consistency_ratio = 0.8
    pipe.valves.shortcut_learning_promote_every = 1
    pipe.valves.shortcut_catalog_path = str(catalog_path)
    pipe.valves.shortcut_index_path = str(index_path)
    pipe.valves.shortcut_meta_path = str(meta_path)
    pipe.valves.shortcut_learning_llm_judge_enabled = True
    pipe.valves.shortcut_learning_llm_judge_min_score = 0.80
    pipe.valves.shortcut_learning_llm_judge_promote_min_score = 0.90
    pipe.valves.shortcut_learning_llm_judge_fail_open = False
    pipe._llm_judge_learning_candidate = lambda **kwargs: (0.84, "acceptable_but_low_for_promote", "ok")

    q = "Знайди всі записи з двома умовами"
    for _ in range(2):
        pipe._maybe_record_success_learning(
            question=q,
            plan="retrieval_intent:filter_multi_conditions",
            analysis_code="result = df.head(2)\n",
            run_status="ok",
            edit_expected=False,
            result_text="ok",
            result_meta={},
        )

    updated = json.loads(catalog_path.read_text(encoding="utf-8"))
    intent = updated["intents"][0]
    assert q not in intent.get("examples", [])
    learned_cases = intent.get("learned_cases") or []
    assert learned_cases == []

    pending_lines = [ln for ln in pending_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(pending_lines) == 2
