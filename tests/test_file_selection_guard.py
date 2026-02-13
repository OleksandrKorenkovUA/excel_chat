from pipelines.spreadsheet_analyst_pipeline import _pick_file_ref, _resolve_active_file_ref


def test_pick_file_ref_ignores_stale_history_messages() -> None:
    messages = [
        {"role": "user", "files": [{"id": "old-file"}]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Скільки рядків?"},
    ]
    file_id, file_obj = _pick_file_ref({}, messages)
    assert file_id is None
    assert file_obj is None


def test_resolve_active_file_prefers_session_over_history() -> None:
    messages = [
        {"role": "user", "files": [{"id": "old-file"}]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Скільки рядків?"},
    ]
    file_id, file_obj, source, ignored = _resolve_active_file_ref({}, messages, {"file_id": "session-file"})
    assert file_id == "session-file"
    assert file_obj is None
    assert source == "session"
    assert ignored == "old-file"


def test_resolve_active_file_uses_explicit_current_turn_file() -> None:
    messages = [
        {"role": "user", "files": [{"id": "old-file"}]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "attachments": [{"id": "new-file"}], "content": "Оновив файл"},
    ]
    file_id, file_obj, source, ignored = _resolve_active_file_ref({}, messages, {"file_id": "session-file"})
    assert file_id == "new-file"
    assert isinstance(file_obj, dict) and file_obj.get("id") == "new-file"
    assert source == "explicit"
    assert ignored is None


def test_resolve_active_file_can_fallback_to_history_without_session() -> None:
    messages = [
        {"role": "user", "files": [{"id": "old-file"}]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Скільки рядків?"},
    ]
    file_id, file_obj, source, ignored = _resolve_active_file_ref({}, messages, None)
    assert file_id == "old-file"
    assert isinstance(file_obj, dict) and file_obj.get("id") == "old-file"
    assert source == "history"
    assert ignored is None

