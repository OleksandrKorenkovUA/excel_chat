from pipelines.spreadsheet_analyst_pipeline import _effective_user_query


def test_effective_user_query_uses_user_message():
    messages = [{"role": "assistant", "content": "ok"}]
    q = _effective_user_query("Скільки рядків?", messages)
    assert q == "Скільки рядків?"


def test_effective_user_query_meta_then_real_after_assistant():
    messages = [
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Скільки рядків і стовпців у таблиці?"},
        {
            "role": "user",
            "content": "### Task: Respond to the user query using the provided context",
        },
    ]
    q = _effective_user_query("### Task: Respond to the user query using the provided context", messages)
    assert q == "Скільки рядків і стовпців у таблиці?"


def test_effective_user_query_meta_only_no_user_query():
    messages = [
        {"role": "assistant", "content": "ok"},
        {
            "role": "user",
            "content": "### Task: Respond to the user query using the provided context",
        },
    ]
    q = _effective_user_query("### Task: Respond to the user query using the provided context", messages)
    assert q == ""
