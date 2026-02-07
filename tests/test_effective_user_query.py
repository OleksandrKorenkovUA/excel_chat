from pipelines.spreadsheet_analyst_pipeline import (
    _effective_user_query,
    _is_search_query_meta_task,
    _parse_row_index,
)


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
    assert q == ""


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


def test_effective_user_query_ignores_trailing_meta_without_user_query():
    messages = [
        {"role": "user", "content": "Скільки брендів у таблиці?"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "### Task: Analyze the chat history to determine the necessity of generating search queries"},
    ]
    q = _effective_user_query("", messages)
    assert q == ""


def test_is_search_query_meta_task_true() -> None:
    text = (
        "### Task:\n"
        "Analyze the chat history to determine the necessity of generating search queries.\n"
        "Return: { \"queries\": [] }"
    )
    assert _is_search_query_meta_task(text) is True


def test_is_search_query_meta_task_false() -> None:
    assert _is_search_query_meta_task("Скільки рядків у таблиці?") is False


def test_effective_user_query_extracts_embedded_user_query() -> None:
    text = (
        "### Task:\n"
        "Respond to the user query using the provided context.\n"
        "<user_query>Яка ціна в рядку 77?</user_query>\n"
    )
    q = _effective_user_query(text, [{"role": "user", "content": text}])
    assert q == "Яка ціна в рядку 77?"


def test_effective_user_query_extracts_user_query_heading_without_colon() -> None:
    text = (
        "### Task:\n"
        "Respond to the user query using the provided context.\n"
        "### User Query\n"
        "Яка ціна в рядку 77?\n"
    )
    q = _effective_user_query(text, [{"role": "user", "content": text}])
    assert q == "Яка ціна в рядку 77?"


def test_effective_user_query_extracts_from_chat_history_block() -> None:
    text = (
        "### Task:\n"
        "Analyze the chat history to determine the necessity of generating search queries.\n"
        "<chat_history>\n"
        "assistant: Вітаю\n"
        "user: Яка ціна в рядку 77?\n"
        "</chat_history>\n"
    )
    q = _effective_user_query(text, [{"role": "user", "content": text}])
    assert q == "Яка ціна в рядку 77?"


def test_parse_row_index_supports_riadku_form() -> None:
    assert _parse_row_index("Яка ціна в рядку 77") == 77
