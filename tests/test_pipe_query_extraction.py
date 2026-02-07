from pipelines.pipe.pipe import _extract_user_query, _last_user_text


def test_extract_user_query_from_user_query_heading_without_colon() -> None:
    text = (
        "### Task:\n"
        "Respond to the user query using the provided context.\n"
        "### User Query\n"
        "Скільки рядків і стовпців у таблиці?\n"
    )
    assert _extract_user_query(text) == "Скільки рядків і стовпців у таблиці?"


def test_last_user_text_skips_trailing_meta_without_embedded_query() -> None:
    messages = [
        {"role": "user", "content": "Покажи кількість рядків."},
        {"role": "assistant", "content": "ok"},
        {
            "role": "user",
            "content": "### Task: Respond to the user query using the provided context",
        },
    ]
    assert _last_user_text(messages) == ""


def test_last_user_text_extracts_from_chat_history_block() -> None:
    text = (
        "### Task:\n"
        "Analyze the chat history to determine the necessity of generating search queries.\n"
        "<chat_history>\n"
        "assistant: Привіт\n"
        "user: Яка ціна в рядку 77?\n"
        "</chat_history>\n"
    )
    messages = [{"role": "user", "content": text}]
    assert _last_user_text(messages) == "Яка ціна в рядку 77?"


def test_extract_user_query_returns_empty_for_meta_without_user_query() -> None:
    text = "### Task: Respond to the user query using the provided context"
    assert _extract_user_query(text) == ""
