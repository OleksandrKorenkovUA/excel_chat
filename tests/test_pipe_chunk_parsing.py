from pipelines.pipe.pipe import (
    _choice_delta_text,
    _choice_message_text,
    _extract_status_from_payload,
    _extract_text_from_payload,
    _looks_like_status_only_text,
)


def test_choice_delta_text_supports_list_content() -> None:
    choice = {"delta": {"content": [{"type": "text", "text": "Привіт"}]}}
    assert _choice_delta_text(choice) == "Привіт"


def test_choice_message_text_supports_text_fallback() -> None:
    choice = {"message": {"text": "Готово"}}
    assert _choice_message_text(choice) == "Готово"


def test_choice_message_text_supports_direct_text_field() -> None:
    choice = {"text": "У таблиці 100 рядків і 10 стовпців."}
    assert _choice_message_text(choice) == "У таблиці 100 рядків і 10 стовпців."


def test_extract_status_from_payload_detects_status_type() -> None:
    payload = {"type": "status", "data": {"description": "Готово. Відповідь сформована (статус: ok)."}}
    assert _extract_status_from_payload(payload) == "Готово. Відповідь сформована (статус: ok)."


def test_extract_text_from_payload_ignores_status_payload() -> None:
    payload = {"type": "status", "data": {"description": "Готово. Відповідь сформована (статус: ok)."}}
    assert _extract_text_from_payload(payload) == ""


def test_extract_text_from_payload_reads_message_event_shape() -> None:
    payload = {"type": "message", "data": {"content": "У таблиці 100 рядків і 10 стовпців."}}
    assert _extract_text_from_payload(payload) == "У таблиці 100 рядків і 10 стовпців."


def test_extract_text_from_payload_reads_nested_choices() -> None:
    payload = {"event": "message", "data": {"choices": [{"delta": {"content": "Готово"}}]}}
    assert _extract_text_from_payload(payload) == "Готово"


def test_looks_like_status_only_text_true_for_status_lines() -> None:
    text = "\n".join(
        [
            "Завантажую файл: test2020.xlsx",
            "Виконую аналіз у sandbox.",
            "Готово. Відповідь сформована (статус: ok).",
        ]
    )
    assert _looks_like_status_only_text(text) is True


def test_looks_like_status_only_text_false_for_real_answer() -> None:
    assert _looks_like_status_only_text("У таблиці 100 рядків і 10 стовпців.") is False
