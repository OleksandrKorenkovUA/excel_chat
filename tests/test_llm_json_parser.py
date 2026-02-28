import pytest

from pipelines.spreadsheet_analyst_pipeline import (
    _extract_analysis_code_from_llm,
    _parse_json_dict_from_llm,
    find_code_blocks,
)


def test_parse_llm_json_plain_dict() -> None:
    text = '{"analysis_code":"result = 1","short_plan":"ok"}'
    parsed = _parse_json_dict_from_llm(text)
    assert parsed["analysis_code"] == "result = 1"


def test_parse_llm_json_fenced_block() -> None:
    text = "```json\n{\"analysis_code\":\"result = 2\",\"short_plan\":\"ok\"}\n```"
    parsed = _parse_json_dict_from_llm(text)
    assert parsed["analysis_code"] == "result = 2"


def test_parse_llm_json_with_noise() -> None:
    text = "Here is JSON:\n{\"analysis_code\":\"result = 3\",\"short_plan\":\"ok\"}\nThanks"
    parsed = _parse_json_dict_from_llm(text)
    assert parsed["analysis_code"] == "result = 3"


def test_parse_llm_json_rejects_non_object() -> None:
    with pytest.raises(ValueError):
        _parse_json_dict_from_llm("[1, 2, 3]")


def test_parse_llm_json_with_think_and_invalid_braces_before_valid_object() -> None:
    text = (
        "<think>\n"
        "I will answer with JSON. {not_json_here}\n"
        "</think>\n"
        '{"analysis_code":"result = 4","short_plan":"ok"}'
    )
    parsed = _parse_json_dict_from_llm(text)
    assert parsed["analysis_code"] == "result = 4"


def test_parse_llm_json_ignores_thinking_fence_with_broken_braces() -> None:
    text = (
        "```thinking\n"
        "{not_json_here\n"
        "```\n"
        '{"analysis_code":"result = 5","short_plan":"ok"}'
    )
    parsed = _parse_json_dict_from_llm(text)
    assert parsed["analysis_code"] == "result = 5"


def test_extract_analysis_code_from_llm_think_and_python_fence() -> None:
    text = "<think>long reasoning</think>\n```python\nresult = 42\n```"
    code = _extract_analysis_code_from_llm(text)
    assert code == "result = 42"


def test_extract_analysis_code_from_llm_prefers_json_analysis_code() -> None:
    text = '{"analysis_code":"result = int(df.shape[0])","short_plan":"ok"}'
    code = _extract_analysis_code_from_llm(text)
    assert code == "result = int(df.shape[0])"


def test_extract_analysis_code_from_llm_rejects_non_code_reasoning() -> None:
    text = "Thinking Process:\nI should analyze the query and then return JSON."
    code = _extract_analysis_code_from_llm(text)
    assert code == ""


def test_find_code_blocks_extracts_python_fence_inside_think() -> None:
    text = "<think>\n```python\nresult = 7\n```\n</think>"
    blocks = find_code_blocks(text)
    assert blocks
    assert blocks[0] == "result = 7"


def test_extract_analysis_code_from_llm_prefers_fenced_code_inside_think() -> None:
    text = (
        "<think>\n"
        "Reasoning...\n"
        "```python\n"
        "result = int(df.shape[0])\n"
        "```\n"
        "</think>\n"
        "Trailing notes"
    )
    code = _extract_analysis_code_from_llm(text)
    assert code == "result = int(df.shape[0])"
