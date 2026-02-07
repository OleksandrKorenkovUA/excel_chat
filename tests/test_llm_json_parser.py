import pytest

from pipelines.spreadsheet_analyst_pipeline import _parse_json_dict_from_llm


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
