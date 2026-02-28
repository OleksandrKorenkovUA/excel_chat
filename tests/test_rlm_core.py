from pipelines.lib.rlm_core import (
    LocalREPL,
    QueryMetadata,
    RLMCore,
    build_rlm_system_prompt,
    find_final_answer,
)


class _FakeLMHandler:
    def __init__(self, responses):
        self._responses = list(responses)

    def completion(self, prompt, max_tokens=700):  # noqa: ANN001
        if self._responses:
            return self._responses.pop(0)
        return ""


def test_rlm_core_completion_without_env_recovers_from_non_code() -> None:
    handler = _FakeLMHandler(["Thinking only", "result = 5"])
    core = RLMCore(
        system_prompt="sys",
        lm_handler_factory=lambda: handler,
        code_extractor=lambda t: t if "result =" in t else "",
        code_validator=lambda c: (c, False, None),
        max_iterations=2,
        max_tokens=64,
        fallback_enabled=False,
    )
    out = core.completion({"question": "q"})
    assert out.analysis_code.strip() == "result = 5"
    assert out.run_resp is None
    assert len(out.iterations) == 2
    assert out.iterations[0].validation_error == "non_code_output"


def test_rlm_core_completion_with_localrepl_runtime_retry() -> None:
    handler = _FakeLMHandler(["result = 1", "result = 2"])
    seen = {"n": 0}

    def _executor(code: str):
        seen["n"] += 1
        if seen["n"] == 1:
            return {"status": "err", "error": "boom"}
        return {"status": "ok", "result_text": "2"}

    core = RLMCore(
        system_prompt="sys",
        lm_handler_factory=lambda: handler,
        environment_factory=lambda _prompt: LocalREPL(_executor),
        code_extractor=lambda t: t if "result =" in t else "",
        code_validator=lambda c: (c, False, None),
        max_iterations=2,
        max_tokens=64,
        fallback_enabled=False,
    )
    out = core.completion({"question": "q"})
    assert out.analysis_code.strip() == "result = 2"
    assert isinstance(out.run_resp, dict)
    assert out.run_resp.get("status") == "ok"
    assert seen["n"] == 2


def test_rlm_core_fallback_answer_used_when_iterations_exhausted() -> None:
    main = _FakeLMHandler(["not code"])
    fallback = _FakeLMHandler(["fallback"])
    core = RLMCore(
        system_prompt="sys",
        lm_handler_factory=lambda: main,
        fallback_lm_handler_factory=lambda: fallback,
        code_extractor=lambda _t: "",
        code_validator=lambda c: (c, False, None),
        max_iterations=1,
        max_tokens=64,
        fallback_enabled=True,
    )
    out = core.completion({"question": "q"})
    assert out.response == "fallback"
    assert out.last_error == "non_code_output"


def test_find_final_answer_from_final_marker() -> None:
    text = "some work\nFINAL(ready)"
    assert find_final_answer(text) == "ready"


def test_build_rlm_system_prompt_contains_context_metadata() -> None:
    meta = QueryMetadata(context_type="str", context_total_length=12, context_lengths=[12])
    out = build_rlm_system_prompt("SYS", meta)
    assert isinstance(out, list)
    assert out[0]["role"] == "system"
    assert out[0]["content"] == "SYS"
    assert "12" in out[1]["content"]


def test_find_final_answer_from_final_var_with_environment() -> None:
    env = LocalREPL(lambda _code: {"status": "ok", "result_text": "value-from-var"})
    text = "prep\nFINAL_VAR(buffer)"
    assert find_final_answer(text, env) == "value-from-var"


def test_query_metadata_from_payload_handles_unserializable_dict_values() -> None:
    meta = QueryMetadata.from_payload({"bad": {1, 2, 3}})
    assert meta.context_type == "dict"
    assert meta.context_total_length > 0
    assert meta.context_lengths and meta.context_lengths[0] == meta.context_total_length


def test_rlm_core_retries_when_code_validator_raises_exception() -> None:
    handler = _FakeLMHandler(["result = 1", "result = 2"])
    seen = {"n": 0}

    def _validator(code: str):
        seen["n"] += 1
        if seen["n"] == 1:
            raise ValueError("validator boom")
        return code, False, None

    core = RLMCore(
        system_prompt="sys",
        lm_handler_factory=lambda: handler,
        code_extractor=lambda t: t if "result =" in t else "",
        code_validator=_validator,
        max_iterations=2,
        max_tokens=64,
        fallback_enabled=False,
    )
    out = core.completion({"question": "q"})
    assert len(out.iterations) == 2
    assert out.iterations[0].validation_error.startswith("code_validator_error:")
    assert out.analysis_code.strip() == "result = 2"


def test_rlm_core_handles_repl_execute_exception_and_uses_fallback() -> None:
    handler = _FakeLMHandler(["result = 1"])
    fallback = _FakeLMHandler(["fallback-after-repl-error"])

    core = RLMCore(
        system_prompt="sys",
        lm_handler_factory=lambda: handler,
        fallback_lm_handler_factory=lambda: fallback,
        environment_factory=lambda _prompt: LocalREPL(lambda _code: (_ for _ in ()).throw(RuntimeError("boom"))),
        code_extractor=lambda t: t if "result =" in t else "",
        code_validator=lambda c: (c, False, None),
        max_iterations=1,
        max_tokens=64,
        fallback_enabled=True,
    )
    out = core.completion({"question": "q"})
    assert out.response == "fallback-after-repl-error"
    assert out.last_error.startswith("repl_execute_error:RuntimeError:")


def test_rlm_core_completion_allows_non_serializable_payload_values() -> None:
    handler = _FakeLMHandler(["result = 7"])
    core = RLMCore(
        system_prompt="sys",
        lm_handler_factory=lambda: handler,
        code_extractor=lambda t: t if "result =" in t else "",
        code_validator=lambda c: (c, False, None),
        max_iterations=1,
        max_tokens=64,
        fallback_enabled=False,
    )
    out = core.completion({"question": "q", "schema": {"bad": {1, 2}}})
    assert out.analysis_code.strip() == "result = 7"
