from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union


def _safe_trunc(text: Any, limit: int) -> str:
    s = str(text or "")
    if len(s) <= limit:
        return s
    return s[:limit] + "...(truncated)"


def _safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


@dataclass
class QueryMetadata:
    context_type: str
    context_total_length: int
    context_lengths: List[int]

    @classmethod
    def from_payload(cls, payload: Any) -> "QueryMetadata":
        if isinstance(payload, list):
            chunks = [str(x) for x in payload]
            lengths = [len(x) for x in chunks]
            return cls(context_type="list", context_total_length=sum(lengths), context_lengths=lengths)
        if isinstance(payload, dict):
            s = _safe_json_dumps(payload)
            return cls(context_type="dict", context_total_length=len(s), context_lengths=[len(s)])
        s = str(payload or "")
        return cls(context_type=type(payload).__name__, context_total_length=len(s), context_lengths=[len(s)])


def build_rlm_system_prompt(system_prompt: str, query_metadata: QueryMetadata) -> List[Dict[str, str]]:
    context_lengths = list(query_metadata.context_lengths or [])
    if len(context_lengths) > 100:
        others = len(context_lengths) - 100
        context_lengths_view = f"{context_lengths[:100]}... [{others} others]"
    else:
        context_lengths_view = str(context_lengths)
    metadata_prompt = (
        f"Your context is a {query_metadata.context_type} with {query_metadata.context_total_length} total characters, "
        f"and is broken up into chunks of char lengths: {context_lengths_view}."
    )
    return [
        {"role": "system", "content": str(system_prompt or "")},
        {"role": "assistant", "content": metadata_prompt},
    ]


def build_user_prompt(
    root_prompt: Optional[str] = None,
    iteration: int = 0,
    context_count: int = 1,
    history_count: int = 0,
) -> Dict[str, str]:
    if iteration == 0:
        safeguard = (
            "You have not interacted with the REPL environment yet. "
            "First inspect and solve via code; avoid final answer without execution.\n\n"
        )
        prompt = safeguard + f"User task: {root_prompt or ''}".strip()
    else:
        prompt = (
            "The history below is your previous interaction with the REPL. "
            f"Continue fixing the solution.\n\nUser task: {root_prompt or ''}"
        ).strip()
    if context_count > 1:
        prompt += f"\n\nNote: You have {context_count} contexts."
    if history_count > 0:
        prompt += f"\n\nNote: You have {history_count} prior history turns."
    return {"role": "user", "content": prompt}


@dataclass
class REPLResult:
    status: str
    payload: Dict[str, Any]
    error: str = ""


class NonIsolatedEnv:
    def setup(self) -> None:
        pass

    def load_context(self, context_payload: Any) -> None:  # noqa: ANN401
        pass

    def execute_code(self, code: str) -> REPLResult:
        raise NotImplementedError


class SandboxREPL(NonIsolatedEnv):
    """
    REPL environment adapted from rlm.md semantics for sandbox-backed execution:
    - explicit setup()/load_context()
    - persistent state/history between execute_code() calls
    - executor-backed runtime (sandbox service in this project)
    """

    def __init__(
        self,
        executor: Callable[[str], Dict[str, Any]],
        context_payload: Optional[Dict[str, Any]] = None,
        persistent: bool = True,
        setup_code: str = "",
    ) -> None:
        self._executor = executor
        self.persistent = bool(persistent)
        self.context_payload: Dict[str, Any] = {}
        self.locals: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self._last_payload: Dict[str, Any] = {}
        self.setup()
        if context_payload is not None:
            self.load_context(context_payload)
        if str(setup_code or "").strip():
            self.execute_code(str(setup_code))

    def setup(self) -> None:
        self.locals = {}
        self.history = []
        self._last_payload = {}

    def load_context(self, context_payload: Any) -> None:  # noqa: ANN401
        if isinstance(context_payload, dict):
            self.context_payload = dict(context_payload)
        else:
            self.context_payload = {"context": context_payload}

    def execute_code(self, code: str) -> REPLResult:
        started = time.perf_counter()
        payload = dict(self._executor(code) or {})
        payload.setdefault("execution_time", time.perf_counter() - started)
        if isinstance(payload.get("locals"), dict):
            self.locals = dict(payload.get("locals") or {})
        else:
            payload["locals"] = dict(self.locals)
        if self.context_payload and "context_payload" not in payload:
            payload["context_payload"] = dict(self.context_payload)
        if self.persistent:
            self.history.append({"code": str(code or ""), "status": str(payload.get("status") or ""), "payload": payload})
            payload.setdefault("history_size", len(self.history))
        self._last_payload = dict(payload)
        status = str(payload.get("status") or "")
        error = str(payload.get("error") or payload.get("stderr") or "")
        return REPLResult(status=status, payload=payload, error=error)

    def cleanup(self) -> None:
        self.history.clear()


class LocalREPL(SandboxREPL):
    """
    Backward-compatible alias used by existing tests and call-sites.
    """

    pass


class LMHandler:
    """
    Lightweight LM handler compatible with OpenAI-like chat completions clients.
    """

    def __init__(
        self,
        openai_client: Any,
        model: str,
        host: str = "127.0.0.1",
        port: int = 0,
        temperature: float = 0.0,
    ) -> None:
        self._client = openai_client
        self._model = model
        self.host = host
        self.port = int(port)
        self.temperature = float(temperature)

    def completion(
        self,
        prompt: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        max_tokens: int = 700,
    ) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, dict):
            messages = [prompt]
        else:
            messages = list(prompt or [])
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=int(max_tokens),
        )
        return str((resp.choices[0].message.content or "")).strip()


@dataclass
class RLMIteration:
    turn: int
    user_payload: Dict[str, Any]
    response: str
    analysis_code: str = ""
    validation_error: str = ""
    repl_status: str = ""
    repl_error: str = ""
    repl_payload: Dict[str, Any] = field(default_factory=dict)
    final_answer: Optional[str] = None


@dataclass
class RLMChatCompletion:
    response: str
    analysis_code: str = ""
    run_resp: Optional[Dict[str, Any]] = None
    edit_expected: bool = False
    iterations: List[RLMIteration] = field(default_factory=list)
    last_error: str = ""
    execution_time: float = 0.0


def _payload_output_text(payload: Dict[str, Any]) -> str:
    p = dict(payload or {})
    for key in ("result_text", "stdout", "error", "stderr", "message"):
        value = p.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def find_final_answer(text: str, environment: Optional[NonIsolatedEnv] = None) -> Optional[str]:
    s = str(text or "")
    if not s:
        return None

    final_var_pattern = r"^\s*FINAL_VAR\((.*?)\)\s*$"
    m_var = re.search(final_var_pattern, s, re.MULTILINE | re.DOTALL)
    if m_var:
        variable_name = str(m_var.group(1) or "").strip().strip('"').strip("'")
        if not variable_name or environment is None:
            return None
        try:
            probe = environment.execute_code(f"result = {variable_name}")
            out = _payload_output_text(probe.payload)
            if out:
                return out
        except Exception:
            return None
        return None

    final_pattern = r"^\s*FINAL\((.*)\)\s*$"
    m_final = re.search(final_pattern, s, re.MULTILINE | re.DOTALL)
    if m_final:
        return str(m_final.group(1) or "").strip()
    return None


def format_execution_result(result: REPLResult) -> str:
    payload = dict(result.payload or {})
    parts: List[str] = []
    stdout = str(payload.get("stdout") or "").strip()
    stderr = str(payload.get("stderr") or result.error or "").strip()
    result_text = str(payload.get("result_text") or "").strip()
    if stdout:
        parts.append(stdout)
    if result_text and result_text != stdout:
        parts.append(result_text)
    if stderr:
        parts.append(stderr)
    locals_obj = payload.get("locals")
    if isinstance(locals_obj, dict) and locals_obj:
        visible = [k for k in locals_obj.keys() if not str(k).startswith("_")]
        if visible:
            parts.append(f"REPL variables: {visible}")
    return "\n\n".join(parts) if parts else "No output"


def format_iteration(iteration: RLMIteration, max_character_length: int = 20000) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "assistant", "content": str(iteration.response or "")}] 
    code = str(iteration.analysis_code or "").strip()
    if not code:
        return messages
    repl = REPLResult(
        status=str(iteration.repl_status or ""),
        payload=dict(iteration.repl_payload or {}),
        error=str(iteration.repl_error or ""),
    )
    output = format_execution_result(repl)
    if len(output) > max_character_length:
        output = output[:max_character_length] + f"... + [{len(output) - max_character_length} chars...]"
    messages.append(
        {
            "role": "user",
            "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{output}",
        }
    )
    return messages


class RLMCore:
    """
    Core recursive loop:
    1) ask LLM for code
    2) validate code
    3) optionally execute in REPL env
    4) feed errors back into next turn
    """

    def __init__(
        self,
        *,
        system_prompt: str,
        lm_handler_factory: Callable[[], Any],
        code_extractor: Callable[[str], str],
        code_validator: Callable[[str], Tuple[str, bool, Optional[str]]],
        environment_factory: Optional[Callable[[Union[str, Dict[str, Any]]], NonIsolatedEnv]] = None,
        fallback_lm_handler_factory: Optional[Callable[[], Any]] = None,
        max_iterations: int = 2,
        max_tokens: int = 700,
        fallback_max_tokens: int = 128,
        fallback_enabled: bool = True,
    ) -> None:
        self.system_prompt = str(system_prompt or "")
        self._lm_handler_factory = lm_handler_factory
        self._environment_factory = environment_factory
        self._fallback_lm_handler_factory = fallback_lm_handler_factory or lm_handler_factory
        self._code_extractor = code_extractor
        self._code_validator = code_validator
        self.max_iterations = max(1, int(max_iterations))
        self.max_tokens = max(64, int(max_tokens))
        self.fallback_max_tokens = max(32, int(fallback_max_tokens))
        self.fallback_enabled = bool(fallback_enabled)

    @contextmanager
    def _spawn_completion_context(
        self,
        prompt: Union[str, Dict[str, Any]],
    ) -> Iterator[Tuple[Any, Optional[NonIsolatedEnv]]]:
        lm_handler = self._lm_handler_factory()
        env = self._environment_factory(prompt) if self._environment_factory else None
        try:
            yield lm_handler, env
        finally:
            cleanup = getattr(env, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    pass

    def _fallback_answer(self, message: Union[str, Dict[str, Any]]) -> str:
        if not self.fallback_enabled:
            return ""
        try:
            handler = self._fallback_lm_handler_factory()
            msg = message if isinstance(message, str) else _safe_json_dumps(message)
            return str(handler.completion(msg, max_tokens=self.fallback_max_tokens) or "").strip()
        except Exception:
            return ""

    def _environment_context_count(self, environment: Optional[NonIsolatedEnv]) -> int:
        getter = getattr(environment, "get_context_count", None)
        if callable(getter):
            try:
                n = int(getter())
                return n if n > 0 else 1
            except Exception:
                return 1
        return 1

    def _environment_history_count(self, environment: Optional[NonIsolatedEnv], message_history: List[Dict[str, str]]) -> int:
        getter = getattr(environment, "get_history_count", None)
        if callable(getter):
            try:
                n = int(getter())
                return n if n >= 0 else 0
            except Exception:
                return max(0, len(message_history) - 2)
        return max(0, len(message_history) - 2)

    def _completion_turn(
        self,
        *,
        turn: int,
        prompt: List[Dict[str, str]],
        user_payload: Dict[str, Any],
        lm_handler: Any,
        environment: Optional[NonIsolatedEnv],
    ) -> RLMIteration:
        iteration = RLMIteration(turn=turn, user_payload=dict(user_payload or {}), response="", analysis_code="")
        try:
            response = str(lm_handler.completion(prompt, max_tokens=self.max_tokens) or "").strip()
            iteration.response = response
        except Exception as exc:
            iteration.validation_error = f"lm_completion_error:{type(exc).__name__}"
            return iteration

        try:
            analysis_code = str(self._code_extractor(iteration.response) or "").strip()
        except Exception as exc:
            iteration.validation_error = f"code_extractor_error:{type(exc).__name__}"
            return iteration
        iteration.analysis_code = analysis_code
        iteration.final_answer = find_final_answer(response, environment=environment)
        if iteration.final_answer is not None:
            return iteration

        if not iteration.analysis_code:
            iteration.validation_error = "non_code_output"
            return iteration

        try:
            normalized_code, _edit_expected, validation_error = self._code_validator(iteration.analysis_code)
            normalized_code = str(normalized_code or "").strip()
            iteration.analysis_code = normalized_code or iteration.analysis_code
        except Exception as exc:
            iteration.validation_error = f"code_validator_error:{type(exc).__name__}"
            return iteration
        if validation_error:
            iteration.validation_error = str(validation_error)
            return iteration

        if environment is None:
            return iteration

        try:
            repl_result = environment.execute_code(iteration.analysis_code)
            iteration.repl_status = str(repl_result.status or "")
            iteration.repl_error = str(repl_result.error or "")
            iteration.repl_payload = dict(repl_result.payload or {})
        except Exception as exc:
            iteration.repl_status = "error"
            iteration.repl_error = f"repl_execute_error:{type(exc).__name__}:{exc}"
            iteration.repl_payload = {"status": "error", "error": str(exc)}
        return iteration

    def _default_answer(self, message_history: List[Dict[str, str]], lm_handler: Any) -> str:
        try:
            prompt = list(message_history or [])
            prompt.append(
                {
                    "role": "assistant",
                    "content": "Please provide a final answer to the user's question based on the information provided.",
                }
            )
            return str(lm_handler.completion(prompt, max_tokens=self.fallback_max_tokens) or "").strip()
        except Exception:
            return ""

    def completion(
        self,
        prompt: Union[str, Dict[str, Any]],
        root_prompt: Optional[str] = None,
    ) -> RLMChatCompletion:
        started = time.perf_counter()
        payload = {"question": str(prompt or "")} if isinstance(prompt, str) else dict(prompt or {})
        question = str(payload.get("question") or payload.get("prompt") or "").strip()
        previous_code = str(payload.get("previous_code") or payload.get("previous_analysis_code") or "").strip()
        previous_error = str(
            payload.get("previous_error")
            or payload.get("runtime_error")
            or payload.get("retry_reason")
            or ""
        ).strip()

        query_metadata = QueryMetadata.from_payload(payload.get("schema") or payload.get("context") or question)
        message_history: List[Dict[str, str]] = build_rlm_system_prompt(self.system_prompt, query_metadata)
        iterations: List[RLMIteration] = []
        with self._spawn_completion_context(payload) as (lm_handler, environment):
            for turn in range(1, self.max_iterations + 1):
                user_payload = dict(payload)
                user_payload["question"] = question
                user_payload["turn"] = turn
                if root_prompt:
                    user_payload["root_prompt"] = root_prompt
                if previous_code:
                    user_payload["previous_code"] = _safe_trunc(previous_code, 3000)
                if previous_error:
                    user_payload["previous_error"] = _safe_trunc(previous_error, 1500)

                user_prompt = build_user_prompt(
                    root_prompt=root_prompt or question,
                    iteration=(turn - 1),
                    context_count=self._environment_context_count(environment),
                    history_count=self._environment_history_count(environment, message_history),
                )
                user_prompt["content"] += "\n\nRLM payload:\n" + _safe_json_dumps(user_payload)
                current_prompt = list(message_history) + [user_prompt]
                it = self._completion_turn(
                    turn=turn,
                    prompt=current_prompt,
                    user_payload=user_payload,
                    lm_handler=lm_handler,
                    environment=environment,
                )
                iterations.append(it)

                if it.final_answer is not None:
                    return RLMChatCompletion(
                        response=str(it.final_answer),
                        analysis_code=str(it.analysis_code or ""),
                        run_resp=dict(it.repl_payload or {}) if it.repl_status == "ok" else None,
                        edit_expected=False,
                        iterations=iterations,
                        last_error="",
                        execution_time=(time.perf_counter() - started),
                    )
                if it.validation_error:
                    previous_code = str(it.analysis_code or previous_code)
                    previous_error = str(it.validation_error)
                    message_history.extend(format_iteration(it))
                    continue
                if environment is None:
                    return RLMChatCompletion(
                        response="planned",
                        analysis_code=str(it.analysis_code or ""),
                        run_resp=None,
                        edit_expected=False,
                        iterations=iterations,
                        last_error="",
                        execution_time=(time.perf_counter() - started),
                    )
                if it.repl_status == "ok":
                    return RLMChatCompletion(
                        response="ok",
                        analysis_code=str(it.analysis_code or ""),
                        run_resp=dict(it.repl_payload or {}),
                        edit_expected=False,
                        iterations=iterations,
                        last_error="",
                        execution_time=(time.perf_counter() - started),
                    )

                previous_code = str(it.analysis_code or previous_code)
                previous_error = str(it.repl_error or _payload_output_text(it.repl_payload) or "runtime_error")
                message_history.extend(format_iteration(it))

            default_answer = self._default_answer(message_history, lm_handler)

        fallback = default_answer or self._fallback_answer(
            {
                "question": question,
                "last_error": previous_error,
                "previous_code": _safe_trunc(previous_code, 3000),
            }
        )
        return RLMChatCompletion(
            response=fallback,
            analysis_code=previous_code,
            run_resp=None,
            edit_expected=False,
            iterations=iterations,
            last_error=previous_error,
            execution_time=(time.perf_counter() - started),
        )
