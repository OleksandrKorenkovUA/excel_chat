import ast
import asyncio
import base64
import contextlib
import contextvars
import inspect
import hashlib
import json
import logging
import os
import re
import textwrap
import threading
import time
import traceback
import uuid
from typing import Any, ClassVar, Dict, Generator, Iterator, List, Optional, Tuple, Union
from pipelines.shortcut_router.shortcut_router import ShortcutRouter, ShortcutRouterConfig
from pipelines.lib.pipeline_prompts import (
    DEFAULT_FINAL_ANSWER_SYSTEM,
    DEFAULT_FINAL_REWRITE_SYSTEM,
    DEFAULT_PLAN_CODE_SYSTEM,
    DEFAULT_RLM_CORE_REPL_SYSTEM,
    DEFAULT_RLM_CODEGEN_SYSTEM,
    META_TASK_HINTS,
    SEARCH_QUERY_META_HINTS,
    _SPREADSHEET_SKILL_PROMPT_MARKER,
)
from pipelines.lib.query_signals import (
    COUNT_CONTEXT_RE as _COUNT_CONTEXT_RE,
    COUNT_NUMBER_OF_RE as _COUNT_NUMBER_OF_RE,
    COUNT_QTY_RE as _COUNT_QTY_RE,
    COUNT_WORD_RE as _COUNT_WORD_RE,
    METRIC_CONTEXT_RE as _METRIC_CONTEXT_RE,
    METRIC_PATTERNS as _METRIC_PATTERNS,
    has_availability_filter_cue,
    has_explicit_subset_filter_words,
    has_grouping_cue,
    has_router_entity_token,
    has_router_filter_context_cue,
    has_router_metric_cue,
)
from pipelines.lib.route_trace import (
    RouteTracer,
    current_route_tracer,
    reset_active_route_tracer,
    set_active_route_tracer,
)
from pipelines.lib.rlm_core import LMHandler, RLMCore, SandboxREPL
import requests
from openai import OpenAI
from pydantic import BaseModel, Field


PIPELINES_DIR = os.path.dirname(__file__)
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(PIPELINES_DIR, os.pardir))
DEFAULT_SPREADSHEET_SKILL_DIR = os.path.join(PROJECT_ROOT_DIR, "skills", "spreadsheet-guardrails")
_SPREADSHEET_SKILL_FILES = (
    "SKILL.md",
    os.path.join("references", "column-matching.md"),
    os.path.join("references", "table-mutation-playbooks.md"),
    os.path.join("references", "forbidden-code-patterns.md"),
)

DEF_TIMEOUT_S = int(os.getenv("PIPELINE_HTTP_TIMEOUT_S", "120"))
_LOCAL_PROMPTS = os.path.join(os.path.dirname(__file__), "prompts.txt")
PROMPTS_PATH = _LOCAL_PROMPTS if os.path.exists(_LOCAL_PROMPTS) else os.path.join(PIPELINES_DIR, "prompts.txt")

SHORTCUT_COL_PLACEHOLDER = "_SHORTCUT_COL_"
GROUP_COL_PLACEHOLDER = "__GROUP_COL__"
SUM_COL_PLACEHOLDER = "__SUM_COL__"
AGG_COL_PLACEHOLDER = "__AGG_COL__"
TOP_N_PLACEHOLDER = "__TOP_N__"
LOOKUP_ALLOWED_FILTER_OPS = {"eq", "ne", "gt", "ge", "lt", "le", "contains", "startswith", "endswith"}
LOOKUP_ALLOWED_AGGREGATIONS = {"none", "count", "sum", "mean", "min", "max", "median"}

_REQUEST_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar("pipeline_request_id", default="-")
_TRACE_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar("pipeline_trace_id", default="-")
_LLM_CALL_STATS_CTX: contextvars.ContextVar[Optional[List[Dict[str, Any]]]] = contextvars.ContextVar(
    "pipeline_llm_call_stats", default=None
)


class _RequestTraceLoggingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "_request_trace_injected", False):
            return True
        request_id = (_REQUEST_ID_CTX.get() or "-").strip() or "-"
        trace_id = (_TRACE_ID_CTX.get() or "-").strip() or "-"
        record.msg = f"request_id={request_id} trace_id={trace_id} {record.msg}"
        record._request_trace_injected = True
        return True


def _extract_request_trace_ids(body: Optional[dict]) -> Tuple[str, str]:
    body = body or {}
    headers_raw = body.get("headers")
    headers: Dict[str, str] = {}
    if isinstance(headers_raw, dict):
        for k, v in headers_raw.items():
            if v is None:
                continue
            headers[str(k).lower()] = str(v).strip()

    request_id = str(
        body.get("request_id")
        or body.get("requestId")
        or body.get("id")
        or headers.get("x-request-id")
        or headers.get("x-requestid")
        or ""
    ).strip()
    trace_id = str(
        body.get("trace_id")
        or body.get("traceId")
        or headers.get("x-trace-id")
        or headers.get("traceparent")
        or headers.get("x-b3-traceid")
        or body.get("conversation_id")
        or ""
    ).strip()

    if not trace_id:
        trace_id = uuid.uuid4().hex[:16]
    if not request_id:
        request_id = trace_id
    return request_id[:96], trace_id[:96]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _normalize_optional_top_n(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        n = int(value)
    except Exception:
        return None
    return n if n > 0 else None


def _read_prompts(path: str) -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return {}
    prompts: Dict[str, str] = {}
    for match in re.finditer(r"\[(?P<name>[a-z_]+)\]\s*(?P<body>.*?)\s*\[/\1\]", text, re.S):
        prompts[match.group("name")] = match.group("body").strip()
    return prompts


def _is_meta_task_text(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    if "<user_query>" in s.lower() or "<chat_history>" in s.lower():
        return True
    if s.startswith("### Task:"):
        return True
    lower = s.lower()
    if lower.startswith("analyze the chat history"):
        return True
    if lower.startswith("respond to the user query using the provided context"):
        return True
    for hint in META_TASK_HINTS:
        if hint.lower() in lower:
            return True
    return False


def _is_search_query_meta_task(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    if not _is_meta_task_text(s):
        return False
    lower = s.lower()
    return any(h in lower for h in SEARCH_QUERY_META_HINTS)


def _extract_user_query_from_meta(text: str) -> str:
    s = (text or "").strip()
    if not s or not _is_meta_task_text(s):
        return ""
    m = re.search(r"<user_query>\s*(.+?)\s*</user_query>", s, re.S | re.I)
    if m:
        return (m.group(1) or "").strip()
    m = re.search(r"###\s*User Query:\s*(.+?)(?:\n###|$)", s, re.S | re.I)
    if m:
        return (m.group(1) or "").strip()
    m = re.search(r"###\s*User Query\s*\n+\s*(.+?)(?:\n###|\n<|$)", s, re.S | re.I)
    if m:
        return (m.group(1) or "").strip()
    m = re.search(r"(?:^|\n)\s*user[_\s-]*query\s*:\s*(.+?)(?:\n[A-Z#][^\n]*:|$)", s, re.S | re.I)
    if m:
        return (m.group(1) or "").strip()
    m = re.search(r"(?:^|\n)\s*user'?s?\s+query\s*:\s*(.+?)(?:\n[A-Z#][^\n]*:|$)", s, re.S | re.I)
    if m:
        return (m.group(1) or "").strip()
    m = re.search(r"(?:\"|')?user_query(?:\"|')?\s*[:=]\s*['\"](.+?)['\"]", s, re.S | re.I)
    if m:
        return (m.group(1) or "").strip()
    chat_block = re.search(r"<chat_history>\s*(.+?)\s*</chat_history>", s, re.S | re.I)
    if chat_block:
        chat = (chat_block.group(1) or "").strip()
        user_lines = re.findall(r"(?:^|\n)\s*(?:user|користувач)\s*[:>-]\s*(.+)", chat, re.I)
        if user_lines:
            return (user_lines[-1] or "").strip()
    return ""


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str) and part.strip():
                parts.append(part.strip())
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        if isinstance(text, str):
            return text
    return ""


def _normalize_query_text(text: str) -> str:
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not s.strip():
        return ""
    # Some clients wrap prompts as "History: ... Query: <actual question>".
    # Keep only the explicit query tail to avoid leaking chat history into planner.
    m_query_tail = re.search(r"(?is)(?:^|[\n\s])query\s*:\s*(.+)$", s)
    if m_query_tail:
        s = (m_query_tail.group(1) or "").strip()
    lines: List[str] = []
    for raw_line in s.split("\n"):
        line = (raw_line or "").strip()
        if not line:
            continue
        if re.fullmatch(r"[*\-•]+", line):
            continue
        line = re.sub(r"^\s*[*\-•]\s+", "", line)
        line = re.sub(r"^\s*\d+[.)]\s+", "", line)
        line = line.strip()
        if line:
            lines.append(line)
    out = " ".join(lines).strip()
    out = re.sub(r"\s{2,}", " ", out)
    if len(out) >= 2 and out[0] == out[-1] and out[0] in {"'", '"', "`"}:
        out = out[1:-1].strip()
    return out


def _last_user_message(messages: List[dict]) -> str:
    for msg in reversed(messages or []):
        if msg.get("role") != "user":
            continue
        content = _message_text(msg.get("content", ""))
        if not isinstance(content, str):
            continue

        s = content.strip()
        if not s:
            continue

        if _is_meta_task_text(s):
            extracted = _extract_user_query_from_meta(s)
            if extracted:
                return _normalize_query_text(extracted)
            return ""
        return _normalize_query_text(s)
    return ""


def _query_selection_debug(messages: List[dict], limit: int = 4) -> List[dict]:
    out: List[dict] = []
    for msg in reversed(messages or []):
        if msg.get("role") != "user":
            continue
        raw = _message_text(msg.get("content", ""))
        if not isinstance(raw, str) or not raw.strip():
            continue
        extracted = _extract_user_query_from_meta(raw) if _is_meta_task_text(raw) else ""
        out.append(
            {
                "preview": _safe_trunc(_normalize_query_text(raw), 160),
                "is_meta": bool(_is_meta_task_text(raw)),
                "has_user_query": bool(extracted),
                "extracted_preview": _safe_trunc(_normalize_query_text(extracted), 160) if extracted else "",
            }
        )
        if len(out) >= limit:
            break
    return out

def _effective_user_query(user_message: str, messages: List[dict]) -> str:
    question = (user_message or "").strip()
    if question:
        if not _is_meta_task_text(question):
            return _normalize_query_text(question)
        extracted = _extract_user_query_from_meta(question)
        if extracted:
            return _normalize_query_text(extracted)
        return ""
    fallback = _last_user_message(messages)
    return _normalize_query_text(fallback) if fallback else ""


def _iter_all_file_objs(body: dict, messages: List[dict]) -> List[dict]:
    out: List[dict] = []
    body = body or {}
    for k in ("files", "attachments"):
        for it in (body.get(k) or []):
            if isinstance(it, dict):
                out.append(it)

    for msg in (messages or []):
        for k in ("files", "attachments"):
            for it in (msg.get(k) or []):
                if isinstance(it, dict):
                    out.append(it)

        c = msg.get("content")
        if isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and (
                    part.get("type") in ("file", "input_file") or part.get("kind") == "file"
                ):
                    out.append(part)
    return out


def _iter_message_file_objs(message: Optional[dict]) -> List[dict]:
    out: List[dict] = []
    if not isinstance(message, dict):
        return out
    for k in ("files", "attachments"):
        for it in (message.get(k) or []):
            if isinstance(it, dict):
                out.append(it)
    c = message.get("content")
    if isinstance(c, list):
        for part in c:
            if isinstance(part, dict) and (
                part.get("type") in ("file", "input_file") or part.get("kind") == "file"
            ):
                out.append(part)
    return out


def _last_user_message_obj(messages: List[dict]) -> Optional[dict]:
    for msg in reversed(messages or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg
    return None


def _pick_file_ref_from_history(body: dict, messages: List[dict]) -> Tuple[Optional[str], Optional[dict]]:
    for obj in reversed(_iter_all_file_objs(body, messages)):
        fid = obj.get("id") or obj.get("file_id")
        if fid:
            return fid, obj
    fid = (body or {}).get("file_id")
    if fid:
        return fid, {"id": fid}
    return None, None


def _pick_file_ref(body: dict, messages: List[dict]) -> Tuple[Optional[str], Optional[dict]]:
    """
    Pick explicit file reference from the current turn only:
    - body.files/body.attachments/body.file_id
    - last user message file parts
    This avoids accidental switches from stale files in older chat history.
    """
    body = body or {}
    last_user = _last_user_message_obj(messages)
    current_objs: List[dict] = []
    for k in ("files", "attachments"):
        for it in (body.get(k) or []):
            if isinstance(it, dict):
                current_objs.append(it)
    current_objs.extend(_iter_message_file_objs(last_user))
    for obj in reversed(current_objs):
        fid = obj.get("id") or obj.get("file_id")
        if fid:
            return fid, obj
    fid = body.get("file_id")
    if fid:
        return fid, {"id": fid}
    return None, None


def _resolve_active_file_ref(
    body: dict,
    messages: List[dict],
    session: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[dict], str, Optional[str]]:
    """
    Resolve active file with guard against silent history-based file switching.
    Priority:
      1) Explicit current-turn file ref
      2) Session file_id
      3) History fallback (only when no session and no explicit file)
    Returns: (file_id, file_obj, source, ignored_history_file_id)
    """
    explicit_file_id, explicit_file_obj = _pick_file_ref(body, messages)
    session_file_id = str((session or {}).get("file_id") or "").strip() or None

    if explicit_file_id:
        return explicit_file_id, explicit_file_obj, "explicit", None

    history_file_id, history_file_obj = _pick_file_ref_from_history(body, messages)
    if session_file_id:
        ignored_history = history_file_id if history_file_id and history_file_id != session_file_id else None
        return session_file_id, None, "session", ignored_history

    if history_file_id:
        return history_file_id, history_file_obj, "history", None

    return None, None, "none", None


def _session_key(body: dict) -> str:
    user = (body or {}).get("user") or {}
    user_id = user.get("id") or "anon"
    chat_id = (body or {}).get("conversation_id") or (body or {}).get("chat_id") or "default"
    return f"{user_id}:{chat_id}"


def _safe_trunc(text: str, limit: int) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def _safe_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _extract_llm_usage(resp: Any) -> Dict[str, Optional[int]]:
    usage: Any = None
    if isinstance(resp, dict):
        usage = resp.get("usage")
    else:
        usage = getattr(resp, "usage", None)

    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

    if isinstance(usage, dict):
        prompt_tokens = _safe_int(usage.get("prompt_tokens"))
        completion_tokens = _safe_int(usage.get("completion_tokens"))
        total_tokens = _safe_int(usage.get("total_tokens"))
    else:
        prompt_tokens = _safe_int(getattr(usage, "prompt_tokens", None))
        completion_tokens = _safe_int(getattr(usage, "completion_tokens", None))
        total_tokens = _safe_int(getattr(usage, "total_tokens", None))

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _normalize_learning_query(text: str) -> str:
    s = _normalize_query_text(text or "")
    s = re.sub(r"^\*{1,3}\s*(.*?)\s*\*{1,3}$", r"\1", s)
    return re.sub(r"\s+", " ", s).strip().casefold()


def _read_json_or_default(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _read_jsonl(path: str) -> List[dict]:
    out: List[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = (line or "").strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
    except FileNotFoundError:
        return []
    except Exception:
        return []
    return out


def _atomic_write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _atomic_write_jsonl(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            if not isinstance(row, dict):
                continue
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
    os.replace(tmp, path)


def _extract_json_candidate(text: str) -> Optional[str]:
    s = (text or "").strip()
    if not s:
        return None
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.S | re.I)
    if fence:
        s = (fence.group(1) or "").strip()
    if not s:
        return None

    if s.startswith("{") and s.endswith("}"):
        return s

    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escaped = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
        start = s.find("{", start + 1)
    return None


def _extract_json_candidates(text: str) -> List[str]:
    s = (text or "").strip()
    if not s:
        return []
    out: List[str] = []
    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escaped = False
        matched = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out.append(s[start : i + 1])
                    start = s.find("{", start + 1)
                    matched = True
                    break
        if not matched:
            break
    return out


def _strip_llm_reasoning_sections(text: str) -> str:
    s = str(text or "")
    if not s:
        return ""
    s = re.sub(r"(?is)<(think|analysis|reasoning)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?is)</?(think|analysis|reasoning)[^>]*>", " ", s)
    s = re.sub(r"(?is)```(?:think|thinking|analysis|reasoning)[^\n]*\n.*?```", " ", s)
    return s.strip()


def _parse_json_dict_from_llm(text: str) -> dict:
    s = (text or "").strip()
    if not s:
        raise ValueError("LLM did not return JSON")
    candidates: List[str] = []
    cleaned = _strip_llm_reasoning_sections(s)
    for source in (s, cleaned):
        if not source:
            continue
        first = _extract_json_candidate(source)
        if first and first not in candidates:
            candidates.append(first)
        for cand in _extract_json_candidates(source):
            if cand not in candidates:
                candidates.append(cand)
    if not candidates:
        candidates = [cleaned or s]
    last_err: Optional[Exception] = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception as exc:
            last_err = exc
            continue
        if isinstance(parsed, dict):
            return parsed
    if last_err is not None:
        raise last_err
    raise ValueError("LLM JSON root must be an object")


def _strip_llm_think_sections(text: str) -> str:
    return _strip_llm_reasoning_sections(text)


def _scalar_text_is_nan_like(text: Any) -> bool:
    s = str(text or "").strip().lower()
    return s in {"nan", "none", "null", "nat", "n/a", "na"}


def _extract_filter_hint_from_analysis_code(code: str, max_items: int = 3) -> Optional[str]:
    s = _strip_llm_reasoning_sections(str(code or ""))
    if not s:
        return None
    hints: List[str] = []
    patterns = [
        (r"\[\s*['\"](?P<col>[^'\"]+)['\"]\s*\]\s*==\s*['\"](?P<val>[^'\"]+)['\"]", "{col} = {val}"),
        (
            r"\[\s*['\"](?P<col>[^'\"]+)['\"]\s*\]\.astype\(str\)\.str\.contains\(\s*(?:str\()?(?P<q>['\"])(?P<val>.*?)(?P=q)",
            "{col} contains {val}",
        ),
        (
            r"\[\s*['\"](?P<col>[^'\"]+)['\"]\s*\]\.astype\(str\)\.str\.lower\(\)\s*==\s*str\(\s*['\"](?P<val>[^'\"]+)['\"]\s*\)\.lower\(\)",
            "{col} = {val}",
        ),
    ]
    for pattern, fmt in patterns:
        for m in re.finditer(pattern, s, re.I):
            col = str(m.group("col") or "").strip()
            val = str(m.group("val") or "").strip()
            if not col or not val:
                continue
            hint = fmt.format(col=col, val=val)
            if hint not in hints:
                hints.append(hint)
            if len(hints) >= max_items:
                return "; ".join(hints)
    if not hints:
        return None
    return "; ".join(hints)


def _looks_like_executable_code(text: str) -> bool:
    s = str(text or "")
    return any(
        token in s
        for token in (
            "result =",
            "df[",
            "df.",
            "pd.",
            "np.",
            "for ",
            "if ",
            "while ",
        )
    )


def find_code_blocks(text: str) -> List[str]:
    s = str(text or "")
    if not s:
        return []
    out: List[str] = []
    seen: set[str] = set()
    patterns = (
        r"```(?:python|py)\s*(.*?)\s*```",
        r"```[a-zA-Z0-9_-]*\s*(.*?)\s*```",
    )
    for pat in patterns:
        for m in re.finditer(pat, s, re.S | re.I):
            code = str(m.group(1) or "").strip()
            if not code or code in seen:
                continue
            seen.add(code)
            out.append(code)
    return out


def _extract_analysis_code_from_llm(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""

    # Prefer think-stripped fenced code first to avoid leaking reasoning sections.
    cleaned_first = _strip_llm_think_sections(raw)
    cleaned_blocks = find_code_blocks(cleaned_first)
    for code in cleaned_blocks:
        if re.search(r"(?m)^\s*result\s*=", code):
            return code
    for code in cleaned_blocks:
        if _looks_like_executable_code(code):
            return code

    # Then inspect raw fenced blocks.
    blocks = find_code_blocks(raw)
    for code in blocks:
        if re.search(r"(?m)^\s*result\s*=", code):
            return code
    for code in blocks:
        if _looks_like_executable_code(code):
            return code

    cleaned = cleaned_first
    if not cleaned:
        return ""

    # First, try structured JSON payloads when model still follows object format.
    with contextlib.suppress(Exception):
        parsed = _parse_json_dict_from_llm(cleaned)
        code = str((parsed or {}).get("analysis_code") or "").strip()
        if code:
            return code

    # Then, prefer fenced code blocks after think-section cleanup.
    for code in cleaned_blocks:
        if re.search(r"(?m)^\s*result\s*=", code):
            return code
    for code in cleaned_blocks:
        if _looks_like_executable_code(code):
            return code

    # Finally, accept raw plain-text code when it looks executable.
    lines = [ln.rstrip() for ln in cleaned.splitlines() if ln.strip()]
    if not lines:
        return ""
    if not _looks_like_executable_code(cleaned):
        return ""
    return "\n".join(lines).strip()


def _extract_analysis_code_from_llm_no_think(text: str) -> str:
    return _extract_analysis_code_from_llm(_strip_llm_think_sections(text))


def _profile_fingerprint(profile: Optional[dict]) -> str:
    if not isinstance(profile, dict):
        return ""
    payload = {
        "rows": profile.get("rows"),
        "cols": profile.get("cols"),
        "columns": profile.get("columns"),
        "dtypes": profile.get("dtypes"),
        "schema_version": profile.get("schema_version"),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _profile_columns(profile: Optional[dict], limit: int = 400) -> List[str]:
    cols = [str(c) for c in ((profile or {}).get("columns") or []) if str(c).strip()]
    return cols[: max(1, int(limit))]


def _profile_numeric_columns(profile: Optional[dict], limit: int = 400) -> List[str]:
    cols = _profile_columns(profile, limit=5000)
    dtypes = (profile or {}).get("dtypes") or {}
    out: List[str] = []
    for c in cols:
        if _is_numeric_dtype_text(dtypes.get(c, "")):
            out.append(c)
    return out[: max(1, int(limit))]


def _compact_profile_for_llm(profile: Optional[dict]) -> Dict[str, Any]:
    return {
        "rows": (profile or {}).get("rows"),
        "cols": (profile or {}).get("cols"),
        "columns": _profile_columns(profile),
        "numeric_columns": _profile_numeric_columns(profile),
    }


def _compact_profile_for_trace(profile: Optional[dict]) -> Dict[str, Any]:
    cols = _profile_columns(profile)
    numeric_cols = _profile_numeric_columns(profile)
    return {
        "rows": (profile or {}).get("rows"),
        "cols": (profile or {}).get("cols"),
        "columns_count": len(cols),
        "numeric_columns_count": len(numeric_cols),
        "columns": cols,
        "numeric_columns": numeric_cols,
    }


def _compact_sandbox_run_output(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload or {})
    result_text = str(out.get("result_text") or "")
    stdout = str(out.get("stdout") or "")
    profile = out.get("profile") if isinstance(out.get("profile"), dict) else {}
    return {
        "status": str(out.get("status") or ""),
        "error": str(out.get("error") or ""),
        "result_text_preview": _safe_trunc(result_text, 1200),
        "result_meta": out.get("result_meta") if isinstance(out.get("result_meta"), dict) else {},
        "stdout_preview": _safe_trunc(stdout, 800),
        "committed": bool(out.get("committed")),
        "auto_committed": bool(out.get("auto_committed")),
        "structure_changed": bool(out.get("structure_changed")),
        "profile": _compact_profile_for_trace(profile) if profile else {},
    }


def _compact_plan_result_for_trace(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload or {})
    analysis_code = str(out.get("analysis_code") or "")
    return {
        "ok": bool(out.get("ok")),
        "status": str(out.get("status") or ""),
        "plan_preview": _safe_trunc(str(out.get("plan") or ""), 400),
        "analysis_code_chars": len(analysis_code),
        "analysis_code_preview": _safe_trunc(analysis_code, 600),
        "op": str(out.get("op") or ""),
        "commit_df": out.get("commit_df"),
        "edit_expected": bool(out.get("edit_expected")),
        "router_meta": out.get("router_meta") if isinstance(out.get("router_meta"), dict) else {},
        "events": out.get("events") if isinstance(out.get("events"), list) else [],
        "message_sync": _safe_trunc(str(out.get("message_sync") or ""), 400),
        "message_stream": _safe_trunc(str(out.get("message_stream") or ""), 400),
    }


def _compact_postprocess_result_for_trace(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload or {})
    return {
        "ok": bool(out.get("ok")),
        "status": str(out.get("status") or ""),
        "profile": _compact_profile_for_trace(out.get("profile")) if isinstance(out.get("profile"), dict) else {},
        "mutation_flags": out.get("mutation_flags") if isinstance(out.get("mutation_flags"), dict) else {},
        "message_sync": _safe_trunc(str(out.get("message_sync") or ""), 400),
        "message_stream": _safe_trunc(str(out.get("message_stream") or ""), 400),
    }


def _profile_change_reason(old: Optional[dict], new: Optional[dict]) -> str:
    if not isinstance(old, dict) or not isinstance(new, dict):
        return "unknown"
    reasons: List[str] = []
    if old.get("rows") != new.get("rows") or old.get("cols") != new.get("cols"):
        reasons.append("shape_changed")
    if old.get("columns") != new.get("columns"):
        reasons.append("columns_changed")
    if old.get("dtypes") != new.get("dtypes"):
        reasons.append("dtypes_changed")
    if old.get("schema_version") != new.get("schema_version"):
        reasons.append("schema_version_changed")
    return "+".join(reasons) if reasons else "unknown"


def _fix_unexpected_indents(code: str) -> str:
    lines = code.splitlines()
    fixed: List[str] = []
    prev_stmt: Optional[str] = None
    open_parens = 0
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            fixed.append("")
            continue
        if open_parens == 0 and (prev_stmt is None or not prev_stmt.endswith(":")) and line[:1].isspace():
            line = stripped
        fixed.append(line)
        open_parens += (
            line.count("(")
            + line.count("[")
            + line.count("{")
            - line.count(")")
            - line.count("]")
            - line.count("}")
        )
        prev_stmt = stripped
    return "\n".join(fixed) + ("\n" if code.endswith("\n") else "")


def _normalize_generated_code(code: str) -> str:
    try:
        ast.parse(code)
        return code
    except IndentationError as exc:
        if "unexpected indent" not in str(exc):
            return code
        fixed = _fix_unexpected_indents(code)
        try:
            ast.parse(fixed)
        except Exception:
            return code
        logging.warning("event=code_indent_fix applied")
        return fixed
    except SyntaxError:
        return code


def _strip_forbidden_imports(code: str) -> Tuple[str, bool]:
    if not code:
        return code, False
    lines = code.splitlines()
    remove_line_indexes: set[int] = set()

    # Prefer AST-based detection first (handles aliases and spacing reliably).
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                start = max(1, getattr(node, "lineno", 1))
                end = max(start, getattr(node, "end_lineno", start))
                for idx in range(start - 1, min(end, len(lines))):
                    remove_line_indexes.add(idx)
    except SyntaxError:
        # Fall back to regex on invalid snippets.
        pass

    kept: List[str] = []
    removed = False
    for i, line in enumerate(lines):
        if i in remove_line_indexes or re.match(r"^\s*(import\s+\S+|from\s+\S+\s+import\s+.+)$", line):
            removed = True
            continue
        kept.append(line)

    out = "\n".join(kept)
    if code.endswith("\n"):
        out += "\n"
    return out, removed


def _has_forbidden_import_nodes(code: str) -> bool:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return bool(
            re.search(r"(?m)^\s*import\s+\S+", code or "")
            or re.search(r"(?m)^\s*from\s+\S+\s+import\s+.+", code or "")
        )
    return any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))


def _strip_commit_flag(code: str) -> str:
    if not code:
        return code
    lines = []
    for line in code.splitlines():
        if re.match(r"^\s*COMMIT_DF\s*=\s*True\s*$", line):
            continue
        lines.append(line)
    return "\n".join(lines) + ("\n" if code.endswith("\n") else "")


def _has_result_assignment(code: str) -> bool:
    return bool(re.search(r"(?m)^\s*result\s*=", code or ""))


def _is_retryable_import_keyerror(error: str) -> bool:
    low = (error or "").strip().lower()
    if not low:
        return False
    return (
        ("keyerror: 'import'" in low)
        or ('keyerror: "import"' in low)
        or ("keyerror: '__import__'" in low)
        or ('keyerror: "__import__"' in low)
    )


def _ensure_result_variable(code: str) -> Tuple[str, bool, bool]:
    """
    Ensure final read value is assigned to `result`.
    Returns: (new_code, was_fixed, has_result_assignment).
    """
    if not (code or "").strip():
        return code, False, False
    if _has_result_assignment(code):
        return code, False, True

    lines = (code or "").splitlines()
    last_assignment_idx = -1
    last_var_name: Optional[str] = None
    skip_vars = {
        "df",
        "COMMIT_DF",
        "UNDO",
        "pd",
        "np",
        "re",
    }
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_]\w*)\s*=\s*.+$", stripped)
        if not m:
            continue
        var_name = m.group(1)
        if var_name in skip_vars or var_name.startswith("_"):
            continue
        last_assignment_idx = i
        last_var_name = var_name

    if last_assignment_idx >= 0 and last_var_name:
        old_line = lines[last_assignment_idx]
        new_line = re.sub(
            rf"^(\s*){re.escape(last_var_name)}\s*=",
            r"\1result =",
            old_line,
            count=1,
        )
        lines[last_assignment_idx] = new_line
        logging.warning(
            "event=auto_fix_result_variable old_var=%s line_idx=%d old=%s new=%s",
            last_var_name,
            last_assignment_idx,
            _safe_trunc(old_line, 120),
            _safe_trunc(new_line, 120),
        )
        out = "\n".join(lines) + ("\n" if code.endswith("\n") else "")
        return out, True, True

    warning_line = "# WARNING: missing result assignment; regenerate with `result = <calculation>`"
    if warning_line not in lines:
        lines.append(warning_line)
    logging.warning(
        "event=missing_result_assignment_no_fix code_preview=%s",
        _safe_trunc(code, 400),
    )
    out = "\n".join(lines) + ("\n" if code.endswith("\n") else "")
    return out, False, False


def _validate_has_result_assignment(code: str, op_norm: str) -> Optional[str]:
    if op_norm != "read":
        return None
    if _has_result_assignment(code):
        return None
    return (
        "missing_result_assignment: Generated code does not assign final value to 'result'. "
        "Please regenerate with: result = <calculation>."
    )

def _harden_common_read_patterns(code: str, df_profile: Optional[dict]) -> str:
    """Disabled due to AST transformation conflicts in read-path hardening."""
    return code

_EDIT_METHODS_RETURN_DF = (
    "drop",
    "rename",
    "assign",
    "replace",
    "fillna",
    "drop_duplicates",
    "sort_values",
    "reset_index",
    "set_index",
    "dropna",
    "astype",
    "reindex",
    "reindex_like",
    "sort_index",
    "insert",
)

_PANDAS_MUTATING_METHODS = {
    "drop",
    "drop_duplicates",
    "dropna",
    "assign",
    "replace",
    "fillna",
    "sort_values",
    "sort_index",
    "reset_index",
    "set_index",
    "reindex",
    "reindex_like",
    "astype",
    "convert_dtypes",
    "rename",
    "rename_axis",
    "mask",
    "where",
}

_PANDAS_ALWAYS_INPLACE_METHODS = {"insert", "update"}

def _is_count_intent(question: str) -> bool:
    q = (question or "").lower()
    return bool(re.search(r"\b(скільк\w*|кільк\w*|count|how\s+many|qty|quantity)\b", q))

def _is_sum_intent(question: str) -> bool:
    q = (question or "").lower()
    return bool(
        re.search(
            r"\b(sum|сума|total|загальн\w*|обсяг|volume|units|залишк\w*|"
            r"revenue|sales|gmv|вируч\w*|дохід\w*|оборот\w*|варт\w*)\b",
            q,
            re.I,
        )
    )


def _has_product_sum_intent(question: str) -> bool:
    q = (question or "").lower()
    if not q:
        return False
    has_mul = ("×" in q) or ("*" in q) or bool(re.search(r"\b(x|mul|помнож|добут)\w*\b", q, re.I))
    has_money = bool(
        re.search(r"\b(ціна|price|cost|amount|revenue|sales|gmv|вируч\w*|дохід\w*|оборот\w*|варт\w*)\b", q, re.I)
    )
    has_qty = bool(re.search(r"\b(кільк\w*|qty|quantity|units?|штук|одиниц\w*)\b", q, re.I))
    return bool((has_mul and (has_money or has_qty)) or (has_money and has_qty))


def _extract_df_subscript_col(node: ast.AST) -> Optional[str]:
    if not isinstance(node, ast.Subscript):
        return None
    if not isinstance(node.value, ast.Name) or node.value.id != "df":
        return None
    sl = node.slice
    if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
        return str(sl.value)
    if hasattr(ast, "Index") and isinstance(sl, ast.Index):  # pragma: no cover (py<3.9 compatibility path)
        inner = sl.value
        if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
            return str(inner.value)
    return None


def _detect_result_sum_of_product_columns(code: str) -> Optional[Tuple[str, str]]:
    try:
        tree = ast.parse(code or "")
    except Exception:
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        t = node.targets[0]
        if not isinstance(t, ast.Name) or t.id != "result":
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        if value.args or value.keywords:
            continue
        if not isinstance(value.func, ast.Attribute) or value.func.attr != "sum":
            continue
        prod = value.func.value
        if not isinstance(prod, ast.BinOp) or not isinstance(prod.op, ast.Mult):
            continue
        c1 = _extract_df_subscript_col(prod.left)
        c2 = _extract_df_subscript_col(prod.right)
        if c1 and c2:
            return c1, c2
    return None


def _detect_result_single_column_sum(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code or "")
    except Exception:
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        t = node.targets[0]
        if not isinstance(t, ast.Name) or t.id != "result":
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        if value.args or value.keywords:
            continue
        if not isinstance(value.func, ast.Attribute) or value.func.attr != "sum":
            continue
        col = _extract_df_subscript_col(value.func.value)
        if col:
            return col
    return None


def _result_sum_uses_derived_df_column(code: str, value_col: str) -> bool:
    if not value_col:
        return False
    try:
        tree = ast.parse(code or "")
    except Exception:
        return False

    result_idx: Optional[int] = None
    for idx, node in enumerate(tree.body):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        t = node.targets[0]
        if not isinstance(t, ast.Name) or t.id != "result":
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        if value.args or value.keywords:
            continue
        if not isinstance(value.func, ast.Attribute) or value.func.attr != "sum":
            continue
        col = _extract_df_subscript_col(value.func.value)
        if col == value_col:
            result_idx = idx
            break
    if result_idx is None:
        return False

    for node in tree.body[:result_idx]:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if _extract_df_subscript_col(target) == value_col:
                    return True
        elif isinstance(node, ast.AugAssign):
            if _extract_df_subscript_col(node.target) == value_col:
                return True
    return False


def _id_like_columns(profile_cols: List[str], excluded: set[str]) -> List[str]:
    return [
        c
        for c in profile_cols
        if c not in excluded and re.search(r"(?:^|_)(id|sku|код|артикул)(?:$|_)", c.lower())
    ][:6]


def _rewrite_sum_of_product_code(code: str, df_profile: Optional[dict], question: str = "") -> str:
    """
    Universal guard for read queries of form:
        result = (df[col_a] * df[col_b]).sum()
    Excludes synthetic summary rows that have product columns populated but identifiers/metadata missing.
    """
    if re.search(
        r"\b(mean|avg|average|median|max(?:imum)?|min(?:imum)?|середн\w*|медіан\w*|макс\w*|мін\w*)\b",
        question or "",
        re.I,
    ):
        logging.info(
            "event=sum_of_product_rewrite skipped reason=non_sum_aggregation question=%s",
            _safe_trunc(question, 200),
        )
        return code
    if _code_has_subset_filter_ops(code or ""):
        logging.info(
            "event=sum_of_product_rewrite skipped reason=code_has_filter question=%s",
            _safe_trunc(question, 200),
        )
        return code

    cols = _detect_result_sum_of_product_columns(code or "")
    if not cols:
        return code
    left_col, right_col = cols

    lines = [
        f"_left_col = {left_col!r}",
        f"_right_col = {right_col!r}",
        "_cols_ok = (_left_col in df.columns) and (_right_col in df.columns)",
        "if not _cols_ok:",
        "    result = None",
        "else:",
        "    _left = pd.to_numeric(df[_left_col], errors='coerce')",
        "    _right = pd.to_numeric(df[_right_col], errors='coerce')",
        "    _valid = _left.notna() & _right.notna()",
        "    result = float((_left[_valid] * _right[_valid]).sum())",
        "    print(f\"sum_of_product_guard rows_total={len(df)} rows_used={int(_valid.sum())} rows_excluded={int((~_valid).sum())}\")",
    ]
    logging.info(
        "event=sum_of_product_rewrite applied left_col=%s right_col=%s",
        left_col,
        right_col,
    )
    return "\n".join(lines) + ("\n" if (code or "").endswith("\n") else "")


def _rewrite_single_column_sum_code(code: str, df_profile: Optional[dict]) -> str:
    """
    Universal guard for read queries of form:
        result = df[col].sum()
    Excludes synthetic summary rows that have summed column populated but identifiers/metadata missing.
    """
    value_col = _detect_result_single_column_sum(code or "")
    if not value_col:
        return code
    profile_cols = [str(c) for c in ((df_profile or {}).get("columns") or [])]
    if _result_sum_uses_derived_df_column(code or "", value_col):
        logging.info(
            "event=single_sum_rewrite skipped reason=derived_value_column value_col=%s",
            value_col,
        )
        return code
    if profile_cols and value_col not in profile_cols:
        logging.info(
            "event=single_sum_rewrite skipped reason=value_col_not_in_profile value_col=%s",
            value_col,
        )
        return code
    id_like_cols = _id_like_columns(profile_cols, {value_col})

    lines = [
        f"_value_col = {value_col!r}",
        "_col_ok = _value_col in df.columns",
        "if not _col_ok:",
        "    result = None",
        "else:",
        "    _valid = pd.Series(True, index=df.index)",
        "    _meta_cols = [c for c in df.columns if c != _value_col]",
        "    if _meta_cols:",
        "        _valid = _valid & (df[_meta_cols].notna().sum(axis=1) > 0)",
        f"    _id_like = {id_like_cols!r}",
        "    if _id_like:",
        "        _id_like = [c for c in _id_like if c in df.columns]",
        "        if _id_like:",
        "            _valid = _valid & df[_id_like].notna().any(axis=1)",
        "    _raw = df.loc[_valid, _value_col].astype(str)",
        r"    _clean = _raw.str.replace(r'[\s\xa0]', '', regex=True).str.replace(r'[^0-9,.\-]', '', regex=True).str.replace(',', '.')",
        r"    _num_mask = _clean.str.match(r'^-?(\d+(\.\d*)?|\.\d+)$')",
        "    _values = _clean.where(_num_mask, np.nan).astype(float)",
        "    result = float(_values.sum())",
        "    print(f\"single_sum_guard rows_total={len(df)} rows_used={int(_valid.sum())} rows_excluded={int((~_valid).sum())}\")",
    ]
    logging.info(
        "event=single_sum_rewrite applied value_col=%s id_like=%s",
        value_col,
        id_like_cols,
    )
    return "\n".join(lines) + ("\n" if (code or "").endswith("\n") else "")


def _is_total_value_scalar_question(question: str, profile: Optional[dict]) -> bool:
    q = (question or "").lower()
    # Do not treat non-sum aggregations as total inventory value.
    if re.search(r"\b(mean|average|avg|median|max(?:imum)?|min(?:imum)?|середн\w*|медіан\w*|макс\w*|мін\w*)\b", q):
        return False
    has_value = bool(re.search(r"\b(варт\w*|value|cost|цін\w*|price|кошту\w*|вируч\w*|дохід\w*)\b", q))
    has_total = bool(re.search(r"\b(всього|загальн\w*|total|sum|сума)\b", q))
    has_mul = ("×" in q) or ("*" in q) or bool(re.search(r"\b(x|mul|помнож|добут)\w*\b", q))
    # Require explicit total/multiply intent to avoid false positives from noisy context.
    return has_value and (has_total or has_mul)


def _is_top_expensive_available_intent(question: str) -> bool:
    q = (question or "").lower()
    has_top = bool(re.search(r"\b(top|топ|найдорожч\w*|найвищ\w*\s+цін\w*|max\s+price)\b", q))
    has_price = bool(re.search(r"\b(цін\w*|price|варт\w*|cost)\b", q))
    has_entity = bool(re.search(r"\b(товар\w*|модел\w*|product\w*|item\w*)\b", q))
    has_available = bool(re.search(r"\b(наявн\w*|in\s+stock|available|склад\w*)\b", q))
    return has_top and (has_price or has_top) and has_entity and has_available


def _detect_top_available_ranking_mode(question: str) -> str:
    q = (question or "").lower()
    if not q:
        return "unit_price"
    has_mul = ("×" in q) or ("*" in q) or bool(re.search(r"\b(x|mul|помнож|добут)\w*\b", q))
    has_total_value = bool(
        re.search(
            r"\b(сумарн\w*|загальн\w*|total|sum|сума|підсум\w*|overall)\b.*\b(варт\w*|value|cost|цін\w*|price)\b"
            r"|\b(варт\w*|value|cost|цін\w*|price)\b.*\b(на\s+склад\w*|in\s*stock|inventory|on\s+hand)\b",
            q,
            re.I,
        )
    )
    return "inventory_value" if (has_mul or has_total_value) else "unit_price"


def _code_has_inventory_value_signal(code: str) -> bool:
    s = code or ""
    if not s.strip():
        return False
    if _detect_result_sum_of_product_columns(s):
        return True
    if re.search(r"\['_metric'\]\s*=\s*.+\*.+", s) and re.search(r"groupby\([^\n]*\)\['_metric'\]", s):
        return True
    if re.search(r"nlargest\([^,\n]+,\s*['_\"](?:inventory_value|metric_sum|_metric)['\"]\)", s, re.I):
        return True
    return False


def _has_explicit_group_dimension_cue(question: str) -> bool:
    q = (question or "").lower()
    if not q:
        return False
    return bool(
        re.search(
            r"\b(group\s*by|by\s+(?:category|categories|brand|brands|model|models|type|types|status|segment|segments|region|regions|country|city))\b"
            r"|(?:по|за)\s+(?:категор\w*|бренд\w*|модел\w*|тип\w*|груп\w*|сегмент\w*|регіон\w*|країн\w*|міст\w*)",
            q,
            re.I,
        )
    )


def _has_ranking_cues(question: str) -> bool:
    q = (question or "").lower()
    return bool(
        re.search(
            r"\b("
            r"top|топ|найдорож\w*|найдешев\w*|найбіль\w*|наймен\w*|"
            r"highest|lowest|largest|smallest|most\s+expensive|cheapest|"
            r"max(?:imum)?|min(?:imum)?|rank(?:ing)?"
            r")\b",
            q,
            re.I,
        )
    )


def _looks_like_value_filter_query(question: str) -> bool:
    q = (question or "").lower()
    has_number = bool(re.search(r"\d", q))
    if not has_number:
        return False
    return bool(
        re.search(
            r"(=|==|<=|>=|<|>)"
            r"|\b(дорівн\w*|рівн\w*|equal(?:s)?|exact(?:ly)?)\b"
            r"|\b(де|where|має|мають|with|having|price\s+of|ціною)\b",
            q,
            re.I,
        )
    )


def _has_explicit_status_constraint(question: str) -> bool:
    q = (question or "").lower()
    return bool(re.search(r"\b(статус\w*|status)\b", q))


def _rewrite_top_expensive_available_code(code: str, question: str, df_profile: Optional[dict]) -> str:
    """
    For intents like "top-N most expensive available products", avoid strict equality
    to one status value (e.g. only 'В наявності') and use inventory-oriented availability.
    """
    if not _is_top_expensive_available_intent(question):
        return code
    if _has_explicit_group_dimension_cue(question):
        logging.info(
            "event=top_expensive_available_rewrite skipped reason=grouping_query question=%s",
            _safe_trunc(question, 200),
        )
        return code
    if _code_has_inventory_value_signal(code):
        logging.info(
            "event=top_expensive_available_rewrite skipped reason=existing_inventory_metric question=%s",
            _safe_trunc(question, 200),
        )
        return code
    profile = df_profile or {}
    price_col = _pick_price_like_column(profile)
    qty_col = _pick_quantity_like_column(profile)
    if not price_col:
        return code
    rank_mode = _detect_top_available_ranking_mode(question)
    if rank_mode == "inventory_value" and (not qty_col or qty_col == price_col):
        logging.info(
            "event=top_expensive_available_rewrite skipped reason=inventory_mode_missing_qty price_col=%s qty_col=%s",
            price_col,
            qty_col,
        )
        return code
    cols = [str(c) for c in (profile.get("columns") or [])]
    dtypes = (profile or {}).get("dtypes") or {}
    status_col = _pick_availability_column(question, profile) if cols else None
    top_n = _extract_top_n_from_question(question, default=5)
    explicit_status = _has_explicit_status_constraint(question)

    out_cols: List[str] = []
    id_col = _pick_id_like_column(cols)
    if id_col:
        out_cols.append(id_col)
    text_cols = [
        c
        for c in cols
        if not _is_numeric_dtype_text(dtypes.get(c, ""))
        and c not in {id_col, status_col, price_col}
    ]
    for c in [*text_cols[:3], price_col]:
        if c in cols and c not in out_cols:
            out_cols.append(c)
    if not out_cols:
        out_cols = [price_col]

    lines = [
        f"_price_col = {price_col!r}",
        f"_qty_col = {qty_col!r}" if qty_col else "_qty_col = None",
        f"_status_col = {status_col!r}" if status_col else "_status_col = None",
        f"_top_n = {int(top_n)}",
        f"_strict_status = {bool(explicit_status)!r}",
        f"_rank_mode = {rank_mode!r}",
        "_df_src = df.copy(deep=False)",
        "_avail = pd.Series(True, index=_df_src.index)",
        "if _qty_col and (_qty_col in _df_src.columns):",
        "    _qty_raw = _df_src[_qty_col].astype(str)",
        r"    _qty_clean = _qty_raw.str.replace(r'[\s\xa0]', '', regex=True).str.replace(r'[^0-9,.\-]', '', regex=True).str.replace(',', '.')",
        r"    _qty_mask = _qty_clean.str.match(r'^-?(\d+(\.\d*)?|\.\d+)$')",
        "    _qty = _qty_clean.where(_qty_mask, np.nan).astype(float)",
        "    _avail = _avail & (_qty > 0)",
        "if _status_col and (_status_col in _df_src.columns):",
        "    _status = _df_src[_status_col].astype(str).str.strip().str.lower()",
        r"    _pos_re = r'(?:в\s*наявн|наявн|in\s*stock|available|доступн|закінч\w*|резерв\w*)'",
        r"    _neg_re = r'(?:нема|відсутн|out\s*of\s*stock|unavailable|not\s*available)'",
        r"    _order_re = r'(?:під\s*замовлення|under\s*order|backorder)'",
        "    if _strict_status:",
        "        _in = _status.str.contains(_pos_re, regex=True, na=False)",
        "        _out = _status.str.contains(_neg_re, regex=True, na=False)",
        "        _avail = _avail & _in & ~_out",
        "    else:",
        "        _under_order = _status.str.contains(_order_re, regex=True, na=False)",
        "        _avail = _avail & (~_under_order)",
        "_top_df = _df_src.loc[_avail].copy()",
        "if _price_col not in _top_df.columns:",
        "    result = []",
        "else:",
        "    _price_raw = _top_df[_price_col].astype(str)",
        r"    _price_clean = _price_raw.str.replace(r'[\s\xa0]', '', regex=True).str.replace(r'[^0-9,.\-]', '', regex=True).str.replace(',', '.')",
        r"    _price_mask = _price_clean.str.match(r'^-?(\d+(\.\d*)?|\.\d+)$')",
        "    _price = _price_clean.where(_price_mask, np.nan).astype(float)",
        "    _top_df = _top_df.loc[_price.notna()].copy()",
        "    _top_df[_price_col] = _price.loc[_price.notna()]",
        "    if _rank_mode == 'inventory_value' and _qty_col and (_qty_col in _top_df.columns):",
        "        _qraw2 = _top_df[_qty_col].astype(str)",
        r"        _qclean2 = _qraw2.str.replace(r'[\s\xa0]', '', regex=True).str.replace(r'[^0-9,.\-]', '', regex=True).str.replace(',', '.')",
        r"        _qmask2 = _qclean2.str.match(r'^-?(\d+(\.\d*)?|\.\d+)$')",
        "        _qty2 = _qclean2.where(_qmask2, np.nan).astype(float)",
        "        _top_df = _top_df.loc[_qty2.notna()].copy()",
        "        _top_df[_qty_col] = _qty2.loc[_qty2.notna()]",
        "        _top_df['_inventory_value'] = _top_df[_price_col] * _top_df[_qty_col]",
        "        _top_df = _top_df.nlargest(_top_n, '_inventory_value')",
        f"        _out_cols = [c for c in {out_cols!r} if c in _top_df.columns]",
        "        if '_inventory_value' not in _out_cols:",
        "            _out_cols.append('_inventory_value')",
        "        result = _top_df[_out_cols] if _out_cols else _top_df",
        "    else:",
        "        _top_df = _top_df.nlargest(_top_n, _price_col)",
        f"        result = _top_df[{out_cols!r}]",
    ]
    logging.info(
        "event=top_expensive_available_rewrite applied top_n=%s price_col=%s qty_col=%s status_col=%s strict_status=%s rank_mode=%s",
        top_n,
        price_col,
        qty_col,
        status_col,
        explicit_status,
        rank_mode,
    )
    return "\n".join(lines) + ("\n" if (code or "").endswith("\n") else "")


def _is_numeric_dtype_text(dtype: str) -> bool:
    d = str(dtype or "").lower()
    return d.startswith(("int", "float", "uint"))


def _is_id_like_col_name(col: str) -> bool:
    return bool(re.search(r"(?:^|_)(id|sku|код|артикул)(?:$|_)", str(col).lower()))


def _id_like_col_score(col: str) -> int:
    c = str(col or "").strip().lower()
    if not c:
        return 0
    score = 0
    if c in {"id", "item_id", "record_id", "product_id"}:
        score += 10
    if c in {"sku", "код", "артикул"}:
        score += 9
    if _is_id_like_col_name(c):
        score += 6
    if c.endswith("_id") or c.startswith("id_"):
        score += 3
    return score


def _pick_id_like_column(columns: List[str]) -> Optional[str]:
    best_col: Optional[str] = None
    best_score = 0
    for col in [str(c) for c in (columns or [])]:
        score = _id_like_col_score(col)
        if score > best_score:
            best_col = col
            best_score = score
    return best_col if best_score > 0 else None


def _pick_price_like_column(profile: dict) -> Optional[str]:
    cols = [str(c) for c in ((profile or {}).get("columns") or [])]
    dtypes = (profile or {}).get("dtypes") or {}
    numeric = [c for c in cols if _is_numeric_dtype_text(dtypes.get(c, ""))]
    if not numeric:
        return None
    pref = [
        c for c in numeric
        if re.search(r"(цін|price|варт|cost|amount|total|revenue|вируч)", c.lower())
    ]
    if pref:
        return pref[0]
    non_id = [c for c in numeric if not _is_id_like_col_name(c)]
    return non_id[0] if non_id else numeric[0]


def _pick_quantity_like_column(profile: dict) -> Optional[str]:
    cols = [str(c) for c in ((profile or {}).get("columns") or [])]
    dtypes = (profile or {}).get("dtypes") or {}
    numeric = [c for c in cols if _is_numeric_dtype_text(dtypes.get(c, ""))]
    if not numeric:
        return None
    pref = [
        c for c in numeric
        if re.search(r"(кільк|qty|quantity|units|stock|залишк)", c.lower())
    ]
    if pref:
        return pref[0]
    non_id = [c for c in numeric if not _is_id_like_col_name(c)]
    return non_id[0] if non_id else numeric[0]


def _total_inventory_value_shortcut_code(question: str, profile: dict) -> Optional[Tuple[str, str]]:
    if not _is_total_value_scalar_question(question, profile):
        return None
    price_col = _pick_price_like_column(profile)
    qty_col = _pick_quantity_like_column(profile)
    if not price_col or not qty_col or price_col == qty_col:
        return None
    code = f"result = (df[{price_col!r}] * df[{qty_col!r}]).sum()\n"
    plan = f"Порахувати загальну вартість як суму добутків {qty_col} × {price_col}."
    return code, plan

def _has_df_assignment(code: str) -> bool:
    return bool(re.search(r"(^|\n)\s*df\s*=", code or "")) or bool(
        re.search(r"df\.(loc|iloc|at|iat)\[.+?\]\s*=", code or "")
    ) or bool(re.search(r"df\[[^\]]+\]\s*=", code or ""))

def _has_inplace_op(code: str) -> bool:
    return bool(re.search(r"inplace\s*=\s*True", code or ""))

def _has_df_changed_flag(code: str) -> bool:
    return bool(re.search(r"(^|\n)\s*df_changed\s*=\s*True\s*$", code or "", re.M))

def _is_df_write_target(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "df"
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id == "df":
            return True
        if isinstance(node.value, ast.Attribute):
            return isinstance(node.value.value, ast.Name) and node.value.value.id == "df" and node.value.attr in {"loc", "iloc", "at", "iat"}
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, ast.Name) and node.value.id == "df" and node.attr in {"loc", "iloc", "at", "iat"}
    return False

def _call_has_inplace_true(call: ast.Call) -> bool:
    for kw in call.keywords or []:
        if kw.arg == "inplace" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
            return True
    return False


def _expr_references_df(node: ast.AST) -> bool:
    try:
        return any(isinstance(n, ast.Name) and n.id == "df" for n in ast.walk(node))
    except Exception:
        return False


def _is_non_mutating_df_copy_call(call: ast.Call) -> bool:
    # Allow read-mode prelude: df = df.copy(deep=False/True)
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr != "copy":
        return False
    if not isinstance(func.value, ast.Name) or func.value.id != "df":
        return False
    return True


def _is_df_filter_like_value(node: ast.AST) -> bool:
    """Return True for read-safe df rebinding patterns like df = df[mask]."""
    cur = node
    # Allow trailing .copy() calls, e.g. df = df[mask].copy()
    while isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute) and cur.func.attr == "copy":
        cur = cur.func.value
    if isinstance(cur, ast.Subscript):
        root = cur.value
        if isinstance(root, ast.Name) and root.id == "df":
            return True
        if (
            isinstance(root, ast.Attribute)
            and isinstance(root.value, ast.Name)
            and root.value.id == "df"
            and root.attr in {"loc", "iloc"}
        ):
            return True
    return False


def _rewrite_read_df_rebinding(code: str) -> Tuple[str, bool]:
    """
    Rewrite read-only df rebinding into a temp variable to avoid false mutation blocks.
    Example:
      df = df[df['<filter_col>'] == 'X']
      result = df['<metric_col>'].mean()
    ->
      _df_read = df.copy(deep=False)
      _df_read = _df_read[_df_read['<filter_col>'] == 'X']
      result = _df_read['<metric_col>'].mean()
    """
    code = _strip_llm_think_sections(code or "")
    text = code
    if not text.strip():
        return text, False
    try:
        tree = ast.parse(text)
    except Exception:
        return text, False

    has_df_rebind = False
    for node in ast.walk(tree):
        if isinstance(node, ast.AugAssign) and _is_df_write_target(node.target):
            return text, False
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, (ast.Subscript, ast.Attribute)) and _is_df_write_target(t):
                    return text, False
                if isinstance(t, ast.Name) and t.id == "df":
                    has_df_rebind = True
                    value = node.value
                    if isinstance(value, ast.Call) and _is_non_mutating_df_copy_call(value):
                        continue
                    if _is_df_filter_like_value(value):
                        continue
                    return text, False
        if isinstance(node, ast.Call) and _call_has_inplace_true(node):
            return text, False

    if not has_df_rebind:
        return text, False

    class _RenameDfToReadVar(ast.NodeTransformer):
        def visit_Name(self, n: ast.Name) -> ast.AST:
            if n.id == "df":
                return ast.copy_location(ast.Name(id="_df_read", ctx=n.ctx), n)
            return n

    try:
        rewritten_tree = _RenameDfToReadVar().visit(tree)
        ast.fix_missing_locations(rewritten_tree)
        rewritten = ast.unparse(rewritten_tree).strip()
    except Exception:
        return text, False

    out = "_df_read = df.copy(deep=False)\n" + rewritten + "\n"
    return out, True

def _auto_detect_commit(code: str) -> bool:
    if not (code or "").strip():
        return False
    if _has_df_changed_flag(code):
        return True
    if _has_inplace_op(code):
        return True
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return bool(
            re.search(r"df\.(loc|iloc|at|iat)\[.+?\]\s*=", code or "", re.S)
            or re.search(r"df\[[^\]]+\]\s*=", code or "")
            or re.search(r"df\s*=\s*df\s*\[", code or "", re.I)
            or re.search(r"df\s*=\s*df\.(loc|iloc)\s*\[", code or "", re.I)
            or re.search(r"df\s*=\s*df\.(drop|rename|assign|replace|fillna|sort_values|reset_index|set_index|astype|reindex|drop_duplicates|dropna|mask|where|convert_dtypes|sort_index)\s*\(", code or "", re.I)
            or re.search(r"df\s*=\s*pd\.concat\s*\(", code or "", re.I)
        )

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(_is_df_write_target(t) for t in targets):
                if any(isinstance(t, (ast.Subscript, ast.Attribute)) for t in targets):
                    return True
                value = node.value
                # Any assignment back into `df` that derives from `df` is a mutation signal,
                # except a plain defensive copy (`df = df.copy(...)`) used in read code.
                if isinstance(value, ast.Call) and _is_non_mutating_df_copy_call(value):
                    continue
                if _expr_references_df(value):
                    return True
                if isinstance(value, ast.Call):
                    # Handle chained calls like:
                    # df = df.drop(...).reset_index(...)
                    methods_in_chain: List[str] = []
                    has_df_root = False
                    cur: ast.AST = value
                    while isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute):
                        methods_in_chain.append(cur.func.attr)
                        cur = cur.func.value
                    if isinstance(cur, ast.Name) and cur.id == "df":
                        has_df_root = True
                    if has_df_root and any(
                        m in _PANDAS_ALWAYS_INPLACE_METHODS or m in _PANDAS_MUTATING_METHODS
                        for m in methods_in_chain
                    ):
                        return True
                    if isinstance(value.func, ast.Attribute):
                        if isinstance(value.func.value, ast.Name) and value.func.value.id == "pd" and value.func.attr == "concat":
                            return True
        if isinstance(node, ast.AugAssign) and _is_df_write_target(node.target):
            return True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "df":
                method = node.func.attr
                if method in _PANDAS_ALWAYS_INPLACE_METHODS:
                    return True
                if method in _PANDAS_MUTATING_METHODS and _call_has_inplace_true(node):
                    return True
    return False

def _infer_op_from_question(question: str) -> str:
    if _is_meta_task_text(question or ""):
        return "read"
    q = (question or "").lower()
    if re.search(
        r"\b(видал|додай|додати|встав|переймен|очист|заміни|замін|set|rename|delete|drop|insert|add|clear|fill|sort|фільтр|відфільтр|сортуй|переміст)\b",
        q,
    ):
        return "edit"
    return "read"

def _should_commit_from_code(code: str) -> bool:
    return _has_df_assignment(code) or _has_inplace_op(code) or _has_df_changed_flag(code)

_EDIT_TRIGGER_RE = re.compile(
    r"\b(видал|додай|додати|встав|переймен|очист|заміни|замін|змі\w*|змін\w*|встанов\w*|редаг\w*|онов\w*|постав|set|rename|delete|drop|insert|add|clear|fill|sort|фільтр|відфільтр|сортуй|переміст)\b",
    re.I,
)

def _has_edit_triggers(question: str) -> bool:
    return bool(_EDIT_TRIGGER_RE.search(question or ""))


_ROUTER_MUTATING_INTENT_HINTS = (
    "drop",
    "delete",
    "remove",
    "add",
    "insert",
    "update",
    "rename",
    "edit",
    "set",
    "clear",
    "fill",
    "sort",
    "concat",
)

def _is_per_item_normalization_query(question: str) -> bool:
    q = (question or "").lower()
    if not q:
        return False

    has_metric = bool(
        re.search(
            r"\b(mean|average|avg|середн\w*|sum|сума|total|загальн\w*|count|кільк\w*|скільк\w*)\b",
            q,
            re.I,
        )
    )
    if not has_metric:
        return False

    blocked_unit_roots = {
        "category",
        "categories",
        "brand",
        "brands",
        "model",
        "models",
        "type",
        "types",
        "status",
        "категор",
        "бренд",
        "модел",
        "тип",
        "груп",
    }
    patterns = [
        r"\bper\s+(?:one|single|each)?\s*([a-z][a-z0-9_-]{2,})\b",
        r"\bна\s+(?:один|одну|одне|1|кож(?:ен|ну|не|ний|на|ну)|по\s+одн\w*)\s+([a-zа-яіїєґ0-9_-]{3,})\b",
        r"\bв\s+середньому\s+на\s+([a-zа-яіїєґ0-9_-]{3,})\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, q, re.I):
            unit = str(m.group(1) or "").strip().lower()
            if not unit:
                continue
            root = unit[:8]
            if any(root.startswith(b[:8]) for b in blocked_unit_roots):
                continue
            return True
    return False


def _router_intent_looks_mutating(intent_id: str) -> bool:
    s = (intent_id or "").strip().lower()
    if not s:
        return False
    return any(h in s for h in _ROUTER_MUTATING_INTENT_HINTS)


def _question_has_filter_context_for_router_guard(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    if _is_per_item_normalization_query(q):
        # "на один товар/позицію" is a normalization cue, not subset filtering.
        return False
    has_metric = has_router_metric_cue(q)
    has_filter_words = has_router_filter_context_cue(q)
    has_explicit_filter_words = has_explicit_subset_filter_words(q)
    has_availability_cue = has_availability_filter_cue(q)
    has_value_filter = _looks_like_value_filter_query(q)
    has_entity_token = has_router_entity_token(q)
    has_grouping_signal = has_grouping_cue(q)
    if has_grouping_signal and not (has_explicit_filter_words or has_availability_cue or has_value_filter):
        # Grouped aggregations ("by/per/по/за ...") should not be treated as subset
        # unless we have an explicit subset signal.
        return False
    if has_filter_words or has_availability_cue or has_value_filter or (has_metric and has_entity_token):
        return True
    if not has_metric:
        return False
    metric_roots = {
        "max",
        "min",
        "mea",
        "avg",
        "sum",
        "tot",
        "cou",
        "кіл",
        "скі",
        "мак",
        "мін",
        "сер",
        "сум",
        "під",
        "цін",
        "вар",
    }
    tokens = [t for t in _subset_word_tokens(q) if t[:3] not in metric_roots]
    # Conservative fallback: trigger only when query has likely entity token
    # (ASCII brand/model token or mixed alnum token), e.g. "відеокарт nvidia".
    has_entity_like = any(re.search(r"[a-z0-9]", t) for t in tokens)
    return has_entity_like and len(tokens) >= 2


def _router_code_has_filter_ops(analysis_code: str) -> bool:
    code = analysis_code or ""
    if not code.strip():
        return False
    if re.search(r"(?m)^\s*result\s*=\s*(?:df|_work)\s*\[", code):
        return True
    if re.search(r"\.query\s*\(", code):
        return True
    if re.search(r"\.str\.contains\s*\(", code):
        return True
    if re.search(r"\.str\.match\s*\(", code):
        return True
    if re.search(r"(?m)^\s*(?:df|_work)\s*=\s*(?:df|_work)\.loc\s*\[", code):
        return True
    if re.search(r"(?m)^\s*(?:df|_work)\s*=\s*(?:df|_work)\s*\[\s*_[A-Za-z]\w*\s*\]", code):
        return True
    if re.search(r"(?m)^\s*(?:df|_work)\s*=\s*(?:df|_work)\s*\[.*(?:==|!=|>=|<=|>|<|\.isin\s*\()", code):
        return True
    return False


def _question_requires_subset_filter(question: str, profile: Optional[dict] = None) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    if _is_per_item_normalization_query(q):
        return False
    has_grouping_signal = has_grouping_cue(q)
    has_explicit_filter_words = has_explicit_subset_filter_words(q)
    has_availability_cue = has_availability_filter_cue(q)
    has_value_filter = _looks_like_value_filter_query(q)
    if has_grouping_signal:
        # "по категоріях/брендах" is usually a grouped aggregation over all rows,
        # not a subset filter. Keep subset mode only for explicit subset cues
        # or when profile evidence shows an actual entity term.
        if not (has_explicit_filter_words or has_availability_cue or has_value_filter):
            if not profile:
                return False
            try:
                maybe_terms = _extract_subset_terms_from_question(q, profile, limit=2)
                if not maybe_terms:
                    return False
            except Exception:
                return False
        elif not profile:
            return False
    has_metric = has_router_metric_cue(q)
    if not has_metric:
        return False
    if _question_has_filter_context_for_router_guard(q):
        return True
    # Profile-aware fallback: if query contains entity-like term present in table preview,
    # treat it as subset request even without explicit words like "серед/where/for".
    if profile:
        try:
            terms = _extract_subset_terms_from_question(q, profile, limit=2)
            if terms:
                return True
        except Exception:
            return False
    return False


def _missing_subset_filter_guard_applies(
    finalize_err: Optional[str],
    question: str,
    profile: Optional[dict] = None,
) -> bool:
    reason = str(finalize_err or "")
    if "missing_subset_filter" not in reason:
        return False
    return _question_requires_subset_filter(question, profile)


def _is_groupby_without_subset_question(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    if not has_grouping_cue(q):
        return False
    if has_explicit_subset_filter_words(q):
        return False
    if has_availability_filter_cue(q):
        return False
    if _looks_like_value_filter_query(q):
        return False
    return True


def _code_has_groupby_aggregation(code: str) -> bool:
    s = code or ""
    if not s.strip():
        return False
    return bool(re.search(r"\.groupby\s*\(", s))


def _code_has_subset_filter_ops(code: str) -> bool:
    s = code or ""
    if not s.strip():
        return False
    # Strong filter signals for subset-style questions.
    if re.search(r"\.str\.(contains|match)\s*\(", s):
        return True
    if re.search(r"\.query\s*\(", s):
        return True
    if re.search(r"\.isin\s*\(", s):
        return True
    if re.search(r"\[\s*['\"][^'\"]+['\"]\s*\].*?(==|!=)\s*['\"][^'\"]+['\"]", s):
        return True
    if re.search(r"(?m)^\s*(?:df|_work)\s*=\s*(?:df|_work)\s*\[\s*_[A-Za-z]\w*\s*\]", s):
        return True
    # Numeric subset filters, e.g. df[df['qty'] < 5], _work = _work[_work['price'] >= 100]
    if re.search(
        r"(?m)^\s*[A-Za-z_]\w*\s*=\s*[A-Za-z_]\w*\s*\[[^\n]*(?:<=|>=|<|>|==|!=)[^\n]*\]\s*$",
        s,
    ):
        return True
    if re.search(
        r"(?m)^\s*[A-Za-z_]\w*\s*=\s*[A-Za-z_]\w*\.loc\s*\[[^\n]*(?:<=|>=|<|>|==|!=)[^\n]*\]\s*$",
        s,
    ):
        return True
    if re.search(r"pd\.to_numeric\s*\([^)]+\)\s*(?:<=|>=|<|>)\s*[-+]?\d", s):
        return True
    return False


def _should_reject_router_hit_for_read(
    has_edit: bool,
    analysis_code: str,
    router_meta: Optional[dict],
    question: str = "",
    profile: Optional[dict] = None,
) -> bool:
    if has_edit:
        return False
    intent_id = str((router_meta or {}).get("intent_id") or "")
    if _router_intent_looks_mutating(intent_id):
        return True
    if _question_has_filter_context_for_router_guard(question) and not _router_code_has_filter_ops(analysis_code):
        logging.warning(
            "event=shortcut_router_guard status=rejected reason=missing_filter intent_id=%s question_preview=%s code_preview=%s",
            intent_id,
            _safe_trunc(question, 200),
            _safe_trunc(analysis_code, 300),
        )
        return True
    price_col = _pick_price_like_column(profile or {})
    qty_col = _pick_quantity_like_column(profile or {})
    if (
        _is_total_value_scalar_question(question, profile)
        and price_col
        and qty_col
        and price_col != qty_col
        and not _code_has_inventory_value_signal(analysis_code)
    ):
        logging.warning(
            "event=shortcut_router_guard status=rejected reason=missing_inventory_value_metric intent_id=%s question_preview=%s code_preview=%s",
            intent_id,
            _safe_trunc(question, 220),
            _safe_trunc(analysis_code, 320),
        )
        return True
    if re.search(r"(?m)^\s*COMMIT_DF\s*=\s*True\s*$", analysis_code or ""):
        return True
    return _auto_detect_commit(analysis_code or "")


def _detect_metrics(question: str) -> List[str]:
    q = (question or "").lower()
    found: List[str] = []
    for name, pat in _METRIC_PATTERNS.items():
        if re.search(pat, q, re.I):
            found.append(name)
    # Guard against English maximum/minimum used outside data context.
    if ("maximum" in q or "minimum" in q) and not _METRIC_CONTEXT_RE.search(q):
        found = [m for m in found if m not in ("max", "min")]
    # count needs stronger context to avoid false positives like "number of years"
    if _COUNT_WORD_RE.search(q):
        found.append("count")
    else:
        m = _COUNT_NUMBER_OF_RE.search(q)
        if m:
            if _COUNT_CONTEXT_RE.search(m.group(1)):
                found.append("count")
        elif _COUNT_QTY_RE.search(q):
            if _COUNT_CONTEXT_RE.search(q):
                found.append("count")
    # stable order
    order = ["mean", "min", "max", "median", "sum", "count"]
    return [m for m in order if m in found]


def _has_grouping_cues(question: str) -> bool:
    q = (question or "").lower()
    if _is_per_item_normalization_query(q):
        return False
    return bool(
        re.search(
            r"\b(груп\w*|розбив\w*|розподіл\w*|кожн\w*|each|group\w*|by)\b",
            q,
            re.I,
        )
        or re.search(
            r"\bper\s+(?:category|categories|brand|brands|model|models|type|types|status|month|year|day)\b",
            q,
            re.I,
        )
        or re.search(r"\b(по|за)\s+\w+", q, re.I)
    )


def _is_aggregate_query_intent(question: str) -> bool:
    q = (question or "").lower()
    if not q:
        return False
    if _has_grouping_cues(question):
        return False

    metrics = _detect_metrics(question)
    if not metrics and not _is_count_intent(question) and not _is_sum_intent(question):
        return False

    has_total_cue = bool(
        re.search(
            r"\b(всього|загальн\w*|total|sum|сума|підсум\w*|скільк\w*|кільк\w*|count|how\s+many)\b",
            q,
            re.I,
        )
    )
    has_display_cue = bool(
        re.search(r"\b(покажи|показати|show|list|перелік|знайди|find|виведи)\b", q, re.I)
    )
    if has_display_cue and not has_total_cue:
        return False
    return True


_SUBSET_TOKEN_STOPWORDS = {
    "the",
    "and",
    "with",
    "for",
    "from",
    "that",
    "this",
    "where",
    "only",
    "among",
    "within",
    "count",
    "price",
    "prices",
    "value",
    "total",
    "sum",
    "avg",
    "mean",
    "median",
    "min",
    "max",
    "maximum",
    "minimum",
    "rows",
    "items",
    "products",
    "data",
    "table",
    "порах",
    "раху",
    "підра",
    "скіл",
    "кіль",
    "сума",
    "суму",
    "ціна",
    "цін",
    "варт",
    "загал",
    "сере",
    "меді",
    "мін",
    "макс",
    "макси",
    "мінім",
    "рядк",
    "запи",
    "това",
    "дани",
    "табл",
    "яка",
    "який",
    "яке",
    "сере",
    "серед",
    "для",
    "з",
    "та",
    "і",
}


def _subset_word_tokens(text: str) -> List[str]:
    if not text:
        return []
    raw = re.findall(r"[A-Za-zА-Яа-яІіЇїЄєҐґ0-9_/-]{2,}", text.lower())
    out: List[str] = []
    for tok in raw:
        tok = tok.strip("_-/")
        if len(tok) < 3:
            continue
        if re.fullmatch(r"\d+", tok):
            continue
        out.append(tok)
    return out


def _preview_value_roots(profile: Optional[dict], limit_rows: int = 150) -> set[str]:
    preview = (profile or {}).get("preview") or []
    roots: set[str] = set()
    if not isinstance(preview, list):
        return roots
    for row in preview[:limit_rows]:
        if not isinstance(row, dict):
            continue
        for val in row.values():
            if val is None:
                continue
            sval = str(val).strip().lower()
            if not sval:
                continue
            if re.fullmatch(r"[\d\s.,:%\-]+", sval):
                continue
            for tok in _subset_word_tokens(sval):
                roots.add(tok[:5])
    return roots


def _extract_subset_terms_from_question(question: str, profile: Optional[dict], limit: int = 4) -> List[str]:
    q_tokens = _subset_word_tokens(question or "")
    if not q_tokens:
        return []
    preview_roots = _preview_value_roots(profile)
    if not preview_roots:
        return []

    keep: List[str] = []
    seen: set[str] = set()
    for tok in q_tokens:
        root = tok[:5]
        if root in _SUBSET_TOKEN_STOPWORDS:
            continue
        if root in seen:
            continue
        # Accept terms that are likely entities/features present in table values.
        if root in preview_roots:
            keep.append(tok)
            seen.add(root)
            if len(keep) >= limit:
                break
            continue
        # Fuzzy inflection fallback (e.g. "мишей" vs preview "миша").
        root3 = root[:3]
        if root3 and any(pr[:3] == root3 for pr in preview_roots):
            keep.append(tok)
            seen.add(root)
            if len(keep) >= limit:
                break
    return keep


def _fallback_subset_terms_from_question(question: str, limit: int = 3) -> List[str]:
    tokens = _subset_word_tokens(question or "")
    if not tokens:
        return []
    generic_roots = set(_SUBSET_TOKEN_STOPWORDS) | {
        "товар",
        "склад",
        "наяв",
        "досту",
        "стату",
        "катег",
        "бренд",
        "модел",
        "group",
        "row",
        "colum",
    }
    out: List[str] = []
    seen: set[str] = set()
    for tok in tokens:
        root = tok[:5]
        if root in generic_roots:
            continue
        if root in seen:
            continue
        out.append(tok)
        seen.add(root)
        if len(out) >= limit:
            break
    return out


def _subset_term_pattern(term: str) -> str:
    t = str(term or "").strip().lower()
    if not t:
        return ""
    # For mixed alnum tokens (e.g. rtx4090) keep exact token matching.
    if re.search(r"\d", t):
        return re.escape(t)
    # Cyrillic orthography tolerance (uk/ru variants).
    # Example: "миша" should also match "мышь".
    cyr_equiv = {
        "и": "иіы",
        "і": "иіы",
        "ы": "иіы",
        "е": "еєёэ",
        "є": "еєёэ",
        "ё": "еєёэ",
        "э": "еєёэ",
        "г": "гґ",
        "ґ": "гґ",
    }
    # Universal fuzzy prefix:
    # - Cyrillic terms: 3-char stem captures inflections ("миша/миші/мишка")
    # - Latin terms: 4-char stem keeps precision for brands/models
    is_cyr = bool(re.search(r"[а-яіїєґ]", t, re.I))
    root_len = 3 if is_cyr else 4
    root_len = max(2, min(root_len, len(t)))
    root = t[:root_len]
    if is_cyr:
        root_re = "".join(
            f"[{re.escape(cyr_equiv[ch])}]" if ch in cyr_equiv else re.escape(ch)
            for ch in root
        )
        return r"\b" + root_re + r"\w*"
    return r"\b" + re.escape(root) + r"\w*"


def _extract_numeric_threshold_condition(text: str) -> Optional[Tuple[str, float]]:
    s = (text or "").strip().lower()
    if not s:
        return None

    symbol = re.search(r"(<=|>=|<|>)\s*(-?\d+(?:[.,]\d+)?)", s)
    if symbol:
        raw = symbol.group(2).replace(",", ".")
        try:
            return symbol.group(1), float(raw)
        except Exception:
            return None

    patterns = [
        (r"\b(?:не\s+більше|no\s+more\s+than|at\s+most)\s*(-?\d+(?:[.,]\d+)?)", "<="),
        (r"\b(?:не\s+менше|no\s+less\s+than|at\s+least)\s*(-?\d+(?:[.,]\d+)?)", ">="),
        (r"\b(?:менше|less\s+than|below|under)\s*(-?\d+(?:[.,]\d+)?)", "<"),
        (r"\b(?:більше|more\s+than|above|over)\s*(-?\d+(?:[.,]\d+)?)", ">"),
    ]
    for pat, op in patterns:
        m = re.search(pat, s, re.I)
        if not m:
            continue
        raw = m.group(1).replace(",", ".")
        try:
            return op, float(raw)
        except Exception:
            return None
    return None


def _subset_keyword_metric_shortcut_code(
    question: str,
    profile: dict,
    preferred_col: Optional[str] = None,
    terms_hint: Optional[List[str]] = None,
    slots_hint: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[str, str]]:
    if not (question or "").strip():
        return None
    if _has_edit_triggers(question):
        return None

    metrics = _detect_metrics(question)
    if not metrics and not _is_count_intent(question):
        return None
    if not metrics and _is_count_intent(question):
        metrics = ["count"]

    terms = [str(t).strip() for t in (terms_hint or []) if str(t).strip()]
    if not terms:
        terms = _extract_subset_terms_from_question(question, profile)
    if not terms:
        terms = _fallback_subset_terms_from_question(question, limit=2)
    if not terms:
        return None

    requires_subset = _question_requires_subset_filter(question)
    if not requires_subset and not _is_count_intent(question):
        return None

    columns = [str(c) for c in ((profile or {}).get("columns") or [])]
    if not columns:
        return None
    dtypes = (profile or {}).get("dtypes") or {}
    text_cols = [
        c for c in columns if not str(dtypes.get(c, "")).lower().startswith(("int", "float", "uint"))
    ]
    if not text_cols:
        text_cols = columns[:]

    q_low = (question or "").lower()
    has_availability_cue = bool(
        re.search(r"\b(на\s+складі|в\s+наявн|у\s+наявн|in\s+stock|available|наявн|склад\w*|залишк\w*)\b", q_low)
    )
    availability_mode = _availability_target_mode(question) if has_availability_cue else ""
    availability_col = _pick_availability_column(question, profile) if availability_mode else None
    qty_like_col = _pick_quantity_like_column(profile)
    llm_agg = str((slots_hint or {}).get("agg") or "").strip().lower()
    llm_metric_col = str((slots_hint or {}).get("metric_col") or "").strip()
    llm_avail_mode = str((slots_hint or {}).get("availability_mode") or "").strip().lower()
    llm_avail_col = str((slots_hint or {}).get("availability_col") or "").strip()
    numeric_threshold = _extract_numeric_threshold_condition(question)

    metric_order = ["count", "max", "min", "mean", "median", "sum"]
    metric = next((m for m in metric_order if m in metrics), "count")
    if llm_agg in {"count", "sum", "mean", "min", "max", "median"}:
        metric = llm_agg
    # Disambiguate "загальна кількість ... на складі": sum stock units, not row count.
    # But keep row-count when query has explicit numeric threshold (e.g. "< 5").
    if metric == "count" and qty_like_col and (_is_sum_intent(question) or availability_mode) and not numeric_threshold:
        metric = "sum"
    if has_availability_cue and llm_avail_mode in {"in", "out", "any"}:
        availability_mode = llm_avail_mode
    if has_availability_cue and availability_mode and llm_avail_col in columns:
        availability_col = llm_avail_col

    metric_col: Optional[str] = None
    if metric != "count":
        if llm_metric_col in columns and str(dtypes.get(llm_metric_col, "")).lower().startswith(("int", "float", "uint")):
            metric_col = llm_metric_col
        if metric == "sum" and qty_like_col in columns:
            metric_col = metric_col or qty_like_col
        if not metric_col and preferred_col and preferred_col in columns:
            metric_col = preferred_col
        if not metric_col:
            metric_col = _choose_column_from_question(question, profile)
        if not metric_col and metric == "sum" and qty_like_col in columns:
            metric_col = qty_like_col
        if not metric_col:
            return None

    if numeric_threshold and qty_like_col:
        terms = [
            t
            for t in terms
            if not re.search(r"\d", str(t))
            and not re.search(r"(менше|більше|less|more|lower|above|below|under|over)", str(t), re.I)
        ]

    patterns: List[str] = []
    for p in (_subset_term_pattern(t) for t in terms):
        if p and p not in patterns:
            patterns.append(p)
    if not patterns and not (numeric_threshold and qty_like_col):
        return None

    preview_rows = [r for r in ((profile or {}).get("preview") or []) if isinstance(r, dict)]

    def _preview_match_count(col: str, pat: str) -> int:
        if not preview_rows:
            return 0
        cnt = 0
        for row in preview_rows:
            val = str(row.get(col, "") or "").lower()
            if val and re.search(pat, val, re.I):
                cnt += 1
        return cnt

    structured_filters: List[Tuple[str, str]] = []
    unresolved_patterns: List[str] = []
    for pat in patterns:
        scored: List[Tuple[int, str]] = []
        for col in text_cols:
            cnt = _preview_match_count(col, pat)
            if cnt > 0:
                scored.append((cnt, col))
        if not scored:
            unresolved_patterns.append(pat)
            continue
        scored.sort(key=lambda x: x[0], reverse=True)
        best_cnt, best_col = scored[0]
        second_cnt = scored[1][0] if len(scored) > 1 else -1
        # Avoid hard assignment when column signal is weak/ambiguous on sparse previews.
        # Weak matches should be resolved later across all text columns to prevent false-zero counts.
        if best_cnt <= 1 or (second_cnt == best_cnt and best_cnt <= 2):
            unresolved_patterns.append(pat)
            continue
        structured_filters.append((best_col, pat))

    use_structured = bool(structured_filters)
    code_lines: List[str] = ["_work = df.copy(deep=False)"]
    if numeric_threshold and qty_like_col:
        op_sym, threshold = numeric_threshold
        code_lines.extend(
            [
                f"_qty_col = {qty_like_col!r}",
                "if _qty_col in _work.columns:",
                f"    _q = pd.to_numeric(_work[_qty_col], errors='coerce')",
                f"    _work = _work.loc[_q {op_sym} {threshold!r}].copy()",
            ]
        )
        if patterns:
            code_lines.append(f"_patterns = {patterns!r}")
            code_lines.extend(
                [
                    f"_text_cols = {text_cols!r}",
                    "if not _text_cols:",
                    "    _text_cols = list(_work.columns)",
                    "_mask = pd.Series(True, index=_work.index)",
                    "for _pat in _patterns:",
                    "    _mask_pat = pd.Series(False, index=_work.index)",
                    "    for _c in _text_cols:",
                    "        if _c in _work.columns:",
                    "            _s = _work[_c].astype(str).str.lower()",
                    "            _mask_pat = _mask_pat | _s.str.contains(_pat, regex=True, na=False)",
                    "    _mask = _mask & _mask_pat",
                    "_work = _work.loc[_mask].copy()",
                ]
            )
    elif use_structured:
        grouped_filters: Dict[str, List[str]] = {}
        for col, pat in structured_filters:
            grouped_filters.setdefault(col, []).append(pat)
        code_lines.append(f"_all_patterns = {patterns!r}")
        code_lines.append(f"_all_text_cols = {text_cols!r}")
        code_lines.append(f"_structured_filters = {grouped_filters!r}")
        code_lines.extend(
            [
                "for _col, _pats in _structured_filters.items():",
                "    if _col not in _work.columns:",
                "        _work = _work.iloc[0:0].copy()",
                "        break",
                "    _series = _work[_col].astype(str).str.lower()",
                "    _mask_col = pd.Series(False, index=_work.index)",
                "    for _pat in _pats:",
                "        _mask_col = _mask_col | _series.str.contains(_pat, regex=True, na=False)",
                "    _work = _work.loc[_mask_col].copy()",
            ]
        )
        if unresolved_patterns:
            # For unresolved terms use broad OR matching across text columns
            # to avoid over-constraining with uncertain column assignment.
            code_lines.extend(
                [
                    f"_unresolved_patterns = {unresolved_patterns!r}",
                    f"_text_cols = {text_cols!r}",
                    "if not _text_cols:",
                    "    _text_cols = list(_work.columns)",
                    "_mask_unresolved = pd.Series(False, index=_work.index)",
                    "for _pat in _unresolved_patterns:",
                    "    _mask_pat = pd.Series(False, index=_work.index)",
                    "    for _c in _text_cols:",
                    "        if _c in _work.columns:",
                    "            _s = _work[_c].astype(str).str.lower()",
                    "            _mask_pat = _mask_pat | _s.str.contains(_pat, regex=True, na=False)",
                    "    _mask_unresolved = _mask_unresolved | _mask_pat",
                    "_work = _work.loc[_mask_unresolved].copy()",
                ]
            )
        code_lines.extend(
            [
                "if len(_work) == 0 and _all_patterns:",
                "    _work_fb = df.copy(deep=False)",
                "    _cols_fb = [c for c in _all_text_cols if c in _work_fb.columns]",
                "    if not _cols_fb:",
                "        _cols_fb = list(_work_fb.columns)",
                "    _mask_fb = pd.Series(True, index=_work_fb.index)",
                "    for _pat in _all_patterns:",
                "        _mask_pat = pd.Series(False, index=_work_fb.index)",
                "        for _c in _cols_fb:",
                "            _s = _work_fb[_c].astype(str).str.lower()",
                "            _mask_pat = _mask_pat | _s.str.contains(_pat, regex=True, na=False)",
                "        _mask_fb = _mask_fb & _mask_pat",
                "    _work = _work_fb.loc[_mask_fb].copy()",
            ]
        )
    else:
        code_lines.extend(
            [
                f"_text_cols = {text_cols!r}",
                "if not _text_cols:",
                "    _text_cols = list(_work.columns)",
                f"_patterns = {patterns!r}",
                "_mask = pd.Series(True, index=_work.index)",
                "for _pat in _patterns:",
                "    _mask_pat = pd.Series(False, index=_work.index)",
                "    for _c in _text_cols:",
                "        if _c in _work.columns:",
                "            _s = _work[_c].astype(str).str.lower()",
                "            _mask_pat = _mask_pat | _s.str.contains(_pat, regex=True, na=False)",
                "    _mask = _mask & _mask_pat",
                "_work = _work.loc[_mask].copy()",
            ]
        )
    if availability_mode:
        code_lines.extend(
            [
                f"_avail_mode = {availability_mode!r}",
                f"_avail_col = {availability_col!r}" if availability_col else "_avail_col = None",
                f"_qty_col = {qty_like_col!r}" if qty_like_col else "_qty_col = None",
                r"_in_re = r'(?:в\s*наявн|наявн|in\s*stock|available|доступн|резерв\w*|закінч\w*)'",
                r"_out_re = r'(?:нема|відсутн|out\s*of\s*stock|unavailable|not\s*available)'",
                "if _avail_col and (_avail_col in _work.columns):",
                "    _st = _work[_avail_col].astype(str).str.strip().str.lower()",
                "    _in = _st.str.contains(_in_re, regex=True, na=False)",
                "    _out = _st.str.contains(_out_re, regex=True, na=False)",
                "elif _qty_col and (_qty_col in _work.columns):",
                "    _q = pd.to_numeric(_work[_qty_col], errors='coerce')",
                "    _in = _q > 0",
                "    _out = _q <= 0",
                "else:",
                "    _in = pd.Series(True, index=_work.index)",
                "    _out = pd.Series(False, index=_work.index)",
                "if _avail_mode == 'out':",
                "    _work = _work.loc[_out & ~_in].copy()",
                "elif _avail_mode == 'any':",
                "    _work = _work.loc[_in | _out].copy()",
                "else:",
                "    _work = _work.loc[_in & ~_out].copy()",
            ]
        )

    if metric == "count":
        code_lines.append("result = int(len(_work))")
    else:
        code_lines.extend(
            [
                f"_metric_col = {metric_col!r}",
                "if _metric_col not in _work.columns:",
                "    result = None",
                "else:",
                "    _raw = _work[_metric_col].astype(str)",
                r"    _clean = _raw.str.replace(r'[\s\xa0]', '', regex=True).str.replace(r'[^0-9,.\-]', '', regex=True).str.replace(',', '.')",
                r"    _num_mask = _clean.str.match(r'^-?(\d+(\.\d*)?|\.\d+)$')",
                "    _num = _clean.where(_num_mask, np.nan).astype(float)",
                f"    _metric = {metric!r}",
                "    if _metric == 'max':",
                "        _v = _num.max()",
                "    elif _metric == 'min':",
                "        _v = _num.min()",
                "    elif _metric == 'mean':",
                "        _v = _num.mean()",
                "    elif _metric == 'median':",
                "        _v = _num.median()",
                "    else:",
                "        _v = _num.sum()",
                "    result = None if pd.isna(_v) else float(_v)",
            ]
        )

    plan = (
        "Виділити підмножину рядків за ключовими словами запиту "
        f"({', '.join(terms)}) і порахувати метрику {metric}."
    )
    return "\n".join(code_lines) + "\n", plan


_ALLOWED_SERIES_AGG = {"mean", "min", "max", "sum", "median", "count"}


def _rewrite_forbidden_getattr(code: str) -> str:
    if not code:
        return code
    pattern = r"getattr\(\s*([^\),]+?)\s*,\s*['\"](" + "|".join(_ALLOWED_SERIES_AGG) + r")['\"]\s*\)\s*\(\s*\)"
    return re.sub(pattern, r"\1.\2()", code)


_ENTITY_COUNT_RE = re.compile(r"\b(скільк\w*|кільк\w*|count|number\s+of|nunique)\b", re.I)
_ENTITY_UNIQUE_RE = re.compile(r"\b(унікальн\w*|distinct|different|unique|nunique)\b", re.I)
_ENTITY_GROUP_RE = re.compile(r"\b(кожн\w*|each|per|group\w*|по\s+кожн\w*|за\s+кожн\w*)\b", re.I)
_ENTITY_TOKEN_STOPWORDS = {
    "скіл",
    "кіль",
    "coun",
    "nuni",
    "uniq",
    "dist",
    "diff",
    "unik",
    "унік",
    "знач",
    "data",
    "tabl",
    "табл",
    "рядк",
    "запи",
    "яки",
    "якi",
    "для",
    "про",
}


def _semantic_roots(text: str) -> List[str]:
    parts = re.split(r"[^a-zа-яіїєґ0-9]+", (text or "").lower())
    roots = [p[:4] for p in parts if len(p) >= 3]
    return [r for r in roots if r not in _ENTITY_TOKEN_STOPWORDS]


def _pick_relevant_column(question: str, columns: List[str]) -> Optional[str]:
    direct = _find_column_in_text(question, columns)
    if direct:
        return direct
    q_roots = set(_semantic_roots(question))
    if not q_roots:
        return None
    best_col: Optional[str] = None
    best_score = 0.0
    for col in columns:
        col_roots = set(_semantic_roots(str(col)))
        if not col_roots:
            continue
        overlap = col_roots & q_roots
        if not overlap:
            # Fuzzy fallback for inflections: match by 3-char root prefix.
            fuzzy = set()
            for cr in col_roots:
                for qr in q_roots:
                    if len(cr) >= 3 and len(qr) >= 3 and cr[:3] == qr[:3]:
                        fuzzy.add(cr)
                        break
            overlap = fuzzy
        if not overlap:
            continue
        score = (2.0 * len(overlap)) - (0.35 * max(0, len(col_roots) - len(overlap)))
        if score > best_score:
            best_score = score
            best_col = str(col)
    return best_col


_AVAILABILITY_VALUE_RE = re.compile(
    r"\b(в\s+наявн|наявн|в\s+запас\w*|запас\w*|залишк\w*|in\s+stock|available|inventory|warehouse)\b",
    re.I,
)
_AVAILABILITY_COL_RE = re.compile(
    r"\b(статус\w*|наявн\w*|доступн\w*|запас\w*|залишк\w*|availability|available|in_stock|stock|inventory|warehouse|status)\b",
    re.I,
)
_AVAILABILITY_NEG_RE = re.compile(r"\b(нема|відсутн|out\s+of\s+stock|unavailable|not\s+available)\b", re.I)
_AVAILABILITY_WAREHOUSE_RE = re.compile(r"\b(склад\w*|запас\w*|залишк\w*|warehouse|inventory)\b", re.I)


def _is_availability_count_intent(question: str) -> bool:
    q = (question or "").lower()
    if not _is_count_intent(q):
        return False
    return bool(
        _AVAILABILITY_VALUE_RE.search(q)
        or _AVAILABILITY_NEG_RE.search(q)
        or _AVAILABILITY_COL_RE.search(q)
        or _AVAILABILITY_WAREHOUSE_RE.search(q)
    )


def _availability_target_mode(question: str) -> str:
    q = (question or "").lower()
    has_in = bool(_AVAILABILITY_VALUE_RE.search(q))
    has_out = bool(_AVAILABILITY_NEG_RE.search(q))
    # Explicit "not in stock" should prefer out-of-stock branch.
    if re.search(r"(не\s+в\s+наявн|not\s+in\s+stock|нема\w*\s+в\s+наявн)", q):
        return "out"
    if has_out and not has_in:
        return "out"
    if has_in and not has_out:
        return "in"
    if _AVAILABILITY_WAREHOUSE_RE.search(q):
        return "in"
    return "any"


def _pick_availability_column(question: str, profile: Optional[dict]) -> Optional[str]:
    columns = [str(c) for c in ((profile or {}).get("columns") or [])]
    preview = (profile or {}).get("preview") or []
    direct = _find_column_in_text(question, columns)
    if direct and _AVAILABILITY_COL_RE.search(str(direct)):
        return direct
    for col in columns:
        c = str(col)
        if _AVAILABILITY_COL_RE.search(c):
            return c
    # Fallback: infer a status-like column by value patterns in preview rows.
    if isinstance(preview, list):
        best_col: Optional[str] = None
        best_hits = 0
        for col in columns:
            hits = 0
            for row in preview[:200]:
                if not isinstance(row, dict):
                    continue
                val = row.get(col)
                if val is None:
                    continue
                s = str(val).strip().lower()
                if not s:
                    continue
                if _AVAILABILITY_VALUE_RE.search(s) or _AVAILABILITY_NEG_RE.search(s):
                    hits += 1
            if hits > best_hits:
                best_hits = hits
                best_col = col
        if best_col and best_hits > 0:
            return best_col
    return direct


def _is_entity_nunique_intent(question: str) -> bool:
    q = (question or "").lower()
    if not q:
        return False
    if not _ENTITY_COUNT_RE.search(q):
        return False
    if not _ENTITY_UNIQUE_RE.search(q):
        return False
    if _ENTITY_GROUP_RE.search(q):
        return False
    return True


def _pick_nunique_column(question: str, df_profile: Optional[dict]) -> Optional[str]:
    columns = [str(c) for c in ((df_profile or {}).get("columns") or [])]
    if not columns:
        return None
    return _pick_relevant_column(question, columns)


def _entity_nunique_code(column: str) -> str:
    lines = [
        f"_entity = df[{column!r}].dropna().astype(str)",
        "_entity = _entity.str.strip()",
        r"_entity = _entity.str.replace(r'[\s\xa0]+', ' ', regex=True)",
        r"_entity = _entity.str.replace(r'^[\s\"\'`.,;:()\[\]{}]+|[\s\"\'`.,;:()\[\]{}]+$', '', regex=True)",
        "_entity = _entity.str.lower()",
        "_entity = _entity[_entity != '']",
        "result = int(_entity.nunique())",
    ]
    return "\n".join(lines) + "\n"


def _enforce_entity_nunique_code(question: str, code: str, df_profile: Optional[dict]) -> str:
    if not _is_entity_nunique_intent(question):
        return code
    col = _pick_nunique_column(question, df_profile)
    if not col:
        return code
    guarded = _entity_nunique_code(col)
    if (code or "").strip() == guarded.strip():
        return code
    logging.info("event=entity_nunique_guard applied column=%s", col)
    return guarded

def _finalize_code_for_sandbox(
    question: str,
    analysis_code: str,
    op: Optional[str]=None,
    commit_df: Optional[bool]=None,
    df_profile: Optional[dict]=None,
) -> Tuple[str, bool, Optional[str]]:
    analysis_code = _strip_llm_think_sections(analysis_code or "")
    code = _normalize_generated_code(analysis_code or "")
    inferred = _infer_op_from_question(question)
    op_norm = (op or "").strip().lower()
    if op_norm not in ("read", "edit"):
        op_norm = inferred

    requires_subset_filter = (
        op_norm == "read"
        and not _has_edit_triggers(question)
        and _question_requires_subset_filter(question, df_profile)
    )
    if (
        requires_subset_filter
        and _is_groupby_without_subset_question(question)
        and _code_has_groupby_aggregation(code)
        and not _code_has_subset_filter_ops(code)
    ):
        logging.info(
            "event=subset_guard_override reason=groupby_without_subset question_preview=%s code_preview=%s",
            _safe_trunc(question, 200),
            _safe_trunc(code, 300),
        )
        requires_subset_filter = False
    if requires_subset_filter and not _code_has_subset_filter_ops(code):
        logging.warning(
            "event=skip_rewrites reason=subset_filter_missing question_preview=%s code_preview=%s",
            _safe_trunc(question, 200),
            _safe_trunc(code, 300),
        )
        return (
            code,
            False,
            "missing_subset_filter: Generated code does not filter data for requested subset. Please add subset filter before aggregation.",
        )

    code = _rewrite_top_expensive_available_code(code, question, df_profile)
    code = _rewrite_sum_of_product_code(code, df_profile, question=question)
    code = _rewrite_single_column_sum_code(code, df_profile)
    code = _rewrite_forbidden_getattr(code)
    code, removed_imports = _strip_forbidden_imports(code)
    if removed_imports:
        logging.warning("event=auto_fix_import removed forbidden import statements from generated code")

    if op_norm == "edit" or _has_edit_triggers(question):
        if re.search(r"(^|\n)\s*result\s*=\s*pd\.concat\(", code):
            code = re.sub(r"(^|\n)(\s*)result(\s*=\s*pd\.concat\()", r"\1\2df\3", code)
            logging.warning("event=auto_fix_concat replaced result=pd.concat(...) with df=pd.concat(...)")
        code, edit_err = _validate_edit_code(code)
        if edit_err:
            logging.warning("event=edit_degraded reason=%s", edit_err)
            op_norm = "read"

    has_mutations = _auto_detect_commit(code)
    if has_mutations and op_norm == "read":
        if _has_edit_triggers(question):
            op_norm = "edit"
            logging.info("event=op_reclassified from=read to=edit reason=mutations_detected")
        else:
            rewritten_code, was_rewritten = _rewrite_read_df_rebinding(code)
            if was_rewritten:
                code = rewritten_code
                has_mutations = _auto_detect_commit(code)
                logging.info(
                    "event=auto_fix_read_rebinding applied=%s still_mutating=%s",
                    was_rewritten,
                    has_mutations,
                )
            if not has_mutations:
                logging.info("event=read_mutation_resolved reason=read_rebinding_rewrite")
            else:
                logging.warning(
                    "event=read_mutation_blocked question_preview=%s code_preview=%s",
                    _safe_trunc(question, 200),
                    _safe_trunc(code, 300),
                )
                return (
                    code,
                    False,
                    "Згенерований код для read-запиту змінює таблицю. Уточніть запит як зміну даних або переформулюйте його як read без модифікацій.",
                )

    if op_norm == "read" and not has_mutations:
        try:
            code = _harden_common_read_patterns(code, df_profile)
        except Exception:
            pass

    commit_requested = bool(commit_df) if commit_df is not None else (op_norm == "edit")
    if op_norm == "read":
        commit_requested = False
    if op_norm == "edit" and has_mutations and not commit_requested:
        logging.info(
            "event=force_commit_from_signal op=%s has_mutations=%s commit_df=%s",
            op_norm,
            has_mutations,
            commit_df,
        )
        commit_requested = True
    if commit_requested and not has_mutations:
        logging.info("event=commit_stripped reason=no_df_change")
        commit_requested = False
        op_norm = "read"

    if commit_requested:
        if "COMMIT_DF" not in code:
            code = code.rstrip() + "\nCOMMIT_DF = True\n"
        if "result" not in code:
            code = code.rstrip() + "\nresult = {'status': 'updated'}\n"
    else:
        code = _strip_commit_flag(code)

    if op_norm == "read" and not has_mutations:
        if not re.search(r"(?m)^\s*df\s*=\s*df\.copy\s*\(", code):
            code = "df = df.copy(deep=False)\n" + code

    # Ensure read-mode code always provides `result`.
    if op_norm == "read":
        code, fixed_result, _ = _ensure_result_variable(code)
        if fixed_result:
            logging.info("event=result_assignment_auto_fixed")
        validation_err = _validate_has_result_assignment(code, op_norm)
        if validation_err:
            logging.error(
                "event=code_validation_failed error=%s code_preview=%s",
                validation_err,
                _safe_trunc(code, 500),
            )
            return code, commit_requested, validation_err

    return code, commit_requested, None



def _repair_edit_code(code: str) -> str:
    lines = (code or "").splitlines()
    out: List[str] = []
    for line in lines:
        stripped = line.strip()
        if "=" in line or "inplace=True" in stripped.replace(" ", ""):
            out.append(line)
            continue
        m = re.match(r"^(\s*)df\.(\w+)\(", line)
        if not m:
            out.append(line)
            continue
        indent, method = m.group(1), m.group(2)
        if method in _EDIT_METHODS_RETURN_DF:
            out.append(re.sub(r"^(\s*)df\.", r"\1df = df.", line, count=1))
        else:
            out.append(line)
    return "\n".join(out)

def _repair_scalar_df_overwrite(code: str) -> str:
    """Avoid destructive rewrites where df is overwritten by a scalar extraction."""
    out: List[str] = []
    for line in (code or "").splitlines():
        if re.match(r"^\s*df\s*=\s*df\.(loc|iloc)\[.*\]\.(iloc|iat|at)\[", line):
            out.append(re.sub(r"^\s*df\s*=", "result =", line, count=1))
            logging.info("event=auto_fix_scalar_overwrite rewrote df=<scalar extraction> to result=<scalar extraction>")
            continue
        if re.match(r"^\s*df\s*=\s*df\[[^\]]+\]\.(iloc|iat)\[", line):
            out.append(re.sub(r"^\s*df\s*=", "result =", line, count=1))
            logging.info("event=auto_fix_scalar_overwrite rewrote df=<series extraction> to result=<series extraction>")
            continue
        out.append(line)
    return "\n".join(out) + ("\n" if (code or "").endswith("\n") else "")

def _validate_edit_code(code: str) -> Tuple[str, Optional[str]]:
    """Перевіряє чи edit-код правильно зберігає зміни назад у df."""
    if not (code or "").strip():
        return code, "Empty code"

    if re.search(r"(^|\n)\s*result\s*=\s*pd\.concat\(", code or ""):
        code = re.sub(
            r"(^|\n)(\s*)result(\s*=\s*pd\.concat\()",
            r"\1\2df\3",
            code,
        )
        logging.info("event=auto_fix_concat_assignment fixed result=pd.concat to df=pd.concat")

    if re.search(r"(^|\n)\s*result\s*=\s*df\.(drop|rename|assign|replace|fillna|sort_values|reset_index|set_index|astype|reindex|drop_duplicates|dropna)\s*\(", code or ""):
        code = re.sub(
            r"(^|\n)(\s*)result(\s*=\s*df\.(drop|rename|assign|replace|fillna|sort_values|reset_index|set_index|astype|reindex|drop_duplicates|dropna)\s*\()",
            r"\1\2df\3",
            code,
        )
        logging.info("event=auto_fix_mutation_assignment fixed result=df.<mutator>(...) to df=df.<mutator>(...)")

    code = _repair_scalar_df_overwrite(code)

    has_df_assign = bool(re.search(r"(^|\n)\s*df\s*=", code or ""))
    has_inplace = bool(re.search(r"inplace\s*=\s*True", code or ""))
    has_index_assign = bool(re.search(r"df\.(loc|iloc|at|iat)\[.+?\]\s*=", code or "", re.S))

    if has_df_assign or has_inplace or has_index_assign:
        return code, None

    repaired = _repair_edit_code(code)
    if _has_df_assignment(repaired) or _has_inplace_op(repaired):
        return repaired, None
    return code, "Edit code must assign back to df or use inplace=True"

def _enforce_count_code(question: str, code: str) -> Tuple[str, Optional[str]]:
    if (
        not _is_count_intent(question)
        or _is_sum_intent(question)
        or _has_product_sum_intent(question)
        or re.search(r"=\s*_left\s*\*\s*_right", code or "")
    ):
        return code, None
    if re.search(r"\.sum\s*\(\s*\)", code or ""):
        lines: List[str] = []
        for line in (code or "").splitlines():
            if not re.search(r"\.sum\s*\(\s*\)", line):
                lines.append(line)
                continue
            # Preserve boolean null checks; sum() is the correct way to count True values.
            if re.search(r"\.(isnull|isna|notnull|notna)\(\)\.sum\s*\(\s*\)", line):
                lines.append(line)
                continue
            # Prefer GroupBy.size() when grouping; otherwise use the .size property.
            if re.search(r"\.groupby\s*\(", line):
                line = re.sub(r"\.sum\s*\(\s*\)", ".size()", line)
            else:
                line = re.sub(r"\.sum\s*\(\s*\)", ".size", line)
            lines.append(line)
        code = "\n".join(lines)
        # Fix any accidental size() on boolean null checks that slipped through.
        code = re.sub(r"\.isnull\(\)\.size\s*\(\s*\)", ".isnull().sum()", code)
        code = re.sub(r"\.isna\(\)\.size\s*\(\s*\)", ".isna().sum()", code)
        code = re.sub(r"\.notnull\(\)\.size\s*\(\s*\)", ".notnull().sum()", code)
        code = re.sub(r"\.notna\(\)\.size\s*\(\s*\)", ".notna().sum()", code)
        code = re.sub(r"\.isnull\(\)\.size\b", ".isnull().sum()", code)
        code = re.sub(r"\.isna\(\)\.size\b", ".isna().sum()", code)
        code = re.sub(r"\.notnull\(\)\.size\b", ".notnull().sum()", code)
        code = re.sub(r"\.notna\(\)\.size\b", ".notna().sum()", code)
    if re.search(r"(?<!\.isnull\(\))(?<!\.isna\(\))(?<!\.notnull\(\))(?<!\.notna\(\))\.sum\s*\(", code or ""):
        return code, "Count intent detected but code uses sum()."
    return code, None


_ORDINAL_UA_ROOTS: Dict[str, int] = {
    # 1-10
    "перш": 1,
    "друг": 2,
    "трет": 3,
    "четверт": 4,
    "п'ят": 5,
    "пʼят": 5,
    "шост": 6,
    "сьом": 7,
    "восьм": 8,
    "дев'ят": 9,
    "девʼят": 9,
    "десят": 10,
    # десятки (20, 30, ..., 90)
    "двадцят": 20,
    "тридцят": 30,
    "сороков": 40,
    "п'ятдесят": 50,
    "пʼятдесят": 50,
    "шістдесят": 60,
    "сімдесят": 70,
    "вісімдесят": 80,
    "дев'яносто": 90,
    "девʼяносто": 90,
    # сотні (100, 200, ..., 900)
    "сот": 100,
    "двісті": 200,
    "трьохсот": 300,
    "чотирьохсот": 400,
    "п'ятисот": 500,
    "пʼятисот": 500,
    "шестисот": 600,
    "семисот": 700,
    "восьмисот": 800,
    "дев'ятисот": 900,
    "девʼятисот": 900,
}


def _match_column_by_index(text: str, columns: List[str]) -> Optional[str]:
    if not text or not columns:
        return None
    t = text.lower()
    m = re.search(r"(?:колонк|стовпц|стовпчик|стовпики|стовпця|стовпці|стовп|columns?)[^\d]*(\d+)", t, re.I)
    if m:
        idx = int(m.group(1))
        if 1 <= idx <= len(columns):
            return columns[idx - 1]
        return None
    m2 = re.search(
        r"(перш\w*|друг\w*|трет\w*|четверт\w*|п[ʼ']ят\w*|шост\w*|сьом\w*|восьм\w*|дев[ʼ']ят\w*|десят\w*|двадцят\w*|тридцят\w*|сороков\w*|шістдесят\w*|сімдесят\w*|вісімдесят\w*|дев[ʼ']яносто\w*|сот\w*|двісті\w*|трьохсот\w*|чотирьохсот\w*|п[ʼ']ятисот\w*|шестисот\w*|семисот\w*|восьмисот\w*|дев[ʼ']ятисот\w*)\s+(?:колонк\w*|стовпц\w*)",
        t,
        re.I,
    )
    if not m2:
        return None
    root = re.sub(r"[^a-zа-яіїєґʼ']", "", m2.group(1).lower())
    for prefix, value in _ORDINAL_UA_ROOTS.items():
        if root.startswith(prefix):
            if 1 <= value <= len(columns):
                return columns[value - 1]
            return None
    return None


def _column_mention_pos(text: str, column_name: str) -> int:
    """Return first mention position of a column name, avoiding short-token false positives inside larger tokens."""
    if not text or not column_name:
        return -1
    hay = str(text)
    lower = hay.lower()
    needle = str(column_name).strip().lower()
    if not needle:
        return -1

    # For short token-like names, require strict token boundaries.
    if re.fullmatch(r"[a-zа-яіїєґ0-9_]+", needle, re.I) and len(needle) <= 3:
        m = re.search(
            rf"(?<![a-zа-яіїєґ0-9_]){re.escape(needle)}(?![a-zа-яіїєґ0-9_])",
            lower,
            re.I,
        )
        return m.start() if m else -1

    # Try bounded match first for token-like names.
    if re.fullmatch(r"[a-zа-яіїєґ0-9_]+", needle, re.I):
        m = re.search(rf"\b{re.escape(needle)}\b", lower, re.I)
        if m:
            return m.start()

    m = re.search(re.escape(needle), lower, re.I)
    return m.start() if m else -1


def _find_column_in_text(text: str, columns: List[str]) -> Optional[str]:
    if not text or not columns:
        return None
    lower = text.lower()

    col = _match_column_by_index(text, columns)
    if col:
        return col
    best_raw: Optional[str] = None
    best_len = 0
    for name in columns:
        if not isinstance(name, str):
            continue
        n = name.strip()
        if not n:
            continue
        if _column_mention_pos(lower, n) >= 0:
            if len(n) > best_len:
                best_raw = name
                best_len = len(n)
    if best_raw is not None:
        return best_raw

    _STOP_ROOTS = {"кіль", "скіл", "count", "coun", "each", "per"}

    def _root_tokens(s: str) -> List[str]:
        parts = re.split(r"[^a-zа-яіїєґ0-9]+", (s or "").lower())
        roots = [p[:4] for p in parts if len(p) >= 3 and p[:4] not in _STOP_ROOTS]
        return roots

    q_roots = set(_root_tokens(lower))
    best_name: Optional[str] = None
    best_score = 0.0
    for name in columns:
        if not isinstance(name, str):
            continue
        roots = _root_tokens(name)
        if not roots or not q_roots:
            continue
        exact_hits = sum(1 for r in roots if r in q_roots)
        fuzzy_hits = 0
        for r in roots:
            if len(r) < 3:
                continue
            if any(len(qr) >= 3 and qr[:3] == r[:3] for qr in q_roots):
                fuzzy_hits += 1
        overlap = max(exact_hits, fuzzy_hits)
        if overlap <= 0:
            continue
        # Reward overlap, slightly penalize unmatched roots (e.g., *_UAH suffix tokens).
        score = (2.0 * overlap) - (0.35 * max(0, len(roots) - overlap))
        if score > best_score:
            best_score = score
            best_name = name
    if best_name:
        return best_name
    return None


def _find_explicit_column_in_text(text: str, columns: List[str]) -> Optional[str]:
    """
    Return a column only when it is explicitly mentioned in the query
    (or referenced by 1-based positional wording like "3rd column").
    """
    if not text or not columns:
        return None
    by_idx = _match_column_by_index(text, columns)
    if by_idx:
        return by_idx

    best_raw: Optional[str] = None
    best_len = 0
    for name in columns:
        if not isinstance(name, str):
            continue
        stripped = name.strip()
        if not stripped:
            continue
        if _column_mention_pos(text, stripped) >= 0 and len(stripped) > best_len:
            best_raw = name
            best_len = len(stripped)
    return best_raw


def _parse_row_index(text: str) -> Optional[int]:
    m = re.search(r"(?:рядок|рядка|рядку|рядком|рядці|row)[^\d]*(\d+)", text, re.I)
    if not m:
        return None
    idx = int(m.group(1))
    return idx if idx > 0 else None


def _parse_set_value(text: str) -> Optional[str]:
    s = (text or "").strip()
    if not s:
        return None

    # Prefer explicit numeric updates like "на 10 000 грн".
    m_num = re.search(r"(?:на|=)\s*(-?\d[\d\s\xa0]*(?:[.,]\d+)?)\s*(?:грн|uah|₴)?\b", s, re.I)
    if m_num:
        return m_num.group(1).strip()

    m_quoted = re.search(
        r"(?:на|=)\s*(?:(?:статус|status|значенн\w*|value)\s*)?(?:[:=-]\s*)?(?:['\"“”«»„`])([^'\"“”«»„`]+)(?:['\"“”«»„`])",
        s,
        re.I,
    )
    if m_quoted:
        return m_quoted.group(1).strip()

    m = re.search(r"(?:на|=)\s*([^\n]+)$", s, re.I)
    if not m:
        return None
    value = m.group(1).strip().strip("\"'“”«»„`")
    value = re.sub(r"^\s*(?:статус|status|значенн\w*|value)\s*[:=-]?\s*", "", value, flags=re.I)
    value = re.split(r"\s+(?:та|і|and)\s+", value, maxsplit=1, flags=re.I)[0]
    value = re.split(r"\s*\(", value, maxsplit=1)[0]
    return value.strip() or None


def _parse_number_literal(raw: str) -> Optional[float]:
    if raw is None:
        return None
    cleaned = re.sub(r"[\s\xa0]", "", str(raw)).replace(",", ".")
    if not re.match(r"^-?(\d+(\.\d*)?|\.\d+)$", cleaned):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_literal(raw: str) -> Any:
    if raw is None:
        return None
    low = str(raw).strip().lower()
    if low in ("null", "none", "порожньо", "пусто"):
        return None
    if low in ("true", "так", "yes"):
        return True
    if low in ("false", "ні", "no"):
        return False
    num = _parse_number_literal(raw)
    if num is not None:
        return num
    return str(raw).strip().strip("\"'")


def _parse_condition(text: str, columns: List[str]) -> Optional[Tuple[str, str, str, Optional[float]]]:
    low = (text or "").lower()
    if "де" not in low and "where" not in low:
        return None
    col = _find_column_in_text(text, columns)
    if not col:
        return None
    m = re.search(r"(>=|<=|!=|=|>|<|містить|contains)\s*([^\n]+)$", text, re.I)
    if not m:
        return None
    op = m.group(1).lower()
    value = m.group(2).strip().strip("\"'")
    num = _parse_number_literal(value) if op in (">", "<", ">=", "<=") else None
    return col, op, value, num


def _find_columns_in_text(text: str, columns: List[str]) -> List[str]:
    if not text or not columns:
        return []
    lower = text.lower()
    hits: List[Tuple[int, str]] = []
    def _roots_no_stopwords(s: str) -> set[str]:
        parts = re.split(r"[^a-zа-яіїєґ0-9]+", (s or "").lower())
        return {p[:4] for p in parts if len(p) >= 3}

    text_roots = _roots_no_stopwords(text)

    # 1) Exact mention first.
    for col in columns:
        if not isinstance(col, str):
            continue
        name = col.strip()
        if not name:
            continue
        pos = _column_mention_pos(text, name)
        if pos >= 0:
            hits.append((pos, str(col)))

    # 2) Fallback by semantic root overlap.
    if not hits and text_roots:
        for i, col in enumerate(columns):
            if not isinstance(col, str):
                continue
            name = col.strip()
            if not name:
                continue
            col_roots = _roots_no_stopwords(name)
            overlap = len(col_roots & text_roots)
            if overlap <= 0:
                continue
            # Keep stable order after exact matches.
            approx_pos = lower.find(name[:4].lower())
            if approx_pos < 0:
                approx_pos = 1000 + i
            hits.append((approx_pos, str(col)))

    hits.sort(key=lambda x: x[0])
    seen: set[str] = set()
    out: List[str] = []
    for _, col in hits:
        if col in seen:
            continue
        seen.add(col)
        out.append(col)
    return out


def _extract_top_n_from_question(question: str, default: int = 10) -> int:
    q = (question or "").lower()
    m = re.search(r"(?:top|топ)\s*[-–]?\s*(\d+)", q, re.I)
    if m:
        return max(1, int(m.group(1)))
    return default


def _classify_columns_by_role(
    question: str,
    found_columns: List[str],
    df_profile: Optional[dict],
) -> Dict[str, Optional[str]]:
    if not found_columns:
        return {"group_by": None, "aggregate": None}

    q_low = (question or "").lower()
    dtypes = (df_profile or {}).get("dtypes") or {}
    has_sum_context = bool(re.search(r"\b(сум\w*|загальн\w*|total|sum|обсяг|volume)\b", q_low))
    has_count_context = bool(re.search(r"\b(кільк\w*|скільк\w*|count|number|к-?ст)\b", q_low))

    aggregate_keywords = (
        "кільк",
        "qty",
        "amount",
        "sum",
        "сума",
        "price",
        "ціна",
        "варт",
        "total",
        "обсяг",
        "volume",
    )
    group_keywords = (
        "категор",
        "category",
        "бренд",
        "brand",
        "тип",
        "type",
        "status",
        "статус",
        "group",
        "груп",
        "клас",
    )

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for col in found_columns:
        dtype = str(dtypes.get(str(col), "")).lower()
        col_low = str(col).lower()
        is_numeric = dtype.startswith(("int", "float", "uint"))
        if any(k in col_low for k in aggregate_keywords):
            numeric_cols.append(col)
            continue
        if any(k in col_low for k in group_keywords):
            categorical_cols.append(col)
            continue
        if is_numeric:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    group_by = categorical_cols[0] if categorical_cols else None
    aggregate = numeric_cols[0] if numeric_cols else None

    if not group_by and found_columns:
        # Prefer first mentioned for grouping.
        group_by = found_columns[0]
    if not aggregate:
        # For sum/count-like asks choose another column, preferably numeric.
        if (has_sum_context or has_count_context) and len(found_columns) > 1:
            aggregate = next((c for c in found_columns if c != group_by), found_columns[0])
        elif has_sum_context or has_count_context:
            aggregate = found_columns[0]

    if aggregate == group_by:
        # Avoid same column for both roles if alternatives exist.
        alt = next((c for c in found_columns if c != group_by), None)
        if alt:
            aggregate = alt

    return {"group_by": group_by, "aggregate": aggregate}


def _excel_col_index(token: str) -> Optional[int]:
    token = (token or "").strip().upper()
    if not token.isalpha():
        return None
    if len(token) > 2:
        return None
    idx = 0
    for ch in token:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _columns_from_spec(spec: str, columns: List[str]) -> List[str]:
    if not spec or not columns:
        return []
    text = spec.strip()
    out: List[str] = []
    for part in re.split(r"[,\s]+", text):
        if not part:
            continue
        if re.search(r"[-–:]", part):
            m = re.split(r"[-–:]", part)
            if len(m) == 2:
                a = _excel_col_index(m[0])
                b = _excel_col_index(m[1])
                if a is not None and b is not None:
                    lo = min(a, b)
                    hi = max(a, b)
                    out.extend([columns[i] for i in range(lo, min(hi + 1, len(columns)))])
                    continue
        idx = _excel_col_index(part)
        if idx is not None and 0 <= idx < len(columns):
            out.append(columns[idx])
            continue
        col = _find_column_in_text(part, columns)
        if col:
            out.append(col)
    seen = set()
    uniq = []
    for col in out:
        if col in seen:
            continue
        seen.add(col)
        uniq.append(col)
    return uniq


def _parse_row_range(text: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"(?:рядки|рядків|rows?)\s*(\d+)\s*[-–]\s*(\d+)", text, re.I)
    if not m:
        return None
    a = int(m.group(1))
    b = int(m.group(2))
    if a <= 0 or b <= 0:
        return None
    return (min(a, b), max(a, b))


def _parse_position_index(text: str) -> Optional[int]:
    m = re.search(r"(?:позиці[яії]|position)\s*(\d+)", text, re.I)
    if not m:
        return None
    idx = int(m.group(1))
    return idx if idx > 0 else None


def _template_shortcut_code(
    question: str,
    profile: dict,
    allow_availability_shortcut: bool = True,
) -> Optional[Tuple[str, str]]:
    q = (question or "").strip()
    q_low = q.lower()
    if q.startswith("### Task:") and "<user_query>" not in q_low and "user query:" not in q_low:
        return None
    columns = (profile or {}).get("columns") or []
    if not columns:
        return None

    code_lines: List[str] = []
    plan = ""

    if allow_availability_shortcut and _is_availability_count_intent(q):
        target_mode = _availability_target_mode(q)
        code_lines.append(
            r"_pos_re = r'(?:в\s*наявн|наявн|in\s*stock|available|доступн)'"
        )
        code_lines.append(
            r"_neg_re = r'(?:нема|відсутн|out\s*of\s*stock|unavailable|not\s*available|не\s*доступн)'"
        )
        code_lines.append(f"_mode = {target_mode!r}")
        # Placeholder is resolved in _resolve_shortcut_placeholders() with LLM-first logic.
        code_lines.append(f"_col = {SHORTCUT_COL_PLACEHOLDER!r}")
        code_lines.append("if _col not in df.columns:")
        code_lines.append("    _col = None")
        code_lines.append("if _col is None:")
        code_lines.append("    _candidates = []")
        code_lines.append("    for _c in df.columns:")
        code_lines.append("        _s = df[_c].astype(str).str.strip().str.lower()")
        code_lines.append("        _hits = int((_s.str.contains(_pos_re, regex=True, na=False) | _s.str.contains(_neg_re, regex=True, na=False)).sum())")
        code_lines.append("        if _hits <= 0:")
        code_lines.append("            continue")
        code_lines.append("        _bonus = 1 if re.search(r'(статус|наявн|availability|available|stock|доступн)', str(_c).lower()) else 0")
        code_lines.append("        _candidates.append((_hits + _bonus, str(_c)))")
        code_lines.append("    if _candidates:")
        code_lines.append("        _candidates.sort(reverse=True)")
        code_lines.append("        _col = _candidates[0][1]")
        code_lines.append("if _col is not None:")
        code_lines.append("    _status = df[_col].astype(str).str.strip().str.lower()")
        code_lines.append("    _in = _status.str.contains(_pos_re, regex=True, na=False)")
        code_lines.append("    _out = _status.str.contains(_neg_re, regex=True, na=False)")
        code_lines.append("    if _mode == 'out':")
        code_lines.append("        result = int(df.loc[_out & ~_in].shape[0])")
        code_lines.append("    elif _mode == 'any':")
        code_lines.append("        result = int(df.loc[_in | _out].shape[0])")
        code_lines.append("    else:")
        code_lines.append("        result = int(df.loc[_in & ~_out].shape[0])")
        code_lines.append("else:")
        code_lines.append("    _row_text = pd.Series('', index=df.index, dtype='object')")
        code_lines.append("    for _c in df.columns:")
        code_lines.append("        _row_text = _row_text + ' ' + df[_c].fillna('').astype(str)")
        code_lines.append("    _row_text = _row_text.str.lower()")
        code_lines.append("    _in_row = _row_text.str.contains(_pos_re, regex=True, na=False)")
        code_lines.append("    _out_row = _row_text.str.contains(_neg_re, regex=True, na=False)")
        code_lines.append("    if _mode == 'out':")
        code_lines.append("        result = int((_out_row & ~_in_row).sum())")
        code_lines.append("    elif _mode == 'any':")
        code_lines.append("        result = int((_in_row | _out_row).sum())")
        code_lines.append("    else:")
        code_lines.append("        result = int((_in_row & ~_out_row).sum())")
        plan = (
            "Порахувати кількість товарів за ознаками наявності "
            f"у колонці {SHORTCUT_COL_PLACEHOLDER} або по значеннях рядка."
        )
        return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(покажи|показати|show)\b.*\b(клітин|комір|cell)\b", q_low):
        row_idx = _parse_row_index(q)
        col = _find_column_in_text(q, columns)
        if row_idx and col:
            code_lines.append(f"result = df.at[{row_idx - 1}, {col!r}]")
            plan = f"Показати значення клітинки в рядку {row_idx}, колонці {col}."
            return "\n".join(code_lines) + "\n", plan

    # Read-only robust extractor for "value in row N": try row position first, then fallback to identifier=N.
    if (
        not _has_edit_triggers(q)
        and re.search(r"\b(покажи|показати|show|яка|який|яке|which|what|значенн\w*)\b", q_low)
    ):
        row_idx = _parse_row_index(q)
        if row_idx:
            col = _pick_relevant_column(q, [str(c) for c in columns])
            if col:
                code_lines.append(f"_row_num = {row_idx}")
                code_lines.append("_pos = _row_num - 1")
                code_lines.append(f"_col = {col!r}")
                code_lines.append("if 0 <= _pos < len(df):")
                code_lines.append("    result = df.iloc[_pos][ _col ]")
                code_lines.append("else:")
                code_lines.append("    result = None")
                plan = f"Показати значення {col} для рядка {row_idx}."
                return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(очисти|clear)\b.*\b(клітин|комір|cell)\b", q_low):
        row_idx = _parse_row_index(q)
        col = _find_column_in_text(q, columns)
        if row_idx and col:
            code_lines.append("df = df.copy()")
            code_lines.append("df = df.reset_index(drop=True)")
            code_lines.append(f"df.at[{row_idx - 1}, {col!r}] = pd.NA")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Очистити клітинку в рядку {row_idx}, колонці {col}."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(заміни|replace)\b.*\b(діапазон|range)\b", q_low):
        row_range = _parse_row_range(q)
        m = re.search(r"(?:колонки|стовпці|columns?)\s+([A-Za-zА-Яа-я0-9_,:\-\s]+)", q, re.I)
        value = _parse_set_value(q)
        if row_range and m and value is not None:
            cols = _columns_from_spec(m.group(1), columns)
            if cols:
                r0, r1 = row_range
                c0 = columns.index(cols[0])
                c1 = columns.index(cols[-1])
                code_lines.append("df = df.copy()")
                code_lines.append(f"df.iloc[{r0 - 1}:{r1}, {c0}:{c1 + 1}] = {_parse_literal(value)!r}")
                code_lines.append("COMMIT_DF = True")
                code_lines.append("result = {'status': 'updated'}")
                plan = "Замінити значення в заданому діапазоні."
                return "\n".join(code_lines) + "\n", plan

    m = re.search(
        r"\b(додай|додати|add)\b\s+(?:колонку|стовпець)\s+([A-Za-zА-Яа-я0-9_ ]+)\s*=\s*([A-Za-zА-Яа-я0-9_ ]+)\s*([+\-*/x×])\s*([A-Za-zА-Яа-я0-9_ ]+)",
        q,
        re.I,
    )
    if m:
        new_col = m.group(2).strip()
        a = _find_column_in_text(m.group(3), columns)
        b = _find_column_in_text(m.group(5), columns)
        op = m.group(4)
        if a and b and new_col:
            code_lines.append("df = df.copy()")
            if op in ("x", "×"):
                op = "*"
            code_lines.append(f"df[{new_col!r}] = df[{a!r}] {op} df[{b!r}]")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Додати колонку {new_col} як обчислення з {a} та {b}."
            return "\n".join(code_lines) + "\n", plan

    m = re.search(
        r"\b(встав|insert)\b.*\b(колонку|стовпець)\b\s+([A-Za-zА-Яа-я0-9_ ]+)\s+.*\b(після|after)\b\s+(?:колонки|стовпця)\s+([A-Za-zА-Яа-я0-9_ ]+)",
        q,
        re.I,
    )
    if m:
        new_col = m.group(3).strip()
        after_col = _find_column_in_text(m.group(5), columns)
        if new_col and after_col:
            code_lines.append("df = df.copy()")
            code_lines.append(f"_loc = list(df.columns).index({after_col!r}) + 1")
            code_lines.append(f"df.insert(_loc, {new_col!r}, pd.NA)")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Вставити колонку {new_col} після {after_col}."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(додай|додати|add)\b.*\b(порожн|empty)\b.*\b(колонку|стовпець)\b", q_low):
        m = re.search(r"(?:колонку|стовпець)\s+([A-Za-zА-Яа-я0-9_ ]+)", q, re.I)
        if m:
            new_col = m.group(1).strip()
            code_lines.append("df = df.copy()")
            code_lines.append(f"df[{new_col!r}] = pd.NA")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Додати порожню колонку {new_col}."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(перемісти|move)\b.*\b(колонку|стовпець)\b.*\b(на початок|в початок|first)\b", q_low):
        col = _find_column_in_text(q, columns)
        if col:
            code_lines.append("df = df.copy()")
            code_lines.append("cols = list(df.columns)")
            code_lines.append(f"cols.insert(0, cols.pop(cols.index({col!r})))")
            code_lines.append("df = df[cols]")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Перемістити колонку {col} на початок."
            return "\n".join(code_lines) + "\n", plan

    m = re.search(r"\b(перейменуй|rename)\b.*\b(колонку|стовпець)\b\s+(.+?)\s+\bна\b\s+(.+)$", q, re.I)
    if m:
        old = _find_column_in_text(m.group(3), columns)
        new = m.group(4).strip().strip("\"'")
        if old and new:
            code_lines.append("df = df.copy()")
            code_lines.append(f"df = df.rename(columns={{ {old!r}: {new!r} }})")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Перейменувати колонку {old} на {new}."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(нижн.*регістр|lower case)\b", q_low):
        code_lines.append("df = df.copy()")
        code_lines.append("df.columns = [str(x).lower() for x in df.columns]")
        code_lines.append("COMMIT_DF = True")
        code_lines.append("result = {'status': 'updated'}")
        plan = "Перейменувати колонки у нижній регістр."
        return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(встав|insert)\b.*\b(рядок|row)\b.*\bпозиці", q_low):
        pos = _parse_position_index(q) or _parse_row_index(q)
        if pos:
            code_lines.append("df = df.copy()")
            code_lines.append(f"_pos = {pos - 1}")
            code_lines.append("top = df.iloc[:_pos]")
            code_lines.append("bot = df.iloc[_pos:]")
            code_lines.append("new_df = pd.DataFrame([{}])")
            code_lines.append("df = pd.concat([top, new_df, bot], ignore_index=True)")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Вставити порожній рядок на позицію {pos}."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(видали|delete)\b.*\b(рядки|rows?)\b", q_low):
        nums = [int(n) for n in re.findall(r"\b(\d+)\b", q)]
        nums = [n for n in nums if n > 0]
        if len(nums) >= 2 and ("і" in q_low or "," in q or "та" in q_low or "and" in q_low):
            idxs = [n - 1 for n in nums]
            code_lines.append("df = df.copy()")
            code_lines.append("df = df.reset_index(drop=True)")
            code_lines.append(f"df = df.drop(index={idxs!r})")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Видалити рядки з позиціями: {', '.join(str(n) for n in nums)}."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(видали|delete)\b.*\b(рядки|rows?)\b", q_low) and re.search(r"\b(де|where)\b", q_low):
        cond = _parse_condition(q, columns)
        if cond:
            col, op, value, num = cond
            code_lines.append("df = df.copy()")
            code_lines.append(f"_col = df[{col!r}]")
            if op in (">", "<", ">=", "<="):
                if num is None:
                    return None
                code_lines.append("_raw = _col.astype(str)")
                code_lines.append(
                    r"_clean = _raw.str.replace(r'[\s\xa0]', '', regex=True).str.replace(r'[^0-9,.\-]', '', regex=True).str.replace(',', '.')"
                )
                code_lines.append(r"_mask = _clean.str.match(r'^-?(\d+(\.\d*)?|\.\d+)$')")
                code_lines.append("_num = _clean.where(_mask, np.nan).astype(float)")
                code_lines.append(f"_value = {num!r}")
                code_lines.append(f"_cond = _num {op} _value")
            elif op in ("=", "!="):
                code_lines.append(f"_cond = _col.astype(str).str.strip() {op} {value!r}")
            else:
                code_lines.append(f"_cond = _col.astype(str).str.contains({value!r}, case=False, na=False)")
            code_lines.append("df = df.loc[~_cond].copy()")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Видалити рядки за умовою по {col}."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(видали|delete)\b.*\b(колонк|стовпц)", q_low):
        m = re.search(r"(?:колонки|стовпці)\s+(.+)$", q, re.I)
        cols = _columns_from_spec(m.group(1), columns) if m else []
        if not cols:
            col = _find_column_in_text(q, columns)
            cols = [col] if col else []
        if cols:
            code_lines.append("df = df.copy()")
            code_lines.append(f"df = df.drop(columns={cols!r})")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Видалити колонки: {', '.join(cols)}."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(сортуй|сортувати|sort)\b", q_low):
        cols = _find_columns_in_text(q, columns)
        if cols:
            desc = bool(re.search(r"\b(спад|desc)\b", q_low))
            if re.search(r"\b(всередині|then|second)\b", q_low) and len(cols) >= 2:
                ascending = [True] * len(cols)
                if desc:
                    ascending[-1] = False
                code_lines.append("df = df.copy()")
                code_lines.append(f"df = df.sort_values(by={cols!r}, ascending={ascending!r})")
                code_lines.append("COMMIT_DF = True")
                code_lines.append("result = {'status': 'updated'}")
                plan = f"Відсортувати за {', '.join(cols)}."
                return "\n".join(code_lines) + "\n", plan
            code_lines.append("df = df.copy()")
            code_lines.append(f"df = df.sort_values(by=[{cols[0]!r}], ascending={not desc})")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Відсортувати за {cols[0]}."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(відфільтр|filter)\b", q_low):
        cond = _parse_condition(q, columns)
        if cond:
            col, op, value, num = cond
            code_lines.append("df = df.copy()")
            if op in (">", "<", ">=", "<="):
                if num is None:
                    return None
                code_lines.append(f"_raw = df[{col!r}].astype(str)")
                code_lines.append(
                    r"_clean = _raw.str.replace(r'[\s\xa0]', '', regex=True).str.replace(r'[^0-9,.\-]', '', regex=True).str.replace(',', '.')"
                )
                code_lines.append(r"_mask = _clean.str.match(r'^-?(\d+(\.\d*)?|\.\d+)$')")
                code_lines.append("_col = _clean.where(_mask, np.nan).astype(float)")
                code_lines.append(f"df = df.loc[_col {op} {num!r}]")
            elif op in ("=", "!="):
                code_lines.append(f"df = df.loc[df[{col!r}].astype(str).str.strip() {op} {value!r}]")
            else:
                code_lines.append(f"df = df.loc[df[{col!r}].astype(str).str.contains({value!r}, na=False)]")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Відфільтрувати за умовою по {col}."
            return "\n".join(code_lines) + "\n", plan
        m = re.search(r"\b(в|in)\s*\[([^\]]+)\]", q)
        if m:
            col = _find_column_in_text(q, columns)
            if col:
                values = [v.strip().strip("\"'") for v in m.group(1).split(",") if v.strip()]
                code_lines.append("df = df.copy()")
                code_lines.append(f"df = df.loc[df[{col!r}].isin({values!r})]")
                code_lines.append("COMMIT_DF = True")
                code_lines.append("result = {'status': 'updated'}")
                plan = f"Відфільтрувати {col} по списку значень."
                return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(заміни|replace)\b.*\b(у всій таблиці|всій таблиці|entire table)\b", q_low):
        m = re.search(r"\b([^\s]+)\s+на\s+(NA|null|порожн|none)", q, re.I)
        if m:
            old = _parse_literal(m.group(1))
            code_lines.append("df = df.copy()")
            code_lines.append(f"df = df.replace({old!r}, pd.NA)")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = "Замінити значення у всій таблиці."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(знайди|replace)\b.*\b(заміни|replace)\b", q_low):
        m = re.search(r"'([^']+)'\s+на\s+'([^']+)'", q)
        col = _find_column_in_text(q, columns)
        if m and col:
            old, new = m.group(1), m.group(2)
            code_lines.append("df = df.copy()")
            code_lines.append(f"df[{col!r}] = df[{col!r}].astype(str).str.replace({old!r}, {new!r}, regex=False)")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Замінити '{old}' на '{new}' у колонці {col}."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(обріж|strip)\b.*\b(пробіли|spaces)\b", q_low):
        col = _find_column_in_text(q, columns)
        if col:
            code_lines.append("df = df.copy()")
            code_lines.append(f"df[{col!r}] = df[{col!r}].astype(str).str.strip()")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Прибрати пробіли по краях у колонці {col}."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(нижній|lower)\b.*\b(регістр|case)\b", q_low):
        col = _find_column_in_text(q, columns)
        if col:
            code_lines.append("df = df.copy()")
            code_lines.append(f"df[{col!r}] = df[{col!r}].astype(str).str.lower()")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Перевести значення колонки {col} у нижній регістр."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(заповни|fill)\b.*\b(порожн|na|null)\b", q_low):
        col = _find_column_in_text(q, columns)
        if col and re.search(r"\b(попередн|ffill)\b", q_low):
            code_lines.append("df = df.copy()")
            code_lines.append(f"df[{col!r}] = df[{col!r}].ffill()")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Заповнити пропуски в {col} попереднім значенням."
            return "\n".join(code_lines) + "\n", plan
        if col and re.search(r"\b(наступн|bfill)\b", q_low):
            code_lines.append("df = df.copy()")
            code_lines.append(f"df[{col!r}] = df[{col!r}].bfill()")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Заповнити пропуски в {col} наступним значенням."
            return "\n".join(code_lines) + "\n", plan
        if col:
            value = _parse_set_value(q)
            if value is not None:
                code_lines.append("df = df.copy()")
                code_lines.append(f"df[{col!r}] = df[{col!r}].fillna({_parse_literal(value)!r})")
                code_lines.append("COMMIT_DF = True")
                code_lines.append("result = {'status': 'updated'}")
                plan = f"Заповнити пропуски в {col} значенням {_parse_literal(value)!r}."
                return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(замін|імпут)\b.*\b(усіх числових|numeric)\b.*\b(середн|mean)\b", q_low):
        code_lines.append("df = df.copy()")
        code_lines.append("num_cols = df.select_dtypes(include='number').columns")
        code_lines.append("df[num_cols] = df[num_cols].fillna(df[num_cols].mean())")
        code_lines.append("COMMIT_DF = True")
        code_lines.append("result = {'status': 'updated'}")
        plan = "Замінити NA у числових колонках їхнім середнім."
        return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(дублік|duplicate)\b", q_low) and re.search(r"\b(познач|mark)\b", q_low):
        cols = _find_columns_in_text(q, columns)
        m = re.search(r"\b(у|в)\s+новій\s+колонці\s+([A-Za-zА-Яа-я0-9_ ]+)", q, re.I)
        new_col = m.group(2).strip() if m else "is_dup"
        if cols:
            code_lines.append("df = df.copy()")
            code_lines.append(f"df[{new_col!r}] = df.duplicated(subset={cols!r}, keep='first')")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Позначити дублікати у колонці {new_col}."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(дублік|duplicate)\b", q_low) and re.search(r"\b(видали|drop)\b", q_low):
        cols = _find_columns_in_text(q, columns)
        if cols:
            code_lines.append("df = df.copy()")
            code_lines.append(f"df = df.drop_duplicates(subset={cols!r}, keep='first')")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Видалити дублікати за колонками {', '.join(cols)}."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(унікальн|unique)\b", q_low):
        col = _find_column_in_text(q, columns)
        if col:
            code_lines.append(f"result = df[{col!r}].dropna().unique().tolist()")
            plan = f"Показати унікальні значення у {col}."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(скільк\w*|кільк\w*|count)\b", q_low) and re.search(r"\b(кожн\w*|each|per)\b", q_low):
        col = _find_column_in_text(q, columns)
        if col:
            code_lines.append(f"result = df[{col!r}].value_counts(dropna=False).to_dict()")
            plan = f"Порахувати частоти значень у {col}."
            return "\n".join(code_lines) + "\n", plan
        code_lines.append(
            f"result = df.groupby({SHORTCUT_COL_PLACEHOLDER!r}).size().reset_index(name='count').sort_values('count', ascending=False)"
        )
        plan = f"Порахувати кількість рядків для кожного значення у колонці {SHORTCUT_COL_PLACEHOLDER}."
        return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(перетвори|convert)\b.*\b(у число|numeric)\b", q_low):
        col = _find_column_in_text(q, columns)
        if col:
            code_lines.append("df = df.copy()")
            code_lines.append(f"_raw = df[{col!r}].astype(str)")
            code_lines.append(
                r"_clean = _raw.str.replace(r'[\s\xa0]', '', regex=True).str.replace(r'[^0-9,.\-]', '', regex=True).str.replace(',', '.')"
            )
            code_lines.append(r"_mask = _clean.str.match(r'^-?(\d+(\.\d*)?|\.\d+)$')")
            code_lines.append(f"df[{col!r}] = _clean.where(_mask, np.nan).astype(float)")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Перетворити {col} у число."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(перетвори|convert)\b.*\b(у дату|datetime|date)\b", q_low):
        col = _find_column_in_text(q, columns)
        dayfirst = bool(re.search(r"\b(dayfirst|дд|день)\b", q_low))
        if col:
            code_lines.append("df = df.copy()")
            code_lines.append(f"df[{col!r}] = pd.to_datetime(df[{col!r}], errors='coerce', dayfirst={dayfirst})")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Перетворити {col} у дату."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(округли|round)\b", q_low):
        col = _find_column_in_text(q, columns)
        m = re.search(r"\bдо\s+(\d+)\s*(знаків|digits)", q_low)
        n = int(m.group(1)) if m else 2
        if col:
            code_lines.append("df = df.copy()")
            code_lines.append(f"_raw = df[{col!r}].astype(str)")
            code_lines.append(
                r"_clean = _raw.str.replace(r'[\s\xa0]', '', regex=True).str.replace(r'[^0-9,.\-]', '', regex=True).str.replace(',', '.')"
            )
            code_lines.append(r"_mask = _clean.str.match(r'^-?(\d+(\.\d*)?|\.\d+)$')")
            code_lines.append(f"df[{col!r}] = _clean.where(_mask, np.nan).astype(float).round({n})")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Округлити {col} до {n} знаків."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(обмеж|clip)\b.*\b(не менше|lower)\b", q_low):
        col = _find_column_in_text(q, columns)
        if col:
            code_lines.append("df = df.copy()")
            code_lines.append(f"_raw = df[{col!r}].astype(str)")
            code_lines.append(
                r"_clean = _raw.str.replace(r'[\s\xa0]', '', regex=True).str.replace(r'[^0-9,.\-]', '', regex=True).str.replace(',', '.')"
            )
            code_lines.append(r"_mask = _clean.str.match(r'^-?(\d+(\.\d*)?|\.\d+)$')")
            code_lines.append(f"df[{col!r}] = _clean.where(_mask, np.nan).astype(float).clip(lower=0)")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Обмежити {col} значеннями не менше 0."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(зведен|pivot)\b", q_low):
        cols = _find_columns_in_text(q, columns)
        if len(cols) >= 3 and re.search(r"\b(та|and)\b", q_low):
            value_col = cols[0]
            index_col = cols[1]
            columns_col = cols[2]
            agg = "sum" if re.search(r"\b(sum|сума)\b", q_low) else "mean"
            code_lines.append(
                f"result = df.pivot_table(index={index_col!r}, columns={columns_col!r}, values={value_col!r}, aggfunc={agg!r})"
            )
            plan = f"Побудувати зведену {agg} для {value_col} по {index_col} та {columns_col}."
            return "\n".join(code_lines) + "\n", plan
        if len(cols) >= 2:
            value_col = cols[0]
            index_col = cols[-1]
            agg = "sum" if re.search(r"\b(sum|сума)\b", q_low) else "mean"
            code_lines.append(f"result = df.pivot_table(index={index_col!r}, values={value_col!r}, aggfunc={agg!r})")
            plan = f"Побудувати зведену таблицю {agg} для {value_col} по {index_col}."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(згрупуй|group)\b", q_low):
        cols = _find_columns_in_text(q, columns)
        if cols:
            key = cols[-1]
            if re.search(r"\b(кільк|count)\b", q_low):
                code_lines.append(f"result = df.groupby({key!r}).size().reset_index(name='count')")
                plan = f"Порахувати кількість рядків по {key}."
                return "\n".join(code_lines) + "\n", plan
            if re.search(r"\b(sum|сума)\b", q_low) and len(cols) >= 2:
                value_col = cols[0]
                code_lines.append(f"result = df.groupby({key!r})[{value_col!r}].sum().reset_index()")
                plan = f"Порахувати суму {value_col} по {key}."
                return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(по місяц|monthly|months)\b", q_low) and re.search(r"\b(sum|сума)\b", q_low):
        cols = _find_columns_in_text(q, columns)
        if len(cols) >= 2:
            value_col = cols[0]
            date_col = cols[-1]
            code_lines.append("df = df.copy()")
            code_lines.append(f"df[{date_col!r}] = pd.to_datetime(df[{date_col!r}], errors='coerce')")
            code_lines.append(
                f"result = df.groupby(pd.Grouper(key={date_col!r}, freq='M'))[{value_col!r}].sum().reset_index()"
            )
            plan = f"Порахувати суму {value_col} по місяцях для {date_col}."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(rozbiy|split)\b.*\b(на)\b", q_low):
        col = _find_column_in_text(q, columns)
        if col:
            m = re.search(r"на\s+([A-Za-zА-Яа-я0-9_]+)\s+та\s+([A-Za-zА-Яа-я0-9_]+)", q, re.I)
            if m:
                c1, c2 = m.group(1), m.group(2)
                code_lines.append("df = df.copy()")
                code_lines.append(f"parts = df[{col!r}].astype(str).str.split(' ', n=1, expand=True)")
                code_lines.append(f"df[{c1!r}] = parts[0]")
                code_lines.append(f"df[{c2!r}] = parts[1]")
                code_lines.append("COMMIT_DF = True")
                code_lines.append("result = {'status': 'updated'}")
                plan = f"Розбити {col} на {c1} та {c2}."
                return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(склей|concat)\b.*\b(колонки|стовпці)\b", q_low):
        cols = _find_columns_in_text(q, columns)
        m = re.search(r"у\s+колонку\s+([A-Za-zА-Яа-я0-9_]+)", q, re.I)
        if len(cols) >= 2 and m:
            new_col = m.group(1).strip()
            code_lines.append("df = df.copy()")
            code_lines.append(
                f"df[{new_col!r}] = df[{cols[0]!r}].astype(str).str.cat(df[{cols[1]!r}].astype(str), sep=' ')"
            )
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Склеїти {cols[0]} та {cols[1]} у {new_col}."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(витягни|extract)\b.*\b(домен|domain)\b", q_low):
        col = _find_column_in_text(q, columns)
        if col:
            m = re.search(r"у\s+колонку\s+([A-Za-zА-Яа-я0-9_]+)", q, re.I)
            new_col = m.group(1).strip() if m else "domain"
            code_lines.append("df = df.copy()")
            code_lines.append(f"df[{new_col!r}] = df[{col!r}].astype(str).str.extract(r'@(.+)$', expand=False)")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Витягнути домен з {col} у {new_col}."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(додай|add)\b.*\b(year|рік)\b.*\b(з дати|date)\b", q_low):
        col = _find_column_in_text(q, columns)
        if col:
            m = re.search(r"колонку\s+([A-Za-zА-Яа-я0-9_]+)", q, re.I)
            new_col = m.group(1).strip() if m else "year"
            code_lines.append("df = df.copy()")
            code_lines.append(f"df[{new_col!r}] = pd.to_datetime(df[{col!r}], errors='coerce').dt.year")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Додати колонку {new_col} з року дати {col}."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(ковзн|rolling)\b.*\b(середн|mean)\b", q_low):
        cols = _find_columns_in_text(q, columns)
        if cols:
            value_col = cols[0]
            date_col = cols[-1] if len(cols) > 1 else None
            if date_col:
                m = re.search(r"(\d+)\s*дн", q_low)
                window = int(m.group(1)) if m else 7
                new_col = f"{value_col}_rolling_{window}"
                code_lines.append("df = df.copy()")
                code_lines.append(f"df[{date_col!r}] = pd.to_datetime(df[{date_col!r}], errors='coerce')")
                code_lines.append(f"df = df.sort_values({date_col!r})")
                code_lines.append(f"df[{new_col!r}] = df[{value_col!r}].rolling({window}).mean()")
                code_lines.append("COMMIT_DF = True")
                code_lines.append("result = {'status': 'updated'}")
                plan = f"Обчислити ковзне середнє {window} для {value_col} по {date_col}."
                return "\n".join(code_lines) + "\n", plan

    return None

def _edit_shortcut_code(question: str, profile: dict) -> Optional[Tuple[str, str]]:
    q=(question or "").strip()
    q_low=q.lower()
    if q.startswith("### Task:") and "<user_query>" not in q_low and "user query:" not in q_low:
        return None

    if re.search(r"\b(undo|відміни|скасуй|відкот|поверни)\b", q_low):
        code="\n".join(["result = {'status': 'undo'}","UNDO = True"])+"\n"
        return code, "Відкотити останню зміну таблиці."

    columns=(profile or {}).get("columns") or []
    if not columns: return None

    # Edit by identifier column (e.g., ID/SKU/код/артикул): "зміни ... для ідентифікатора 1003 на ..."
    if _has_edit_triggers(q):
        m_id = re.search(r"\b(?:id|sku|код|артикул)\s*[:=]?\s*([A-Za-zА-Яа-яІіЇїЄєҐґ0-9._\-]+)\b", q, re.I)
        if m_id:
            item_id = (m_id.group(1) or "").strip()
            col = _find_column_in_text(q, columns)
            raw_value = _parse_set_value(q)
            value = _parse_literal(raw_value) if raw_value is not None else None
            id_col = _pick_id_like_column([str(c) for c in columns])
            if col and value is not None and id_col and str(col) != str(id_col):
                code_lines: List[str] = []
                code_lines.append("df = df.copy()")
                code_lines.append(f"_id = {item_id!r}")
                code_lines.append(f"_id_column = {id_col!r}")
                code_lines.append("if _id_column in df.columns:")
                code_lines.append("    _id_target = str(_id).strip()")
                code_lines.append("    _id_target_norm = _id_target")
                code_lines.append("    while _id_target_norm.endswith('.0'):")
                code_lines.append("        _id_target_norm = _id_target_norm[:-2]")
                code_lines.append("    _id_col = df[_id_column].astype(str).str.strip()")
                code_lines.append("    _id_col_norm = _id_col.str.replace(r'\\.0+$', '', regex=True)")
                code_lines.append("    _mask = (_id_col == _id_target) | (_id_col_norm == _id_target_norm)")
                code_lines.append("    if _mask.any():")
                code_lines.append(f"        df.loc[_mask, {col!r}] = {value!r}")
                code_lines.append("        COMMIT_DF = True")
                code_lines.append(f"        result = {{'status': 'updated', 'id': _id, 'column': {col!r}, 'new_value': {value!r}}}")
                code_lines.append("    else:")
                code_lines.append("        result = {'status': 'not_found', 'id': _id}")
                code_lines.append("else:")
                code_lines.append("    result = {'status': 'no_id_column'}")
                plan = f"Змінити {col} для запису з ідентифікатором {item_id}."
                return "\n".join(code_lines) + "\n", plan

    is_add_row=bool(re.search(r"\b(додай|додати|add)\b.*\b(рядок|row)\b", q_low))
    is_del_row=bool(re.search(r"\b(видали|видалити|delete)\b.*\b(рядок|row)\b", q_low))
    is_add_col=bool(re.search(r"\b(додай|додати|add)\b.*\b(стовпец|стовпець|колонк|column)\b", q_low))
    is_del_col=bool(re.search(r"\b(видали|видалити|delete)\b.*\b(стовпец|стовпець|колонк|column)\b", q_low))
    is_edit_cell=bool(re.search(r"\b(зміни|змініть|редагуй|поміняй)\b.*\b(клітин|комір|cell)\b", q_low))
    if not is_edit_cell and _has_edit_triggers(q):
        row_idx_hint = _parse_row_index(q)
        col_hint = _find_column_in_text(q, columns)
        value_hint = _parse_set_value(q)
        if row_idx_hint and col_hint and value_hint is not None:
            is_edit_cell = True

    code_lines: List[str]=[]
    plan=""

    if is_add_col:
        col_name=_find_column_in_text(q, columns) or "new_column"
        if col_name in columns: col_name=f"{col_name}_new"
        raw_value=_parse_set_value(q)
        value=_parse_literal(raw_value) if raw_value is not None else None
        code_lines.append("df = df.copy()")
        if value is None or value=="": code_lines.append(f"df[{col_name!r}] = None")
        else: code_lines.append(f"df[{col_name!r}] = {value!r}")
        plan=f"Додати стовпець {col_name}."

    elif is_del_col:
        col_name=_find_column_in_text(q, columns)
        if not col_name: return None
        code_lines.append("df = df.copy()")
        code_lines.append(f"df = df.drop(columns=[{col_name!r}])")
        plan=f"Видалити стовпець {col_name}."

    elif is_add_row:
        m=re.findall(r"([A-Za-zА-Яа-я0-9_ ]+?)\s*[:=]\s*([^,;]+)", q)
        row={}
        for key,val in m:
            col=_find_column_in_text(key, columns)
            if col: row[col]=_parse_literal(val)
        code_lines.append("df = df.copy()")
        if row:
            code_lines.append(f"_row = {row!r}")
            code_lines.append("df = pd.concat([df, pd.DataFrame([_row])], ignore_index=True)")
        else:
            code_lines.append("df = pd.concat([df, pd.DataFrame([{}])], ignore_index=True)")
        plan="Додати рядок."

    elif is_del_row:
        row_idx=_parse_row_index(q)
        cond=_parse_condition(q, columns)
        code_lines.append("df = df.copy()")
        code_lines.append("df = df.reset_index(drop=True)")
        if row_idx:
            code_lines.append(f"df = df.drop(index=[{row_idx - 1}])")
            plan=f"Видалити рядок {row_idx}."
        elif cond:
            col,op,value,num=cond
            code_lines.append(f"_col = df[{col!r}]")
            if op in (">","<",">=","<="):
                if num is None: return None
                code_lines.append("_raw = _col.astype(str)")
                code_lines.append(r"_clean = _raw.str.replace(r'[\s\xa0]', '', regex=True).str.replace(r'[^0-9,.\-]', '', regex=True).str.replace(',', '.')")
                code_lines.append(r"_mask = _clean.str.match(r'^-?(\d+(\.\d*)?|\.\d+)$')")
                code_lines.append("_num = _clean.where(_mask, np.nan).astype(float)")
                code_lines.append(f"_value = {num!r}")
                code_lines.append(f"_cond = _num {op} _value")
            elif op in ("=","!="):
                code_lines.append(f"_cond = _col.astype(str).str.strip() {op} {value!r}")
            else:
                code_lines.append(f"_cond = _col.astype(str).str.contains({value!r}, case=False, na=False)")
            code_lines.append("df = df.loc[~_cond].copy()")
            plan=f"Видалити рядки за умовою по {col}."
        else:
            return None

    elif is_edit_cell:
        row_idx=_parse_row_index(q)
        col=_find_column_in_text(q, columns)
        raw_value=_parse_set_value(q)
        value=_parse_literal(raw_value) if raw_value is not None else None
        if not row_idx or not col or value is None: return None
        code_lines.append("df = df.copy()")
        code_lines.append("df = df.reset_index(drop=True)")
        code_lines.append(f"df.at[{row_idx - 1}, {col!r}] = {value!r}")
        plan=f"Змінити значення в рядку {row_idx}, колонці {col}."

    else:
        return None

    if not re.search(r"(?m)^\s*COMMIT_DF\s*=", "\n".join(code_lines)):
        code_lines.append("COMMIT_DF = True")
    if not re.search(r"(?m)^\s*result\s*=", "\n".join(code_lines)):
        code_lines.append("result = {'status': 'updated'}")

    return "\n".join(code_lines) + "\n", plan

def _choose_column_from_question(question: str, profile: dict) -> Optional[str]:
    cols = (profile or {}).get("columns") or []
    if not cols:
        return None
    q = (question or "").lower()
    # Intent-aware bias for common numeric semantics.
    if re.search(r"\b(варт|price|cost|amount|вируч|сума)\w*", q):
        pref = _pick_price_like_column(profile)
        if pref:
            return pref
    if re.search(r"\b(кільк|qty|quantity|units|stock|залишк)\w*", q):
        pref = _pick_quantity_like_column(profile)
        if pref:
            return pref
    picked = _pick_relevant_column(question, [str(c) for c in cols])
    if picked:
        return picked
    dtypes = (profile or {}).get("dtypes") or {}
    for col in cols:
        dtype = str(dtypes.get(str(col), "")).lower()
        if dtype.startswith(("int", "float", "uint")) and not _is_id_like_col_name(str(col)):
            return col
    for col in cols:
        dtype = str(dtypes.get(str(col), "")).lower()
        if dtype.startswith(("int", "float", "uint")):
            return col
    return cols[0]


def _stats_shortcut_code(
    question: str,
    profile: dict,
    preferred_col: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    q = (question or "").lower()
    if (question or "").lstrip().startswith("### Task:") and "<user_query>" not in q and "user query:" not in q:
        return None
    wants_min = bool(re.search(r"\b(min|мінімал|мінімум)\b", q))
    wants_max = bool(re.search(r"\b(max|максимал|максимум)\b", q))
    wants_mean = bool(re.search(r"\b(mean|average|avg|середн\w*)\b", q))
    wants_sum = bool(re.search(r"\b(sum|сума)\b", q))
    wants_count = bool(re.search(r"\b(count|кільк|кількість)\b", q))
    wants_median = bool(re.search(r"\b(median|медіан)\b", q))
    # Total inventory shortcut is only valid for plain total/sum intent.
    if not (wants_min or wants_max or wants_mean or wants_count or wants_median):
        total_value_shortcut = _total_inventory_value_shortcut_code(question, profile)
        if total_value_shortcut:
            return total_value_shortcut
    if not (wants_min or wants_max or wants_mean or wants_sum or wants_count or wants_median):
        return None
    cols = [str(c) for c in ((profile or {}).get("columns") or [])]
    col: Optional[str] = None
    if preferred_col and preferred_col in cols:
        # Prefer numeric target from LLM sloting; fallback logic handles non-numeric safely.
        col = preferred_col
    if not col:
        col = _choose_column_from_question(question, profile)
    if not col:
        return None
    plan_parts = []
    if wants_min:
        plan_parts.append("мінімум")
    if wants_max:
        plan_parts.append("максимум")
    if wants_mean:
        plan_parts.append("середнє")
    if wants_sum:
        plan_parts.append("суму")
    if wants_count:
        plan_parts.append("кількість")
    if wants_median:
        plan_parts.append("медіану")
    plan = f"Порахувати {', '.join(plan_parts)} для колонки {col} після приведення значень до числового типу."
    code_lines = []
    code_lines.append(f"_col = df[{col!r}]")
    code_lines.append("_dtype = str(_col.dtype).lower()")
    code_lines.append("if _dtype.startswith(('int', 'float', 'uint')):")
    code_lines.append("    _num = _col.astype(float)")
    code_lines.append("else:")
    code_lines.append("    _raw = _col.astype(str)")
    code_lines.append(
        r"    _clean = _raw.str.replace(r'[\s\xa0]', '', regex=True).str.replace(r'[^0-9,.\-]', '', regex=True).str.replace(',', '.')"
    )
    code_lines.append(r"    _mask = _clean.str.match(r'^-?(\d+(\.\d*)?|\.\d+)$')")
    code_lines.append("    _num = _clean.where(_mask, np.nan).astype(float)")
    code_lines.append(
        "print(f\"num_col source=%s dtype=%s nan_count=%d\" % (" + repr(col) + ", str(_num.dtype), int(_num.isna().sum())))"
    )
    code_lines.append("_min = _num.min()")
    code_lines.append("_max = _num.max()")
    code_lines.append("_mean = _num.mean()")
    code_lines.append("_sum = _num.sum()")
    code_lines.append("_count = _num.count()")
    code_lines.append("_median = _num.median()")
    code_lines.append("result = {}")
    if wants_min:
        code_lines.append("result['min'] = None if pd.isna(_min) else float(_min)")
    if wants_max:
        code_lines.append("result['max'] = None if pd.isna(_max) else float(_max)")
    if wants_mean:
        code_lines.append("result['mean'] = None if pd.isna(_mean) else float(_mean)")
    if wants_sum:
        code_lines.append("result['sum'] = None if pd.isna(_sum) else float(_sum)")
    if wants_count:
        code_lines.append("result['count'] = int(_count)")
    if wants_median:
        code_lines.append("result['median'] = None if pd.isna(_median) else float(_median)")
    return "\n".join(code_lines) + "\n", plan


def _emit_status(event_emitter: Any, description: str, done: bool = False, hidden: bool = False) -> None:
    if not event_emitter:
        return
    payload = {"type": "status", "data": {"description": description, "done": done, "hidden": hidden}}
    try:
        if hasattr(event_emitter, "emit"):
            event_emitter.emit(
                "status", {"description": description, "message": description, "done": done, "hidden": hidden}
            )
            return
        if callable(event_emitter):
            res = event_emitter(payload)
            if inspect.isawaitable(res):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(res)
                else:
                    loop.create_task(res)
    except Exception:
        pass


_STATUS_MARKER_PREFIX = "[[PIPELINE_STATUS:"
_STATUS_MARKER_SUFFIX = "]]"


def _status_marker(description: str, done: bool = False, hidden: bool = False) -> str:
    payload = {"description": description, "done": done, "hidden": hidden}
    return f"{_STATUS_MARKER_PREFIX}{json.dumps(payload, ensure_ascii=True)}{_STATUS_MARKER_SUFFIX}"


def _guess_filename(meta: dict) -> str:
    for key in ("filename", "name", "path"):
        value = (meta or {}).get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _status_message(event: str, payload: Optional[dict]) -> str:
    if event == "start":
        return "Стартую обробку запиту."
    if event == "file_id":
        file_id = (payload or {}).get("file_id")
        if file_id:
            return f"Файл знайдено: {file_id}"
        return "Файл знайдено."
    if event == "no_file":
        return "Файл не знайдено у запиті."
    if event == "fetch_meta":
        return "Отримую метадані файлу."
    if event == "fetch_bytes":
        return "Завантажую файл."
    if event == "sandbox_load":
        return "Завантажую таблицю в sandbox."
    if event == "sandbox_load_failed":
        return "Не вдалося завантажити таблицю в sandbox."
    if event == "codegen":
        return "Генерую план та код аналізу."
    if event == "codegen_shortcut":
        return "Використовую швидкий режим для генерації коду."
    if event == "codegen_rlm_tool":
        phase = str((payload or {}).get("phase") or "").strip()
        if phase == "runtime_error":
            return "Повторно генерую код за помилкою виконання (RLM-tool)."
        if phase == "core_repl":
            return "Запускаю рекурсивний RLM-REPL цикл для виправлення коду."
        return "Генерую код через RLM-tool і перевіряю обмеження."
    if event == "codegen_retry":
        reason = str((payload or {}).get("reason") or "").strip()
        if reason == "missing_subset_filter":
            return "Повторно генерую код: додам обов'язкову фільтрацію підмножини."
        if reason == "subset_guard_conflict_non_subset":
            return "Повторно генерую код: subset-фільтр не потрібен, залишаю агрегацію по всій таблиці."
        if reason == "read_mutation":
            return "Повторно генерую код: прибираю модифікації таблиці для read-запиту."
        if reason == "runtime_keyerror_import":
            return "Повторно генерую код: виправляю помилку виконання."
        return "Повторно генерую код: у попередній версії не було присвоєння в result."
    if event == "codegen_empty":
        return "Не вдалося згенерувати код аналізу."
    if event == "sandbox_run":
        return "Виконую аналіз у sandbox."
    if event == "wait":
        label = (payload or {}).get("label") or ""
        seconds = (payload or {}).get("seconds")
        if label == "codegen":
            base = "Генерую план та код аналізу."
        elif label == "sandbox_run":
            base = "Виконую аналіз у sandbox."
        elif label == "final_answer":
            base = "Формую відповідь."
        else:
            base = "Обробляю запит."
        if isinstance(seconds, int) and seconds > 0:
            return f"{base} Минуло ~{seconds}с."
        return base
    if event == "final_answer":
        status = (payload or {}).get("status")
        if status:
            return f"Готово. Відповідь сформована (статус: {status})."
        return "Готово. Відповідь сформована."
    if event == "error":
        err = (payload or {}).get("error")
        if err:
            return f"Помилка: {err}"
        return "Сталася помилка."
    return _safe_trunc(f"{event}: {payload}", 200)


class Pipeline(object):
    class Valves(BaseModel):
        id: str = Field(default=os.getenv("PIPELINE_ID", "spreadsheet-analyst"))
        name: str = Field(default=os.getenv("PIPELINE_NAME", "Spreadsheet Analyst"))
        description: str = Field(
            default=os.getenv(
                "PIPELINE_DESCRIPTION", "Upload CSV/XLSX and ask questions; pandas runs in a sandbox."
            )
        )
        debug: bool = Field(default=os.getenv("PIPELINE_DEBUG", "").lower() in ("1", "true", "yes", "on"))

        webui_base_url: str = Field(default=os.getenv("WEBUI_BASE_URL", "http://host.docker.internal:3000"))
        webui_api_key: str = Field(default=os.getenv("WEBUI_API_KEY", ""))

        base_llm_base_url: str = Field(
            default=os.getenv("BASE_LLM_BASE_URL", "https://ai-gateway-test.noone.pw/v1")
        )
        base_llm_api_key: str = Field(default=os.getenv("BASE_LLM_API_KEY", "sk-bf-91983094-c759-4f74-b765-b794731b47ca"))
        base_llm_model: str = Field(default=os.getenv("BASE_LLM_MODEL", "qwen3-coder-next"))
        base_llm_timeout_s: int = Field(default=_env_int("BASE_LLM_TIMEOUT_S", 45), ge=1)
        base_llm_max_retries: int = Field(default=_env_int("BASE_LLM_MAX_RETRIES", 1), ge=0)
        llm_json_max_tokens: int = Field(default=_env_int("LLM_JSON_MAX_TOKENS", 256), ge=64)
        rlm_codegen_enabled: bool = Field(
            default=os.getenv("RLM_CODEGEN_ENABLED", "true").lower() in ("1", "true", "yes", "on")
        )
        rlm_primary_planner_enabled: bool = Field(
            default=os.getenv("RLM_PRIMARY_PLANNER_ENABLED", "true").lower() in ("1", "true", "yes", "on")
        )
        rlm_codegen_max_turns: int = Field(default=_env_int("RLM_CODEGEN_MAX_TURNS", 2), ge=1)
        rlm_codegen_max_tokens: int = Field(default=_env_int("RLM_CODEGEN_MAX_TOKENS", 700), ge=128)
        rlm_codegen_runtime_retry_enabled: bool = Field(
            default=os.getenv("RLM_CODEGEN_RUNTIME_RETRY_ENABLED", "true").lower() in ("1", "true", "yes", "on")
        )
        rlm_core_repl_enabled: bool = Field(
            default=os.getenv("RLM_CORE_REPL_ENABLED", "true").lower() in ("1", "true", "yes", "on")
        )
        rlm_core_repl_max_iterations: int = Field(
            default=_env_int("RLM_CORE_REPL_MAX_ITERATIONS", 2),
            ge=1,
        )
        rlm_core_repl_max_tokens: int = Field(default=_env_int("RLM_CORE_REPL_MAX_TOKENS", 700), ge=128)
        final_answer_max_tokens: int = Field(default=_env_int("FINAL_ANSWER_MAX_TOKENS", 384), ge=64)

        sandbox_url: str = Field(default=os.getenv("SANDBOX_URL", "http://sandbox:8081"))
        sandbox_api_key: str = Field(default=os.getenv("SANDBOX_API_KEY", ""))

        max_rows: int = Field(default=_env_int("PIPELINE_MAX_ROWS", 200000), ge=1)
        preview_rows: int = Field(default=_env_int("PIPELINE_PREVIEW_ROWS", 200000), ge=1)
        max_cell_chars: int = Field(default=_env_int("PIPELINE_MAX_CELL_CHARS", 200), ge=10)
        code_timeout_s: int = Field(default=_env_int("PIPELINE_CODE_TIMEOUT_S", 120), ge=1)
        max_stdout_chars: int = Field(default=_env_int("PIPELINE_MAX_STDOUT_CHARS", 8000), ge=1000)
        answer_table_max_rows: int = Field(default=_env_int("ANSWER_TABLE_MAX_ROWS", 0), ge=0)
        answer_list_max_items: int = Field(default=_env_int("ANSWER_LIST_MAX_ITEMS", 30), ge=1)
        answer_pairs_max_rows: int = Field(default=_env_int("ANSWER_PAIRS_MAX_ROWS", 0), ge=0)

        session_cache_ttl_s: int = Field(default=_env_int("PIPELINE_SESSION_CACHE_TTL_S", 1800), ge=60)
        wait_tick_s: int = Field(default=_env_int("PIPELINE_WAIT_TICK_S", 5), ge=0)
        route_trace_enabled: bool = Field(
            default=os.getenv("ROUTE_TRACE_ENABLED", "true").lower() in ("1", "true", "yes", "on")
        )
        route_trace_sink_url: str = Field(default=os.getenv("ROUTE_TRACE_SINK_URL", ""))
        route_trace_sink_api_key: str = Field(default=os.getenv("ROUTE_TRACE_SINK_API_KEY", ""))
        route_trace_public_url: str = Field(
            default=os.getenv("ROUTE_TRACE_PUBLIC_URL", "http://localhost:8081/v1/traces/dashboard")
        )
        route_trace_include_link: bool = Field(
            default=os.getenv("ROUTE_TRACE_INCLUDE_LINK", "true").lower() in ("1", "true", "yes", "on")
        )
        route_trace_max_payload_chars: int = Field(
            default=_env_int("ROUTE_TRACE_MAX_PAYLOAD_CHARS", 4000),
            ge=256,
        )
        route_trace_local_path: str = Field(default=os.getenv("ROUTE_TRACE_LOCAL_PATH", ""))
        spreadsheet_skill_runtime_enabled: bool = Field(
            default=os.getenv("SPREADSHEET_SKILL_RUNTIME_ENABLED", "true").lower() in ("1", "true", "yes", "on")
        )
        spreadsheet_skill_force_on_all_queries: bool = Field(
            default=os.getenv("SPREADSHEET_SKILL_FORCE_ON_ALL_QUERIES", "true").lower()
            in ("1", "true", "yes", "on")
        )
        spreadsheet_skill_dir: str = Field(
            default=os.getenv("SPREADSHEET_SKILL_DIR", DEFAULT_SPREADSHEET_SKILL_DIR)
        )
        spreadsheet_skill_max_chars: int = Field(default=_env_int("SPREADSHEET_SKILL_MAX_CHARS", 16000), ge=1000)

        shortcut_enabled: bool = Field(
            default=os.getenv("SHORTCUT_ENABLED", "true").lower() in ("1", "true", "yes", "on")
        )
        shortcut_catalog_path: str = Field(
            default=os.getenv("SHORTCUT_CATALOG_PATH", "sandbox_service/catalog.json")
        )
        shortcut_index_path: str = Field(
            default=os.getenv("SHORTCUT_INDEX_PATH", "retrieval_index/index.faiss")
        )
        shortcut_meta_path: str = Field(default=os.getenv("SHORTCUT_META_PATH", "retrieval_index/meta.json"))
        shortcut_top_k: int = Field(default=_env_int("SHORTCUT_TOP_K", 5), ge=1)
        shortcut_threshold: float = Field(default=float(os.getenv("SHORTCUT_THRESHOLD", "0.35")))
        shortcut_margin: float = Field(default=float(os.getenv("SHORTCUT_MARGIN", "0.05")))
        shortcut_llm_intent_min_confidence: float = Field(
            default=float(os.getenv("SHORTCUT_LLM_INTENT_MIN_CONFIDENCE", "0.45"))
        )
        shortcut_llm_intent_max_candidates: int = Field(
            default=_env_int("SHORTCUT_LLM_INTENT_MAX_CANDIDATES", 8),
            ge=1,
        )
        shortcut_query_ir_enabled: bool = Field(
            default=os.getenv("SHORTCUT_QUERY_IR_ENABLED", "true").lower() in ("1", "true", "yes", "on")
        )
        shortcut_query_ir_llm_enabled: bool = Field(
            default=os.getenv("SHORTCUT_QUERY_IR_LLM_ENABLED", "true").lower() in ("1", "true", "yes", "on")
        )
        shortcut_query_ir_require_hard_coverage: bool = Field(
            default=os.getenv("SHORTCUT_QUERY_IR_REQUIRE_HARD_COVERAGE", "true").lower()
            in ("1", "true", "yes", "on")
        )
        shortcut_query_ir_block_soft_promotion: bool = Field(
            default=os.getenv("SHORTCUT_QUERY_IR_BLOCK_SOFT_PROMOTION", "true").lower()
            in ("1", "true", "yes", "on")
        )
        shortcut_query_ir_llm_verify_enabled: bool = Field(
            default=os.getenv("SHORTCUT_QUERY_IR_LLM_VERIFY_ENABLED", "false").lower()
            in ("1", "true", "yes", "on")
        )
        shortcut_query_ir_llm_verify_fail_open: bool = Field(
            default=os.getenv("SHORTCUT_QUERY_IR_LLM_VERIFY_FAIL_OPEN", "true").lower()
            in ("1", "true", "yes", "on")
        )
        shortcut_learning_enabled: bool = Field(
            default=os.getenv("SHORTCUT_LEARNING_ENABLED", "true").lower() in ("1", "true", "yes", "on")
        )
        shortcut_learning_pending_path: str = Field(
            default=os.getenv(
                "SHORTCUT_LEARNING_PENDING_PATH",
                "pipelines/shortcut_router/learning/pending_success_queries.jsonl",
            )
        )
        shortcut_learning_min_support: int = Field(
            default=_env_int("SHORTCUT_LEARNING_MIN_SUPPORT", 2),
            ge=2,
        )
        shortcut_learning_consistency_ratio: float = Field(
            default=float(os.getenv("SHORTCUT_LEARNING_CONSISTENCY_RATIO", "0.8"))
        )
        shortcut_learning_max_pending: int = Field(
            default=_env_int("SHORTCUT_LEARNING_MAX_PENDING", 4000),
            ge=100,
        )
        shortcut_learning_max_code_chars: int = Field(
            default=_env_int("SHORTCUT_LEARNING_MAX_CODE_CHARS", 1600),
            ge=200,
        )
        shortcut_learning_max_examples_per_intent: int = Field(
            default=_env_int("SHORTCUT_LEARNING_MAX_EXAMPLES_PER_INTENT", 300),
            ge=20,
        )
        shortcut_learning_max_cases_per_intent: int = Field(
            default=_env_int("SHORTCUT_LEARNING_MAX_CASES_PER_INTENT", 200),
            ge=10,
        )
        shortcut_learning_promote_every: int = Field(
            default=_env_int("SHORTCUT_LEARNING_PROMOTE_EVERY", 1),
            ge=1,
        )
        shortcut_learning_llm_judge_enabled: bool = Field(
            default=os.getenv("SHORTCUT_LEARNING_LLM_JUDGE_ENABLED", "false").lower()
            in ("1", "true", "yes", "on")
        )
        shortcut_learning_llm_judge_min_score: float = Field(
            default=float(os.getenv("SHORTCUT_LEARNING_LLM_JUDGE_MIN_SCORE", "0.90"))
        )
        shortcut_learning_llm_judge_promote_min_score: float = Field(
            default=float(os.getenv("SHORTCUT_LEARNING_LLM_JUDGE_PROMOTE_MIN_SCORE", "0.90"))
        )
        shortcut_learning_llm_judge_fail_open: bool = Field(
            default=os.getenv("SHORTCUT_LEARNING_LLM_JUDGE_FAIL_OPEN", "true").lower()
            in ("1", "true", "yes", "on")
        )
        shortcut_learning_llm_judge_max_result_chars: int = Field(
            default=_env_int("SHORTCUT_LEARNING_LLM_JUDGE_MAX_RESULT_CHARS", 1200),
            ge=200,
        )
        shortcut_debug_trace_enabled: bool = Field(
            default=os.getenv("SHORTCUT_DEBUG_TRACE_ENABLED", "true").lower() in ("1", "true", "yes", "on")
        )
        shortcut_debug_trace_path: str = Field(
            default=os.getenv(
                "SHORTCUT_DEBUG_TRACE_PATH",
                "pipelines/shortcut_router/learning/debug_trace.jsonl",
            )
        )
        shortcut_debug_trace_max_rows: int = Field(
            default=_env_int("SHORTCUT_DEBUG_TRACE_MAX_ROWS", 2000),
            ge=100,
        )
        shortcut_debug_trace_max_text_chars: int = Field(
            default=_env_int("SHORTCUT_DEBUG_TRACE_MAX_TEXT_CHARS", 6000),
            ge=500,
        )

        vllm_base_url: str = Field(
            default=os.getenv(
                "EMB_BASE_URL", os.getenv("VLLM_BASE_URL", "http://alph-gpu.silly.billy:8022/v1")
            )
        )
        vllm_embed_model: str = Field(
            default=os.getenv("EMB_MODEL", os.getenv("VLLM_EMBED_MODEL", "multilingual-embeddings"))
        )
        vllm_api_key: str = Field(
            default=os.getenv("EMB_API_KEY", os.getenv("VLLM_API_KEY", "DUMMY_KEY"))
        )
        vllm_timeout_s: int = Field(default=_env_int("VLLM_TIMEOUT_S", 30), ge=1)

    api_version: ClassVar[str] = "v1"

    def __init__(self) -> None:
        self.valves = self.Valves()
        logging.basicConfig(level=logging.INFO)
        root_logger = logging.getLogger()
        if not any(isinstance(f, _RequestTraceLoggingFilter) for f in root_logger.filters):
            root_logger.addFilter(_RequestTraceLoggingFilter())
        self._llm = OpenAI(
            base_url=self.valves.base_llm_base_url,
            api_key=self.valves.base_llm_api_key or "DUMMY_KEY",
            timeout=float(self.valves.base_llm_timeout_s),
            max_retries=int(self.valves.base_llm_max_retries),
        )
        logging.info(
            "event=shortcut_paths enabled=%s catalog_path=%s index_path=%s meta_path=%s "
            "catalog_exists=%s index_exists=%s meta_exists=%s",
            bool(self.valves.shortcut_enabled),
            self.valves.shortcut_catalog_path,
            self.valves.shortcut_index_path,
            self.valves.shortcut_meta_path,
            os.path.exists(self.valves.shortcut_catalog_path),
            os.path.exists(self.valves.shortcut_index_path),
            os.path.exists(self.valves.shortcut_meta_path),
        )
        if self.valves.debug:
            logging.info(
                "event=embeddings_config url=%s model=%s api_key_set=%s timeout_s=%s",
                self.valves.vllm_base_url,
                self.valves.vllm_embed_model,
                bool(self.valves.vllm_api_key),
                self.valves.vllm_timeout_s,
            )
        self._session_cache: Dict[str, Dict[str, Any]] = {}
        self._prompts = _read_prompts(PROMPTS_PATH)
        self._skill_prompt_cache: Dict[str, Dict[str, str]] = {}
        self._skill_prompt_lock = threading.Lock()
        self._learning_lock = threading.Lock()
        self._shortcut_debug_trace_lock = threading.Lock()
        self._learning_record_count = 0
        router_cfg = ShortcutRouterConfig(
            catalog_path=self.valves.shortcut_catalog_path,
            index_path=self.valves.shortcut_index_path,
            meta_path=self.valves.shortcut_meta_path,
            top_k=self.valves.shortcut_top_k,
            threshold=float(self.valves.shortcut_threshold),
            margin=float(self.valves.shortcut_margin),
            llm_intent_min_confidence=float(self.valves.shortcut_llm_intent_min_confidence),
            llm_intent_max_candidates=int(self.valves.shortcut_llm_intent_max_candidates),
            vllm_base_url=self.valves.vllm_base_url,
            vllm_embed_model=self.valves.vllm_embed_model,
            vllm_api_key=self.valves.vllm_api_key,
            vllm_timeout_s=self.valves.vllm_timeout_s,
            enabled=bool(self.valves.shortcut_enabled),
            query_ir_enabled=bool(self.valves.shortcut_query_ir_enabled),
            query_ir_llm_enabled=bool(self.valves.shortcut_query_ir_llm_enabled),
            query_ir_require_hard_coverage=bool(self.valves.shortcut_query_ir_require_hard_coverage),
            query_ir_block_soft_promotion=bool(self.valves.shortcut_query_ir_block_soft_promotion),
            query_ir_llm_verify_enabled=bool(self.valves.shortcut_query_ir_llm_verify_enabled),
            query_ir_llm_verify_fail_open=bool(self.valves.shortcut_query_ir_llm_verify_fail_open),
        )
        self._shortcut_router = ShortcutRouter(router_cfg, llm_json=self._llm_json)

    def pipelines(self) -> List[dict]:
        return [{"id": self.valves.id, "name": self.valves.name, "description": self.valves.description}]

    def _webui_headers(self) -> dict:
        headers = {"Accept": "application/json"}
        if self.valves.webui_api_key:
            headers["Authorization"] = f"Bearer {self.valves.webui_api_key}"
        return headers

    def _sandbox_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.valves.sandbox_api_key:
            headers["Authorization"] = f"Bearer {self.valves.sandbox_api_key}"
        return headers

    def _route_trace_sink_url(self) -> str:
        configured = str(self.valves.route_trace_sink_url or "").strip()
        if configured:
            return configured
        return f"{self.valves.sandbox_url.rstrip('/')}/v1/traces/upsert"

    def _new_route_tracer(self, request_id: str, trace_id: str, mode: str) -> Optional[RouteTracer]:
        if not bool(self.valves.route_trace_enabled):
            return None
        try:
            return RouteTracer(
                request_id=request_id,
                trace_id=trace_id,
                sink_url=self._route_trace_sink_url(),
                sink_api_key=str(self.valves.route_trace_sink_api_key or self.valves.sandbox_api_key or "").strip(),
                max_payload_chars=int(self.valves.route_trace_max_payload_chars),
                persist_path=str(self.valves.route_trace_local_path or "").strip(),
                meta={
                    "mode": mode,
                    "pipeline_id": self.valves.id,
                    "pipeline_name": self.valves.name,
                    "model": self.valves.base_llm_model,
                },
            )
        except Exception as exc:
            logging.warning("event=route_trace_init_failed error=%s", _safe_trunc(str(exc), 200))
            return None

    def _build_route_trace_url(self, trace_id: str, request_id: str) -> str:
        base = str(self.valves.route_trace_public_url or "").strip()
        if not base:
            return ""
        sep = "&" if "?" in base else "?"
        return f"{base}{sep}trace_id={trace_id}&request_id={request_id}"

    def _append_route_trace_link(self, text: str, trace_id: str, request_id: str) -> str:
        # User-facing answers should not include internal trace links.
        if not (text or "").strip():
            return text
        cleaned = re.sub(
            r"\n*\s*View route trace:\s*https?://\S+\s*",
            "",
            text or "",
            flags=re.I,
        )
        return cleaned.rstrip()

    def _sandbox_get_profile(self, df_id: str) -> Optional[dict]:
        tracer = current_route_tracer()
        stage_id = ""
        if tracer:
            stage_id = tracer.start_stage(
                stage_key="sandbox_get_profile",
                stage_name="Sandbox Profile Refresh",
                purpose="Refresh DataFrame profile for a cached sandbox dataframe id.",
                input_payload={"df_id": df_id},
                processing_summary="Call sandbox /v1/dataframe/{df_id}/profile endpoint.",
            )
        url = f"{self.valves.sandbox_url.rstrip('/')}/v1/dataframe/{df_id}/profile"
        try:
            resp = requests.get(url, headers=self._sandbox_headers(), timeout=10)
            if resp.status_code != 200:
                if tracer and stage_id:
                    tracer.end_stage(
                        stage_id,
                        status="warn",
                        output_payload={"status_code": resp.status_code},
                        processing_summary="Sandbox profile endpoint returned non-200 status.",
                    )
                return None
            payload = resp.json()
        except Exception as exc:
            if tracer and stage_id:
                tracer.end_stage(
                    stage_id,
                    status="warn",
                    output_payload={},
                    processing_summary="Failed to refresh sandbox profile.",
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
            return None
        profile = payload.get("profile") or {}
        if tracer and stage_id:
            tracer.end_stage(
                stage_id,
                status="ok",
                output_payload={"profile": profile},
                processing_summary="Sandbox profile refreshed from active dataframe.",
            )
        return profile

    def _fetch_file_meta(self, file_id: str, file_obj: Optional[dict]) -> dict:
        tracer = current_route_tracer()
        stage_id = ""
        if tracer:
            stage_id = tracer.start_stage(
                stage_key="file_meta_fetch",
                stage_name="File Metadata Fetch",
                purpose="Resolve file metadata used for downstream sandbox load.",
                input_payload={"file_id": file_id, "file_obj": file_obj or {}},
                processing_summary="Read metadata from current payload or OpenWebUI file API.",
            )
        try:
            if isinstance(file_obj, dict):
                if (
                    file_obj.get("filename")
                    or file_obj.get("content_type")
                    or file_obj.get("name")
                    or file_obj.get("path")
                ):
                    if tracer and stage_id:
                        tracer.end_stage(
                            stage_id,
                            status="ok",
                            output_payload={"meta_source": "request_file_obj", "meta": file_obj},
                            processing_summary="Metadata resolved directly from request payload.",
                        )
                    return file_obj
            url = f"{self.valves.webui_base_url.rstrip('/')}/api/v1/files/{file_id}"
            resp = requests.get(url, headers=self._webui_headers(), timeout=DEF_TIMEOUT_S)
            resp.raise_for_status()
            if resp.headers.get("content-type", "").startswith("application/json"):
                meta = resp.json()
                if tracer and stage_id:
                    tracer.end_stage(
                        stage_id,
                        status="ok",
                        output_payload={"meta_source": "webui_api", "meta": meta},
                        processing_summary="Metadata fetched from OpenWebUI JSON endpoint.",
                    )
                return meta
            raw_meta = {"raw": resp.text}
            if tracer and stage_id:
                tracer.end_stage(
                    stage_id,
                    status="warn",
                    output_payload={"meta_source": "webui_api_text", "meta": raw_meta},
                    processing_summary="Metadata endpoint returned non-JSON payload.",
                )
            return raw_meta
        except Exception as exc:
            if tracer and stage_id:
                tracer.end_stage(
                    stage_id,
                    status="error",
                    output_payload={},
                    processing_summary="Failed to fetch file metadata.",
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
            raise

    def _fetch_file_bytes(self, file_id: str, meta: dict, file_obj: Optional[dict]) -> bytes:
        tracer = current_route_tracer()
        stage_id = ""
        if tracer:
            stage_id = tracer.start_stage(
                stage_key="file_bytes_fetch",
                stage_name="File Bytes Fetch",
                purpose="Retrieve uploaded file bytes for sandbox ingestion.",
                input_payload={"file_id": file_id, "meta": meta or {}, "file_obj": file_obj or {}},
                processing_summary="Resolve bytes from inline base64, URL, path, or OpenWebUI content endpoint.",
            )
        try:
            if isinstance(file_obj, dict):
                b64 = file_obj.get("data_b64") or file_obj.get("data") or file_obj.get("content_b64")
                if b64:
                    payload = base64.b64decode(b64)
                    if tracer and stage_id:
                        tracer.end_stage(
                            stage_id,
                            status="ok",
                            output_payload={"bytes_len": len(payload), "source": "inline_base64"},
                            processing_summary="File bytes decoded from inline base64 payload.",
                        )
                    return payload

                url = file_obj.get("url") or file_obj.get("content_url")
                if url:
                    if url.startswith("/"):
                        url = f"{self.valves.webui_base_url.rstrip('/')}{url}"
                    resp = requests.get(url, headers=self._webui_headers(), timeout=DEF_TIMEOUT_S)
                    resp.raise_for_status()
                    payload = resp.content
                    if tracer and stage_id:
                        tracer.end_stage(
                            stage_id,
                            status="ok",
                            output_payload={"bytes_len": len(payload), "source": "content_url"},
                            processing_summary="File bytes downloaded from content URL.",
                        )
                    return payload

                path = file_obj.get("path") or file_obj.get("filepath") or file_obj.get("file_path")
                if path and os.path.exists(path):
                    with open(path, "rb") as f:
                        payload = f.read()
                    if tracer and stage_id:
                        tracer.end_stage(
                            stage_id,
                            status="ok",
                            output_payload={"bytes_len": len(payload), "source": "local_path"},
                            processing_summary="File bytes loaded from filesystem path reference.",
                        )
                    return payload

            base = self.valves.webui_base_url.rstrip("/")
            url = f"{base}/api/v1/files/{file_id}/content"
            resp = requests.get(url, headers=self._webui_headers(), timeout=DEF_TIMEOUT_S)
            resp.raise_for_status()
            payload = resp.content
            if tracer and stage_id:
                tracer.end_stage(
                    stage_id,
                    status="ok",
                    output_payload={"bytes_len": len(payload), "source": "webui_content_api"},
                    processing_summary="File bytes downloaded from OpenWebUI content endpoint.",
                )
            return payload
        except Exception as exc:
            if tracer and stage_id:
                tracer.end_stage(
                    stage_id,
                    status="error",
                    output_payload={},
                    processing_summary="Failed to fetch file bytes.",
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
            raise

    def _session_get(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self._session_cache.get(key)
        if not entry:
            return None
        if time.time() - entry.get("ts", 0) > self.valves.session_cache_ttl_s:
            self._session_cache.pop(key, None)
            return None
        return entry

    def _session_set(self, key: str, file_id: str, df_id: str, profile: dict) -> None:
        profile_fp = _profile_fingerprint(profile)
        self._session_cache[key] = {
            "file_id": file_id,
            "df_id": df_id,
            "profile": profile,
            "profile_fp": profile_fp,
            "ts": time.time(),
        }

    def _apply_dynamic_limits(self, profile: dict) -> None:
        rows = (profile or {}).get("rows")
        if not isinstance(rows, int) or rows <= 0:
            return
        new_max = max(1, int(rows * 1.2))
        self.valves.max_rows = new_max
        if self.valves.preview_rows > new_max:
            self.valves.preview_rows = new_max

    def _intent_from_plan(self, plan: str) -> Optional[str]:
        m = re.search(r"\bretrieval_intent:([A-Za-z0-9_.:-]+)\b", str(plan or ""))
        if not m:
            return None
        intent_id = str(m.group(1) or "").strip()
        return intent_id or None

    def _selector_mode_from_plan(self, plan: str) -> str:
        m = re.search(r"\bselector_mode:([A-Za-z0-9_.:-]+)\b", str(plan or ""))
        if not m:
            return ""
        return str(m.group(1) or "").strip().lower()

    def _result_non_empty(self, result_text: str, result_meta: Dict[str, Any]) -> bool:
        if str(result_text or "").strip():
            return True
        rows = (result_meta or {}).get("rows")
        if isinstance(rows, int):
            return rows > 0
        return False

    def _coerce_unit_score(self, value: Any) -> Optional[float]:
        try:
            score = float(value)
        except Exception:
            return None
        if score != score:  # NaN guard
            return None
        return max(0.0, min(1.0, score))

    def _median_score(self, values: List[float]) -> Optional[float]:
        if not values:
            return None
        items = sorted(float(v) for v in values)
        n = len(items)
        mid = n // 2
        if n % 2 == 1:
            return float(items[mid])
        return float((items[mid - 1] + items[mid]) / 2.0)

    def _llm_judge_learning_candidate(
        self,
        *,
        question: str,
        intent_id: str,
        analysis_code: str,
        result_text: str,
        result_meta: Dict[str, Any],
    ) -> Tuple[Optional[float], str, str]:
        """
        Returns: (score, reason, status_code)
        status_code in {"disabled","ok","invalid","error"}.
        """
        if not bool(self.valves.shortcut_learning_llm_judge_enabled):
            return None, "llm_judge_disabled", "disabled"

        max_result_chars = int(self.valves.shortcut_learning_llm_judge_max_result_chars)
        system = (
            "You are judging whether generated pandas analysis code matches the spreadsheet user intent. "
            "Return ONLY JSON: "
            "{\"score\": 0..1, \"reason\": \"<short_reason>\", \"decision\": \"accept|reject\"}. "
            "Score near 1.0 only when constraints and intent are preserved with high confidence."
        )
        payload = {
            "question": _normalize_query_text(question or ""),
            "intent_id": str(intent_id or ""),
            "analysis_code": _safe_trunc(str(analysis_code or ""), int(self.valves.shortcut_learning_max_code_chars)),
            "result_text_preview": _safe_trunc(str(result_text or ""), max_result_chars),
            "result_meta": result_meta or {},
        }
        try:
            parsed = self._llm_json(system, json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            return None, f"llm_judge_error:{_safe_trunc(str(exc), 120)}", "error"
        if not isinstance(parsed, dict):
            return None, "llm_judge_invalid_response", "invalid"
        score = self._coerce_unit_score(parsed.get("score"))
        reason = str(parsed.get("reason") or "").strip() or "llm_judge_no_reason"
        if score is None:
            return None, reason, "invalid"
        return score, reason, "ok"

    def _json_safe(self, value: Any) -> Any:
        try:
            return json.loads(json.dumps(value, ensure_ascii=False, default=str))
        except Exception:
            return _safe_trunc(str(value), 1200)

    def _record_llm_call_stat(self, item: Dict[str, Any]) -> None:
        stats = _LLM_CALL_STATS_CTX.get()
        if stats is None:
            return
        stats.append(self._json_safe(item))

    def _llm_call_usage_summary(self) -> Dict[str, Any]:
        calls = _LLM_CALL_STATS_CTX.get() or []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_latency_ms = 0.0
        has_prompt = False
        has_completion = False
        has_total = False

        for call in calls:
            if not isinstance(call, dict):
                continue
            p = _safe_int(call.get("prompt_tokens"))
            c = _safe_int(call.get("completion_tokens"))
            t = _safe_int(call.get("total_tokens"))
            lat = call.get("latency_ms")

            if p is not None:
                total_prompt_tokens += p
                has_prompt = True
            if c is not None:
                total_completion_tokens += c
                has_completion = True
            if t is not None:
                total_tokens += t
                has_total = True
            else:
                if p is not None and c is not None:
                    total_tokens += p + c
                    has_total = True
            try:
                if lat is not None:
                    total_latency_ms += float(lat)
            except Exception:
                pass

        return {
            "calls_count": len(calls),
            "prompt_tokens": total_prompt_tokens if has_prompt else None,
            "completion_tokens": total_completion_tokens if has_completion else None,
            "total_tokens": total_tokens if has_total else None,
            "total_latency_ms": round(total_latency_ms, 3) if calls else None,
            "calls": calls[-20:],
        }

    def _maybe_record_shortcut_debug_trace(
        self,
        *,
        mode: str,
        question: str,
        router_meta: Optional[Dict[str, Any]],
        analysis_code: str,
        run_status: str,
        result_text: str,
        result_meta: Dict[str, Any],
        final_answer: str,
        error: str = "",
        learning_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not bool(self.valves.shortcut_debug_trace_enabled):
            return
        path = str(self.valves.shortcut_debug_trace_path or "").strip()
        if not path:
            return

        text_limit = int(self.valves.shortcut_debug_trace_max_text_chars)
        meta = router_meta if isinstance(router_meta, dict) else {}
        learning = learning_meta if isinstance(learning_meta, dict) else {}
        llm_usage = self._llm_call_usage_summary()
        retrieval_candidates = meta.get("retrieval_candidates")
        if isinstance(retrieval_candidates, list):
            retrieval_candidates = retrieval_candidates[:12]
        else:
            retrieval_candidates = []
        entry = {
            "ts": int(time.time()),
            "request_id": _REQUEST_ID_CTX.get(),
            "trace_id": _TRACE_ID_CTX.get(),
            "mode": str(mode or ""),
            "question": _normalize_query_text(question or ""),
            "intent_id": str(meta.get("intent_id") or ""),
            "selector_mode": str(meta.get("selector_mode") or ""),
            "score": meta.get("score"),
            "selector_confidence": meta.get("selector_confidence"),
            "retrieval_query_used": str(meta.get("retrieval_query_used") or ""),
            "normalized_query": str(meta.get("normalized_query") or ""),
            "retrieval_threshold": meta.get("retrieval_threshold"),
            "retrieval_margin": meta.get("retrieval_margin"),
            "retrieval_candidate_count": meta.get("retrieval_candidate_count"),
            "retrieval_top_score": meta.get("retrieval_top_score"),
            "retrieval_second_score": meta.get("retrieval_second_score"),
            "retrieval_candidates": self._json_safe(retrieval_candidates),
            "query_ir": self._json_safe(meta.get("query_ir") or {}),
            "query_ir_summary": self._json_safe(meta.get("query_ir_summary") or {}),
            "slots": self._json_safe(meta.get("slots") or {}),
            "result_rows": (result_meta or {}).get("rows"),
            "run_status": str(run_status or ""),
            "error": _safe_trunc(str(error or ""), max(200, text_limit // 2)),
            "analysis_code": _safe_trunc(str(analysis_code or ""), text_limit),
            "result_text": _safe_trunc(str(result_text or ""), text_limit),
            "final_answer": _safe_trunc(str(final_answer or ""), text_limit),
            "llm_judge_enabled": bool(learning.get("llm_judge_enabled", False)),
            "llm_judge_status": str(learning.get("llm_judge_status") or ""),
            "llm_judge_score": learning.get("llm_judge_score"),
            "llm_judge_reason": _safe_trunc(str(learning.get("llm_judge_reason") or ""), 300),
            "learning_recorded": bool(learning.get("learning_recorded", False)),
            "llm_calls_count": llm_usage.get("calls_count"),
            "llm_prompt_tokens": llm_usage.get("prompt_tokens"),
            "llm_completion_tokens": llm_usage.get("completion_tokens"),
            "llm_total_tokens": llm_usage.get("total_tokens"),
            "llm_total_latency_ms": llm_usage.get("total_latency_ms"),
            "llm_calls": self._json_safe(llm_usage.get("calls") or []),
        }
        try:
            with self._shortcut_debug_trace_lock:
                rows = _read_jsonl(path)
                rows.append(entry)
                max_rows = int(self.valves.shortcut_debug_trace_max_rows)
                if len(rows) > max_rows:
                    rows = rows[-max_rows:]
                _atomic_write_jsonl(path, rows)
        except Exception as exc:
            logging.warning("event=shortcut_debug_trace_record_failed error=%s", _safe_trunc(str(exc), 300))

    def _catalog_has_query(self, catalog: Dict[str, Any], query_norm: str) -> bool:
        if not query_norm:
            return False
        for intent in (catalog.get("intents") or []):
            for ex in (intent.get("examples") or []):
                if _normalize_learning_query(str(ex or "")) == query_norm:
                    return True
            for case in (intent.get("learned_cases") or []):
                if not isinstance(case, dict):
                    continue
                if _normalize_learning_query(str(case.get("query") or "")) == query_norm:
                    return True
        return False

    def _append_index_row_for_learned_query(self, intent_id: str, query: str) -> bool:
        try:
            if not self._shortcut_router._ensure_loaded():
                return False
            index = getattr(self._shortcut_router, "_index", None)
            meta = getattr(self._shortcut_router, "_meta", None)
            if index is None or not isinstance(meta, dict):
                return False
            rows = meta.get("rows")
            if not isinstance(rows, list):
                rows = []
                meta["rows"] = rows

            q_norm = _normalize_learning_query(query)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if (
                    str(row.get("intent_id") or "").strip() == intent_id
                    and _normalize_learning_query(str(row.get("text") or "")) == q_norm
                ):
                    return True

            vec = self._shortcut_router._embed_query(query)
            if vec is None:
                return False
            index.add(vec)
            rows.append({"intent_id": intent_id, "text": query})

            # Persist index and meta atomically.
            import faiss  # type: ignore

            os.makedirs(os.path.dirname(self.valves.shortcut_index_path) or ".", exist_ok=True)
            faiss.write_index(index, self.valves.shortcut_index_path)
            _atomic_write_json(self.valves.shortcut_meta_path, meta)
            return True
        except Exception as exc:
            logging.warning("event=learning_index_append_failed error=%s", _safe_trunc(str(exc), 300))
            return False

    def _promote_success_learning_locked(self) -> Dict[str, Any]:
        pending_path = str(self.valves.shortcut_learning_pending_path or "").strip()
        if not pending_path:
            return {"promoted": 0}
        pending_rows = _read_jsonl(pending_path)
        if not pending_rows:
            return {"promoted": 0}

        catalog_path = str(self.valves.shortcut_catalog_path or "").strip()
        catalog = _read_json_or_default(catalog_path, {})
        intents = catalog.get("intents") or []
        if not isinstance(intents, list) or not intents:
            return {"promoted": 0}
        intent_map: Dict[str, Dict[str, Any]] = {
            str(i.get("id") or "").strip(): i for i in intents if isinstance(i, dict) and i.get("id")
        }

        min_support = int(self.valves.shortcut_learning_min_support)
        consistency_ratio = float(self.valves.shortcut_learning_consistency_ratio)
        max_examples = int(self.valves.shortcut_learning_max_examples_per_intent)
        max_cases = int(self.valves.shortcut_learning_max_cases_per_intent)
        judge_enabled = bool(self.valves.shortcut_learning_llm_judge_enabled)
        judge_fail_open = bool(self.valves.shortcut_learning_llm_judge_fail_open)
        judge_promote_min_score = self._coerce_unit_score(self.valves.shortcut_learning_llm_judge_promote_min_score)
        if judge_promote_min_score is None:
            judge_promote_min_score = 0.90

        grouped: Dict[Tuple[str, str], List[Tuple[int, dict]]] = {}
        for idx, row in enumerate(pending_rows):
            if not isinstance(row, dict):
                continue
            intent_id = str(row.get("intent_id") or "").strip()
            query_norm = str(row.get("query_norm") or "").strip()
            if not intent_id or not query_norm:
                continue
            grouped.setdefault((intent_id, query_norm), []).append((idx, row))

        processed_indexes: set[int] = set()
        promoted: List[Tuple[str, str, dict, List[int], Dict[str, Any]]] = []

        for (intent_id, query_norm), items in grouped.items():
            if len(items) < min_support:
                continue
            intent = intent_map.get(intent_id)
            if not intent:
                continue

            non_empty_count = sum(1 for _, row in items if bool(row.get("result_non_empty")))
            if non_empty_count < min_support:
                continue

            code_counts: Dict[str, int] = {}
            for _, row in items:
                code_hash = str(row.get("code_hash") or "").strip()
                if not code_hash:
                    continue
                code_counts[code_hash] = code_counts.get(code_hash, 0) + 1
            if not code_counts:
                continue
            dominant_hash, dominant_count = max(code_counts.items(), key=lambda kv: kv[1])
            ratio = float(dominant_count) / float(len(items))
            if ratio < consistency_ratio:
                continue

            selected_rows = [row for _, row in items if str(row.get("code_hash") or "").strip() == dominant_hash]
            if not selected_rows:
                continue

            judge_scores: List[float] = []
            for row in selected_rows:
                score = self._coerce_unit_score(row.get("llm_judge_score"))
                if score is not None:
                    judge_scores.append(score)
            judge_median = self._median_score(judge_scores)
            if judge_enabled:
                if not judge_scores and not judge_fail_open:
                    continue
                if judge_median is not None and judge_median < judge_promote_min_score:
                    continue

            chosen = selected_rows[-1]
            query_text = _normalize_query_text(str(chosen.get("query") or ""))
            query_text = re.sub(r"^\*{1,3}\s*(.*?)\s*\*{1,3}$", r"\1", query_text)
            query_text = query_text.strip()
            if not query_text:
                continue

            if self._catalog_has_query(catalog, query_norm):
                for idx, _ in items:
                    processed_indexes.add(idx)
                continue

            promoted.append(
                (
                    intent_id,
                    query_text,
                    chosen,
                    [idx for idx, _ in items],
                    {
                        "llm_judge_enabled": judge_enabled,
                        "llm_judge_count": len(judge_scores),
                        "llm_judge_median": judge_median,
                        "llm_judge_min": min(judge_scores) if judge_scores else None,
                        "llm_judge_max": max(judge_scores) if judge_scores else None,
                        "llm_judge_promote_threshold": judge_promote_min_score,
                    },
                )
            )

        if not promoted:
            if len(pending_rows) > int(self.valves.shortcut_learning_max_pending):
                pending_rows = pending_rows[-int(self.valves.shortcut_learning_max_pending) :]
                _atomic_write_jsonl(pending_path, pending_rows)
            return {"promoted": 0}

        promoted_count = 0
        indexed_count = 0
        for intent_id, query_text, chosen, item_indexes, judge_stats in promoted:
            intent = intent_map.get(intent_id)
            if not intent:
                continue
            examples = intent.get("examples")
            if not isinstance(examples, list):
                examples = []
                intent["examples"] = examples
            examples.append(query_text)
            if len(examples) > max_examples:
                intent["examples"] = examples[-max_examples:]

            learned_cases = intent.get("learned_cases")
            if not isinstance(learned_cases, list):
                learned_cases = []
                intent["learned_cases"] = learned_cases
            learned_cases.append(
                {
                    "query": query_text,
                    "code": _safe_trunc(str(chosen.get("analysis_code") or ""), int(self.valves.shortcut_learning_max_code_chars)),
                    "code_hash": str(chosen.get("code_hash") or ""),
                    "support": int(len(item_indexes)),
                    "result_non_empty_support": int(len(item_indexes)),
                    "promoted_at": int(time.time()),
                    "source": "auto_success_learning",
                    "llm_judge": judge_stats,
                }
            )
            if len(learned_cases) > max_cases:
                intent["learned_cases"] = learned_cases[-max_cases:]

            for idx in item_indexes:
                processed_indexes.add(idx)

            promoted_count += 1
            if self._append_index_row_for_learned_query(intent_id, query_text):
                indexed_count += 1

        _atomic_write_json(catalog_path, catalog)

        remaining = [row for i, row in enumerate(pending_rows) if i not in processed_indexes]
        if len(remaining) > int(self.valves.shortcut_learning_max_pending):
            remaining = remaining[-int(self.valves.shortcut_learning_max_pending) :]
        _atomic_write_jsonl(pending_path, remaining)

        return {"promoted": promoted_count, "indexed": indexed_count}

    def _maybe_record_success_learning(
        self,
        *,
        question: str,
        plan: str,
        analysis_code: str,
        run_status: str,
        edit_expected: bool,
        result_text: str,
        result_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        learning_meta: Dict[str, Any] = {
            "llm_judge_enabled": bool(self.valves.shortcut_learning_llm_judge_enabled),
            "llm_judge_status": "skipped",
            "llm_judge_score": None,
            "llm_judge_reason": "",
            "learning_recorded": False,
        }
        if not bool(self.valves.shortcut_learning_enabled):
            learning_meta["llm_judge_reason"] = "learning_disabled"
            return learning_meta
        if run_status != "ok" or edit_expected:
            learning_meta["llm_judge_reason"] = "not_ok_or_edit_expected"
            return learning_meta
        intent_id = self._intent_from_plan(plan)
        if not intent_id:
            learning_meta["llm_judge_reason"] = "intent_not_detected"
            return learning_meta
        selector_mode = self._selector_mode_from_plan(plan)
        if selector_mode and selector_mode not in {"retrieval", "retrieval_guarded"}:
            learning_meta["llm_judge_reason"] = f"selector_mode_blocked:{selector_mode}"
            return learning_meta
        query = _normalize_query_text(question or "")
        query = re.sub(r"^\*{1,3}\s*(.*?)\s*\*{1,3}$", r"\1", query).strip()
        query_norm = _normalize_learning_query(query)
        if not query_norm:
            learning_meta["llm_judge_reason"] = "query_norm_empty"
            return learning_meta
        if _has_forbidden_import_nodes(analysis_code):
            learning_meta["llm_judge_reason"] = "forbidden_import"
            return learning_meta

        judge_score: Optional[float] = None
        judge_reason = "llm_judge_disabled"
        judge_status = "disabled"
        judge_min_score = self._coerce_unit_score(self.valves.shortcut_learning_llm_judge_min_score)
        if judge_min_score is None:
            judge_min_score = 0.90
        judge_fail_open = bool(self.valves.shortcut_learning_llm_judge_fail_open)

        if bool(self.valves.shortcut_learning_llm_judge_enabled):
            judge_score, judge_reason, judge_status = self._llm_judge_learning_candidate(
                question=question,
                intent_id=intent_id,
                analysis_code=analysis_code,
                result_text=result_text,
                result_meta=result_meta or {},
            )
            if judge_score is None:
                if not judge_fail_open:
                    logging.info(
                        "event=learning_record_skipped reason=llm_judge_unavailable fail_open=false intent_id=%s judge_status=%s judge_reason=%s",
                        intent_id,
                        judge_status,
                        _safe_trunc(judge_reason, 160),
                    )
                    learning_meta["llm_judge_status"] = judge_status
                    learning_meta["llm_judge_score"] = judge_score
                    learning_meta["llm_judge_reason"] = judge_reason
                    return learning_meta
            elif judge_score < judge_min_score:
                logging.info(
                    "event=learning_record_skipped reason=llm_judge_low_score intent_id=%s score=%.3f threshold=%.3f judge_reason=%s",
                    intent_id,
                    judge_score,
                    judge_min_score,
                    _safe_trunc(judge_reason, 160),
                )
                learning_meta["llm_judge_status"] = judge_status
                learning_meta["llm_judge_score"] = judge_score
                learning_meta["llm_judge_reason"] = judge_reason
                return learning_meta

        entry = {
            "ts": int(time.time()),
            "request_id": _REQUEST_ID_CTX.get(),
            "trace_id": _TRACE_ID_CTX.get(),
            "intent_id": intent_id,
            "query": query,
            "query_norm": query_norm,
            "analysis_code": _safe_trunc(str(analysis_code or ""), int(self.valves.shortcut_learning_max_code_chars)),
            "code_hash": hashlib.sha256((analysis_code or "").encode("utf-8")).hexdigest()[:20],
            "result_non_empty": self._result_non_empty(result_text, result_meta or {}),
            "support": 1,
            "result_non_empty_support": 1 if self._result_non_empty(result_text, result_meta or {}) else 0,
            "llm_judge_enabled": bool(self.valves.shortcut_learning_llm_judge_enabled),
            "llm_judge_status": judge_status,
            "llm_judge_score": judge_score,
            "llm_judge_reason": _safe_trunc(judge_reason, 300),
        }

        pending_path = str(self.valves.shortcut_learning_pending_path or "").strip()
        if not pending_path:
            learning_meta["llm_judge_status"] = judge_status
            learning_meta["llm_judge_score"] = judge_score
            learning_meta["llm_judge_reason"] = judge_reason
            return learning_meta

        try:
            with self._learning_lock:
                pending_rows = _read_jsonl(pending_path)
                pending_rows.append(entry)
                if len(pending_rows) > int(self.valves.shortcut_learning_max_pending):
                    pending_rows = pending_rows[-int(self.valves.shortcut_learning_max_pending) :]
                _atomic_write_jsonl(pending_path, pending_rows)

                self._learning_record_count += 1
                promote_every = int(self.valves.shortcut_learning_promote_every)
                if promote_every <= 1 or (self._learning_record_count % promote_every == 0):
                    promote_res = self._promote_success_learning_locked()
                    if int(promote_res.get("promoted") or 0) > 0:
                        logging.info(
                            "event=learning_promote_success promoted=%s indexed=%s",
                            int(promote_res.get("promoted") or 0),
                            int(promote_res.get("indexed") or 0),
                        )
        except Exception as exc:
            logging.warning("event=learning_record_failed error=%s", _safe_trunc(str(exc), 300))
            learning_meta["llm_judge_status"] = judge_status
            learning_meta["llm_judge_score"] = judge_score
            learning_meta["llm_judge_reason"] = f"{judge_reason};record_failed:{_safe_trunc(str(exc), 120)}"
            return learning_meta

        learning_meta["llm_judge_status"] = judge_status
        learning_meta["llm_judge_score"] = judge_score
        learning_meta["llm_judge_reason"] = judge_reason
        learning_meta["learning_recorded"] = True
        return learning_meta

    def _spreadsheet_skill_file_paths(self, focus: str = "plan") -> List[Tuple[str, str]]:
        skill_dir = str(self.valves.spreadsheet_skill_dir or "").strip()
        if not skill_dir:
            return []
        all_files: List[Tuple[str, str]] = [
            (rel, os.path.join(skill_dir, rel))
            for rel in _SPREADSHEET_SKILL_FILES
        ]
        if focus == "column":
            include = {
                "SKILL.md",
                os.path.join("references", "column-matching.md"),
                os.path.join("references", "forbidden-code-patterns.md"),
            }
            return [it for it in all_files if it[0] in include]
        return all_files

    def _strip_markdown_frontmatter(self, text: str) -> str:
        s = str(text or "")
        return re.sub(r"(?s)\A---\s*\n.*?\n---\s*\n?", "", s, count=1).strip()

    def _load_spreadsheet_skill_prompt(self, focus: str = "plan") -> str:
        cache_key = f"{focus}:{self.valves.spreadsheet_skill_dir}:{self.valves.spreadsheet_skill_max_chars}"
        files = self._spreadsheet_skill_file_paths(focus=focus)
        if not files:
            logging.info("event=spreadsheet_skill_cache status=disabled reason=no_files_configured focus=%s", focus)
            return ""

        stat_items: List[str] = []
        existing: List[Tuple[str, str]] = []
        for rel, path in files:
            if not os.path.exists(path):
                continue
            try:
                st = os.stat(path)
                stat_items.append(f"{rel}:{int(st.st_mtime_ns)}:{int(st.st_size)}")
                existing.append((rel, path))
            except OSError:
                continue
        if not existing:
            logging.info(
                "event=spreadsheet_skill_cache status=miss reason=no_existing_files focus=%s skill_dir=%s",
                focus,
                self.valves.spreadsheet_skill_dir,
            )
            return ""
        version = "|".join(stat_items)

        with self._skill_prompt_lock:
            cached = self._skill_prompt_cache.get(cache_key)
            if cached and cached.get("version") == version:
                logging.info(
                    "event=spreadsheet_skill_cache status=hit focus=%s file_count=%s chars=%s",
                    focus,
                    len(existing),
                    len(str(cached.get("text") or "")),
                )
                return str(cached.get("text") or "")

        sections: List[str] = []
        for rel, path in existing:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                continue
            if rel == "SKILL.md":
                text = self._strip_markdown_frontmatter(text)
            label = rel.replace(os.sep, "/")
            body = text.strip()
            if not body:
                continue
            sections.append(f"[{label}]\n{body}")

        if not sections:
            logging.info(
                "event=spreadsheet_skill_cache status=miss reason=empty_contents focus=%s skill_dir=%s",
                focus,
                self.valves.spreadsheet_skill_dir,
            )
            return ""

        out = "\n\n".join(sections).strip()
        max_chars = int(self.valves.spreadsheet_skill_max_chars)
        truncated = False
        if len(out) > max_chars:
            out = out[:max_chars].rstrip() + "\n\n[skill context truncated]"
            truncated = True

        with self._skill_prompt_lock:
            self._skill_prompt_cache[cache_key] = {"version": version, "text": out}
        logging.info(
            "event=spreadsheet_skill_cache status=miss reason=loaded focus=%s file_count=%s chars=%s truncated=%s",
            focus,
            len(existing),
            len(out),
            truncated,
        )
        return out

    def _should_apply_spreadsheet_skill(self, question: str, profile: Optional[dict]) -> bool:
        if not bool(self.valves.spreadsheet_skill_runtime_enabled):
            return False
        if bool(self.valves.spreadsheet_skill_force_on_all_queries):
            return True
        q = str(question or "")
        if _has_edit_triggers(q):
            return True
        if isinstance(profile, dict):
            cols = profile.get("columns")
            if isinstance(cols, list) and len(cols) > 0:
                return True
        return bool(
            re.search(r"\b(column|columns|table|sheet|cell|row|rows|колон|таблиц|клітин|рядк|збереж)\b", q, re.I)
        )

    def _with_spreadsheet_skill_prompt(
        self,
        base_system: str,
        question: str,
        profile: Optional[dict],
        focus: str = "plan",
    ) -> str:
        system = str(base_system or "")
        if _SPREADSHEET_SKILL_PROMPT_MARKER in system:
            logging.info("event=spreadsheet_skill_prompt status=skipped reason=already_injected focus=%s", focus)
            return system
        should_apply = self._should_apply_spreadsheet_skill(question, profile)
        if not should_apply:
            logging.info("event=spreadsheet_skill_prompt status=skipped reason=guard focus=%s", focus)
            return system
        skill_context = self._load_spreadsheet_skill_prompt(focus=focus)
        if not skill_context:
            logging.info("event=spreadsheet_skill_prompt status=skipped reason=empty_context focus=%s", focus)
            return system
        header = (
            f"{_SPREADSHEET_SKILL_PROMPT_MARKER}\n"
            "Use the following local skill rules as strict guidance for table-safe code generation. "
            "Do not invent columns, and prefer clarification over guessing for ambiguous matches."
        )
        file_labels = re.findall(r"(?m)^\[([^\]]+)\]", skill_context)
        logging.info(
            "event=spreadsheet_skill_prompt status=applied focus=%s files=%s skill_chars=%s question_preview=%s",
            focus,
            ",".join(file_labels[:8]),
            len(skill_context),
            _safe_trunc(question, 200),
        )
        if self.valves.debug:
            logging.info(
                "event=spreadsheet_skill_prompt_preview focus=%s preview=%s",
                focus,
                _safe_trunc(skill_context, 1200),
            )
        return f"{system}\n\n{header}\n\n{skill_context}"

    def _llm_json(self, system: str, user: str) -> dict:
        json_only_guard = (
            "STRICT JSON MODE.\n"
            "You are a backend formatting component.\n\n"
            "You must output exactly one JSON object and nothing else.\n"
            "The first character must be {.\n"
            "The last character must be }.\n\n"
            "Forbidden:\n"
            "- any explanation\n"
            "- any reasoning\n"
            "- any markdown\n"
            "- any code fences\n"
            "- any comments\n"
            "- any prefix or suffix text\n"
            "- any labels like \"Answer\", \"JSON\", \"Result\"\n"
            "- any extra keys not in schema\n\n"
            "If unsure, output a valid JSON object with empty values allowed by schema.\n"
            "Never ask questions.\n"
            "Never refuse.\n"
            "Never describe the schema.\n"
            "Never echo the user query."
        )
        guarded_system = f"{json_only_guard}\n\n{system or ''}".strip()
        user_keys: List[str] = []
        user_size = 0
        try:
            user_size = len((user or "").encode("utf-8", errors="ignore"))
        except Exception:
            user_size = len(str(user or ""))
        try:
            user_obj = json.loads(user or "{}")
            if isinstance(user_obj, dict):
                user_keys = [str(k) for k in list(user_obj.keys())[:80]]
        except Exception:
            user_keys = []

        tracer = current_route_tracer()
        stage_id = ""
        if tracer:
            stage_id = tracer.start_stage(
                stage_key="llm_json_call",
                stage_name="LLM Structured JSON Call",
                purpose="Send system/user messages to base model and parse strict JSON object output.",
                input_payload={
                    "model": self.valves.base_llm_model,
                    "temperature": 0,
                    "system_chars": len(guarded_system or ""),
                    "user_chars": len(user or ""),
                    "user_size_bytes": user_size,
                    "user_keys": user_keys,
                },
                processing_summary="Call chat.completions with deterministic settings and parse JSON payload.",
                details={
                    "llm": {
                        "model": self.valves.base_llm_model,
                        "temperature": 0,
                        "messages": [
                            {"role": "system", "chars": len(guarded_system or ""), "content_preview": _safe_trunc(guarded_system, 1200)},
                            {"role": "user", "chars": len(user or ""), "content_preview": _safe_trunc(user, 1200)},
                        ],
                    }
                },
            )
        skill_injected = _SPREADSHEET_SKILL_PROMPT_MARKER in (guarded_system or "")
        logging.info(
            "event=llm_json_skill_injection active=%s prompt_hash=%s",
            skill_injected,
            hashlib.sha256((guarded_system or "").encode("utf-8")).hexdigest()[:16],
        )
        logging.info(
            "event=llm_json_request system_preview=%s user_preview=%s",
            _safe_trunc(guarded_system, 800),
            _safe_trunc(user, 1200),
        )
        def _chat_json_completion(messages: List[Dict[str, str]], max_tokens: int, force_json_mode: bool = False):
            kwargs: Dict[str, Any] = {
                "model": self.valves.base_llm_model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": int(max_tokens),
            }
            if force_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                return self._llm.chat.completions.create(**kwargs)
            except Exception as exc:
                if not force_json_mode:
                    raise
                err = str(exc).lower()
                if not any(token in err for token in ("response_format", "json_object", "unsupported")):
                    raise
                logging.info(
                    "event=llm_json_response_format_fallback reason=unsupported_json_mode model=%s",
                    self.valves.base_llm_model,
                )
                kwargs.pop("response_format", None)
                return self._llm.chat.completions.create(**kwargs)

        raw_text = ""
        started = time.monotonic()
        est_prompt_tokens = len(re.findall(r"\S+", f"{guarded_system or ''}\n{user or ''}"))
        try:
            messages = [{"role": "system", "content": guarded_system}, {"role": "user", "content": user}]
            resp = _chat_json_completion(
                messages=messages,
                max_tokens=int(self.valves.llm_json_max_tokens),
                force_json_mode=True,
            )
            raw_text = (resp.choices[0].message.content or "").strip()
            try:
                parsed = _parse_json_dict_from_llm(raw_text)
            except Exception as first_parse_exc:
                logging.warning(
                    "event=llm_json_parse_retry reason=first_parse_failed error=%s raw_preview=%s",
                    str(first_parse_exc),
                    _safe_trunc(raw_text, 800),
                )
                retry_guard = (
                    "STRICT JSON RETRY MODE. "
                    "Output exactly one minified JSON object. "
                    "No prose, no markdown, no explanations, no <think>, no extra keys, no prefix/suffix."
                )
                retry_messages = [
                    {"role": "system", "content": f"{retry_guard}\n\n{guarded_system}"},
                    {"role": "user", "content": user},
                ]
                resp = _chat_json_completion(
                    messages=retry_messages,
                    max_tokens=max(int(self.valves.llm_json_max_tokens), 384),
                    force_json_mode=True,
                )
                raw_text = (resp.choices[0].message.content or "").strip()
                parsed = _parse_json_dict_from_llm(raw_text)
            latency_ms = round((time.monotonic() - started) * 1000.0, 3)
            usage = _extract_llm_usage(resp)
            est_completion_tokens = len(re.findall(r"\S+", raw_text or ""))
            self._record_llm_call_stat(
                {
                    "status": "ok",
                    "model": self.valves.base_llm_model,
                    "latency_ms": latency_ms,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "prompt_tokens_estimate": est_prompt_tokens,
                    "completion_tokens_estimate": est_completion_tokens,
                    "user_keys": user_keys[:20],
                }
            )
            logging.info("event=llm_json_response preview=%s", _safe_trunc(parsed, 1200))
            if tracer and stage_id:
                tracer.end_stage(
                    stage_id,
                    status="ok",
                    output_payload={"parsed": parsed},
                    processing_summary="LLM returned JSON and parser accepted object schema.",
                    details={
                        "llm": {
                            "model": self.valves.base_llm_model,
                            "temperature": 0,
                            "latency_ms": latency_ms,
                            "usage": usage,
                            "raw_response": _safe_trunc(raw_text, 4000),
                            "parsed_output": parsed,
                        }
                    },
                )
            return parsed
        except Exception as exc:
            latency_ms = round((time.monotonic() - started) * 1000.0, 3)
            est_completion_tokens = len(re.findall(r"\S+", raw_text or ""))
            self._record_llm_call_stat(
                {
                    "status": "error",
                    "model": self.valves.base_llm_model,
                    "latency_ms": latency_ms,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "prompt_tokens_estimate": est_prompt_tokens,
                    "completion_tokens_estimate": est_completion_tokens,
                    "user_keys": user_keys[:20],
                    "error_type": type(exc).__name__,
                }
            )
            if raw_text:
                logging.warning("event=llm_json_non_json_response preview=%s", _safe_trunc(raw_text, 1200))
            if tracer and stage_id:
                tracer.end_stage(
                    stage_id,
                    status="error",
                    output_payload={"raw_response_preview": _safe_trunc(raw_text, 1200)},
                    processing_summary="LLM call or JSON parsing failed.",
                    error={"type": type(exc).__name__, "message": str(exc)},
                    details={
                        "llm": {
                            "model": self.valves.base_llm_model,
                            "temperature": 0,
                            "raw_response": _safe_trunc(raw_text, 4000),
                            "parse_error": str(exc),
                        }
                    },
                )
            logging.warning(
                "event=llm_json_fail_open error_type=%s error=%s",
                type(exc).__name__,
                str(exc),
            )
            return {}

    def _plan_code(
        self,
        question: str,
        profile: dict,
        lookup_hints: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, str, Optional[bool]]:
        system = self._with_spreadsheet_skill_prompt(
            self._prompts.get("plan_code_system", DEFAULT_PLAN_CODE_SYSTEM),
            question,
            profile,
            focus="plan",
        )
        payload_obj: Dict[str, Any] = {"question": question, "schema": _compact_profile_for_llm(profile)}
        if isinstance(lookup_hints, dict) and lookup_hints:
            system += (
                " CRITICAL LOOKUP HINTS: if lookup_hints.filters are present, you MUST preserve them "
                "in analysis_code and combine them with any additional constraints from the question."
            )
            payload_obj["lookup_hints"] = lookup_hints
        payload = json.dumps(payload_obj, ensure_ascii=False)
        parsed = self._llm_json(system, payload)
        commit_df = parsed.get("commit_df")
        return (
            parsed.get("analysis_code", ""),
            parsed.get("short_plan", ""),
            (parsed.get("op") or "read"),
            commit_df if isinstance(commit_df, bool) else None,
        )

    def _plan_code_with_rlm_tool(
        self,
        question: str,
        profile: dict,
        lookup_hints: Optional[Dict[str, Any]] = None,
        retry_reason: str = "",
        previous_code: str = "",
        runtime_error: str = "",
    ) -> Tuple[str, str, str, Optional[bool]]:
        if not bool(self.valves.rlm_codegen_enabled):
            return "", "", "read", None

        system = self._with_spreadsheet_skill_prompt(
            self._prompts.get("rlm_codegen_system", DEFAULT_RLM_CODEGEN_SYSTEM),
            question,
            profile,
            focus="plan",
        )
        op_guess = _infer_op_from_question(question)
        commit_guess: Optional[bool] = True if op_guess == "edit" else False

        def _validator(code: str) -> Tuple[str, bool, Optional[str]]:
            normalized = _strip_llm_think_sections(textwrap.dedent(str(code or "")).strip())
            if not normalized:
                return "", False, "empty_or_non_code_output"
            if _has_forbidden_import_nodes(normalized):
                return normalized, False, "forbidden_import_detected"
            if op_guess != "edit" and not _has_result_assignment(normalized):
                return normalized, False, "missing_result_assignment"
            return normalized, bool(op_guess == "edit"), None

        core = RLMCore(
            system_prompt=system,
            lm_handler_factory=lambda: LMHandler(
                openai_client=self._llm,
                model=self.valves.base_llm_model,
                host="127.0.0.1",
                port=0,
                temperature=0.0,
            ),
            code_extractor=_extract_analysis_code_from_llm_no_think,
            code_validator=_validator,
            max_iterations=int(self.valves.rlm_codegen_max_turns),
            max_tokens=int(self.valves.rlm_codegen_max_tokens),
            fallback_enabled=False,
        )
        payload_obj: Dict[str, Any] = {
            "question": question,
            "schema": _compact_profile_for_llm(profile),
            "op_hint": op_guess,
        }
        if isinstance(lookup_hints, dict) and lookup_hints:
            payload_obj["lookup_hints"] = lookup_hints
        if retry_reason:
            payload_obj["retry_reason"] = retry_reason
        if previous_code:
            payload_obj["previous_code"] = _safe_trunc(previous_code, 3000)
        if runtime_error:
            payload_obj["previous_error"] = _safe_trunc(runtime_error, 1200)

        res = core.completion(payload_obj)
        for it in res.iterations:
            logging.info(
                "event=rlm_codegen_iteration turn=%s validation_error=%s response_preview=%s",
                it.turn,
                _safe_trunc(it.validation_error, 120),
                _safe_trunc(it.response, 320),
            )
        analysis_code = str(res.analysis_code or "").strip()
        if not analysis_code:
            return "", "", op_guess, commit_guess
        plan = f"rlm_codegen_tool:turn_{max(1, len(res.iterations))}"
        return analysis_code, plan, op_guess, commit_guess

    def _plan_code_retry_missing_result(
        self,
        question: str,
        profile: dict,
        previous_code: str,
        reason: str,
        lookup_hints: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, str, Optional[bool]]:
        system = self._with_spreadsheet_skill_prompt(
            self._prompts.get("plan_code_system", DEFAULT_PLAN_CODE_SYSTEM),
            question,
            profile,
            focus="plan",
        )
        payload_obj: Dict[str, Any] = {
            "question": question,
            "schema": _compact_profile_for_llm(profile),
            "retry_reason": reason,
            "previous_analysis_code": previous_code,
            "retry_constraints": [
                "CRITICAL: final answer must be assigned to variable `result`.",
                "If previous code used another variable, rewrite to `result = ...`.",
                "Do not return code without `result =` assignment for read operations.",
            ],
        }
        if isinstance(lookup_hints, dict) and lookup_hints:
            system += (
                " CRITICAL LOOKUP HINTS: if lookup_hints.filters are present, you MUST preserve them "
                "in analysis_code and combine them with any additional constraints from the question."
            )
            payload_obj["lookup_hints"] = lookup_hints
        payload = json.dumps(payload_obj, ensure_ascii=False)
        parsed = self._llm_json(system, payload)
        commit_df = parsed.get("commit_df")
        return (
            parsed.get("analysis_code", ""),
            parsed.get("short_plan", ""),
            (parsed.get("op") or "read"),
            commit_df if isinstance(commit_df, bool) else None,
        )

    def _plan_code_retry_missing_filter(
        self,
        question: str,
        profile: dict,
        previous_code: str,
        reason: str,
        lookup_hints: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, str, Optional[bool]]:
        system = self._with_spreadsheet_skill_prompt(
            self._prompts.get("plan_code_system", DEFAULT_PLAN_CODE_SYSTEM),
            question,
            profile,
            focus="plan",
        )
        payload_obj: Dict[str, Any] = {
            "question": question,
            "schema": _compact_profile_for_llm(profile),
            "retry_reason": reason,
            "previous_analysis_code": previous_code,
            "retry_constraints": [
                "CRITICAL: for subset questions (among/for/with/where/тільки/лише/серед), filter df first and only then aggregate.",
                "Use explicit pandas filter (e.g., str.contains, query, boolean mask) before min/max/mean/sum/count.",
                "CRITICAL: final answer must be assigned to variable `result`.",
                "No import/from statements.",
            ],
        }
        if isinstance(lookup_hints, dict) and lookup_hints:
            system += (
                " CRITICAL LOOKUP HINTS: if lookup_hints.filters are present, you MUST preserve them "
                "in analysis_code and combine them with any additional constraints from the question."
            )
            payload_obj["lookup_hints"] = lookup_hints
        payload = json.dumps(payload_obj, ensure_ascii=False)
        parsed = self._llm_json(system, payload)
        commit_df = parsed.get("commit_df")
        return (
            parsed.get("analysis_code", ""),
            parsed.get("short_plan", ""),
            (parsed.get("op") or "read"),
            commit_df if isinstance(commit_df, bool) else None,
        )

    def _plan_code_retry_subset_guard_conflict(
        self,
        question: str,
        profile: dict,
        previous_code: str,
        reason: str,
        lookup_hints: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, str, Optional[bool]]:
        system = self._with_spreadsheet_skill_prompt(
            self._prompts.get("plan_code_system", DEFAULT_PLAN_CODE_SYSTEM),
            question,
            profile,
            focus="plan",
        )
        payload_obj: Dict[str, Any] = {
            "question": question,
            "schema": _compact_profile_for_llm(profile),
            "retry_reason": reason,
            "previous_analysis_code": previous_code,
            "retry_constraints": [
                "CRITICAL: this query does NOT require subset filtering.",
                "For grouped queries (by/per/по/за each), aggregate over full df unless an explicit filter is requested.",
                "Do not inject subset filters from ambiguous soft hints.",
                "CRITICAL: final answer must be assigned to variable `result`.",
                "No import/from statements.",
            ],
        }
        if isinstance(lookup_hints, dict) and lookup_hints:
            system += (
                " CRITICAL LOOKUP HINTS: if lookup_hints.filters are present, you MUST preserve them "
                "in analysis_code and combine them with any additional constraints from the question."
            )
            payload_obj["lookup_hints"] = lookup_hints
        payload = json.dumps(payload_obj, ensure_ascii=False)
        parsed = self._llm_json(system, payload)
        commit_df = parsed.get("commit_df")
        return (
            parsed.get("analysis_code", ""),
            parsed.get("short_plan", ""),
            (parsed.get("op") or "read"),
            commit_df if isinstance(commit_df, bool) else None,
        )

    def _plan_code_retry_read_mutation(
        self,
        question: str,
        profile: dict,
        previous_code: str,
        reason: str,
        lookup_hints: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, str, Optional[bool]]:
        system = self._with_spreadsheet_skill_prompt(
            self._prompts.get("plan_code_system", DEFAULT_PLAN_CODE_SYSTEM),
            question,
            profile,
            focus="plan",
        )
        payload_obj: Dict[str, Any] = {
            "question": question,
            "schema": _compact_profile_for_llm(profile),
            "retry_reason": reason,
            "previous_analysis_code": previous_code,
            "retry_constraints": [
                "CRITICAL: this is strictly a read-only query.",
                "Do NOT mutate DataFrame state: no `df = ...`, no `df[...] = ...`, no `.loc/.iloc/.at/.iat` assignments, no `inplace=True`.",
                "If filtering is needed, use a temporary variable (e.g., `_df = df[...]`) and compute from `_df`.",
                "CRITICAL: final answer must be assigned to variable `result`.",
                "Set op='read' and commit_df=false.",
                "No import/from statements.",
            ],
        }
        if isinstance(lookup_hints, dict) and lookup_hints:
            system += (
                " CRITICAL LOOKUP HINTS: if lookup_hints.filters are present, you MUST preserve them "
                "in analysis_code and combine them with any additional constraints from the question."
            )
            payload_obj["lookup_hints"] = lookup_hints
        payload = json.dumps(payload_obj, ensure_ascii=False)
        parsed = self._llm_json(system, payload)
        commit_df = parsed.get("commit_df")
        return (
            parsed.get("analysis_code", ""),
            parsed.get("short_plan", ""),
            (parsed.get("op") or "read"),
            commit_df if isinstance(commit_df, bool) else None,
        )

    def _plan_code_retry_runtime_error(
        self,
        question: str,
        profile: dict,
        previous_code: str,
        runtime_error: str,
    ) -> Tuple[str, str, str, Optional[bool]]:
        system = self._with_spreadsheet_skill_prompt(
            self._prompts.get("plan_code_system", DEFAULT_PLAN_CODE_SYSTEM),
            question,
            profile,
            focus="plan",
        )
        retry_constraints = [
            "CRITICAL: final answer must be assigned to variable `result`.",
            "No import/from statements.",
            "Use exact column names from schema.columns only.",
            "Do not access dict key 'import' directly.",
            "Prefer robust pandas code that handles missing/invalid values safely.",
        ]
        if _question_requires_subset_filter(question):
            retry_constraints.insert(
                0,
                "CRITICAL: this is a subset query; apply explicit pandas filtering before aggregation/count.",
            )
        payload = json.dumps(
            {
                "question": question,
                "schema": _compact_profile_for_llm(profile),
                "retry_reason": runtime_error,
                "previous_analysis_code": previous_code,
                "retry_constraints": retry_constraints,
            },
            ensure_ascii=False,
        )
        parsed = self._llm_json(system, payload)
        commit_df = parsed.get("commit_df")
        return (
            parsed.get("analysis_code", ""),
            parsed.get("short_plan", ""),
            (parsed.get("op") or "read"),
            commit_df if isinstance(commit_df, bool) else None,
        )

    def _rlm_core_repl_repair(
        self,
        question: str,
        profile: dict,
        df_id: str,
        failed_code: str,
        failed_error: str,
        op: Optional[str],
        commit_df: Optional[bool],
    ) -> Optional[Dict[str, Any]]:
        if not bool(self.valves.rlm_core_repl_enabled):
            return None
        # Keep edit flow on existing deterministic path for now.
        if _infer_op_from_question(question) == "edit":
            return None

        system = self._with_spreadsheet_skill_prompt(
            self._prompts.get("rlm_core_repl_system", DEFAULT_RLM_CORE_REPL_SYSTEM),
            question,
            profile,
            focus="plan",
        )
        def _validator(code: str) -> Tuple[str, bool, Optional[str]]:
            normalized, retry_edit_expected, finalize_err = _finalize_code_for_sandbox(
                question,
                code,
                op,
                commit_df,
                df_profile=profile,
            )
            if finalize_err:
                return normalized, retry_edit_expected, finalize_err
            if _has_forbidden_import_nodes(normalized):
                return normalized, retry_edit_expected, "forbidden_import_detected"
            normalized, _ = self._resolve_shortcut_placeholders(normalized, "", question, profile)
            normalized = textwrap.dedent(normalized or "").strip() + "\n"
            if "df_profile" in (normalized or ""):
                normalized = f"df_profile = {_compact_profile_for_llm(profile)!r}\n" + normalized
            normalized = _normalize_generated_code(normalized)
            return normalized, retry_edit_expected, None

        core = RLMCore(
            system_prompt=system,
            lm_handler_factory=lambda: LMHandler(
                openai_client=self._llm,
                model=self.valves.base_llm_model,
                host="127.0.0.1",
                port=0,
                temperature=0.0,
            ),
            environment_factory=lambda _prompt: SandboxREPL(
                executor=lambda code: self._sandbox_run(df_id, code),
                context_payload=_prompt if isinstance(_prompt, dict) else {"prompt": _prompt},
                persistent=True,
            ),
            code_extractor=_extract_analysis_code_from_llm_no_think,
            code_validator=_validator,
            max_iterations=int(self.valves.rlm_core_repl_max_iterations),
            max_tokens=int(self.valves.rlm_core_repl_max_tokens),
            fallback_enabled=False,
        )

        payload = {
            "question": question,
            "schema": _compact_profile_for_llm(profile),
            "previous_code": _safe_trunc(failed_code, 3000),
            "previous_error": _safe_trunc(failed_error, 1500),
            "op_hint": str(op or ""),
        }
        res = core.completion(payload)
        for it in res.iterations:
            logging.info(
                "event=rlm_core_repl_iteration turn=%s validation_error=%s repl_status=%s repl_error=%s",
                it.turn,
                _safe_trunc(it.validation_error, 120),
                _safe_trunc(it.repl_status, 40),
                _safe_trunc(it.repl_error, 180),
            )
        run_resp = dict(res.run_resp or {})
        if str(run_resp.get("status") or "") != "ok":
            return None
        return {
            "analysis_code": str(res.analysis_code or ""),
            "run_resp": run_resp,
            "edit_expected": bool(res.edit_expected),
        }

    def _llm_pick_column_for_shortcut(self, question: str, profile: dict) -> Optional[str]:
        columns = [str(c) for c in (profile or {}).get("columns") or []]
        if not columns:
            return None
        system = (
            "Pick the single best column name from the provided list that matches the user's question. "
            "Return ONLY JSON: {\"column\": \"<exact column name from list or empty>\"}."
        )
        system = self._with_spreadsheet_skill_prompt(system, question, profile, focus="column")
        payload = {
            "question": question,
            "columns": columns[:200],
            "numeric_columns": _profile_numeric_columns(profile, limit=200),
        }
        try:
            parsed = self._llm_json(system, json.dumps(payload, ensure_ascii=False))
        except Exception:
            return None
        col = str((parsed or {}).get("column") or "").strip()
        if not col or col not in columns:
            return None
        return col

    def _llm_pick_semantic_lookup_column(
        self,
        question: str,
        alias: str,
        profile: dict,
        role: str = "filter",
    ) -> Optional[str]:
        alias = str(alias or "").strip()
        columns = [str(c) for c in (profile or {}).get("columns") or []]
        if not alias or not columns:
            return None

        # Fast local heuristic before spending an extra LLM call.
        heuristic = _pick_relevant_column(alias, columns)
        if not heuristic and role != "output_column":
            heuristic = _pick_relevant_column(f"{question} {alias}", columns)
        if heuristic and heuristic in columns:
            logging.info(
                "event=lookup_semantic_column_resolve source=heuristic role=%s alias=%s column=%s",
                role,
                _safe_trunc(alias, 80),
                heuristic,
            )
            return heuristic

        system = (
            "Resolve a semantic alias from a user request to the closest dataframe column name. "
            "Return ONLY JSON: {\"column\": \"<exact column name from list or empty>\", \"confidence\": 0..1}. "
            "Use semantic similarity from question, alias, column names, and numeric column hints. "
            "If uncertain, return empty column and confidence 0."
        )
        system = self._with_spreadsheet_skill_prompt(system, question, profile, focus="column")
        payload = {
            "question": question,
            "alias": alias,
            "role": role,
            "columns": columns[:200],
            "numeric_columns": _profile_numeric_columns(profile, limit=200),
        }
        try:
            parsed = self._llm_json(system, json.dumps(payload, ensure_ascii=False))
        except Exception:
            return None

        col = str((parsed or {}).get("column") or "").strip()
        if col not in columns:
            return None

        confidence: Optional[float] = None
        conf_raw = (parsed or {}).get("confidence")
        if isinstance(conf_raw, (int, float)):
            confidence = float(conf_raw)
        elif isinstance(conf_raw, str):
            try:
                confidence = float(conf_raw.strip())
            except Exception:
                confidence = None
        if confidence is not None:
            confidence = max(0.0, min(1.0, confidence))
            if confidence < 0.35:
                logging.info(
                    "event=lookup_semantic_column_resolve source=llm role=%s alias=%s column=%s confidence=%.2f accepted=false",
                    role,
                    _safe_trunc(alias, 80),
                    col,
                    confidence,
                )
                return None

        logging.info(
            "event=lookup_semantic_column_resolve source=llm role=%s alias=%s column=%s confidence=%s accepted=true",
            role,
            _safe_trunc(alias, 80),
            col,
            "n/a" if confidence is None else f"{confidence:.2f}",
        )
        return col

    def _llm_pick_numeric_metric_column(self, question: str, profile: dict) -> Optional[str]:
        columns = [str(c) for c in (profile or {}).get("columns") or []]
        if not columns:
            return None
        dtypes = (profile or {}).get("dtypes") or {}
        numeric_cols = [
            c for c in columns if str(dtypes.get(c, "")).lower().startswith(("int", "float", "uint"))
        ]
        if not numeric_cols:
            return None
        system = (
            "Pick the best numeric column for the user's aggregation question. "
            "Return ONLY JSON: {\"column\": \"<exact column name from list or empty>\"}. "
            "Choose only from numeric_columns."
        )
        system = self._with_spreadsheet_skill_prompt(system, question, profile, focus="column")
        payload = {
            "question": question,
            "columns": columns[:200],
            "numeric_columns": numeric_cols[:200],
        }
        try:
            parsed = self._llm_json(system, json.dumps(payload, ensure_ascii=False))
        except Exception:
            return None
        col = str((parsed or {}).get("column") or "").strip()
        if not col or col not in numeric_cols:
            return None
        return col

    def _llm_pick_ranking_slots(self, question: str, profile: dict) -> Dict[str, Any]:
        columns = [str(c) for c in (profile or {}).get("columns") or []]
        if not columns:
            return {}
        dtypes = (profile or {}).get("dtypes") or {}
        numeric_cols = [
            c for c in columns if str(dtypes.get(c, "")).lower().startswith(("int", "float", "uint"))
        ]
        system = (
            "Analyze the query and map it to ranking slots over DataFrame columns. "
            "Return ONLY JSON with keys: query_mode, metric_col, group_col, target_col, agg, top_n, order, require_available, availability_col, entity_cols. "
            "query_mode must be one of: row_ranking, group_ranking, other. "
            "metric_col, group_col, target_col and availability_col must be exact names from columns or empty string. "
            "agg must be one of: count, sum, mean, min, max, median. "
            "entity_cols must be a list of exact column names from columns (can be empty). "
            "top_n must be integer or null. order must be desc or asc. "
            "Use row_ranking only when user asks for top/bottom rows/items/models by a metric. "
            "Use group_ranking when ranking grouped entities (e.g., categories by sum/count). "
            "For group_ranking: use group_col as grouping dimension and target_col+agg as metric; "
            "for agg='count' target_col may be empty. "
            "IMPORTANT: If query_mode is row_ranking/group_ranking, top_n MUST be an integer "
            "(use defaults: 5 for row_ranking, 10 for group_ranking when user did not specify). "
            "If the query is not ranking (e.g., exact match/filter lookup), set query_mode='other' and top_n=null."
        )
        system = self._with_spreadsheet_skill_prompt(system, question, profile, focus="column")
        payload = {
            "question": question,
            "columns": columns[:200],
            "numeric_columns": numeric_cols[:200],
            "rows": (profile or {}).get("rows"),
        }
        try:
            parsed = self._llm_json(system, json.dumps(payload, ensure_ascii=False))
        except Exception:
            return {}

        out: Dict[str, Any] = {}
        mode = str((parsed or {}).get("query_mode") or "").strip().lower()
        if mode not in {"row_ranking", "group_ranking", "other"}:
            mode = "other"
        out["query_mode"] = mode

        metric_col = str((parsed or {}).get("metric_col") or "").strip()
        if metric_col in numeric_cols:
            out["metric_col"] = metric_col

        group_col = str((parsed or {}).get("group_col") or "").strip()
        if group_col in columns:
            out["group_col"] = group_col

        target_col = str((parsed or {}).get("target_col") or "").strip()
        if target_col in numeric_cols:
            out["target_col"] = target_col

        agg = str((parsed or {}).get("agg") or "").strip().lower()
        if agg in {"count", "sum", "mean", "min", "max", "median"}:
            out["agg"] = agg

        availability_col = str((parsed or {}).get("availability_col") or "").strip()
        if availability_col in columns:
            out["availability_col"] = availability_col

        order = str((parsed or {}).get("order") or "").strip().lower()
        out["order"] = "asc" if order in {"asc", "ascending", "lowest", "smallest"} else "desc"

        top_n_raw = (parsed or {}).get("top_n")
        top_n = _normalize_optional_top_n(top_n_raw)
        if top_n is not None:
            out["top_n"] = top_n

        out["require_available"] = bool((parsed or {}).get("require_available"))

        entity_cols_raw = (parsed or {}).get("entity_cols")
        entity_cols: List[str] = []
        if isinstance(entity_cols_raw, list):
            for c in entity_cols_raw:
                s = str(c).strip()
                if s and s in columns and s not in entity_cols:
                    entity_cols.append(s)
        out["entity_cols"] = entity_cols[:4]
        return out

    def _lookup_has_explicit_eq_cue(self, question: str) -> bool:
        q_low = (question or "").lower()
        return bool(
            re.search(
                r"(?:==|=)"
                r"|\b(дорівн\w*|рівн\w*|exact(?:ly)?|точно|саме)\b"
                r"|\b(?:id|sku|код|артикул)\s*[:=]?\s*[a-zа-яіїєґ0-9._\-]+\b",
                q_low,
                re.I,
            )
        )

    def _lookup_has_prefix_cue(self, question: str) -> bool:
        q_low = (question or "").lower()
        return bool(
            re.search(
                r"\b(starts?\s+with|begin(?:s|ning)?\s+with|почина\w*\s+на|на\s+літер\w*|з\s+літер\w*)\b",
                q_low,
                re.I,
            )
        )

    def _lookup_has_suffix_cue(self, question: str) -> bool:
        q_low = (question or "").lower()
        return bool(
            re.search(
                r"\b(ends?\s+with|закінчу\w*\s+на|на\s+кінц\w*)\b",
                q_low,
                re.I,
            )
        )

    def _lookup_filter_is_exact_field(self, column: str) -> bool:
        c = str(column or "").strip().lower()
        if not c:
            return False
        if re.search(r"(?:^|_)(id|uuid|uid|sku|код|артикул)(?:$|_)", c):
            return True
        return bool(re.search(r"(status|статус|available|наявн|flag|is_|active|enabled|disabled|bool)", c))

    def _lookup_is_status_like_column(self, column: str) -> bool:
        c = str(column or "").strip().lower()
        if not c:
            return False
        return bool(re.search(r"(status|статус|available|наявн|stock|склад|доступн|inventory|warehouse|запас|залишк)", c))

    def _lookup_quoted_literals(self, question: str) -> List[str]:
        q = str(question or "")
        if not q:
            return []
        out: List[str] = []
        for m in re.finditer(r"['\"]([^'\"]+)['\"]", q):
            s = str(m.group(1) or "").strip().lower()
            if s and s not in out:
                out.append(s)
        return out

    def _lookup_question_requests_exact_value(self, question: str, value: Any) -> bool:
        s = str(value or "").strip().lower()
        if not s:
            return False
        if self._lookup_has_explicit_eq_cue(question):
            return True
        quoted = self._lookup_quoted_literals(question)
        return s in set(quoted)

    def _lookup_status_pattern(self, value: Any) -> Optional[str]:
        s = str(value or "").strip().lower()
        if not s:
            return None
        # Low-stock intent (must stay separate from generic "in stock").
        if re.search(r"(закінч\w*|заканч\w*|майже\s+скінч\w*|running\s*out|low\s*stock|limited\s*stock)", s, re.I):
            return r"(?:закінч\w*|заканч\w*|майже\s+скінч\w*|running\s*out|low\s*stock|limited\s*stock)"
        # Negative availability intent.
        if re.search(r"(нема|відсутн|out\s*of\s*stock|unavailable|not\s*available)", s):
            return r"(?:нема|відсутн|out\s*of\s*stock|unavailable|not\s*available)"
        # Under-order intent.
        if re.search(r"(під\s*замовлення|under\s*order|backorder)", s):
            return r"(?:під\s*замовлення|under\s*order|backorder)"
        # Positive availability intent.
        if re.search(r"(на\s+склад|в\s+наявн|наявн|в\s+запас\w*|запас\w*|залишк\w*|in\s*stock|available|inventory|warehouse|доступн|резерв)", s):
            return r"(?:в\s*наявн|наявн|в\s*запас\w*|запас\w*|залишк\w*|in\s*stock|available|inventory|warehouse|доступн|на\s*склад|резерв\w*)"
        return None

    def _lookup_filter_value_is_entity_like(self, value: Any) -> bool:
        if value is None:
            return False
        s = str(value).strip()
        if len(s) < 2:
            return False
        if re.fullmatch(r"[\d\s.,:%\-]+", s):
            return False
        return bool(re.search(r"[A-Za-zА-Яа-яІіЇїЄєҐґ]", s))

    def _lookup_ambiguous_filter_candidate_columns(
        self,
        question: str,
        filter_col: str,
        value: Any,
        columns: List[str],
        dtypes: Dict[str, Any],
    ) -> List[str]:
        if not columns:
            return []
        if not (isinstance(filter_col, str) and filter_col in columns):
            return []
        if not self._lookup_filter_value_is_entity_like(value):
            return []
        if self._lookup_filter_is_exact_field(filter_col):
            return []
        explicit_col = _find_explicit_column_in_text(question, columns)
        if explicit_col:
            return []
        mentioned_cols = _find_columns_in_text(question, columns)
        if filter_col in mentioned_cols:
            return []

        text_cols = [
            c
            for c in columns
            if not str((dtypes or {}).get(c, "")).lower().startswith(("int", "float", "uint"))
            and not self._lookup_is_status_like_column(c)
        ]
        if filter_col not in text_cols:
            return []

        entity_col_pat = r"(категор|category|тип|type|group|груп|клас|class|product|товар|модел|model|назв|name|опис|desc|description|spec|характер|бренд|brand)"
        candidate_pool = [c for c in text_cols if re.search(entity_col_pat, c.lower())]
        if not candidate_pool:
            return []

        priority_patterns = (
            r"(категор|category|тип|type|group|груп|class|клас|product|товар)",
            r"(модел|model|назв|name|бренд|brand)",
            r"(опис|desc|description|spec|характер)",
        )
        ordered: List[str] = []
        for pat in priority_patterns:
            for col in candidate_pool:
                if col == filter_col:
                    continue
                if col in ordered:
                    continue
                if re.search(pat, col.lower()):
                    ordered.append(col)
        for col in candidate_pool:
            if col == filter_col:
                continue
            if col not in ordered:
                ordered.append(col)
        return ordered[:4]

    def _normalize_lookup_filters(
        self,
        question: str,
        filters: List[Dict[str, Any]],
        dtypes: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        explicit_eq = self._lookup_has_explicit_eq_cue(question)
        has_prefix_cue = self._lookup_has_prefix_cue(question)
        has_suffix_cue = self._lookup_has_suffix_cue(question)
        out: List[Dict[str, Any]] = []
        for f in filters:
            cur = dict(f)
            col = str(cur.get("column") or "").strip()
            op = str(cur.get("op") or "").strip().lower()
            val = cur.get("value")
            dtype = str((dtypes or {}).get(col, "")).lower()
            is_numeric = dtype.startswith(("int", "float", "uint"))
            is_status_like = self._lookup_is_status_like_column(col)
            exact_value_requested = self._lookup_question_requests_exact_value(question, val)
            text_val = str(val).strip() if isinstance(val, str) else ""

            if op == "contains" and text_val:
                if text_val.startswith("^") and len(text_val) > 1:
                    cur["op"] = "startswith"
                    cur["value"] = text_val[1:]
                    op = "startswith"
                elif text_val.endswith("$") and len(text_val) > 1:
                    cur["op"] = "endswith"
                    cur["value"] = text_val[:-1]
                    op = "endswith"
                elif has_prefix_cue:
                    cur["op"] = "startswith"
                    op = "startswith"
                elif has_suffix_cue:
                    cur["op"] = "endswith"
                    op = "endswith"

            if op == "eq" and is_status_like and exact_value_requested:
                cur["match_mode"] = "exact"
                out.append(cur)
                continue

            if (
                op == "eq"
                and not explicit_eq
                and is_status_like
                and self._lookup_filter_value_is_entity_like(val)
            ):
                cur["op"] = "startswith" if has_prefix_cue else ("endswith" if has_suffix_cue else "contains")
                cur["match_mode"] = "semantic"
                logging.info(
                    "event=lookup_filter_operator_adjust column=%s from=eq to=%s reason=status_semantic value_preview=%s",
                    col,
                    cur["op"],
                    _safe_trunc(val, 80),
                )
                out.append(cur)
                continue

            if (
                op == "eq"
                and not explicit_eq
                and not is_numeric
                and not self._lookup_filter_is_exact_field(col)
                and self._lookup_filter_value_is_entity_like(val)
            ):
                cur["op"] = "startswith" if has_prefix_cue else ("endswith" if has_suffix_cue else "contains")
                logging.info(
                    "event=lookup_filter_operator_adjust column=%s from=eq to=%s value_preview=%s",
                    col,
                    cur["op"],
                    _safe_trunc(val, 80),
                )
            out.append(cur)
        return out

    def _lookup_requires_multicol_fallback(
        self,
        question: str,
        filters: List[Dict[str, Any]],
        columns: List[str],
        dtypes: Dict[str, Any],
    ) -> bool:
        if not filters:
            return False
        explicit_col = _find_explicit_column_in_text(question, columns)
        q_low = (question or "").lower()
        metric_context = bool(
            re.search(
                r"\b(max|min|mean|average|avg|median|sum|total|макс\w*|мін\w*|середн\w*|сума|підсум\w*)\b",
                q_low,
                re.I,
            )
        )
        has_entity_filter = False
        search_like_cols = [
            c
            for c in columns
            if not str((dtypes or {}).get(c, "")).lower().startswith(("int", "float", "uint"))
            and re.search(r"(бренд|brand|model|name|назв|опис|desc|spec|характер|категор|category|type|тип)", c.lower())
        ]
        for f in filters:
            col = str(f.get("column") or "").strip()
            op = str(f.get("op") or "").strip().lower()
            val = f.get("value")
            dtype = str((dtypes or {}).get(col, "")).lower()
            is_numeric = dtype.startswith(("int", "float", "uint"))
            if op not in {"eq", "contains"}:
                continue
            if is_numeric:
                continue
            if not self._lookup_filter_value_is_entity_like(val):
                continue
            has_entity_filter = True
            if explicit_col and explicit_col == col:
                continue
            if metric_context or len(search_like_cols) >= 2:
                if has_entity_filter and not metric_context:
                    logging.info(
                        "event=lookup_fallback_skip reason=entity_filter_present question=%s",
                        _safe_trunc(question, 200),
                    )
                    return False
                return True
        return False

    def _lookup_infer_aggregation_from_question(self, question: str) -> str:
        q = (question or "").lower()
        if not q:
            return "none"
        if re.search(r"\b(найб\w*|макс\w*|max(?:imum)?|highest|largest|найвищ\w*)\b", q, re.I):
            return "max"
        if re.search(r"\b(наймен\w*|мін\w*|min(?:imum)?|lowest|smallest|найниж\w*)\b", q, re.I):
            return "min"
        if re.search(r"\b(середн\w*|average|avg|mean)\b", q, re.I):
            return "mean"
        if re.search(r"\b(медіан\w*|median)\b", q, re.I):
            return "median"
        if re.search(r"\b(сума|sum|total|підсум\w*|загальн\w*)\b", q, re.I):
            return "sum"
        return "none"

    def _lookup_pick_aggregation_column(
        self,
        question: str,
        profile: dict,
        filters: List[Dict[str, Any]],
        output_columns: List[str],
    ) -> Optional[str]:
        columns = [str(c) for c in ((profile or {}).get("columns") or [])]
        dtypes = (profile or {}).get("dtypes") or {}
        numeric_cols = [c for c in columns if _is_numeric_dtype_text(dtypes.get(c, ""))]
        if not numeric_cols:
            return None

        numeric_out = [c for c in output_columns if c in numeric_cols]
        if numeric_out:
            return numeric_out[0]

        explicit_col = _find_explicit_column_in_text(question, columns)
        if explicit_col in numeric_cols:
            return explicit_col

        metric_hint = self._llm_pick_numeric_metric_column(question, profile)
        if metric_hint in numeric_cols:
            return metric_hint

        price_col = _pick_price_like_column(profile)
        qty_col = _pick_quantity_like_column(profile)
        if re.search(r"\b(цін\w*|price|cost|варт\w*)\b", question or "", re.I) and price_col in numeric_cols:
            return price_col
        if re.search(r"\b(кільк\w*|qty|quantity|units?|stock|залишк\w*)\b", question or "", re.I) and qty_col in numeric_cols:
            return qty_col

        filter_numeric_cols = [
            str(f.get("column") or "").strip()
            for f in (filters or [])
            if str(f.get("column") or "").strip() in numeric_cols
        ]
        if filter_numeric_cols:
            return filter_numeric_cols[0]

        non_id_numeric = [c for c in numeric_cols if not _is_id_like_col_name(c)]
        if non_id_numeric:
            return non_id_numeric[0]
        return numeric_cols[0]

    def _llm_pick_lookup_slots(self, question: str, profile: dict) -> Dict[str, Any]:
        columns = [str(c) for c in (profile or {}).get("columns") or []]
        if not columns:
            return {}
        dtypes = (profile or {}).get("dtypes") or {}
        system = (
            "Map the query to table lookup slots. "
            "Return ONLY JSON with keys: mode, filters, output_columns, limit, aggregation. "
            "mode must be 'lookup' or 'other'. "
            "filters must be a list of objects with keys: column, op, value. "
            "column must be exact from columns. op must be one of: eq, ne, gt, ge, lt, le, contains, startswith, endswith. "
            "output_columns must be a list of exact column names from columns. "
            "limit must be integer or null. "
            "aggregation must be one of: none, count, sum, mean, min, max, median. "
            "CRITICAL FILTER RULES: "
            "Use 'eq' for exact IDs/status/boolean flags and explicit exact-match asks. "
            "Use 'contains' for brand/model/product names, features/materials, colors, categories/types. "
            "If user asks 'starts with / починаються / на літеру', use op='startswith' with the prefix. "
            "If user asks 'ends with / закінчується на', use op='endswith' with the suffix. "
            "Do NOT use regex anchors like '^' or '$' in values. "
            "'contains' is plain substring matching, not regex. "
            "When in doubt between 'eq' and 'contains' for text values, prefer 'contains'. "
            "If query likely needs search across multiple text columns, set mode='other'. "
            "Use mode='lookup' when query asks to find/show rows by conditions (e.g., where price equals X). "
            "For single-value row questions like 'яка модель ...', set limit=1 and aggregation='none'. "
            "For metric questions with aggregate intent (max/min/sum/avg/count), set aggregation accordingly and limit=null. "
            "Never emulate max/min/sum/avg/count via limit=1. "
            "AMBIGUITY RULE: if an entity value is given without explicit column (e.g., 'товар Миша'), "
            "prefer category/type columns over model/description when assigning filter column."
        )
        system = self._with_spreadsheet_skill_prompt(system, question, profile, focus="column")
        payload = {
            "question": question,
            "columns": columns[:200],
            "numeric_columns": _profile_numeric_columns(profile, limit=200),
            "rows": (profile or {}).get("rows"),
        }
        try:
            parsed = self._llm_json(system, json.dumps(payload, ensure_ascii=False))
        except Exception:
            return {}

        out: Dict[str, Any] = {}
        mode = str((parsed or {}).get("mode") or "").strip().lower()
        if mode not in {"lookup", "other"}:
            mode = "other"
        out["mode"] = mode

        filters_raw = (parsed or {}).get("filters")
        filters: List[Dict[str, Any]] = []
        if isinstance(filters_raw, list):
            for f in filters_raw:
                if not isinstance(f, dict):
                    continue
                raw_col = str(f.get("column") or "").strip()
                col = raw_col
                if col and col not in columns:
                    mapped = self._llm_pick_semantic_lookup_column(
                        question=question,
                        alias=col,
                        profile=profile,
                        role="filter_column",
                    )
                    if mapped:
                        col = mapped
                op = str(f.get("op") or "").strip().lower()
                if col not in columns:
                    continue
                if op not in LOOKUP_ALLOWED_FILTER_OPS:
                    continue
                val = f.get("value")
                filters.append({"column": col, "op": op, "value": val})
        filters = self._normalize_lookup_filters(question, filters, dtypes)
        out["filters"] = filters

        out_cols: List[str] = []
        out_cols_raw = (parsed or {}).get("output_columns")
        if isinstance(out_cols_raw, list):
            for c in out_cols_raw:
                s = str(c).strip()
                if s and s not in columns:
                    mapped = self._llm_pick_semantic_lookup_column(
                        question=question,
                        alias=s,
                        profile=profile,
                        role="output_column",
                    )
                    if mapped:
                        s = mapped
                if s in columns and s not in out_cols:
                    out_cols.append(s)
        out["output_columns"] = out_cols

        aggregation = str((parsed or {}).get("aggregation") or "").strip().lower()
        if aggregation not in LOOKUP_ALLOWED_AGGREGATIONS:
            aggregation = "none"
        inferred_agg = self._lookup_infer_aggregation_from_question(question)
        if aggregation == "none" and inferred_agg != "none":
            aggregation = inferred_agg
        out["aggregation"] = aggregation

        limit_raw = (parsed or {}).get("limit")
        if isinstance(limit_raw, (int, float)):
            out["limit"] = max(1, int(limit_raw))
        elif isinstance(limit_raw, str) and limit_raw.strip().isdigit():
            out["limit"] = max(1, int(limit_raw.strip()))
        if aggregation != "none":
            out.pop("limit", None)

        if (
            mode == "lookup"
            and aggregation == "none"
            and self._lookup_requires_multicol_fallback(question, filters, columns, dtypes)
        ):
            logging.info("event=lookup_slots_fallback mode=other reason=multi_column_keyword_search")
            out["mode"] = "other"
            out["fallback_reason"] = "multi_column_keyword_search"

        return out

    def _lookup_hints_from_slots(self, slots: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(slots, dict):
            return None
        filters = [dict(f) for f in ((slots or {}).get("filters") or []) if isinstance(f, dict)]
        out_cols = [str(c) for c in ((slots or {}).get("output_columns") or []) if str(c).strip()]
        limit = (slots or {}).get("limit")
        aggregation = str((slots or {}).get("aggregation") or "").strip().lower()
        fallback_reason = str((slots or {}).get("fallback_reason") or "").strip()
        mode = str((slots or {}).get("mode") or "").strip().lower()
        if not filters and not out_cols:
            return None
        hints: Dict[str, Any] = {
            "mode": mode or "other",
            "filters": filters,
        }
        if out_cols:
            hints["output_columns"] = out_cols
        if isinstance(limit, int) and limit > 0:
            hints["limit"] = limit
        if aggregation in LOOKUP_ALLOWED_AGGREGATIONS and aggregation != "none":
            hints["aggregation"] = aggregation
        if fallback_reason:
            hints["reason"] = fallback_reason
        return hints

    def _lookup_question_has_detail_rows_cue(self, question: str) -> bool:
        q = (question or "").lower()
        if not q:
            return False
        return bool(
            re.search(
                r"\b(усі\s+рядк\w*|всі\s+рядк\w*|all\s+rows?|every\s+row|"
                r"кож\w*\s+товар\w*|every\s+item|детальн\w*|повн\w*\s+інформац\w*|"
                r"full\s+details?|всі\s+колонк\w*|all\s+columns?)\b",
                q,
                re.I,
            )
        )

    def _lookup_question_has_unique_cue(self, question: str) -> bool:
        q = (question or "").lower()
        if not q:
            return False
        return bool(
            re.search(
                r"\b(унікальн\w*|без\s+дублік\w*|distinct|unique|different|deduplicat\w*)\b",
                q,
                re.I,
            )
        )

    def _lookup_should_deduplicate_output(
        self,
        question: str,
        output_columns: List[str],
        profile: dict,
    ) -> bool:
        cols = [str(c) for c in (output_columns or []) if str(c).strip()]
        if not cols:
            return False
        if self._lookup_question_has_unique_cue(question):
            return True
        if self._lookup_question_has_detail_rows_cue(question):
            return False

        dtypes = (profile or {}).get("dtypes") or {}
        for col in cols:
            if _is_id_like_col_name(col):
                return False
            if _is_numeric_dtype_text(dtypes.get(col, "")):
                return False
        return True

    def _lookup_shortcut_code_from_slots(
        self,
        question: str,
        profile: dict,
        slots: Dict[str, Any],
    ) -> Optional[Tuple[str, str]]:
        if str((slots or {}).get("mode") or "") != "lookup":
            return None
        filters = [f for f in ((slots or {}).get("filters") or []) if isinstance(f, dict)]
        if not filters:
            return None

        columns = [str(c) for c in ((profile or {}).get("columns") or [])]
        dtypes = (profile or {}).get("dtypes") or {}
        aggregation = str((slots or {}).get("aggregation") or "").strip().lower()
        if aggregation not in LOOKUP_ALLOWED_AGGREGATIONS:
            aggregation = "none"
        inferred_aggregation = self._lookup_infer_aggregation_from_question(question)
        if aggregation == "none" and inferred_aggregation != "none":
            aggregation = inferred_aggregation

        output_columns = [str(c) for c in ((slots or {}).get("output_columns") or []) if str(c) in columns]
        if aggregation == "count":
            output_columns = []
        elif aggregation != "none":
            if not output_columns:
                agg_col = self._lookup_pick_aggregation_column(question, profile, filters, output_columns)
                if agg_col:
                    output_columns = [agg_col]
        else:
            if not output_columns:
                mentioned = _find_columns_in_text(question, columns)
                output_columns = [c for c in mentioned if c not in [str(f.get("column")) for f in filters]]
            if not output_columns:
                text_cols = [c for c in columns if not str(dtypes.get(c, "")).lower().startswith(("int", "float", "uint"))]
                output_columns = text_cols[:2] if text_cols else columns[:1]

        dedupe_output = aggregation == "none" and self._lookup_should_deduplicate_output(question, output_columns, profile)
        limit = (slots or {}).get("limit")
        if aggregation != "none":
            limit = None
        fallback_candidates_by_idx: Dict[int, List[str]] = {}
        explicit_col = _find_explicit_column_in_text(question, columns)
        if not explicit_col:
            for i, f in enumerate(filters):
                col = str(f.get("column") or "").strip()
                op = str(f.get("op") or "").strip().lower()
                val = f.get("value")
                if op not in {"eq", "contains"}:
                    continue
                if isinstance(val, (int, float)):
                    continue
                candidates = self._lookup_ambiguous_filter_candidate_columns(
                    question=question,
                    filter_col=col,
                    value=val,
                    columns=columns,
                    dtypes=dtypes,
                )
                if candidates:
                    fallback_candidates_by_idx[i] = candidates

        op_text = {
            "eq": "=",
            "ne": "!=",
            "gt": ">",
            "ge": ">=",
            "lt": "<",
            "le": "<=",
            "contains": "contains",
            "startswith": "startswith",
            "endswith": "endswith",
        }
        filter_hint_parts: List[str] = []
        for f in filters:
            col = str(f.get("column") or "").strip()
            op = str(f.get("op") or "").strip().lower()
            val = "" if f.get("value") is None else str(f.get("value"))
            if not col or not val:
                continue
            filter_hint_parts.append(f"{col} {op_text.get(op, op)} {val}")
        filter_hint = "; ".join(filter_hint_parts) if filter_hint_parts else "задані умови"

        lines: List[str] = [
            "_work = df.copy(deep=False)",
        ]
        for i, f in enumerate(filters):
            col = str(f.get("column"))
            op = str(f.get("op")).lower()
            val = f.get("value")
            match_mode = str(f.get("match_mode") or "").strip().lower()
            if op not in LOOKUP_ALLOWED_FILTER_OPS:
                continue
            mask_name = f"_m{i}"
            fallback_cols = fallback_candidates_by_idx.get(i) or []
            fallback_enabled = bool(fallback_cols and op in {"eq", "contains"} and not isinstance(val, (int, float)))
            if fallback_enabled:
                lines.append(f"_base{i} = _work.copy()")
            lines.append(f"_c{i} = {col!r}")
            if op in {"gt", "ge", "lt", "le"}:
                try:
                    num_val = float(val)
                except Exception:
                    continue
                lines.append(f"_raw{i} = _work[_c{i}].astype(str)")
                lines.append(
                    rf"_clean{i} = _raw{i}.str.replace(r'[\s\xa0]', '', regex=True).str.replace(r'[^0-9,.\-]', '', regex=True).str.replace(',', '.')"
                )
                lines.append(rf"_masknum{i} = _clean{i}.str.match(r'^-?(\d+(\.\d*)?|\.\d+)$')")
                lines.append(f"_num{i} = _clean{i}.where(_masknum{i}, np.nan).astype(float)")
                op_map = {"gt": ">", "ge": ">=", "lt": "<", "le": "<="}
                lines.append(f"{mask_name} = _num{i} {op_map[op]} {num_val!r}")
            elif op == "startswith":
                sval = "" if val is None else str(val)
                lines.append(f"{mask_name} = _work[_c{i}].astype(str).str.lower().str.startswith(str({sval!r}).lower(), na=False)")
            elif op == "endswith":
                sval = "" if val is None else str(val)
                lines.append(f"{mask_name} = _work[_c{i}].astype(str).str.lower().str.endswith(str({sval!r}).lower(), na=False)")
            elif op == "contains":
                sval = "" if val is None else str(val)
                status_pat = self._lookup_status_pattern(val) if self._lookup_is_status_like_column(col) else None
                if status_pat:
                    lines.append(f"_pat{i} = {status_pat!r}")
                    lines.append(f"{mask_name} = _work[_c{i}].astype(str).str.contains(_pat{i}, case=False, regex=True, na=False)")
                else:
                    lines.append(f"{mask_name} = _work[_c{i}].astype(str).str.contains({sval!r}, case=False, regex=False, na=False)")
            else:
                # eq/ne: decide numeric or string compare by value shape
                if isinstance(val, (int, float)):
                    lines.append(f"_raw{i} = _work[_c{i}].astype(str)")
                    lines.append(
                        rf"_clean{i} = _raw{i}.str.replace(r'[\s\xa0]', '', regex=True).str.replace(r'[^0-9,.\-]', '', regex=True).str.replace(',', '.')"
                    )
                    lines.append(rf"_masknum{i} = _clean{i}.str.match(r'^-?(\d+(\.\d*)?|\.\d+)$')")
                    lines.append(f"_num{i} = _clean{i}.where(_masknum{i}, np.nan).astype(float)")
                    eq_op = "==" if op == "eq" else "!="
                    lines.append(f"{mask_name} = _num{i} {eq_op} {float(val)!r}")
                else:
                    sval = "" if val is None else str(val)
                    cmp_op = "==" if op == "eq" else "!="
                    strict_status_eq = (
                        self._lookup_is_status_like_column(col)
                        and (
                            match_mode == "exact"
                            or self._lookup_question_requests_exact_value(question, val)
                        )
                    )
                    status_pat = (
                        None
                        if strict_status_eq
                        else (self._lookup_status_pattern(val) if self._lookup_is_status_like_column(col) else None)
                    )
                    if status_pat:
                        lines.append(f"_pat{i} = {status_pat!r}")
                        lines.append(f"_hit{i} = _work[_c{i}].astype(str).str.contains(_pat{i}, case=False, regex=True, na=False)")
                        if op == "eq":
                            lines.append(f"{mask_name} = _hit{i}")
                        else:
                            lines.append(f"{mask_name} = ~_hit{i}")
                    else:
                        lines.append(
                            f"{mask_name} = _work[_c{i}].astype(str).str.strip().str.lower() {cmp_op} {sval!r}.strip().lower()"
                        )
            lines.append(f"_work = _work.loc[{mask_name}].copy()")
            if fallback_enabled:
                sval = "" if val is None else str(val)
                lines.append(f"_resolved_col{i} = _c{i}")
                lines.append(f"if len(_work) == 0:")
                lines.append(f"    _alt_cols{i} = {fallback_cols!r}")
                lines.append(f"    for _alt_col{i} in _alt_cols{i}:")
                if op == "contains":
                    lines.append(
                        f"        _alt_mask{i} = _base{i}[_alt_col{i}].astype(str).str.contains({sval!r}, case=False, regex=False, na=False)"
                    )
                else:
                    lines.append(
                        f"        _alt_eq{i} = _base{i}[_alt_col{i}].astype(str).str.strip().str.lower() == {sval!r}.strip().lower()"
                    )
                    lines.append(f"        _alt_mask{i} = _alt_eq{i}")
                    lines.append(f"        if not bool(_alt_mask{i}.any()):")
                    lines.append(
                        f"            _alt_mask{i} = _base{i}[_alt_col{i}].astype(str).str.contains({sval!r}, case=False, regex=False, na=False)"
                    )
                lines.append(f"        _alt_work{i} = _base{i}.loc[_alt_mask{i}].copy()")
                lines.append(f"        if len(_alt_work{i}) > 0:")
                lines.append(f"            _work = _alt_work{i}")
                lines.append(f"            _resolved_col{i} = _alt_col{i}")
                lines.append("            break")

        if aggregation != "none":
            if aggregation == "count":
                lines.append("result = int(len(_work))")
                plan = "Відфільтрувати рядки за умовами та порахувати їх кількість."
                logging.info("event=lookup_shortcut_llm slots=%s", _safe_trunc(slots, 600))
                return "\n".join(lines) + "\n", plan

            agg_col = output_columns[0] if output_columns else None
            if not agg_col:
                agg_col = self._lookup_pick_aggregation_column(question, profile, filters, output_columns)
            if not agg_col or agg_col not in columns:
                logging.info(
                    "event=lookup_shortcut_skip reason=missing_aggregation_column aggregation=%s question=%s",
                    aggregation,
                    _safe_trunc(question, 200),
                )
                return None
            if not _is_numeric_dtype_text(dtypes.get(agg_col, "")):
                logging.info(
                    "event=lookup_shortcut_skip reason=non_numeric_aggregation_column aggregation=%s column=%s",
                    aggregation,
                    agg_col,
                )
                return None
            lines.append(f"_filter_hint = {filter_hint!r}")
            lines.append("if len(_work) == 0:")
            lines.append("    result = f\"Не знайдено рядків для фільтра: {_filter_hint}\"")
            lines.append("else:")
            lines.append(f"    _agg_col = {agg_col!r}")
            lines.append("    _raw_agg = _work[_agg_col].astype(str)")
            lines.append(
                r"    _clean_agg = _raw_agg.str.replace(r'[\s\xa0]', '', regex=True).str.replace(r'[^0-9,.\-]', '', regex=True).str.replace(',', '.')"
            )
            lines.append(r"    _masknum_agg = _clean_agg.str.match(r'^-?(\d+(\.\d*)?|\.\d+)$')")
            lines.append("    _num_agg = _clean_agg.where(_masknum_agg, np.nan).astype(float)")
            lines.append("    _num_non_na = _num_agg.dropna()")
            lines.append("    if len(_num_non_na) == 0:")
            lines.append("        result = f\"Не знайдено числових значень у колонці {_agg_col} для вибраних рядків.\"")
            lines.append("    else:")
            if aggregation == "max":
                lines.append("        _agg_value = _num_non_na.max()")
            elif aggregation == "min":
                lines.append("        _agg_value = _num_non_na.min()")
            elif aggregation == "sum":
                lines.append("        _agg_value = _num_non_na.sum()")
            elif aggregation == "mean":
                lines.append("        _agg_value = _num_non_na.mean()")
            elif aggregation == "median":
                lines.append("        _agg_value = _num_non_na.median()")
            else:
                lines.append("        _agg_value = _num_non_na.max()")
            lines.append("        result = float(_agg_value)")
            agg_label = {
                "max": "максимум",
                "min": "мінімум",
                "sum": "суму",
                "mean": "середнє",
                "median": "медіану",
            }.get(aggregation, aggregation)
            plan = f"Відфільтрувати рядки за умовами та обчислити {agg_label} по колонці {agg_col}."
            logging.info("event=lookup_shortcut_llm slots=%s", _safe_trunc(slots, 600))
            return "\n".join(lines) + "\n", plan

        lines.append(f"_out_cols = {output_columns!r}")
        lines.append("_out_cols = [c for c in _out_cols if c in _work.columns]")
        lines.append("if _out_cols:")
        lines.append("    _out = _work[_out_cols]")
        lines.append("else:")
        lines.append("    _out = _work")
        if dedupe_output:
            lines.append("_out = _out.drop_duplicates()")
        if isinstance(limit, int) and limit > 0:
            lines.append(f"_out = _out.head({limit})")
        lines.append("if len(_out) == 0:")
        lines.append("    result = []")
        lines.append("elif len(_out.columns) == 1 and len(_out) == 1:")
        lines.append("    result = _out.iloc[0, 0]")
        lines.append("else:")
        lines.append("    result = _out")

        plan = "Виконати пошук рядків за умовами та повернути релевантні колонки."
        if dedupe_output:
            plan = "Виконати пошук рядків за умовами та повернути унікальні значення релевантних колонок."
        logging.info("event=lookup_shortcut_llm slots=%s", _safe_trunc(slots, 600))
        return "\n".join(lines) + "\n", plan

    def _lookup_shortcut_code(self, question: str, profile: dict) -> Optional[Tuple[str, str]]:
        slots = self._llm_pick_lookup_slots(question, profile)
        return self._lookup_shortcut_code_from_slots(question, profile, slots)

    def _ranking_shortcut_code(self, question: str, profile: dict) -> Optional[Tuple[str, str]]:
        slots = self._llm_pick_ranking_slots(question, profile)
        mode = str((slots or {}).get("query_mode") or "")
        if mode not in {"row_ranking", "group_ranking"}:
            return None
        top_n_slot = (slots or {}).get("top_n")
        top_n = _normalize_optional_top_n(top_n_slot)
        if top_n is None:
            logging.info("event=ranking_shortcut_skip reason=missing_top_n mode=%s", mode)
            return None

        columns = [str(c) for c in ((profile or {}).get("columns") or [])]
        dtypes = (profile or {}).get("dtypes") or {}
        if mode == "row_ranking":
            if not _has_ranking_cues(question) and _looks_like_value_filter_query(question):
                logging.info("event=ranking_shortcut_skip reason=filter_like_query mode=row")
                return None
            top_n = int(top_n)
        else:
            top_n = int(top_n)
        order = "asc" if str((slots or {}).get("order") or "").lower() == "asc" else "desc"
        require_available = bool((slots or {}).get("require_available"))
        availability_col = str((slots or {}).get("availability_col") or "").strip() or None
        lines = [
            "_src = df.copy(deep=False)",
            f"_avail_col = {availability_col!r}" if availability_col else "_avail_col = None",
            f"_require_avail = {require_available!r}",
            f"_top_n = {top_n}",
            f"_order = {order!r}",
            "_work = _src",
            "if _require_avail and _avail_col and (_avail_col in _work.columns):",
            "    _st = _work[_avail_col].astype(str).str.strip().str.lower()",
            r"    _in = _st.str.contains(r'(?:в\s*наявн|наявн|in\s*stock|available|доступн|резерв\w*|закінч\w*)', regex=True, na=False)",
            r"    _out = _st.str.contains(r'(?:нема|відсутн|out\s*of\s*stock|unavailable|not\s*available)', regex=True, na=False)",
            r"    _ord = _st.str.contains(r'(?:під\s*замовлення|under\s*order|backorder)', regex=True, na=False)",
            "    _work = _work.loc[_in & ~_out & ~_ord].copy()",
        ]

        if mode == "row_ranking":
            metric_col = str((slots or {}).get("metric_col") or "").strip()
            if not metric_col:
                return None

            entity_cols = [str(c) for c in ((slots or {}).get("entity_cols") or []) if str(c) in columns]
            if not entity_cols:
                text_cols = [
                    c for c in columns
                    if not str(dtypes.get(c, "")).lower().startswith(("int", "float", "uint"))
                    and c != availability_col
                    and c != metric_col
                ]
                entity_cols = text_cols[:3]

            out_cols: List[str] = []
            id_col = _pick_id_like_column([str(c) for c in columns])
            preferred_cols = ([id_col] if id_col else []) + entity_cols + [metric_col]
            for c in preferred_cols:
                if c in columns and c not in out_cols:
                    out_cols.append(c)
            if not out_cols:
                out_cols = [metric_col]

            lines.extend(
                [
                    f"_metric_col = {metric_col!r}",
                    "_metric_raw = _work[_metric_col].astype(str)",
                    r"_metric_clean = _metric_raw.str.replace(r'[\s\xa0]', '', regex=True).str.replace(r'[^0-9,.\-]', '', regex=True).str.replace(',', '.')",
                    r"_metric_mask = _metric_clean.str.match(r'^-?(\d+(\.\d*)?|\.\d+)$')",
                    "_metric = _metric_clean.where(_metric_mask, np.nan).astype(float)",
                    "_work = _work.loc[_metric.notna()].copy()",
                    "_work[_metric_col] = _metric.loc[_metric.notna()]",
                    "if _order == 'asc':",
                    "    _work = _work.nsmallest(_top_n, _metric_col)",
                    "else:",
                    "    _work = _work.nlargest(_top_n, _metric_col)",
                    f"_out_cols = {out_cols!r}",
                    "_out_cols = [c for c in _out_cols if c in _work.columns]",
                    "result = _work[_out_cols] if _out_cols else _work",
                ]
            )
            plan = f"Побудувати {'топ' if order == 'desc' else 'нижні'}-{top_n} рядків за метрикою {metric_col}."
            logging.info("event=ranking_shortcut_llm mode=row slots=%s", _safe_trunc(slots, 600))
            return "\n".join(lines) + "\n", plan

        group_col = str((slots or {}).get("group_col") or "").strip()
        agg = str((slots or {}).get("agg") or "").strip().lower()
        target_col = str((slots or {}).get("target_col") or "").strip()
        metric_col = str((slots or {}).get("metric_col") or "").strip()
        if not group_col or agg not in {"count", "sum", "mean", "min", "max", "median"}:
            return None
        if agg != "count" and not target_col and metric_col in columns and _is_numeric_dtype_text(dtypes.get(metric_col, "")):
            target_col = metric_col
        if agg == "sum" and not target_col:
            qty_col = _pick_quantity_like_column(profile)
            if qty_col and qty_col in columns:
                target_col = qty_col
        if agg != "count" and not target_col:
            return None

        out_name = "count" if agg == "count" else f"{agg}_{target_col}"
        lines.append(f"_group_col = {group_col!r}")
        lines.append(f"_agg = {agg!r}")
        lines.append(f"_target_col = {target_col!r}" if target_col else "_target_col = None")
        lines.append("if _agg == 'count':")
        lines.append("    if _target_col:")
        lines.append("        _res = _work.groupby(_group_col)[_target_col].nunique(dropna=True).reset_index(name='count')")
        lines.append("    else:")
        lines.append("        _res = _work.groupby(_group_col).size().reset_index(name='count')")
        lines.append("else:")
        lines.append("    _num_raw = _work[_target_col].astype(str)")
        lines.append(r"    _num_clean = _num_raw.str.replace(r'[\s\xa0]', '', regex=True).str.replace(r'[^0-9,.\-]', '', regex=True).str.replace(',', '.')")
        lines.append(r"    _num_mask = _num_clean.str.match(r'^-?(\d+(\.\d*)?|\.\d+)$')")
        lines.append("    _num = _num_clean.where(_num_mask, np.nan).astype(float)")
        lines.append("    _work = _work.loc[_num.notna()].copy()")
        lines.append("    _work[_target_col] = _num.loc[_num.notna()]")
        lines.append("    if _agg == 'sum':")
        lines.append(f"        _res = _work.groupby(_group_col)[_target_col].sum().reset_index(name={out_name!r})")
        lines.append("    elif _agg == 'mean':")
        lines.append(f"        _res = _work.groupby(_group_col)[_target_col].mean().reset_index(name={out_name!r})")
        lines.append("    elif _agg == 'min':")
        lines.append(f"        _res = _work.groupby(_group_col)[_target_col].min().reset_index(name={out_name!r})")
        lines.append("    elif _agg == 'max':")
        lines.append(f"        _res = _work.groupby(_group_col)[_target_col].max().reset_index(name={out_name!r})")
        lines.append("    else:")
        lines.append(f"        _res = _work.groupby(_group_col)[_target_col].median().reset_index(name={out_name!r})")
        lines.append("if _order == 'asc':")
        lines.append(f"    _res = _res.sort_values({out_name!r}, ascending=True)")
        lines.append("else:")
        lines.append(f"    _res = _res.sort_values({out_name!r}, ascending=False)")
        lines.append("result = _res.head(_top_n)")
        plan = f"Побудувати {'топ' if order == 'desc' else 'нижні'}-{top_n} груп за {group_col} ({agg})."
        logging.info("event=ranking_shortcut_llm mode=group slots=%s", _safe_trunc(slots, 600))
        return "\n".join(lines) + "\n", plan

    def _llm_pick_subset_plan_slots(self, question: str, profile: dict) -> Dict[str, Any]:
        columns = [str(c) for c in (profile or {}).get("columns") or []]
        if not columns:
            return {"terms": [], "slots": {}}
        numeric_columns = _profile_numeric_columns(profile, limit=200)
        system = (
            "Map subset filtering query to one compact plan. "
            "Return ONLY JSON with keys: agg, metric_col, filter_terms, availability_mode, availability_col. "
            "agg must be one of: count, sum, mean, min, max, median. "
            "metric_col and availability_col must be exact names from columns or empty string. "
            "availability_mode must be one of: in, out, any, none. "
            "filter_terms must include product/entity/category/feature terms only. "
            "Exclude metric words and aggregation words like max/min/avg/sum/count/price/ціна/кількість. "
            "If useful, include 1-2 short aliases in other likely languages (uk/ru/en) that may appear in table values. "
            "Prefer 1-6 concise terms."
        )
        system = self._with_spreadsheet_skill_prompt(system, question, profile, focus="column")
        payload = {
            "question": question,
            "columns": columns[:200],
            "numeric_columns": numeric_columns,
        }
        try:
            parsed = self._llm_json(system, json.dumps(payload, ensure_ascii=False))
        except Exception:
            return {"terms": [], "slots": {}}
        out_terms: List[str] = []
        raw = (parsed or {}).get("filter_terms")
        if isinstance(raw, list):
            for t in raw:
                s = str(t).strip()
                if len(s) < 2 or len(s) > 40:
                    continue
                if re.fullmatch(r"[\d\s.,:%\-]+", s):
                    continue
                if s not in out_terms:
                    out_terms.append(s)
        out_slots: Dict[str, Any] = {}
        agg = str((parsed or {}).get("agg") or "").strip().lower()
        metric_col = str((parsed or {}).get("metric_col") or "").strip()
        availability_mode = str((parsed or {}).get("availability_mode") or "").strip().lower()
        availability_col = str((parsed or {}).get("availability_col") or "").strip()
        if agg in {"count", "sum", "mean", "min", "max", "median"}:
            out_slots["agg"] = agg
        if metric_col in columns:
            out_slots["metric_col"] = metric_col
        if availability_mode in {"in", "out", "any", "none"}:
            out_slots["availability_mode"] = availability_mode
        if availability_col in columns:
            out_slots["availability_col"] = availability_col
        logging.info(
            "event=subset_plan_llm terms=%s slots=%s",
            _safe_trunc(out_terms, 240),
            _safe_trunc(out_slots, 280),
        )
        return {"terms": out_terms[:6], "slots": out_slots}

    def _llm_extract_subset_terms(self, question: str, profile: dict) -> List[str]:
        return list((self._llm_pick_subset_plan_slots(question, profile) or {}).get("terms") or [])

    def _llm_pick_subset_metric_slots(self, question: str, profile: dict) -> Dict[str, Any]:
        slots = (self._llm_pick_subset_plan_slots(question, profile) or {}).get("slots")
        return dict(slots) if isinstance(slots, dict) else {}

    def _build_subset_keyword_metric_shortcut(
        self,
        question: str,
        profile: dict,
        preferred_col: Optional[str] = None,
    ) -> Optional[Tuple[str, str]]:
        # Heuristic-first path avoids extra LLM latency and is robust to non-JSON "thinking" outputs.
        shortcut = _subset_keyword_metric_shortcut_code(
            question,
            profile,
            preferred_col=preferred_col,
            terms_hint=[],
            slots_hint={},
        )
        if shortcut:
            return shortcut

        subset_plan = self._llm_pick_subset_plan_slots(question, profile)
        terms_hint = list((subset_plan or {}).get("terms") or [])
        slots_hint = dict((subset_plan or {}).get("slots") or {})
        return _subset_keyword_metric_shortcut_code(
            question,
            profile,
            preferred_col=preferred_col,
            terms_hint=terms_hint,
            slots_hint=slots_hint,
        )

    def _select_router_or_shortcut(
        self,
        question: str,
        profile: dict,
        has_edit: bool,
    ) -> Tuple[Optional[Tuple[str, Dict[str, Any]]], Optional[Tuple[str, str]], Dict[str, Any], Dict[str, Any]]:
        router_hit: Optional[Tuple[str, Dict[str, Any]]] = None
        shortcut: Optional[Tuple[str, str]] = None
        router_meta: Dict[str, Any] = {}
        planner_hints: Dict[str, Any] = {}

        candidate_hit = None if has_edit else self._shortcut_router.shortcut_to_sandbox_code(question, profile)
        if candidate_hit:
            candidate_code, candidate_meta = candidate_hit
            if _should_reject_router_hit_for_read(has_edit, candidate_code, candidate_meta, question, profile):
                logging.warning(
                    "event=shortcut_router status=rejected reason=read_guard intent_id=%s question=%s",
                    (candidate_meta or {}).get("intent_id"),
                    _safe_trunc(question, 200),
                )
            else:
                router_meta = candidate_meta or {}
                router_hit = (candidate_code, router_meta)
                logging.info("event=shortcut_router status=ok meta=%s", _safe_trunc(router_meta, 800))
                return router_hit, None, router_meta, planner_hints

        if has_edit:
            logging.info("event=shortcut_router status=skipped reason=edit_intent")
        else:
            logging.info("event=shortcut_router status=miss question=%s", _safe_trunc(question, 200))

        metrics = _detect_metrics(question)
        is_meta = _is_meta_task_text(question)
        inferred_op = _infer_op_from_question(question)
        requires_subset_filter = bool(
            inferred_op == "read" and not is_meta and not has_edit and _question_requires_subset_filter(question, profile)
        )
        aggregate_intent = _is_aggregate_query_intent(question)

        metric_col_hint: Optional[str] = None
        if (
            inferred_op == "read"
            and not is_meta
            and not has_edit
            and metrics
            and not (len(metrics) == 1 and metrics[0] == "count")
        ):
            metric_col_hint = self._llm_pick_numeric_metric_column(question, profile)

        # Prefer deterministic subset+aggregation shortcut for aggregate filtered requests.
        if (
            inferred_op == "read"
            and not is_meta
            and not has_edit
            and requires_subset_filter
            and aggregate_intent
        ):
            shortcut = self._build_subset_keyword_metric_shortcut(
                question,
                profile,
                preferred_col=metric_col_hint,
            )

        if inferred_op == "read" and not is_meta and not has_edit:
            if not shortcut:
                lookup_slots = self._llm_pick_lookup_slots(question, profile)
                lookup_hints = self._lookup_hints_from_slots(lookup_slots)
                if lookup_hints:
                    planner_hints["lookup_hints"] = lookup_hints
                shortcut = self._lookup_shortcut_code_from_slots(question, profile, lookup_slots)
        if not shortcut and inferred_op == "read" and not is_meta and not has_edit:
            shortcut = self._ranking_shortcut_code(question, profile)
        if not shortcut and inferred_op == "read" and not is_meta and not has_edit and requires_subset_filter:
            shortcut = self._build_subset_keyword_metric_shortcut(
                question,
                profile,
                preferred_col=metric_col_hint,
            )
        if (
            not shortcut
            and inferred_op == "read"
            and not is_meta
            and not has_edit
            and not requires_subset_filter
            and len(metrics) >= 2
        ):
            shortcut = _stats_shortcut_code(question, profile, preferred_col=metric_col_hint)
        if not shortcut:
            allow_availability_shortcut = True
            if _is_availability_count_intent(question):
                allow_availability_shortcut = self._should_use_availability_shortcut(question, profile)
            shortcut = _template_shortcut_code(
                question,
                profile,
                allow_availability_shortcut=allow_availability_shortcut,
            )
        if not shortcut:
            shortcut = _edit_shortcut_code(question, profile)
        if not shortcut and not requires_subset_filter:
            shortcut = _stats_shortcut_code(question, profile, preferred_col=metric_col_hint)

        return None, shortcut, router_meta, planner_hints

    def _format_scalar_list_from_result(self, result_text: str, max_items: int = 100) -> Optional[str]:
        text = (result_text or "").strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except Exception:
            return None
        if not isinstance(data, list):
            return None
        if not data:
            return "Нічого не знайдено."
        if any(isinstance(x, (dict, list, tuple)) for x in data):
            return None
        values = [str(x).strip() for x in data if str(x).strip()]
        if not values:
            return "Нічого не знайдено."
        if len(values) == 1:
            return values[0]
        return "\n".join(f"- {v}" for v in values[: max(1, max_items)])

    def _should_use_availability_shortcut(self, question: str, profile: dict) -> bool:
        """
        Availability shortcut is safe only for global counts.
        For filtered intents (brand/category/model/etc.) we should let codegen build explicit filters.
        """
        columns = [str(c) for c in (profile or {}).get("columns") or []]
        if not columns:
            return True
        mentioned = _find_columns_in_text(question, columns)
        non_status_mentions = [
            c
            for c in mentioned
            if not _AVAILABILITY_COL_RE.search(str(c))
        ]
        if non_status_mentions:
            logging.info(
                "event=availability_shortcut_scope source=heuristic scope=filtered mentioned=%s",
                _safe_trunc(non_status_mentions, 200),
            )
            return False
        # Entity-like tokens (brand/model codes) usually indicate filtered subset.
        if re.search(r"\b[A-Z][A-Z0-9_-]{1,}\b", question or ""):
            logging.info("event=availability_shortcut_scope source=heuristic scope=filtered reason=entity_token")
            return False
        # Quoted explicit value also indicates a filter.
        if re.search(r"[\"'“”«»][^\"'“”«»]{2,}[\"'“”«»]", question or ""):
            logging.info("event=availability_shortcut_scope source=heuristic scope=filtered reason=quoted_value")
            return False

        system = (
            "Decide if the user asks a GLOBAL availability count over all rows, "
            "or a FILTERED subset count. Return ONLY JSON: "
            "{\"scope\":\"global\"|\"filtered\"}. "
            "Use filtered when query constrains category/brand/model/id/color/spec/price or named entities."
        )
        system = self._with_spreadsheet_skill_prompt(system, question, profile, focus="column")
        payload = {
            "question": question,
            "columns": columns[:200],
            "numeric_columns": _profile_numeric_columns(profile, limit=200),
        }
        try:
            parsed = self._llm_json(system, json.dumps(payload, ensure_ascii=False))
            scope = str((parsed or {}).get("scope") or "").strip().lower()
            if scope in {"filtered", "subset", "specific"}:
                logging.info("event=availability_shortcut_scope source=llm scope=filtered")
                return False
            if scope in {"global", "all", "overall"}:
                logging.info("event=availability_shortcut_scope source=llm scope=global")
                return True
        except Exception as exc:
            logging.warning("event=availability_shortcut_scope source=llm error=%s", str(exc))
        return True

    def _llm_pick_columns_by_role_for_shortcut(self, question: str, profile: dict) -> Dict[str, Any]:
        columns = [str(c) for c in (profile or {}).get("columns") or []]
        if not columns:
            return {}
        system = (
            "Map the user query to dataframe column roles. "
            "Return ONLY JSON with keys: group_by, aggregate, top_n. "
            "group_by and aggregate must be exact names from columns or empty string. "
            "top_n must be an integer or null. "
            "Use group_by for categorical grouping dimension and aggregate for numeric value to sum/count/average."
        )
        system = self._with_spreadsheet_skill_prompt(system, question, profile, focus="column")
        payload = {
            "question": question,
            "columns": columns[:200],
            "numeric_columns": _profile_numeric_columns(profile, limit=200),
        }
        try:
            parsed = self._llm_json(system, json.dumps(payload, ensure_ascii=False))
        except Exception:
            return {}
        out: Dict[str, Any] = {}
        group_by = str((parsed or {}).get("group_by") or "").strip()
        aggregate = str((parsed or {}).get("aggregate") or "").strip()
        top_n_raw = (parsed or {}).get("top_n")
        if group_by and group_by in columns:
            out["group_by"] = group_by
        if aggregate and aggregate in columns:
            out["aggregate"] = aggregate
        top_n = _normalize_optional_top_n(top_n_raw)
        if top_n is not None:
            out["top_n"] = top_n
        return out

    def _resolve_shortcut_placeholders(
        self, analysis_code: str, plan: str, question: str, profile: dict
    ) -> Tuple[str, str]:
        legacy_placeholder = "__SHORTCUT_COL__"
        has_single_col_ph = (
            SHORTCUT_COL_PLACEHOLDER in (analysis_code or "")
            or SHORTCUT_COL_PLACEHOLDER in (plan or "")
            or legacy_placeholder in (analysis_code or "")
            or legacy_placeholder in (plan or "")
        )
        has_group_ph = (
            GROUP_COL_PLACEHOLDER in (analysis_code or "")
            or GROUP_COL_PLACEHOLDER in (plan or "")
        )
        has_agg_ph = (
            SUM_COL_PLACEHOLDER in (analysis_code or "")
            or SUM_COL_PLACEHOLDER in (plan or "")
            or AGG_COL_PLACEHOLDER in (analysis_code or "")
            or AGG_COL_PLACEHOLDER in (plan or "")
        )
        has_top_n_ph = (
            TOP_N_PLACEHOLDER in (analysis_code or "")
            or TOP_N_PLACEHOLDER in (plan or "")
        )
        if (
            not has_single_col_ph
            and not has_group_ph
            and not has_agg_ph
            and not has_top_n_ph
        ):
            return analysis_code, plan
        columns = [str(c) for c in ((profile or {}).get("columns") or [])]

        found_columns = _find_columns_in_text(question, columns)
        role_map: Dict[str, Any] = {}
        if has_group_ph or has_agg_ph or has_top_n_ph:
            role_map.update(self._llm_pick_columns_by_role_for_shortcut(question, profile))
            fallback_roles = _classify_columns_by_role(question, found_columns, profile)
            if not role_map.get("group_by"):
                role_map["group_by"] = fallback_roles.get("group_by")
            if not role_map.get("aggregate"):
                role_map["aggregate"] = fallback_roles.get("aggregate")
            if not role_map.get("top_n"):
                role_map["top_n"] = _extract_top_n_from_question(question, default=10)

        llm_question = question
        if _is_availability_count_intent(question):
            llm_question = (
                f"{question}\n"
                "Intent: choose the column that stores availability/in-stock/out-of-stock status."
            )
        # LLM-first for semantic column resolution.
        col = self._llm_pick_column_for_shortcut(llm_question, profile) if has_single_col_ph else None
        if not col:
            col = role_map.get("group_by") or _find_column_in_text(question, columns)
        if not col and _is_availability_count_intent(question):
            col = _pick_availability_column(question, profile)
        replacements: Dict[str, str] = {}
        if col:
            replacements[SHORTCUT_COL_PLACEHOLDER] = col
            replacements[legacy_placeholder] = col
        group_by_col = role_map.get("group_by")
        aggregate_col = role_map.get("aggregate")
        top_n = role_map.get("top_n") or _extract_top_n_from_question(question, default=10)
        if group_by_col:
            replacements[GROUP_COL_PLACEHOLDER] = str(group_by_col)
        if aggregate_col:
            replacements[SUM_COL_PLACEHOLDER] = str(aggregate_col)
            replacements[AGG_COL_PLACEHOLDER] = str(aggregate_col)
        if top_n:
            replacements[TOP_N_PLACEHOLDER] = str(int(top_n))

        if not replacements:
            return analysis_code, plan
        for ph, val in replacements.items():
            analysis_code = (analysis_code or "").replace(ph, val)
            plan = (plan or "").replace(ph, val)
        logging.info(
            "event=placeholder_resolution found_cols=%s role_map=%s replacements=%s",
            _safe_trunc(found_columns, 300),
            _safe_trunc(role_map, 300),
            _safe_trunc(replacements, 300),
        )
        return analysis_code or "", plan or ""

    def _format_top_pairs_from_result(self, result_text: str, top_n: int = 15) -> Optional[str]:
        text = (result_text or "").strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except Exception:
            return None

        top_n_norm = int(top_n) if isinstance(top_n, int) else 0
        if top_n_norm <= 0:
            top_n_norm = 0

        lines: List[str] = []
        if isinstance(data, dict):
            items = list(data.items())
            try:
                items.sort(key=lambda kv: float(kv[1]) if kv[1] is not None else float("-inf"), reverse=True)
            except Exception:
                pass
            rows = items if top_n_norm == 0 else items[:top_n_norm]
            if rows:
                header = "| Ключ | Значення |"
                sep = "|---|---|"
                body = [f"| {k} | {v} |" for k, v in rows]
                return "\n".join([header, sep] + body)
        elif isinstance(data, list) and data and all(isinstance(x, dict) for x in data):
            first = data[0]
            keys = [str(k) for k in first.keys()]
            if keys:
                header = "| " + " | ".join(keys) + " |"
                sep = "| " + " | ".join(["---"] * len(keys)) + " |"
                top_rows = data if top_n_norm == 0 else data[:top_n_norm]
                body = []
                for row in top_rows:
                    body.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
                return "\n".join([header, sep] + body)
        if not lines:
            return None
        header = "Топ значень:"
        return header + "\n" + "\n".join(lines)

    def _format_table_from_result(self, result_text: str, top_n: int = 50) -> Optional[str]:
        text = (result_text or "").strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except Exception:
            return None
        top_n_norm = int(top_n) if isinstance(top_n, int) else 0
        if top_n_norm <= 0:
            top_n_norm = 0
        if isinstance(data, dict):
            rows = list(data.items()) if top_n_norm == 0 else list(data.items())[:top_n_norm]
            if not rows:
                return None
            header = "| Ключ | Значення |"
            sep = "|---|---|"
            body = [f"| {k} | {v} |" for k, v in rows]
            return "\n".join([header, sep] + body)
        if isinstance(data, list) and data and all(isinstance(x, dict) for x in data):
            keys = [str(k) for k in data[0].keys()]
            if not keys:
                return None
            header = "| " + " | ".join(keys) + " |"
            sep = "| " + " | ".join(["---"] * len(keys)) + " |"
            top_rows = data if top_n_norm == 0 else data[:top_n_norm]
            body = []
            for row in top_rows:
                body.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
            return "\n".join([header, sep] + body)
        return None

    def _format_structure_from_result(self, result_text: str) -> Optional[str]:
        text = (result_text or "").strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        rows = data.get("rows")
        cols = data.get("cols")
        if isinstance(rows, (int, float)) and isinstance(cols, (int, float)):
            return f"У таблиці {int(rows)} рядків і {int(cols)} стовпців."
        if isinstance(rows, (int, float)):
            return f"У таблиці {int(rows)} рядків."
        if isinstance(cols, (int, float)):
            return f"У таблиці {int(cols)} стовпців."
        return None

    def _format_availability_from_result(self, result_text: str) -> Optional[str]:
        text = (result_text or "").strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except Exception:
            data = None

        def _normalize_values(values: List[Any]) -> List[str]:
            out: List[str] = []
            for v in values:
                if v is None:
                    continue
                s = str(v).strip()
                if not s:
                    continue
                out.append(s)
            return out

        in_pat = re.compile(r"(в\s+наявн|наявн|in\s+stock|available)", re.I)
        out_pat = re.compile(r"(нема|відсутн|out\s+of\s+stock|unavailable)", re.I)

        if isinstance(data, list):
            values = _normalize_values(data)
            if not values:
                return "Немає даних про наявність."
            any_in = any(in_pat.search(v) for v in values)
            any_out = any(out_pat.search(v) for v in values)
            if any_in and not any_out:
                return "Так, є в наявності."
            if any_out and not any_in:
                return "Ні, немає в наявності."
            uniq: List[str] = []
            seen = set()
            for v in values:
                if v in seen:
                    continue
                seen.add(v)
                uniq.append(v)
            return "Статуси: " + ", ".join(uniq)
        if isinstance(data, dict):
            keys = _normalize_values(list(data.keys()))
            if not keys:
                return "Немає даних про наявність."
            stats_map = {
                "mean": "Середнє",
                "avg": "Середнє",
                "average": "Середнє",
                "min": "Мінімум",
                "max": "Максимум",
                "sum": "Сума",
                "count": "Кількість",
                "median": "Медіана",
            }
            stat_keys = [k for k in keys if k.lower() in stats_map]
            if stat_keys:
                parts: List[str] = []
                order = ["mean", "min", "max", "median", "sum", "count"]
                for k in order:
                    for orig in keys:
                        if orig.lower() == k:
                            val = data.get(orig)
                            if val is not None:
                                parts.append(f"{stats_map[k]}: {val}")
                if not parts:
                    for orig in stat_keys:
                        val = data.get(orig)
                        if val is not None:
                            parts.append(f"{stats_map[orig.lower()]}: {val}")
                if parts:
                    return "; ".join(parts)
            any_in = any(in_pat.search(v) for v in keys)
            any_out = any(out_pat.search(v) for v in keys)
            if any_in and not any_out:
                return "Так, є в наявності."
            if any_out and not any_in:
                return "Ні, немає в наявності."
            pairs: List[str] = []
            for k in keys:
                v = data.get(k)
                if v is None:
                    pairs.append(k)
                else:
                    pairs.append(f"{k}: {v}")
            return "Статуси: " + ", ".join(pairs)
        if isinstance(data, str):
            if in_pat.search(data) and not out_pat.search(data):
                return "Так, є в наявності."
            if out_pat.search(data) and not in_pat.search(data):
                return "Ні, немає в наявності."
            return f"Статус: {data}"

        if in_pat.search(text) and not out_pat.search(text):
            return "Так, є в наявності."
        if out_pat.search(text) and not in_pat.search(text):
            return "Ні, немає в наявності."
        return None

    def _aggregate_from_tabular_result(
        self,
        question: str,
        result_text: str,
        profile: Optional[dict],
    ) -> Optional[str]:
        if not _is_aggregate_query_intent(question):
            return None
        if _has_grouping_cues(question):
            return None
        text = (result_text or "").strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except Exception:
            return None
        if not isinstance(data, list) or not data or not all(isinstance(x, dict) for x in data):
            return None

        def _to_float(v: Any) -> Optional[float]:
            if v is None:
                return None
            if isinstance(v, (int, float)):
                try:
                    fv = float(v)
                except Exception:
                    return None
                if fv != fv:
                    return None
                return fv
            s = str(v).strip()
            if not s:
                return None
            s = re.sub(r"[\s\xa0]", "", s)
            s = re.sub(r"[^0-9,.\-]", "", s).replace(",", ".")
            if not re.fullmatch(r"-?(\d+(\.\d*)?|\.\d+)", s or ""):
                return None
            try:
                return float(s)
            except Exception:
                return None

        def _fmt_num(v: float) -> str:
            if float(v).is_integer():
                return str(int(v))
            return f"{float(v):.6f}".rstrip("0").rstrip(".")

        cols = [str(c) for c in (data[0] or {}).keys()]
        if not cols:
            return None

        metrics = _detect_metrics(question)
        metric = metrics[0] if metrics else ("sum" if _is_sum_intent(question) else ("count" if _is_count_intent(question) else ""))
        if not metric:
            return None

        # Disambiguate "загальна кількість ...": usually sum over qty-like column.
        if metric == "count" and _is_sum_intent(question):
            metric = "sum"

        if metric == "count":
            return f"Кількість — {len(data)}."

        numeric_by_col: Dict[str, List[float]] = {}
        for c in cols:
            nums: List[float] = []
            for row in data:
                val = _to_float((row or {}).get(c))
                if val is not None:
                    nums.append(val)
            if nums:
                numeric_by_col[c] = nums
        if not numeric_by_col:
            return None

        explicit_col = _find_explicit_column_in_text(question, cols) or _find_column_in_text(question, cols)
        qty_like = _pick_quantity_like_column(profile or {})
        target_col = None
        if explicit_col in numeric_by_col:
            target_col = explicit_col
        elif metric == "sum" and qty_like in numeric_by_col:
            target_col = qty_like
        if not target_col:
            target_col = max(numeric_by_col.keys(), key=lambda c: len(numeric_by_col[c]))

        values = numeric_by_col.get(target_col) or []
        if not values:
            return None

        if metric == "sum":
            val = float(sum(values))
        elif metric == "mean":
            val = float(sum(values) / len(values))
        elif metric == "min":
            val = float(min(values))
        elif metric == "max":
            val = float(max(values))
        elif metric == "median":
            vals = sorted(values)
            mid = len(vals) // 2
            if len(vals) % 2 == 0:
                val = float((vals[mid - 1] + vals[mid]) / 2.0)
            else:
                val = float(vals[mid])
        else:
            return None

        col_low = str(target_col or "").lower()
        if metric == "sum" and re.search(r"(кільк|qty|quantity|units|stock|залишк)", col_low):
            label = "Загальна кількість"
        elif metric == "sum":
            label = "Сума"
        elif metric == "mean":
            label = "Середнє"
        elif metric == "min":
            label = "Мінімум"
        elif metric == "max":
            label = "Максимум"
        else:
            label = "Медіана"
        return f"{label} — {_fmt_num(val)}."

    def _deterministic_answer(self, question: str, result_text: str, profile: Optional[dict]) -> Optional[str]:
        q = (question or "").lower()
        if any(word in q for word in ["видали", "додай", "зміни", "онов", "переймен"]):
            try:
                data = json.loads(result_text)
                if isinstance(data, dict) and 'status' in data:
                    return "Зміни успішно внесено та збережено."
            except:
                pass
            return result_text 

        aggregate_answer = self._aggregate_from_tabular_result(question, result_text, profile)
        if aggregate_answer:
            return aggregate_answer

        scalar = (result_text or "").strip()
        if scalar and "\n" not in scalar and not scalar.startswith(("{", "[", "|")):
            if _is_total_value_scalar_question(question, profile):
                if re.search(r"\b(uah|грн|₴)\b", scalar, re.I):
                    return f"Загальна вартість — {scalar}."
                return f"Загальна вартість — {scalar} UAH."
            row_idx = _parse_row_index(question or "")
            columns = [str(c) for c in (profile or {}).get("columns") or []]
            explicit_col = _find_explicit_column_in_text(question, columns)
            if row_idx and explicit_col:
                return f"{explicit_col} в рядку {row_idx} — {scalar}."
            if explicit_col:
                return f"{explicit_col} — {scalar}."
            if row_idx:
                return f"Значення в рядку {row_idx} — {scalar}."
            return scalar

        list_answer = self._format_scalar_list_from_result(
            result_text, max_items=int(self.valves.answer_list_max_items)
        )
        if list_answer:
            return list_answer

        table = self._format_table_from_result(result_text, top_n=int(self.valves.answer_table_max_rows))
        is_preview_request = bool(
            re.search(r"\b(перш\w*|head|first|покаж\w*|show|preview|прев'ю|превю)\b", q)
        )
        if is_preview_request and table:
            return table
        if re.search(r"\b(рядк\w*|стовпц\w*|columns?|rows?)\b", q):
            structural = self._format_structure_from_result(result_text)
            if structural:
                return structural
        wants_availability = bool(
            re.search(r"\b(наявн|в\s+наявності|доступн|availability|available|in\s+stock|out\s+of\s+stock)\b", q)
        )
        if table and not wants_availability:
            return table
        wants_pairs = bool(re.search(r"\b(кожн\w*|each|per|по\s+кожн\w*|для\s+кожн\w*)\b", q))
        wants_counts = bool(re.search(r"\b(скільк\w*|кільк\w*|count|к-?ст[ьі])\b", q))
        is_grouping = bool(
            re.search(
                r"\b(груп\w*|розбив\w*|розподіл\w*|структур\w*|"
                r"by|group\w*|breakdown|distribution|segment\w*|"
                r"по\s+\w+|за\s+\w+)\b",
                q,
            )
        )
        columns = [str(c) for c in (profile or {}).get("columns") or []]
        matched_col = _find_column_in_text(question, columns)
        if not matched_col and columns and (wants_counts or wants_pairs):
            matched_col = self._llm_pick_column_for_shortcut(question, profile or {})
        if matched_col:
            is_grouping = True
        if wants_availability:
            availability = self._format_availability_from_result(result_text)
            if availability:
                return availability
        if table:
            return table
        if is_grouping and (wants_counts or wants_pairs):
            return self._format_top_pairs_from_result(result_text, top_n=int(self.valves.answer_pairs_max_rows))
        if not (wants_pairs and wants_counts):
            return None
        return self._format_top_pairs_from_result(result_text, top_n=int(self.valves.answer_pairs_max_rows))

    def _fmt_cell_value(self, value: Any) -> str:
        if value is None:
            return "порожньо"
        text = str(value)
        if len(text) > 80:
            return text[:77] + "..."
        return text

    def _has_meaningful_mutation(self, mutation_summary: Optional[dict], mutation_flags: Optional[dict] = None) -> bool:
        summary = mutation_summary if isinstance(mutation_summary, dict) else {}
        try:
            if int(summary.get("changed_cells_count", 0) or 0) > 0:
                return True
        except Exception:
            pass
        try:
            if int(summary.get("added_rows", 0) or 0) > 0:
                return True
        except Exception:
            pass
        try:
            if int(summary.get("removed_rows", 0) or 0) > 0:
                return True
        except Exception:
            pass
        added_cols = summary.get("added_columns")
        removed_cols = summary.get("removed_columns")
        if (isinstance(added_cols, list) and added_cols) or (isinstance(removed_cols, list) and removed_cols):
            return True
        flags = mutation_flags if isinstance(mutation_flags, dict) else {}
        if bool(flags.get("committed")):
            return True
        if bool(flags.get("auto_committed")):
            return True
        if bool(flags.get("structure_changed")):
            return True
        if bool(flags.get("profile_changed")):
            return True
        return False

    def _edit_success_answer_from_result(
        self, result_text: str, profile: Optional[dict], mutation_summary: Optional[dict] = None
    ) -> Optional[str]:
        summary = mutation_summary if isinstance(mutation_summary, dict) else {}
        changed_cells = summary.get("changed_cells") if isinstance(summary.get("changed_cells"), list) else []
        changed_count = summary.get("changed_cells_count")
        try:
            changed_count_int = int(changed_count)
        except Exception:
            changed_count_int = len(changed_cells)
        if changed_cells:
            first = changed_cells[0] if isinstance(changed_cells[0], dict) else {}
            row = first.get("row")
            col = first.get("column")
            old_value = self._fmt_cell_value(first.get("old_value"))
            new_value = self._fmt_cell_value(first.get("new_value"))
            base = "Оновлено"
            if row is not None and col:
                base = f"Оновлено: рядок {row}, колонка {col}, було {old_value}, стало {new_value}."
            elif col:
                base = f"Оновлено: колонка {col}, було {old_value}, стало {new_value}."
            elif row is not None:
                base = f"Оновлено: рядок {row}, було {old_value}, стало {new_value}."
            if changed_count_int > 1:
                return f"{base} Змінено комірок: {changed_count_int}."
            return base

        added_rows = int(summary.get("added_rows", 0) or 0)
        removed_rows = int(summary.get("removed_rows", 0) or 0)
        added_cols = summary.get("added_columns") if isinstance(summary.get("added_columns"), list) else []
        removed_cols = summary.get("removed_columns") if isinstance(summary.get("removed_columns"), list) else []
        shape_changes: List[str] = []
        if added_rows:
            shape_changes.append(f"додано рядків: {added_rows}")
        if removed_rows:
            shape_changes.append(f"видалено рядків: {removed_rows}")
        if added_cols:
            shape_changes.append("додано колонки: " + ", ".join(str(x) for x in added_cols[:5]))
        if removed_cols:
            shape_changes.append("видалено колонки: " + ", ".join(str(x) for x in removed_cols[:5]))
        if shape_changes:
            return "Оновлено: " + "; ".join(shape_changes) + "."

        text = (result_text or "").strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        if str(data.get("status", "")).lower() not in {"updated", "ok", "success"}:
            return None
        row = data.get("row")
        col = data.get("column")
        new_value = data.get("new_value")
        item_id = data.get("id")
        parts: List[str] = []
        if row is not None:
            parts.append(f"рядок {row}")
        if col:
            parts.append(f"колонка {col}")
        if item_id is not None:
            parts.append(f"ідентифікатор {item_id}")
        if new_value is not None:
            parts.append(f"нове значення: {new_value}")
        if parts:
            return "Оновлено: " + ", ".join(parts) + "."
        rows = (profile or {}).get("rows")
        cols = (profile or {}).get("cols")
        if isinstance(rows, int) and isinstance(cols, int):
            return f"Оновлено. Тепер у таблиці {rows} рядків і {cols} стовпців."
        return "Оновлено таблицю."

    def _final_answer(
        self,
        question: str,
        profile: dict,
        plan: str,
        code: str,
        edit_expected: bool,
        run_status: str,
        stdout: str,
        result_text: str,
        result_meta: dict,
        mutation_summary: Optional[dict],
        mutation_flags: Optional[dict],
        error: str,
    ) -> str:
        tracer = current_route_tracer()

        def _log_return(mode: str, text: str) -> None:
            logging.info("event=final_answer_return mode=%s preview=%s", mode, _safe_trunc(text, 300))

        def _question_is_error_diagnostic(q: str) -> bool:
            return bool(
                re.search(
                    r"\b("
                    r"помил\w*|error|exception|traceback|debug|діагност\w*|"
                    r"чому\s+.*(?:помил|error)|"
                    r"why\s+.*(?:error|fail)"
                    r")\b",
                    q or "",
                    re.I,
                )
            )

        def _answer_looks_like_unprompted_code_diagnostic(ans: str, q: str) -> bool:
            if not (ans or "").strip():
                return False
            if _question_is_error_diagnostic(q):
                return False
            a = ans.lower()
            strong_signals = [
                "analysis_code",
                "у коді",
                "виникла помилка",
                "потрібно замінити",
                "```python",
            ]
            if any(sig in a for sig in strong_signals):
                return True
            return False

        def _rewrite_from_result_text(reason: str) -> Optional[str]:
            if not (result_text or "").strip():
                return None
            rewrite_system = self._prompts.get("final_rewrite_system", DEFAULT_FINAL_REWRITE_SYSTEM)
            rewrite_payload = {"question": question, "result_text": result_text}
            llm_stage_id = ""
            if tracer:
                llm_stage_id = tracer.start_stage(
                    stage_key="llm_final_rewrite",
                    stage_name="LLM Final Rewrite",
                    purpose="Rewrite result_text into concise user-facing answer when primary answer is unsafe.",
                    input_payload={
                        "model": self.valves.base_llm_model,
                        "temperature": 0.1,
                        "messages": [
                            {"role": "system", "content": rewrite_system},
                            {"role": "user", "content": json.dumps(rewrite_payload, ensure_ascii=False)},
                        ],
                    },
                    processing_summary="Call model with rewrite prompt and validate safe narrative output.",
                    details={
                        "llm": {
                            "model": self.valves.base_llm_model,
                            "temperature": 0.1,
                            "messages": [
                                {"role": "system", "content": rewrite_system},
                                {"role": "user", "content": json.dumps(rewrite_payload, ensure_ascii=False)},
                            ],
                            "reason": reason,
                        }
                    },
                )
            try:
                rewrite_resp = self._llm.chat.completions.create(
                    model=self.valves.base_llm_model,
                    messages=[
                        {"role": "system", "content": rewrite_system},
                        {"role": "user", "content": json.dumps(rewrite_payload, ensure_ascii=False)},
                    ],
                    temperature=0.1,
                    max_tokens=int(self.valves.final_answer_max_tokens),
                )
                rewrite = (rewrite_resp.choices[0].message.content or "").strip()
                if tracer and llm_stage_id:
                    tracer.end_stage(
                        llm_stage_id,
                        status="ok",
                        output_payload={"rewrite": rewrite},
                        processing_summary="Rewrite response generated.",
                        details={"llm": {"raw_response": rewrite}},
                    )
                if rewrite:
                    logging.info(
                        "event=final_answer mode=rewrite reason=%s preview=%s",
                        reason,
                        _safe_trunc(rewrite, 300),
                    )
                    return rewrite
            except Exception as exc:
                if tracer and llm_stage_id:
                    tracer.end_stage(
                        llm_stage_id,
                        status="error",
                        output_payload={},
                        processing_summary="Rewrite model call failed.",
                        error={"type": type(exc).__name__, "message": str(exc)},
                    )
                logging.warning("event=final_rewrite_failed reason=%s err=%s", reason, str(exc))
            return None

        if run_status == "ok" and edit_expected and "COMMIT_DF = True" in (code or "") and not error:
            has_mutation = self._has_meaningful_mutation(mutation_summary, mutation_flags)
            logging.info(
                "event=edit_answer_attempt has_mutation=%s summary_preview=%s flags=%s",
                has_mutation,
                _safe_trunc(mutation_summary, 400),
                _safe_trunc(mutation_flags, 400),
            )
            if not has_mutation:
                msg = "Запит виконано, але фактичних змін у таблиці не виявлено."
                _log_return("edit_no_mutation", msg)
                return msg
            edit_answer = self._edit_success_answer_from_result(result_text, profile, mutation_summary=mutation_summary)
            logging.info(
                "event=edit_answer_generated has_answer=%s preview=%s",
                bool(edit_answer),
                _safe_trunc(edit_answer or "", 300),
            )
            if edit_answer:
                _log_return("edit_success", edit_answer)
                return edit_answer
            rows = (profile or {}).get("rows")
            cols = (profile or {}).get("cols")
            if isinstance(rows, int) and isinstance(cols, int):
                msg = f"Оновлено. Тепер у таблиці {rows} рядків і {cols} стовпців."
                _log_return("edit_fallback", msg)
                return msg
            msg = "Оновлено таблицю."
            _log_return("edit_fallback", msg)
            return msg
        # Strict success mode: format answer only from result_text without free-form LLM generation.
        if run_status == "ok" and not (error or "").strip():
            if not (result_text or "").strip():
                msg = "За умовою запиту не знайдено значень."
                _log_return("empty_result", msg)
                return msg
            if _scalar_text_is_nan_like(result_text):
                filter_hint = _extract_filter_hint_from_analysis_code(code)
                if filter_hint:
                    msg = f"Не знайдено рядків для фільтра: {filter_hint}."
                    _log_return("nan_filter_empty", msg)
                    return msg
                msg = "За умовою запиту не знайдено значень."
                _log_return("nan_result", msg)
                return msg
            deterministic = self._deterministic_answer(question, result_text, profile)
            if deterministic:
                logging.info("event=final_answer mode=deterministic preview=%s", _safe_trunc(deterministic, 300))
                _log_return("deterministic", deterministic)
                return deterministic
            table = self._format_table_from_result(result_text, top_n=int(self.valves.answer_table_max_rows))
            if table:
                _log_return("table_fallback", table)
                return table
            scalar_list = self._format_scalar_list_from_result(
                result_text, max_items=int(self.valves.answer_list_max_items)
            )
            if scalar_list:
                _log_return("list_fallback", scalar_list)
                return scalar_list
            raw = (result_text or "").strip()
            if raw:
                _log_return("raw_result_text", raw)
                return raw
        system = self._prompts.get("final_answer_system", DEFAULT_FINAL_ANSWER_SYSTEM)
        payload = {
            "question": question,
            "schema": _compact_profile_for_llm(profile),
            "plan": plan,
            "analysis_code": _safe_trunc(code, 1600),
            "exec_status": run_status,
            "stdout": _safe_trunc(stdout, 1200),
            "result_text": result_text,
            "result_meta": result_meta,
            "error": error,
        }
        logging.info(
            "event=llm_final_request question_preview=%s payload_preview=%s",
            _safe_trunc(question, 400),
            _safe_trunc(json.dumps(payload, ensure_ascii=False), 1200),
        )
        final_llm_stage_id = ""
        if tracer:
            final_llm_stage_id = tracer.start_stage(
                stage_key="llm_final_answer",
                stage_name="LLM Final Answer",
                purpose="Generate final natural language answer from execution payload.",
                input_payload={
                    "model": self.valves.base_llm_model,
                    "temperature": 0.2,
                    "system_chars": len(system or ""),
                    "user_chars": len(json.dumps(payload, ensure_ascii=False)),
                    "payload_keys": list(payload.keys()),
                },
                processing_summary="Call base model for final answer when deterministic formatter did not return.",
                details={
                    "llm": {
                        "model": self.valves.base_llm_model,
                        "temperature": 0.2,
                        "messages": [
                            {"role": "system", "chars": len(system or ""), "content_preview": _safe_trunc(system, 1200)},
                            {
                                "role": "user",
                                "chars": len(json.dumps(payload, ensure_ascii=False)),
                                "content_preview": _safe_trunc(json.dumps(payload, ensure_ascii=False), 1200),
                            },
                        ],
                    }
                },
            )
        answer = ""
        try:
            resp = self._llm.chat.completions.create(
                model=self.valves.base_llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0.2,
                max_tokens=int(self.valves.final_answer_max_tokens),
            )
            answer = (resp.choices[0].message.content or "").strip()
            if tracer and final_llm_stage_id:
                tracer.end_stage(
                    final_llm_stage_id,
                    status="ok",
                    output_payload={"answer": answer},
                    processing_summary="Final-answer model call completed.",
                    details={"llm": {"raw_response": answer}},
                )
        except Exception as exc:
            if tracer and final_llm_stage_id:
                tracer.end_stage(
                    final_llm_stage_id,
                    status="error",
                    output_payload={"answer": answer},
                    processing_summary="Final-answer model call failed.",
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
            raise
        logging.info("event=llm_final_response preview=%s", _safe_trunc(answer, 1200))
        if run_status == "ok" and not (error or "").strip():
            if _answer_looks_like_unprompted_code_diagnostic(answer, question):
                logging.warning("event=final_answer mode=llm_unprompted_code_diagnostic")
                rewrite = _rewrite_from_result_text("unprompted_code_diagnostic")
                if rewrite:
                    _log_return("rewrite_unprompted_code_diagnostic", rewrite)
                    return rewrite
                fallback = (
                    self._deterministic_answer(question, result_text, profile)
                    or self._format_table_from_result(result_text, top_n=int(self.valves.answer_table_max_rows))
                    or self._format_scalar_list_from_result(
                        result_text, max_items=int(self.valves.answer_list_max_items)
                    )
                )
                if fallback:
                    _log_return("fallback_unprompted_code_diagnostic", fallback)
                    return fallback
        if result_text and re.search(r"\d+", result_text or ""):
            real_numbers = set(re.findall(r"\d+", result_text))
            llm_numbers = set(re.findall(r"\d+", answer))
            if llm_numbers and not (llm_numbers & real_numbers):
                logging.warning(
                    "event=final_answer mode=llm_hallucinated real=%s llm=%s", real_numbers, llm_numbers
                )
                rewrite = _rewrite_from_result_text("numeric_mismatch")
                if rewrite:
                    rewrite_numbers = set(re.findall(r"\d+", rewrite))
                    if not rewrite_numbers or (real_numbers & rewrite_numbers):
                        _log_return("rewrite", rewrite)
                        return rewrite
                msg = "Не можу безпечно сформувати відповідь. Спробуйте уточнити запит."
                _log_return("unsafe_numbers", msg)
                return msg
        logging.info("event=final_answer mode=llm preview=%s", _safe_trunc(answer, 300))
        _log_return("llm", answer)
        return answer

    def _emit(
        self, event_emitter: Any, event: str, payload: Dict[str, Any], status_queue: Optional[List[str]] = None
    ) -> None:
        description = _status_message(event, payload)
        done = event in ("final_answer", "error")
        if status_queue is not None:
            status_queue.append(_status_marker(description, done=done))
        _emit_status(event_emitter, description, done=done)
        logging.info("event=%s payload=%s", event, _safe_trunc(payload, 500))

    def _start_wait(self, event_emitter: Any, label: str) -> Optional[threading.Event]:
        interval_s = int(self.valves.wait_tick_s or 0)
        if not event_emitter or interval_s <= 0:
            return None
        stop_event = threading.Event()

        def _ticker() -> None:
            elapsed = 0
            while not stop_event.wait(interval_s):
                elapsed += interval_s
                self._emit(event_emitter, "wait", {"label": label, "seconds": elapsed})

        thread = threading.Thread(target=_ticker, daemon=True)
        thread.start()
        return stop_event

    def _stop_wait(self, stop_event: Optional[threading.Event]) -> None:
        if stop_event:
            stop_event.set()

    def _debug_body(self, body: dict, messages: List[dict]) -> None:
        body = body or {}
        msg = messages[-1] if messages else {}
        body_files = body.get("files") or body.get("attachments") or []
        msg_files = msg.get("files") or msg.get("attachments") or []
        msg_content = msg.get("content")
        logging.warning(
            "no_file_debug body_keys=%s body_files=%s msg_keys=%s msg_files=%s msg_content_type=%s",
            list(body.keys()),
            _safe_trunc(body_files, 500),
            list(msg.keys()),
            _safe_trunc(msg_files, 500),
            type(msg_content).__name__,
        )

    def _sandbox_load(self, file_id: str, meta: dict, data: bytes) -> Dict[str, Any]:
        tracer = current_route_tracer()
        stage_id = ""
        if tracer:
            stage_id = tracer.start_stage(
                stage_key="sandbox_load",
                stage_name="Sandbox DataFrame Load",
                purpose="Decode uploaded file and create sandbox DataFrame with schema profile.",
                input_payload={
                    "file_id": file_id,
                    "filename": _guess_filename(meta),
                    "content_type": (meta or {}).get("content_type") or (meta or {}).get("mime"),
                    "bytes_len": len(data or b""),
                    "max_rows": self.valves.max_rows,
                    "preview_rows": self.valves.preview_rows,
                },
                processing_summary="POST /v1/dataframe/load to sandbox service.",
            )
        url = f"{self.valves.sandbox_url.rstrip('/')}/v1/dataframe/load"
        payload = {
            "file_id": file_id,
            "filename": _guess_filename(meta),
            "content_type": (meta or {}).get("content_type") or (meta or {}).get("mime"),
            "data_b64": base64.b64encode(data).decode("ascii"),
            "max_rows": self.valves.max_rows,
            "preview_rows": self.valves.preview_rows,
        }
        try:
            resp = requests.post(url, headers=self._sandbox_headers(), json=payload, timeout=DEF_TIMEOUT_S)
            resp.raise_for_status()
            out = resp.json()
            if tracer and stage_id:
                tracer.end_stage(
                    stage_id,
                    status="ok",
                    output_payload=out,
                    processing_summary="Sandbox accepted file and returned df_id/profile.",
                )
            return out
        except Exception as exc:
            if tracer and stage_id:
                tracer.end_stage(
                    stage_id,
                    status="error",
                    output_payload={},
                    processing_summary="Sandbox load failed.",
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
            raise

    def _sandbox_run(self, df_id: str, code: str) -> Dict[str, Any]:
        tracer = current_route_tracer()
        stage_id = ""
        started = time.monotonic()
        if tracer:
            stage_id = tracer.start_stage(
                stage_key="sandbox_run",
                stage_name="Sandbox Code Execution",
                purpose="Execute generated pandas code inside isolated sandbox runtime.",
                input_payload={
                    "df_id": df_id,
                    "timeout_s": self.valves.code_timeout_s,
                    "preview_rows": self.valves.preview_rows,
                    "max_cell_chars": self.valves.max_cell_chars,
                    "max_stdout_chars": self.valves.max_stdout_chars,
                    "analysis_code": code,
                },
                processing_summary="POST /v1/dataframe/run with compiled analysis code.",
                details={"sandbox": {"code": code}},
            )
        url = f"{self.valves.sandbox_url.rstrip('/')}/v1/dataframe/run"
        payload = {
            "df_id": df_id,
            "code": code,
            "timeout_s": self.valves.code_timeout_s,
            "preview_rows": self.valves.preview_rows,
            "max_cell_chars": self.valves.max_cell_chars,
            "max_stdout_chars": self.valves.max_stdout_chars,
        }
        try:
            resp = requests.post(url, headers=self._sandbox_headers(), json=payload, timeout=DEF_TIMEOUT_S)
            resp.raise_for_status()
            out = resp.json()
            if tracer and stage_id:
                sandbox_status = str(out.get("status") or "")
                if sandbox_status == "ok":
                    exit_code = 0
                elif sandbox_status == "timeout":
                    exit_code = 124
                else:
                    exit_code = 1
                tracer.end_stage(
                    stage_id,
                    status="ok" if sandbox_status == "ok" else "warn",
                    output_payload=_compact_sandbox_run_output(out),
                    processing_summary="Sandbox execution completed.",
                    details={
                        "sandbox": {
                            "code": code,
                            "runtime_ms": round((time.monotonic() - started) * 1000.0, 3),
                            "status": sandbox_status,
                            "exit_code": exit_code,
                            "stdout": out.get("stdout", ""),
                            "stderr": out.get("error", "") if sandbox_status != "ok" else "",
                            "error": out.get("error", ""),
                            "artifacts": out.get("artifacts", []) if isinstance(out.get("artifacts"), list) else [],
                        }
                    },
                )
            return out
        except Exception as exc:
            if tracer and stage_id:
                tracer.end_stage(
                    stage_id,
                    status="error",
                    output_payload={},
                    processing_summary="Sandbox execution request failed.",
                    error={"type": type(exc).__name__, "message": str(exc)},
                    details={
                        "sandbox": {
                            "code": code,
                            "runtime_ms": round((time.monotonic() - started) * 1000.0, 3),
                            "exit_code": 1,
                            "stderr": str(exc),
                            "artifacts": [],
                        }
                    },
                )
            raise

    def _prepare_analysis_code_for_question(
        self,
        question: str,
        profile: dict,
        has_edit: bool,
    ) -> Dict[str, Any]:
        tracer = current_route_tracer()
        stage_id = ""
        if tracer:
            stage_id = tracer.start_stage(
                stage_key="analysis_planning",
                stage_name="Analysis Plan And Code Build",
                purpose="Select shortcut or planner path and produce validated sandbox code.",
                input_payload={
                    "question": question,
                    "has_edit": has_edit,
                    "profile": _compact_profile_for_trace(profile),
                },
                processing_summary="Route query through shortcut router or LLM planner and validate generated code.",
            )

        def _finish(result: Dict[str, Any], status: str = "ok", processing: str = "") -> Dict[str, Any]:
            if tracer and stage_id:
                tracer.end_stage(
                    stage_id,
                    status=status,
                    output_payload=_compact_plan_result_for_trace(result),
                    processing_summary=processing
                    or (
                        "Planning finished successfully."
                        if status == "ok"
                        else "Planning failed before sandbox execution."
                    ),
                )
            return result

        events: List[Tuple[str, Dict[str, Any]]] = []
        events.append(("codegen", {"question": _safe_trunc(question, 200)}))

        edit_expected = False
        op: Optional[str] = None
        commit_df: Optional[bool] = None
        plan = ""
        analysis_code = ""
        used_router = False
        used_shortcut = False

        router_hit, shortcut, router_meta, planner_hints = self._select_router_or_shortcut(question, profile, has_edit)
        lookup_hints = planner_hints.get("lookup_hints") if isinstance(planner_hints, dict) else None
        if router_hit:
            analysis_code, router_meta = router_hit
            plan = (
                f"retrieval_intent:{router_meta.get('intent_id')};"
                f"selector_mode:{router_meta.get('selector_mode') or ''}"
            )
            events.append(("codegen_shortcut", {"intent_id": router_meta.get("intent_id")}))
            used_router = True
        else:
            rlm_primary_enabled = bool(self.valves.rlm_primary_planner_enabled) and bool(self.valves.rlm_codegen_enabled)
            if rlm_primary_enabled:
                rlm_code, rlm_plan, rlm_op, rlm_commit = self._plan_code_with_rlm_tool(
                    question=question,
                    profile=profile,
                    lookup_hints=lookup_hints,
                    retry_reason="primary_planner",
                    previous_code="",
                )
                if (rlm_code or "").strip():
                    analysis_code, plan, op, commit_df = rlm_code, rlm_plan, rlm_op, rlm_commit
                    events.append(("codegen_rlm_tool", {"phase": "primary_planner"}))
            if not (analysis_code or "").strip() and shortcut:
                analysis_code, plan = shortcut
                used_shortcut = True
                events.append(("codegen_shortcut", {}))

            try:
                if not (analysis_code or "").strip():
                    analysis_code, plan, op, commit_df = self._plan_code(question, profile, lookup_hints=lookup_hints)
            except Exception as exc:
                logging.warning("event=plan_code_error error=%s", str(exc))
                analysis_code = ""
                plan = ""
                op = None
                commit_df = None
                read_fallback: Optional[Tuple[str, str]] = None
                if _infer_op_from_question(question) == "read":
                    read_fallback = (
                        self._build_subset_keyword_metric_shortcut(question, profile, preferred_col=None)
                        or _stats_shortcut_code(question, profile)
                        or _template_shortcut_code(question, profile)
                    )
                if read_fallback:
                    analysis_code, plan = read_fallback
                    used_shortcut = True
                    events.append(("codegen_shortcut", {"fallback": "plan_error"}))
            if not (analysis_code or "").strip():
                rlm_code, rlm_plan, rlm_op, rlm_commit = self._plan_code_with_rlm_tool(
                    question=question,
                    profile=profile,
                    lookup_hints=lookup_hints,
                    retry_reason="planner_empty_or_error",
                    previous_code=analysis_code,
                )
                if (rlm_code or "").strip():
                    analysis_code, plan, op, commit_df = rlm_code, rlm_plan, rlm_op, rlm_commit
                    events.append(("codegen_rlm_tool", {"phase": "primary"}))
            if not (analysis_code or "").strip() and _infer_op_from_question(question) == "read":
                read_fallback = (
                    self._build_subset_keyword_metric_shortcut(question, profile, preferred_col=None)
                    or _stats_shortcut_code(question, profile)
                    or _template_shortcut_code(question, profile)
                )
                if read_fallback:
                    analysis_code, plan = read_fallback
                    used_shortcut = True
                    events.append(("codegen_shortcut", {"fallback": "plan_empty"}))
            if not (analysis_code or "").strip():
                events.append(("codegen_empty", {}))
                return _finish({
                    "ok": False,
                    "events": events,
                    "status": "codegen_empty",
                    "message_sync": "Я не зміг згенерувати код для цього запиту. Спробуйте сформулювати інакше.",
                    "message_stream": "Не вдалося згенерувати код аналізу. Спробуйте інше формулювання.",
                }, status="warn", processing="LLM planner returned empty analysis_code.")

        analysis_code, count_err = _enforce_count_code(question, analysis_code)
        if count_err:
            return _finish({
                "ok": False,
                "events": events,
                "status": "invalid_code",
                "message_sync": f"Неможливо виконати запит: {count_err}",
                "message_stream": f"Неможливо виконати: {count_err}",
            }, status="warn", processing="Generated code failed count-intent guardrails.")
        analysis_code = _enforce_entity_nunique_code(question, analysis_code, profile)

        analysis_code, edit_expected, finalize_err = _finalize_code_for_sandbox(
            question, analysis_code, op, commit_df, df_profile=profile
        )
        subset_guard_applies = _missing_subset_filter_guard_applies(finalize_err, question, profile)
        subset_guard_conflict = bool(
            finalize_err and "missing_subset_filter" in str(finalize_err or "") and not subset_guard_applies
        )
        need_retry_missing_result = bool(
            finalize_err and "missing_result_assignment" in finalize_err and not used_shortcut and not used_router
        )
        need_retry_missing_filter = bool(subset_guard_applies)
        need_retry_subset_guard_conflict = bool(subset_guard_conflict)
        need_retry_read_mutation = bool(
            finalize_err and "read-запиту змінює таблицю" in finalize_err and not used_shortcut and not used_router
        )
        if (
            need_retry_missing_result
            or need_retry_missing_filter
            or need_retry_subset_guard_conflict
            or need_retry_read_mutation
        ):
            if need_retry_missing_filter:
                retry_reason = "missing_subset_filter"
            elif need_retry_subset_guard_conflict:
                retry_reason = "subset_guard_conflict_non_subset"
            elif need_retry_read_mutation:
                retry_reason = "read_mutation"
            else:
                retry_reason = "missing_result_assignment"
            events.append(("codegen_retry", {"reason": retry_reason}))
            if need_retry_missing_filter:
                analysis_code, plan, op, commit_df = self._plan_code_retry_missing_filter(
                    question=question,
                    profile=profile,
                    previous_code=analysis_code,
                    reason=finalize_err or "",
                    lookup_hints=lookup_hints,
                )
            elif need_retry_subset_guard_conflict:
                analysis_code, plan, op, commit_df = self._plan_code_retry_subset_guard_conflict(
                    question=question,
                    profile=profile,
                    previous_code=analysis_code,
                    reason=finalize_err or "",
                    lookup_hints=lookup_hints,
                )
            elif need_retry_read_mutation:
                analysis_code, plan, op, commit_df = self._plan_code_retry_read_mutation(
                    question=question,
                    profile=profile,
                    previous_code=analysis_code,
                    reason=finalize_err or "",
                    lookup_hints=lookup_hints,
                )
            else:
                analysis_code, plan, op, commit_df = self._plan_code_retry_missing_result(
                    question=question,
                    profile=profile,
                    previous_code=analysis_code,
                    reason=finalize_err or "",
                    lookup_hints=lookup_hints,
                )
            if not (analysis_code or "").strip():
                rlm_code, rlm_plan, rlm_op, rlm_commit = self._plan_code_with_rlm_tool(
                    question=question,
                    profile=profile,
                    lookup_hints=lookup_hints,
                    retry_reason=retry_reason,
                    previous_code=analysis_code,
                )
                if (rlm_code or "").strip():
                    analysis_code, plan, op, commit_df = rlm_code, rlm_plan, rlm_op, rlm_commit
                    events.append(("codegen_rlm_tool", {"phase": "retry"}))
            if not (analysis_code or "").strip() and _infer_op_from_question(question) == "read":
                read_fallback = (
                    self._build_subset_keyword_metric_shortcut(question, profile, preferred_col=None)
                    or _stats_shortcut_code(question, profile)
                    or _template_shortcut_code(question, profile)
                )
                if read_fallback:
                    analysis_code, plan = read_fallback
                    op = None
                    commit_df = None
                    used_shortcut = True
                    events.append(("codegen_shortcut", {"fallback": "retry_plan_empty"}))
            if not (analysis_code or "").strip():
                events.append(("codegen_empty", {}))
                return _finish({
                    "ok": False,
                    "events": events,
                    "status": "codegen_empty",
                    "message_sync": "Я не зміг згенерувати код для цього запиту. Спробуйте сформулювати інакше.",
                    "message_stream": "Не вдалося згенерувати код аналізу. Спробуйте інше формулювання.",
                }, status="warn", processing="Retry planner attempt returned empty code.")
            analysis_code, count_err = _enforce_count_code(question, analysis_code)
            if count_err:
                return _finish({
                    "ok": False,
                    "events": events,
                    "status": "invalid_code",
                    "message_sync": f"Неможливо виконати запит: {count_err}",
                    "message_stream": f"Неможливо виконати: {count_err}",
                }, status="warn", processing="Retry code failed count-intent guardrails.")
            analysis_code = _enforce_entity_nunique_code(question, analysis_code, profile)
            analysis_code, edit_expected, finalize_err = _finalize_code_for_sandbox(
                question, analysis_code, op, commit_df, df_profile=profile
            )

        if finalize_err:
            if not used_shortcut and not used_router:
                rlm_code, rlm_plan, rlm_op, rlm_commit = self._plan_code_with_rlm_tool(
                    question=question,
                    profile=profile,
                    lookup_hints=lookup_hints,
                    retry_reason=str(finalize_err or ""),
                    previous_code=analysis_code,
                )
                if (rlm_code or "").strip():
                    events.append(("codegen_rlm_tool", {"phase": "finalize_error"}))
                    analysis_code, plan, op, commit_df = rlm_code, rlm_plan, rlm_op, rlm_commit
                    analysis_code, count_err = _enforce_count_code(question, analysis_code)
                    if count_err:
                        return _finish({
                            "ok": False,
                            "events": events,
                            "status": "invalid_code",
                            "message_sync": f"Неможливо виконати запит: {count_err}",
                            "message_stream": f"Неможливо виконати: {count_err}",
                        }, status="warn", processing="RLM tool code failed count-intent guardrails.")
                    analysis_code = _enforce_entity_nunique_code(question, analysis_code, profile)
                    analysis_code, edit_expected, finalize_err = _finalize_code_for_sandbox(
                        question, analysis_code, op, commit_df, df_profile=profile
                    )

        if finalize_err:
            if "missing_result_assignment" in finalize_err:
                status = "invalid_missing_result"
            elif "missing_subset_filter" in finalize_err:
                if _missing_subset_filter_guard_applies(finalize_err, question, profile):
                    status = "invalid_subset_filter"
                else:
                    status = "invalid_guardrail_conflict"
                    finalize_err = (
                        "Guardrail conflict: query does not request subset filtering. "
                        "Use full-table aggregation/group-by unless an explicit subset is requested."
                    )
            else:
                status = "invalid_read_mutation"
            return _finish({
                "ok": False,
                "events": events,
                "status": status,
                "message_sync": f"Неможливо виконати запит: {finalize_err}",
                "message_stream": f"Неможливо виконати: {finalize_err}",
            }, status="warn", processing="Code finalization failed guardrail validation.")

        if _has_forbidden_import_nodes(analysis_code):
            return _finish({
                "ok": False,
                "events": events,
                "status": "invalid_import",
                "message_sync": "Неможливо виконати запит: згенерований код містить заборонений import.",
                "message_stream": "Неможливо виконати: згенерований код містить заборонений import.",
            }, status="warn", processing="Forbidden import detected in generated code.")

        analysis_code, plan = self._resolve_shortcut_placeholders(analysis_code, plan, question, profile)
        analysis_code = textwrap.dedent(analysis_code or "").strip() + "\n"
        if "df_profile" in (analysis_code or ""):
            analysis_code = f"df_profile = {_compact_profile_for_llm(profile)!r}\n" + analysis_code
        analysis_code = _normalize_generated_code(analysis_code)
        logging.info("event=analysis_code preview=%s", _safe_trunc(analysis_code, 4000))
        if _has_forbidden_import_nodes(analysis_code):
            return _finish({
                "ok": False,
                "events": events,
                "status": "invalid_import",
                "message_sync": "Неможливо виконати запит: згенерований код містить заборонений import.",
                "message_stream": "Неможливо виконати: згенерований код містить заборонений import.",
            }, status="warn", processing="Forbidden import detected after placeholder resolution.")

        router_meta = dict(router_meta or {})
        router_meta["confidence_guard_enabled"] = False
        router_meta["confidence_guard_status"] = "removed"
        router_meta["confidence_guard_score"] = None
        router_meta["confidence_guard_reason"] = "removed"

        return _finish({
            "ok": True,
            "events": events,
            "analysis_code": analysis_code,
            "plan": plan,
            "op": op,
            "commit_df": commit_df,
            "edit_expected": edit_expected,
            "router_meta": router_meta,
        }, status="ok", processing="Code prepared and validated for sandbox execution.")

    def _run_analysis_with_retry(
        self,
        question: str,
        profile: dict,
        df_id: str,
        analysis_code: str,
        plan: str,
        op: Optional[str],
        commit_df: Optional[bool],
        edit_expected: bool,
    ) -> Dict[str, Any]:
        tracer = current_route_tracer()
        stage_id = ""
        if tracer:
            stage_id = tracer.start_stage(
                stage_key="analysis_execute",
                stage_name="Analysis Execution And Retry",
                purpose="Execute analysis code in sandbox and optionally retry for retryable runtime errors.",
                input_payload={
                    "question": question,
                    "df_id": df_id,
                    "analysis_code": analysis_code,
                    "plan": plan,
                    "op": op,
                    "commit_df": commit_df,
                    "edit_expected": edit_expected,
                },
                processing_summary="Run sandbox execution, inspect status/error, and trigger one retry when allowed.",
            )
        events: List[Tuple[str, Dict[str, Any]]] = []
        events.append(("sandbox_run", {"df_id": df_id}))
        run_resp = self._sandbox_run(df_id, analysis_code)

        run_status = run_resp.get("status", "")
        run_error = run_resp.get("error", "") or ""
        if run_status != "ok":
            try:
                repl_repair = self._rlm_core_repl_repair(
                    question=question,
                    profile=profile,
                    df_id=df_id,
                    failed_code=analysis_code,
                    failed_error=run_error,
                    op=op,
                    commit_df=commit_df,
                )
                if repl_repair:
                    events.append(("codegen_rlm_tool", {"phase": "core_repl"}))
                    events.append(("sandbox_run", {"df_id": df_id, "retry": "rlm_core_repl"}))
                    run_resp = dict(repl_repair.get("run_resp") or {})
                    analysis_code = str(repl_repair.get("analysis_code") or analysis_code)
                    edit_expected = bool(repl_repair.get("edit_expected", edit_expected))
                    logging.info("event=analysis_code_retry_rlm_core preview=%s", _safe_trunc(analysis_code, 4000))
                    run_status = run_resp.get("status", "")
                    run_error = run_resp.get("error", "") or ""
            except Exception as retry_exc:
                logging.warning(
                    "event=sandbox_retry_failed reason=rlm_core_repl error=%s",
                    _safe_trunc(str(retry_exc), 500),
                )

        if run_status != "ok" and _is_retryable_import_keyerror(run_error):
            try:
                logging.warning("event=sandbox_retry reason=import_keyerror error=%s", _safe_trunc(run_error, 500))
                events.append(("codegen_retry", {"reason": "runtime_keyerror_import"}))

                retry_code = ""
                retry_plan = plan
                retry_op = op
                retry_commit = commit_df

                shortcut_retry = None
                if not _question_requires_subset_filter(question, profile):
                    shortcut_retry = _stats_shortcut_code(question, profile)
                if shortcut_retry:
                    retry_code, retry_plan = shortcut_retry
                    retry_op = None
                    retry_commit = None
                else:
                    retry_code, retry_plan, retry_op, retry_commit = self._plan_code_retry_runtime_error(
                        question=question,
                        profile=profile,
                        previous_code=analysis_code,
                        runtime_error=run_error,
                    )

                if (retry_code or "").strip():
                    retry_code, retry_edit_expected, retry_finalize_err = _finalize_code_for_sandbox(
                        question, retry_code, retry_op, retry_commit, df_profile=profile
                    )
                    if not retry_finalize_err and not _has_forbidden_import_nodes(retry_code):
                        retry_code, retry_plan = self._resolve_shortcut_placeholders(
                            retry_code, retry_plan, question, profile
                        )
                        retry_code = textwrap.dedent(retry_code or "").strip() + "\n"
                        if "df_profile" in (retry_code or ""):
                            retry_code = f"df_profile = {_compact_profile_for_llm(profile)!r}\n" + retry_code
                        retry_code = _normalize_generated_code(retry_code)
                        logging.info("event=analysis_code_retry preview=%s", _safe_trunc(retry_code, 4000))

                        events.append(("sandbox_run", {"df_id": df_id, "retry": True}))
                        retry_resp = self._sandbox_run(df_id, retry_code)
                        if retry_resp is not None:
                            run_resp = retry_resp
                            analysis_code = retry_code
                            plan = retry_plan
                            edit_expected = retry_edit_expected
            except Exception as retry_exc:
                logging.warning(
                    "event=sandbox_retry_failed reason=import_keyerror error=%s",
                    _safe_trunc(str(retry_exc), 500),
                )
        run_status = run_resp.get("status", "")
        run_error = run_resp.get("error", "") or ""
        if run_status != "ok" and bool(self.valves.rlm_codegen_runtime_retry_enabled):
            try:
                logging.warning(
                    "event=sandbox_retry reason=rlm_codegen_runtime status=%s error=%s",
                    run_status,
                    _safe_trunc(run_error, 500),
                )
                events.append(("codegen_rlm_tool", {"phase": "runtime_error"}))
                retry_code, retry_plan, retry_op, retry_commit = self._plan_code_with_rlm_tool(
                    question=question,
                    profile=profile,
                    retry_reason=f"runtime_error:{run_status}",
                    previous_code=analysis_code,
                    runtime_error=run_error,
                )
                if (retry_code or "").strip():
                    retry_code, retry_edit_expected, retry_finalize_err = _finalize_code_for_sandbox(
                        question, retry_code, retry_op, retry_commit, df_profile=profile
                    )
                    if not retry_finalize_err and not _has_forbidden_import_nodes(retry_code):
                        retry_code, retry_plan = self._resolve_shortcut_placeholders(
                            retry_code, retry_plan, question, profile
                        )
                        retry_code = textwrap.dedent(retry_code or "").strip() + "\n"
                        if "df_profile" in (retry_code or ""):
                            retry_code = f"df_profile = {_compact_profile_for_llm(profile)!r}\n" + retry_code
                        retry_code = _normalize_generated_code(retry_code)
                        logging.info("event=analysis_code_retry_rlm preview=%s", _safe_trunc(retry_code, 4000))
                        events.append(("sandbox_run", {"df_id": df_id, "retry": "rlm_codegen"}))
                        retry_resp = self._sandbox_run(df_id, retry_code)
                        if retry_resp is not None:
                            run_resp = retry_resp
                            analysis_code = retry_code
                            plan = retry_plan
                            edit_expected = retry_edit_expected
            except Exception as retry_exc:
                logging.warning(
                    "event=sandbox_retry_failed reason=rlm_codegen_runtime error=%s",
                    _safe_trunc(str(retry_exc), 500),
                )

        result = {
            "events": events,
            "run_resp": run_resp,
            "analysis_code": analysis_code,
            "plan": plan,
            "edit_expected": edit_expected,
        }
        if tracer and stage_id:
            run_status = str((run_resp or {}).get("status") or "")
            tracer.end_stage(
                stage_id,
                status="ok" if run_status == "ok" else "warn",
                output_payload={
                    "events": events,
                    "run_resp": _compact_sandbox_run_output(run_resp),
                    "analysis_code_chars": len(str(analysis_code or "")),
                    "analysis_code_preview": _safe_trunc(str(analysis_code or ""), 600),
                    "plan_preview": _safe_trunc(str(plan or ""), 400),
                    "edit_expected": bool(edit_expected),
                },
                processing_summary="Sandbox execution stage completed with retry handling.",
            )
        return result

    def _postprocess_run_result(
        self,
        *,
        run_resp: Dict[str, Any],
        analysis_code: str,
        edit_expected: bool,
        profile: dict,
        cached_fp: str,
        session_key: str,
        file_id: str,
        df_id: str,
    ) -> Dict[str, Any]:
        tracer = current_route_tracer()
        stage_id = ""
        if tracer:
            stage_id = tracer.start_stage(
                stage_key="postprocess_result",
                stage_name="Postprocess Execution Result",
                purpose="Apply commit rules, refresh profile/session cache, and map sandbox status to pipeline status.",
                input_payload={
                    "run_resp": _compact_sandbox_run_output(run_resp),
                    "edit_expected": edit_expected,
                    "file_id": file_id,
                    "df_id": df_id,
                    "cached_profile_fingerprint": cached_fp,
                },
                processing_summary="Inspect commit flags and profile diffs, then derive pipeline post-status.",
            )
        profile_out = (run_resp or {}).get("profile")
        was_committed = bool((run_resp or {}).get("committed"))
        structure_changed = bool((run_resp or {}).get("structure_changed"))
        profile_changed = False

        if profile_out is not None:
            profile_fp = _profile_fingerprint(profile_out)
            profile_changed = bool(profile_fp and profile_fp != cached_fp)
            if was_committed or structure_changed or (profile_fp and profile_fp != cached_fp):
                profile = profile_out
                self._session_set(session_key, file_id, df_id, profile_out)
                if self.valves.debug:
                    logging.info(
                        "event=profile_updated committed=%s structure_changed=%s fp_old=%s fp_new=%s",
                        was_committed,
                        structure_changed,
                        cached_fp[:8] if cached_fp else "none",
                        profile_fp[:8] if profile_fp else "none",
                    )

        run_status = str((run_resp or {}).get("status", ""))
        if run_status != "ok":
            error = (run_resp or {}).get("error", "") or "Sandbox execution failed."
            result = {
                "ok": False,
                "status": run_status,
                "profile": profile,
                "message_sync": f"Не вдалося виконати аналіз (статус: {run_status}). Помилка: {error}",
                "message_stream": f"Помилка виконання у sandbox (status: {run_status}). {error}",
                "mutation_flags": {
                    "committed": was_committed,
                    "auto_committed": bool((run_resp or {}).get("auto_committed")),
                    "structure_changed": structure_changed,
                    "profile_changed": profile_changed,
                },
            }
            if tracer and stage_id:
                tracer.end_stage(
                    stage_id,
                    status="warn",
                    output_payload=_compact_postprocess_result_for_trace(result),
                    processing_summary="Sandbox did not return ok status; mapped to pipeline error response.",
                )
            return result

        has_commit_marker = "COMMIT_DF" in (analysis_code or "")
        auto_committed = bool((run_resp or {}).get("auto_committed"))
        logging.info(
            "event=commit_result edit_expected=%s commit_marker=%s committed=%s auto_committed=%s structure_changed=%s profile_changed=%s",
            edit_expected,
            has_commit_marker,
            was_committed,
            auto_committed,
            structure_changed,
            profile_changed,
        )

        if edit_expected and has_commit_marker:
            if not (was_committed or auto_committed or structure_changed or profile_changed):
                result = {
                    "ok": False,
                    "status": "commit_failed",
                    "profile": profile,
                    "message_sync": "Зміни не були зафіксовані. Спробуйте ще раз або вкажіть COMMIT_DF = True явно.",
                    "message_stream": "Зміни не були застосовані. Будь ласка, перевірте код або спробуйте ще раз.",
                    "mutation_flags": {
                        "committed": was_committed,
                        "auto_committed": auto_committed,
                        "structure_changed": structure_changed,
                        "profile_changed": profile_changed,
                    },
                }
                if tracer and stage_id:
                    tracer.end_stage(
                        stage_id,
                        status="warn",
                        output_payload=_compact_postprocess_result_for_trace(result),
                        processing_summary="Edit was expected but commit signals show no persisted mutation.",
                    )
                return result

        result = {
            "ok": True,
            "status": run_status,
            "profile": profile,
            "mutation_flags": {
                "committed": was_committed,
                "auto_committed": auto_committed,
                "structure_changed": structure_changed,
                "profile_changed": profile_changed,
            },
        }
        if tracer and stage_id:
            tracer.end_stage(
                stage_id,
                status="ok",
                output_payload=_compact_postprocess_result_for_trace(result),
                processing_summary="Postprocess complete; result accepted.",
            )
        return result

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
        __event_emitter__: Any = None,
    ) -> Union[str, Iterator, Generator]:
        event_emitter = __event_emitter__ or (body or {}).get("event_emitter") or (body or {}).get("__event_emitter__")
        stream_requested = bool((body or {}).get("stream"))
        if stream_requested and not event_emitter:
            return self._pipe_stream(user_message, model_id, messages, body)
        return self._pipe_sync(user_message, model_id, messages, body, event_emitter)

    def _pipe_sync(
        self, user_message: str, model_id: str, messages: List[dict], body: dict, event_emitter: Any
    ) -> str:
        request_id, trace_id = _extract_request_trace_ids(body)
        request_token = _REQUEST_ID_CTX.set(request_id)
        trace_token = _TRACE_ID_CTX.set(trace_id)
        llm_stats_token = _LLM_CALL_STATS_CTX.set([])
        route_tracer = self._new_route_tracer(request_id=request_id, trace_id=trace_id, mode="sync")
        route_trace_token = set_active_route_tracer(route_tracer) if route_tracer else None
        route_trace_status = "ok"
        try:
            logging.info("event=request_context mode=sync")
            self._emit(event_emitter, "start", {"model_id": model_id, "has_messages": bool(messages)})
            
            question = _effective_user_query(user_message, messages)
            user_preview = _safe_trunc(_normalize_query_text(user_message or ""), 200)
            logging.info(
                "event=query_selection incoming_preview=%s selected_preview=%s selected_empty=%s",
                user_preview,
                _safe_trunc(question, 200),
                not bool(question),
            )
            if route_tracer:
                route_tracer.record_stage(
                    stage_key="request_intake",
                    stage_name="Request Intake",
                    purpose="Normalize user query and determine if this request should run the spreadsheet pipeline.",
                    input_payload={
                        "user_message_preview": _safe_trunc(user_message, 300),
                        "messages_count": len(messages or []),
                        "model_id": model_id,
                    },
                    output_payload={"question": question, "selected_empty": not bool(question)},
                    processing_summary="Extract effective user query and skip pure meta-task traffic.",
                    status="ok" if bool(question) else "warn",
                )
            if self.valves.debug:
                logging.info("event=query_selection_tail payload=%s", _safe_trunc(_query_selection_debug(messages), 1200))
            if not question:
                raw_user_message = (user_message or "").strip()
                reason = "meta_without_user_query" if _is_meta_task_text(raw_user_message) else "no_user_query_found"
                logging.info("event=skip_meta_task_sync reason=%s", reason)
                self._emit(event_emitter, "final_answer", {"status": "skipped_no_user_query"})
                route_trace_status = "warn"
                return ""
            if _is_search_query_meta_task(question):
                logging.info("event=skip_meta_task_sync reason=search_query_meta_task")
                self._emit(event_emitter, "final_answer", {"status": "skipped_meta_task"})
                route_trace_status = "warn"
                return ""

            logging.info("event=question_selected source=effective_user_query preview=%s", _safe_trunc(question, 200))

            session_key = _session_key(body)
            session = self._session_get(session_key)
            cached_fp = session.get("profile_fp") if session else ""

            file_id, file_obj, file_source, ignored_history_file_id = _resolve_active_file_ref(body, messages, session)
            logging.info(
                "event=file_selection source=%s session_file_id=%s selected_file_id=%s ignored_history_file_id=%s",
                file_source,
                (session or {}).get("file_id"),
                file_id,
                ignored_history_file_id,
            )
            if route_tracer:
                route_tracer.record_stage(
                    stage_key="file_selection",
                    stage_name="File Selection",
                    purpose="Choose active file reference for this turn with session/history guards.",
                    input_payload={
                        "body": body,
                        "messages": messages,
                        "session": session or {},
                    },
                    output_payload={
                        "file_id": file_id,
                        "file_source": file_source,
                        "ignored_history_file_id": ignored_history_file_id,
                    },
                    processing_summary="Resolved explicit turn file, session file, or history fallback.",
                    status="ok" if bool(file_id) else "warn",
                )

            self._emit(event_emitter, "file_id", {"file_id": file_id})
            if not file_id:
                self._emit(event_emitter, "no_file", {"body_keys": list((body or {}).keys())})
                self._debug_body(body, messages)
                route_trace_status = "warn"
                return "Будь ласка, прикріпіть файл CSV/XLSX для аналізу."

            if session and session.get("file_id") == file_id and session.get("df_id"):
                df_id = session["df_id"]
                cached_profile = session.get("profile") or {}
                fresh_profile = self._sandbox_get_profile(df_id)
                if fresh_profile:
                    profile = fresh_profile
                    if fresh_profile != cached_profile:
                        self._session_set(session_key, file_id, df_id, fresh_profile)
                        cached_fp = _profile_fingerprint(fresh_profile)
                else:
                    profile = cached_profile
            else:
                self._emit(event_emitter, "fetch_meta", {"file_id": file_id})
                meta = self._fetch_file_meta(file_id, file_obj)
                self._emit(event_emitter, "fetch_bytes", {"file_id": file_id})
                data = self._fetch_file_bytes(file_id, meta, file_obj)
                self._emit(event_emitter, "sandbox_load", {"file_id": file_id})
                load_resp = self._sandbox_load(file_id, meta, data)
                df_id = load_resp.get("df_id")
                profile = load_resp.get("profile") or {}
                if not df_id:
                    self._emit(event_emitter, "sandbox_load_failed", {"file_id": file_id})
                    route_trace_status = "error"
                    return "Не вдалося завантажити таблицю в пісочницю (sandbox)."
                self._apply_dynamic_limits(profile)
                self._session_set(session_key, file_id, df_id, profile)

            edit_expected = False
            op = None
            commit_df = None
            router_meta: Dict[str, Any] = {}
            learning_meta: Dict[str, Any] = {}
            
            has_edit = _has_edit_triggers(question)
            prep = self._prepare_analysis_code_for_question(question, profile, has_edit)
            for ev, payload in prep.get("events", []):
                self._emit(event_emitter, ev, payload)
            if not prep.get("ok"):
                status = str(prep.get("status") or "")
                if status in {"invalid_missing_result", "invalid_subset_filter", "invalid_read_mutation", "invalid_import"}:
                    self._emit(event_emitter, "final_answer", {"status": status})
                message_sync = str(prep.get("message_sync") or "Неможливо виконати запит.")
                prep_router_meta = prep.get("router_meta") if isinstance(prep.get("router_meta"), dict) else {}
                self._maybe_record_shortcut_debug_trace(
                    mode="sync",
                    question=question,
                    router_meta=prep_router_meta,
                    analysis_code="",
                    run_status=status or "prepare_error",
                    result_text="",
                    result_meta={},
                    final_answer=message_sync,
                    error=status,
                    learning_meta=learning_meta,
                )
                route_trace_status = "warn"
                return message_sync

            analysis_code = str(prep.get("analysis_code") or "")
            plan = str(prep.get("plan") or "")
            op = prep.get("op")
            commit_df = prep.get("commit_df")
            edit_expected = bool(prep.get("edit_expected"))
            router_meta = prep.get("router_meta") if isinstance(prep.get("router_meta"), dict) else {}

            run_stage = self._run_analysis_with_retry(
                question=question,
                profile=profile,
                df_id=df_id,
                analysis_code=analysis_code,
                plan=plan,
                op=op,
                commit_df=commit_df,
                edit_expected=edit_expected,
            )
            for ev, payload in run_stage.get("events", []):
                self._emit(event_emitter, ev, payload)
            run_resp = run_stage.get("run_resp") or {}
            analysis_code = str(run_stage.get("analysis_code") or analysis_code)
            plan = str(run_stage.get("plan") or plan)
            edit_expected = bool(run_stage.get("edit_expected"))
            post = self._postprocess_run_result(
                run_resp=run_resp,
                analysis_code=analysis_code,
                edit_expected=edit_expected,
                profile=profile,
                cached_fp=cached_fp,
                session_key=session_key,
                file_id=file_id,
                df_id=df_id,
            )
            profile = post.get("profile") or profile
            run_status = str(post.get("status") or "")
            if not post.get("ok"):
                self._emit(event_emitter, "final_answer", {"status": run_status})
                message_sync = str(post.get("message_sync") or "Не вдалося виконати аналіз.")
                self._maybe_record_shortcut_debug_trace(
                    mode="sync",
                    question=question,
                    router_meta=router_meta,
                    analysis_code=analysis_code,
                    run_status=run_status,
                    result_text=run_resp.get("result_text", ""),
                    result_meta=run_resp.get("result_meta", {}) or {},
                    final_answer=message_sync,
                    error=run_resp.get("error", ""),
                    learning_meta=learning_meta,
                )
                route_trace_status = "warn"
                return message_sync

            mutation_flags = dict(post.get("mutation_flags") or {})
            learning_meta = self._maybe_record_success_learning(
                question=question,
                plan=plan,
                analysis_code=analysis_code,
                run_status=run_status,
                edit_expected=edit_expected,
                result_text=run_resp.get("result_text", ""),
                result_meta=run_resp.get("result_meta", {}) or {},
            )
            wait = self._start_wait(event_emitter, "final_answer")
            final_answer_text = ""
            final_stage_id = ""
            if route_tracer:
                final_stage_id = route_tracer.start_stage(
                    stage_key="response_render",
                    stage_name="Response Rendering",
                    purpose="Build final user-visible response text from execution outputs.",
                    input_payload={
                        "question": question,
                        "run_status": run_status,
                        "result_text": run_resp.get("result_text", ""),
                        "result_meta": run_resp.get("result_meta", {}) or {},
                        "mutation_summary": run_resp.get("mutation_summary", {}) or {},
                    },
                    processing_summary="Run deterministic formatter with optional LLM fallback.",
                )
            try:
                final_answer_text = self._final_answer(
                    question=question,
                    profile=profile,
                    plan=plan,
                    code=analysis_code,
                    edit_expected=edit_expected,
                    run_status=run_status,
                    stdout=run_resp.get("stdout", ""),
                    result_text=run_resp.get("result_text", ""),
                    result_meta=run_resp.get("result_meta", {}) or {},
                    mutation_summary=run_resp.get("mutation_summary", {}) or {},
                    mutation_flags=mutation_flags,
                    error=run_resp.get("error", ""),
                )
                if route_tracer and final_stage_id:
                    route_tracer.end_stage(
                        final_stage_id,
                        status="ok",
                        output_payload={"final_answer_text": final_answer_text},
                        processing_summary="Final answer rendered.",
                    )
            except Exception as final_exc:
                if route_tracer and final_stage_id:
                    route_tracer.end_stage(
                        final_stage_id,
                        status="error",
                        output_payload={},
                        processing_summary="Final answer rendering failed.",
                        error={"type": type(final_exc).__name__, "message": str(final_exc)},
                    )
                raise
            finally:
                self._stop_wait(wait)
            final_answer_text = self._append_route_trace_link(final_answer_text, trace_id, request_id)
            self._maybe_record_shortcut_debug_trace(
                mode="sync",
                question=question,
                router_meta=router_meta,
                analysis_code=analysis_code,
                run_status=run_status,
                result_text=run_resp.get("result_text", ""),
                result_meta=run_resp.get("result_meta", {}) or {},
                final_answer=final_answer_text,
                error=run_resp.get("error", ""),
                learning_meta=learning_meta,
            )
            self._emit(event_emitter, "final_answer", {"status": run_status})
            return final_answer_text

        except Exception as exc:
            logging.error("event=pipe_sync_error error=%s", str(exc))
            tb = _safe_trunc(traceback.format_exc(), 2000)
            try:
                self._emit(event_emitter, "error", {"error": str(exc)})
            except Exception:
                pass
            route_trace_status = "error"
            return f"Сталася помилка в пайплайні: {type(exc).__name__}: {exc}\n\n{tb}"
        finally:
            if route_tracer:
                route_tracer.finalize(status=route_trace_status)
            if route_trace_token is not None:
                reset_active_route_tracer(route_trace_token)
            _LLM_CALL_STATS_CTX.reset(llm_stats_token)
            _REQUEST_ID_CTX.reset(request_token)
            _TRACE_ID_CTX.reset(trace_token)

    def _pipe_stream(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Generator[str, None, None]:
        inner = self._pipe_stream_inner(user_message, model_id, messages, body)
        stream_ctx = contextvars.copy_context()
        try:
            while True:
                try:
                    yield stream_ctx.run(next, inner)
                except StopIteration:
                    return
        finally:
            try:
                stream_ctx.run(inner.close)
            except Exception:
                pass

    def _pipe_stream_inner(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Generator[str, None, None]:
        request_id, trace_id = _extract_request_trace_ids(body)
        request_token = _REQUEST_ID_CTX.set(request_id)
        trace_token = _TRACE_ID_CTX.set(trace_id)
        llm_stats_token = _LLM_CALL_STATS_CTX.set([])
        route_tracer = self._new_route_tracer(request_id=request_id, trace_id=trace_id, mode="stream")
        route_trace_token = set_active_route_tracer(route_tracer) if route_tracer else None
        route_trace_status = "ok"
        status_queue: List[str] = []

        def emit(event: str, payload: Dict[str, Any]) -> None:
            self._emit(None, event, payload, status_queue=status_queue)

        def drain() -> Iterator[str]:
            while status_queue:
                yield status_queue.pop(0)

        try:
            logging.info("event=request_context mode=stream")
            emit("start", {"model_id": model_id, "has_messages": bool(messages)})
            yield from drain()

            question = _effective_user_query(user_message, messages)
            user_preview = _safe_trunc(_normalize_query_text(user_message or ""), 200)
            logging.info(
                "event=query_selection incoming_preview=%s selected_preview=%s selected_empty=%s",
                user_preview,
                _safe_trunc(question, 200),
                not bool(question),
            )
            if route_tracer:
                route_tracer.record_stage(
                    stage_key="request_intake",
                    stage_name="Request Intake",
                    purpose="Normalize user query and determine if this request should run the spreadsheet pipeline.",
                    input_payload={
                        "user_message_preview": _safe_trunc(user_message, 300),
                        "messages_count": len(messages or []),
                        "model_id": model_id,
                    },
                    output_payload={"question": question, "selected_empty": not bool(question)},
                    processing_summary="Extract effective user query and skip pure meta-task traffic.",
                    status="ok" if bool(question) else "warn",
                )
            if self.valves.debug:
                logging.info("event=query_selection_tail payload=%s", _safe_trunc(_query_selection_debug(messages), 1200))
            if not question:
                raw_user_message = (user_message or "").strip()
                reason = "meta_without_user_query" if _is_meta_task_text(raw_user_message) else "no_user_query_found"
                logging.info("event=skip_meta_task_stream reason=%s", reason)
                emit("final_answer", {"status": "skipped_no_user_query"})
                yield from drain()
                route_trace_status = "warn"
                return
            if _is_search_query_meta_task(question):
                logging.info("event=skip_meta_task_stream reason=search_query_meta_task")
                emit("final_answer", {"status": "skipped_meta_task"})
                yield from drain()
                route_trace_status = "warn"
                return

            logging.info("event=question_selected source=effective_user_query preview=%s", _safe_trunc(question, 200))

            session_key = _session_key(body)
            session = self._session_get(session_key)
            cached_fp = session.get("profile_fp") if session else ""

            file_id, file_obj, file_source, ignored_history_file_id = _resolve_active_file_ref(body, messages, session)
            logging.info(
                "event=file_selection source=%s session_file_id=%s selected_file_id=%s ignored_history_file_id=%s",
                file_source,
                (session or {}).get("file_id"),
                file_id,
                ignored_history_file_id,
            )
            if route_tracer:
                route_tracer.record_stage(
                    stage_key="file_selection",
                    stage_name="File Selection",
                    purpose="Choose active file reference for this turn with session/history guards.",
                    input_payload={
                        "body": body,
                        "messages": messages,
                        "session": session or {},
                    },
                    output_payload={
                        "file_id": file_id,
                        "file_source": file_source,
                        "ignored_history_file_id": ignored_history_file_id,
                    },
                    processing_summary="Resolved explicit turn file, session file, or history fallback.",
                    status="ok" if bool(file_id) else "warn",
                )

            emit("file_id", {"file_id": file_id})
            yield from drain()

            if not file_id:
                emit("no_file", {"body_keys": list((body or {}).keys())})
                yield from drain()
                self._debug_body(body, messages)
                route_trace_status = "warn"
                yield "Будь ласка, прикріпіть файл (CSV/XLSX) для аналізу."
                return

            if session and session.get("file_id") == file_id and session.get("df_id"):
                df_id = session["df_id"]
                profile = session.get("profile") or {}
            else:
                emit("fetch_meta", {"file_id": file_id})
                yield from drain()
                meta = self._fetch_file_meta(file_id, file_obj)
                
                emit("fetch_bytes", {"file_id": file_id})
                yield from drain()
                data = self._fetch_file_bytes(file_id, meta, file_obj)
                
                emit("sandbox_load", {"file_id": file_id})
                yield from drain()
                load_resp = self._sandbox_load(file_id, meta, data)
                df_id = load_resp.get("df_id")
                profile = load_resp.get("profile") or {}
                
                if not df_id:
                    emit("sandbox_load_failed", {"file_id": file_id})
                    yield from drain()
                    route_trace_status = "error"
                    yield "Не вдалося завантажити файл у sandbox."
                    return
                
                self._apply_dynamic_limits(profile)
                self._session_set(session_key, file_id, df_id, profile)

            edit_expected = False
            op = None
            commit_df = None
            router_meta: Dict[str, Any] = {}
            learning_meta: Dict[str, Any] = {}
            
            has_edit = _has_edit_triggers(question)
            prep = self._prepare_analysis_code_for_question(question, profile, has_edit)
            for ev, payload in prep.get("events", []):
                emit(ev, payload)
                yield from drain()
            if not prep.get("ok"):
                status = str(prep.get("status") or "")
                if status in {"invalid_code", "invalid_missing_result", "invalid_subset_filter", "invalid_read_mutation", "invalid_import"}:
                    emit("final_answer", {"status": status})
                    yield from drain()
                message_stream = str(prep.get("message_stream") or "Неможливо виконати запит.")
                prep_router_meta = prep.get("router_meta") if isinstance(prep.get("router_meta"), dict) else {}
                self._maybe_record_shortcut_debug_trace(
                    mode="stream",
                    question=question,
                    router_meta=prep_router_meta,
                    analysis_code="",
                    run_status=status or "prepare_error",
                    result_text="",
                    result_meta={},
                    final_answer=message_stream,
                    error=status,
                    learning_meta=learning_meta,
                )
                route_trace_status = "warn"
                yield message_stream
                return

            analysis_code = str(prep.get("analysis_code") or "")
            plan = str(prep.get("plan") or "")
            op = prep.get("op")
            commit_df = prep.get("commit_df")
            edit_expected = bool(prep.get("edit_expected"))
            router_meta = prep.get("router_meta") if isinstance(prep.get("router_meta"), dict) else {}

            run_stage = self._run_analysis_with_retry(
                question=question,
                profile=profile,
                df_id=df_id,
                analysis_code=analysis_code,
                plan=plan,
                op=op,
                commit_df=commit_df,
                edit_expected=edit_expected,
            )
            for ev, payload in run_stage.get("events", []):
                emit(ev, payload)
                yield from drain()
            run_resp = run_stage.get("run_resp") or {}
            analysis_code = str(run_stage.get("analysis_code") or analysis_code)
            plan = str(run_stage.get("plan") or plan)
            edit_expected = bool(run_stage.get("edit_expected"))
            post = self._postprocess_run_result(
                run_resp=run_resp,
                analysis_code=analysis_code,
                edit_expected=edit_expected,
                profile=profile,
                cached_fp=cached_fp,
                session_key=session_key,
                file_id=file_id,
                df_id=df_id,
            )
            profile = post.get("profile") or profile
            run_status = str(post.get("status") or "")
            if not post.get("ok"):
                emit("final_answer", {"status": run_status})
                yield from drain()
                message_stream = str(post.get("message_stream") or "Помилка виконання у sandbox.")
                self._maybe_record_shortcut_debug_trace(
                    mode="stream",
                    question=question,
                    router_meta=router_meta,
                    analysis_code=analysis_code,
                    run_status=run_status,
                    result_text=run_resp.get("result_text", ""),
                    result_meta=run_resp.get("result_meta", {}) or {},
                    final_answer=message_stream,
                    error=run_resp.get("error", ""),
                    learning_meta=learning_meta,
                )
                route_trace_status = "warn"
                yield message_stream
                return

            mutation_flags = dict(post.get("mutation_flags") or {})
            learning_meta = self._maybe_record_success_learning(
                question=question,
                plan=plan,
                analysis_code=analysis_code,
                run_status=run_status,
                edit_expected=edit_expected,
                result_text=run_resp.get("result_text", ""),
                result_meta=run_resp.get("result_meta", {}) or {},
            )
            final_stage_id = ""
            if route_tracer:
                final_stage_id = route_tracer.start_stage(
                    stage_key="response_render",
                    stage_name="Response Rendering",
                    purpose="Build final user-visible response text from execution outputs.",
                    input_payload={
                        "question": question,
                        "run_status": run_status,
                        "result_text": run_resp.get("result_text", ""),
                        "result_meta": run_resp.get("result_meta", {}) or {},
                        "mutation_summary": run_resp.get("mutation_summary", {}) or {},
                    },
                    processing_summary="Run deterministic formatter with optional LLM fallback.",
                )
            final_answer_text = ""
            try:
                final_answer_text = self._final_answer(
                    question=question,
                    profile=profile,
                    plan=plan,
                    code=analysis_code,
                    edit_expected=edit_expected,
                    run_status=run_status,
                    stdout=run_resp.get("stdout", ""),
                    result_text=run_resp.get("result_text", ""),
                    result_meta=run_resp.get("result_meta", {}) or {},
                    mutation_summary=run_resp.get("mutation_summary", {}) or {},
                    mutation_flags=mutation_flags,
                    error=run_resp.get("error", ""),
                )
                if route_tracer and final_stage_id:
                    route_tracer.end_stage(
                        final_stage_id,
                        status="ok",
                        output_payload={"final_answer_text": final_answer_text},
                        processing_summary="Final answer rendered.",
                    )
            except Exception as final_exc:
                if route_tracer and final_stage_id:
                    route_tracer.end_stage(
                        final_stage_id,
                        status="error",
                        output_payload={},
                        processing_summary="Final answer rendering failed.",
                        error={"type": type(final_exc).__name__, "message": str(final_exc)},
                    )
                raise
            final_answer_text = self._append_route_trace_link(final_answer_text, trace_id, request_id)
            self._maybe_record_shortcut_debug_trace(
                mode="stream",
                question=question,
                router_meta=router_meta,
                analysis_code=analysis_code,
                run_status=run_status,
                result_text=run_resp.get("result_text", ""),
                result_meta=run_resp.get("result_meta", {}) or {},
                final_answer=final_answer_text,
                error=run_resp.get("error", ""),
                learning_meta=learning_meta,
            )
            emit("final_answer", {"status": run_status})
            yield from drain()
            yield final_answer_text

        except Exception as exc:
            logging.error("event=pipe_stream_error error=%s", str(exc))
            tb = _safe_trunc(traceback.format_exc(), 2000)
            emit("error", {"error": str(exc)})
            yield from drain()
            route_trace_status = "error"
            yield f"Pipeline error: {type(exc).__name__}: {exc}\n\n{tb}"
        finally:
            if route_tracer:
                route_tracer.finalize(status=route_trace_status)
            if route_trace_token is not None:
                reset_active_route_tracer(route_trace_token)
            _LLM_CALL_STATS_CTX.reset(llm_stats_token)
            _REQUEST_ID_CTX.reset(request_token)
            _TRACE_ID_CTX.reset(trace_token)


pipeline = Pipeline()
