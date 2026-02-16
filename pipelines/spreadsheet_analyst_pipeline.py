### Поточна версія коду 

import ast
import asyncio
import base64
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
import requests
from openai import OpenAI
from pydantic import BaseModel, Field


PIPELINES_DIR = os.path.dirname(__file__)
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(PIPELINES_DIR, os.pardir))
DEFAULT_SPREADSHEET_SKILL_DIR = os.path.join(PROJECT_ROOT_DIR, "skills", "spreadsheet-guardrails")
_SPREADSHEET_SKILL_PROMPT_MARKER = "RUNTIME SKILL HOOK: spreadsheet-guardrails"
_SPREADSHEET_SKILL_FILES = (
    "SKILL.md",
    os.path.join("references", "column-matching.md"),
    os.path.join("references", "table-mutation-playbooks.md"),
    os.path.join("references", "forbidden-code-patterns.md"),
)

DEF_TIMEOUT_S = int(os.getenv("PIPELINE_HTTP_TIMEOUT_S", "120"))
_LOCAL_PROMPTS = os.path.join(os.path.dirname(__file__), "prompts.txt")
PROMPTS_PATH = _LOCAL_PROMPTS if os.path.exists(_LOCAL_PROMPTS) else os.path.join(PIPELINES_DIR, "prompts.txt")
DEFAULT_PLAN_CODE_SYSTEM = (
    "You write pandas code to answer questions about a DataFrame named df. "
    "CRITICAL SUBSET RULE: When user asks about a subset (e.g., among/for/with/where/тільки/лише/серед), "
    "you MUST filter DataFrame first, then compute metric on filtered rows only. "
    "Do NOT compute metric on full df when subset is requested. "
    "Example: 'max price among mice' => "
    "result = df[df['Категорія'].astype(str).str.contains('миш', case=False, na=False)]['Ціна_UAH'].max(). "
    "Return ONLY valid JSON with keys: analysis_code, short_plan, op, commit_df. "
    "CRITICAL: The FINAL value MUST be assigned to variable named 'result'. "
    "NEVER use custom final variable names like 'total_value', 'sum_price', 'avg_qty'. "
    "ALWAYS use: result = <your calculation>. "
    "For read queries that compute total via product of two columns and sum (e.g., colA * colB).sum(), "
    "ignore synthetic summary rows with missing identifiers/metadata. "
    "op must be 'edit' if the user asks to modify data (delete, add, rename, update values), otherwise 'read'. "
    "commit_df must be true when DataFrame is modified, otherwise false. "
    "CRITICAL: NO IMPORTS ALLOWED. Do NOT write any import statements (import/from). "
    "CRITICAL: DO NOT USE lambda expressions. "
    "pd and np are already available in the execution environment. "
    "CRITICAL: For op='edit', your analysis_code MUST ALWAYS include these lines at the end:\n"
    "COMMIT_DF = True\n"
    "result = {'status': 'updated'}\n"
    "If op == 'edit', your code MUST assign the updated DataFrame back to variable df (e.g. df = df.drop(...)) "
    "or use df.loc/at assignment or inplace=True. "
    "After updates like df.loc[...] = value, DO NOT overwrite df with a scalar/series extraction "
    "(e.g. DO NOT do df = df.loc[...].iloc[0]). "
    "CRITICAL ROW DELETION RULES:\n"
    "- When user mentions row numbers/positions (e.g. 'delete rows 98 and 99', 'видали рядки 98 і 99'), "
    "these are 1-based row positions in the visible table.\n"
    "- Phrase 'рядки з номерами X, Y' also means row positions, not ID values.\n"
    "- You MUST use df.drop(index=[...]) with 0-based indices (subtract 1 from each number).\n"
    "- You MUST call df.reset_index(drop=True) BEFORE dropping to ensure correct positional indexing.\n"
    "- DO NOT filter by ID column unless user explicitly says 'where ID equals' or 'з ID='.\n"
    "- DO NOT assign filtered result to result variable; ALWAYS assign back to df.\n"
    "ROW/ID DISAMBIGUATION RULE:\n"
    "- Phrase like 'рядок N' usually means 1-based row position, but if N is far beyond table length and "
    "'ID' column exists with value N, treat it as ID lookup.\n"
    "CRITICAL ROW ADDITION RULES:\n"
    "- When adding rows with pd.concat, ALWAYS assign back to df: df = pd.concat([df, new_rows], ignore_index=True).\n"
    "- NEVER assign pd.concat result only to result variable.\n"
    "CRITICAL MUTATION ASSIGNMENT RULES:\n"
    "- NEVER write result = df[...] for edit operations. Use df = df[...] instead.\n"
    "- NEVER write result = df.drop(...) / result = df.rename(...). Use df = ... instead.\n"
    "\nExample for 'delete rows 98 and 99':\n"
    "{\n"
    '  "analysis_code": "df = df.copy()\\ndf = df.reset_index(drop=True)\\ndf = df.drop(index=[97, 98])\\nCOMMIT_DF = True\\nresult = {\'status\': \'updated\'}",\n'
    '  "short_plan": "Видалити рядки 98 та 99 за позицією",\n'
    '  "op": "edit",\n'
    '  "commit_df": true\n'
    "}\n"
    "\nExample WRONG (DO NOT DO THIS):\n"
    "{\n"
    '  "analysis_code": "df = df[df[\'ID\'] != 98]\\nresult = df[df[\'ID\'] != 99]",\n'
    '  "short_plan": "Видалити рядки з ID 98 та 99"\n'
    "}\n"
    "\nExample for 'add 3 empty rows':\n"
    "{\n"
    '  "analysis_code": "df = df.copy()\\nnew_rows = pd.DataFrame([{} for _ in range(3)], columns=df.columns)\\ndf = pd.concat([df, new_rows], ignore_index=True)\\nCOMMIT_DF = True\\nresult = {\'status\': \'updated\'}",\n'
    '  "short_plan": "Додати 3 порожні рядки",\n'
    '  "op": "edit",\n'
    '  "commit_df": true\n'
    "}\n"
)

DEFAULT_FINAL_ANSWER_SYSTEM = (
    "You are a data analysis assistant. Answer in Ukrainian. "
    "CRITICAL: Use ONLY the data from result_text field. NEVER generate or invent numbers. "
    "If you mention any numbers, they must appear in result_text. "
    "Do not mention model numbers or SKUs unless they appear in result_text. "
    "If result_text contains the answer, format it clearly. "
    "If result_text is empty or has an error, explain what to change. "
    "If a keyword for grouping is not explicitly mentioned, first try to infer the grouping column from df_profile column names. "
    "DO NOT use data from df_profile to answer questions about counts or aggregations."
)

DEFAULT_FINAL_REWRITE_SYSTEM = (
    "You rewrite a validated result into a concise, human-friendly Ukrainian answer. "
    "Use ONLY result_text to determine facts. "
    "Do not add any numbers or details not present in result_text. "
    "If result_text is empty or unclear, say that the result is unclear and ask the user to уточнити запит."
)

SHORTCUT_COL_PLACEHOLDER = "_SHORTCUT_COL_"
GROUP_COL_PLACEHOLDER = "__GROUP_COL__"
SUM_COL_PLACEHOLDER = "__SUM_COL__"
AGG_COL_PLACEHOLDER = "__AGG_COL__"
TOP_N_PLACEHOLDER = "__TOP_N__"

_REQUEST_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar("pipeline_request_id", default="-")
_TRACE_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar("pipeline_trace_id", default="-")


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


META_TASK_HINTS = (
    "### Task:",
    "Generate 1-3 broad tags",
    "Create a concise, 3-5 word title",
    "Suggest 3-5 relevant follow-up",
    "determine whether a search is necessary",
    "<chat_history>",
    "Respond to the user query using the provided context",
    "<user_query>",
)


SEARCH_QUERY_META_HINTS = (
    "determine the necessity of generating search queries",
    "prioritize generating 1-3 broad and relevant search queries",
    "return: { \"queries\": [] }",
    "\"queries\": [",
)


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


def _parse_json_dict_from_llm(text: str) -> dict:
    s = (text or "").strip()
    if not s:
        raise ValueError("LLM did not return JSON")
    candidate = _extract_json_candidate(s) or s
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON root must be an object")
    return parsed


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
    return bool(re.search(r"\b(sum|сума|total|загальн\w*|обсяг|volume|units|залишк\w*)\b", q))


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
    profile = df_profile or {}
    price_col = _pick_price_like_column(profile)
    qty_col = _pick_quantity_like_column(profile)
    if not price_col:
        return code
    cols = [str(c) for c in (profile.get("columns") or [])]
    status_col = _pick_availability_column(question, profile) if cols else None
    top_n = _extract_top_n_from_question(question, default=5)
    explicit_status = _has_explicit_status_constraint(question)

    out_cols: List[str] = []
    for c in ("ID", "Категорія", "Бренд", "Модель", price_col):
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
        "    _top_df = _top_df.nlargest(_top_n, _price_col)",
        f"    result = _top_df[{out_cols!r}]",
    ]
    logging.info(
        "event=top_expensive_available_rewrite applied top_n=%s price_col=%s qty_col=%s status_col=%s strict_status=%s",
        top_n,
        price_col,
        qty_col,
        status_col,
        explicit_status,
    )
    return "\n".join(lines) + ("\n" if (code or "").endswith("\n") else "")


def _is_numeric_dtype_text(dtype: str) -> bool:
    d = str(dtype or "").lower()
    return d.startswith(("int", "float", "uint"))


def _is_id_like_col_name(col: str) -> bool:
    return bool(re.search(r"(?:^|_)(id|sku|код|артикул)(?:$|_)", str(col).lower()))


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
      df = df[df['Категорія'] == 'X']
      result = df['Ціна'].mean()
    ->
      _df_read = df.copy(deep=False)
      _df_read = _df_read[_df_read['Категорія'] == 'X']
      result = _df_read['Ціна'].mean()
    """
    text = code or ""
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

_ROUTER_FILTER_CONTEXT_RE = re.compile(
    r"\b(серед|among|within|where|де|тільки|only|лише|having)\b"
    r"|(?:що|які)\s+мають"
    r"|(?:with|for)\s+[a-zа-яіїєґ0-9_]{2,}",
    re.I,
)

_AVAILABILITY_FILTER_CUE_RE = re.compile(
    r"\b(на\s+склад\w*|в\s+наявн\w*|наявн\w*|в\s+запас\w*|запас\w*|залишк\w*|in\s*stock|available|inventory|warehouse|доступн\w*)\b",
    re.I,
)

_ROUTER_METRIC_CUE_RE = re.compile(
    r"\b("
    r"max(?:imum)?|minimum|min|mean|average|avg|median|sum|total|count|"
    r"макс\w*|мін\w*|середн\w*|сума|підсум\w*|кільк\w*|скільк\w*"
    r")\b",
    re.I,
)

_ROUTER_ENTITY_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9_-]{2,}\b")
_GROUPING_CUE_RE = re.compile(
    r"\b("
    r"group\s*by|"
    r"by\s+\w+|"
    r"by\s+(?:category|categories|brand|brands|model|models|type|types)|"
    r"per\s+\w+|"
    r"по\s+(?:категор\w*|бренд\w*|модел\w*|тип\w*|груп\w*)|"
    r"за\s+(?:категор\w*|бренд\w*|модел\w*|тип\w*|груп\w*)"
    r")\b",
    re.I,
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
    has_metric = bool(_ROUTER_METRIC_CUE_RE.search(q))
    has_filter_words = bool(_ROUTER_FILTER_CONTEXT_RE.search(q))
    has_availability_cue = bool(_AVAILABILITY_FILTER_CUE_RE.search(q))
    has_value_filter = _looks_like_value_filter_query(q)
    has_entity_token = bool(_ROUTER_ENTITY_TOKEN_RE.search(q))
    has_grouping_cue = bool(_GROUPING_CUE_RE.search(q))
    if has_grouping_cue and not (has_filter_words or has_availability_cue or has_value_filter or has_entity_token):
        # Grouped aggregations ("by/per/по/за ...") should not be treated as subset
        # unless we have an explicit filter signal.
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
    has_grouping_cue = bool(_GROUPING_CUE_RE.search(q))
    if has_grouping_cue:
        # "по категоріях/брендах" is usually a grouped aggregation over all rows,
        # not a subset filter. Keep subset mode only when we can detect an actual entity term.
        if not profile:
            return False
        try:
            maybe_terms = _extract_subset_terms_from_question(q, profile, limit=2)
            if not maybe_terms:
                return False
        except Exception:
            return False
    has_metric = bool(_ROUTER_METRIC_CUE_RE.search(q))
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
    return False


def _should_reject_router_hit_for_read(
    has_edit: bool,
    analysis_code: str,
    router_meta: Optional[dict],
    question: str = "",
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
    if re.search(r"(?m)^\s*COMMIT_DF\s*=\s*True\s*$", analysis_code or ""):
        return True
    return _auto_detect_commit(analysis_code or "")


_METRIC_CONTEXT_RE = re.compile(
    r"\b(value|price|amount|total|sum|qty|count|number|rows?|records?|items?|products?|sales?|revenue|profit)\b"
    r"|\b(значенн\w*|цін\w*|варт\w*|сум\w*|кільк\w*|рядк\w*|запис\w*|елемент\w*|товар\w*)\b",
    re.I,
)

_METRIC_PATTERNS = {
    "mean": r"\b(mean|average|avg|середн\w*)\b",
    "min": r"\b(min|мін(ім(ум|ал\w*)?)?)\b|\bminimum\b(?=\s+\S+)",
    "max": r"\b(max|макс(имум|имал\w*)?)\b|\bmaximum\b(?=\s+\S+)",
    "sum": r"\b(sum|сума|підсум\w*)\b",
    "median": r"\b(median|медіан\w*)\b",
}

_COUNT_CONTEXT_RE = re.compile(
    r"\b(items?|products?|rows?|records?|entries|values?|brands?|categories|orders?|customers?)\b"
    r"|\b(товар\w*|рядк\w*|запис\w*|елемент\w*|бренд\w*|категор\w*|замовлен\w*|клієнт\w*)\b",
    re.I,
)

_COUNT_WORD_RE = re.compile(r"\b(count|кільк\w*)\b", re.I)
_COUNT_NUMBER_OF_RE = re.compile(r"\bnumber\s+of\s+([a-z]+)\b", re.I)
_COUNT_QTY_RE = re.compile(r"\bqty\b", re.I)

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

    metric_order = ["count", "max", "min", "mean", "median", "sum"]
    metric = next((m for m in metric_order if m in metrics), "count")
    if llm_agg in {"count", "sum", "mean", "min", "max", "median"}:
        metric = llm_agg
    # Disambiguate "загальна кількість ... на складі": sum stock units, not row count.
    if metric == "count" and qty_like_col and (_is_sum_intent(question) or availability_mode):
        metric = "sum"
    if llm_avail_mode in {"in", "out", "any"}:
        availability_mode = llm_avail_mode
    if availability_mode and llm_avail_col in columns:
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

    patterns = [p for p in (_subset_term_pattern(t) for t in terms) if p]
    if not patterns:
        return None

    code_lines: List[str] = [
        "_work = df.copy(deep=False)",
        f"_text_cols = {text_cols!r}",
        "if not _text_cols:",
        "    _text_cols = list(_work.columns)",
        "_text = _work[_text_cols].fillna('').astype(str).apply(' '.join, axis=1).str.lower()",
        f"_patterns = {patterns!r}",
        "_mask_and = pd.Series(True, index=_work.index)",
        "for _pat in _patterns:",
        "    _mask_and = _mask_and & _text.str.contains(_pat, regex=True, na=False)",
        "_mask_or = pd.Series(False, index=_work.index)",
        "for _pat in _patterns:",
        "    _mask_or = _mask_or | _text.str.contains(_pat, regex=True, na=False)",
        "_mask = _mask_and if _mask_and.any() else _mask_or",
        "_work = _work.loc[_mask].copy()",
    ]
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
    if not _is_count_intent(question) or _is_sum_intent(question):
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
    """Return first mention position of a column name, avoiding short-token false positives like ID in NVIDIA."""
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
        code_lines.append("    _row_text = df.fillna('').astype(str).apply(' '.join, axis=1).str.lower()")
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

    # Read-only robust extractor for "value in row N": try row position first, then fallback to ID=N.
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

    # Edit by ID: "зміни ... для ID 1003 на ..."
    if _has_edit_triggers(q):
        m_id = re.search(r"\bID\s*[:=]?\s*(\d+)\b", q, re.I)
        if m_id:
            item_id = int(m_id.group(1))
            col = _find_column_in_text(q, columns)
            raw_value = _parse_set_value(q)
            value = _parse_literal(raw_value) if raw_value is not None else None
            if col and value is not None and str(col).lower() != "id":
                code_lines: List[str] = []
                code_lines.append("df = df.copy()")
                code_lines.append(f"_id = {item_id}")
                code_lines.append("if 'ID' in df.columns:")
                code_lines.append("    _id_target = str(_id).strip()")
                code_lines.append("    _id_col = df['ID'].astype(str).str.strip()")
                code_lines.append("    _id_col_norm = _id_col.str.replace(r'\\.0+$', '', regex=True)")
                code_lines.append("    _mask = (_id_col == _id_target) | (_id_col_norm == _id_target)")
                code_lines.append("    if _mask.any():")
                code_lines.append(f"        df.loc[_mask, {col!r}] = {value!r}")
                code_lines.append("        COMMIT_DF = True")
                code_lines.append(f"        result = {{'status': 'updated', 'id': _id, 'column': {col!r}, 'new_value': {value!r}}}")
                code_lines.append("    else:")
                code_lines.append("        result = {'status': 'not_found', 'id': _id}")
                code_lines.append("else:")
                code_lines.append("    result = {'status': 'no_id_column'}")
                plan = f"Змінити {col} для ID {item_id}."
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
    if event == "codegen_retry":
        reason = str((payload or {}).get("reason") or "").strip()
        if reason == "missing_subset_filter":
            return "Повторно генерую код: додам обов'язкову фільтрацію підмножини."
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
        base_llm_api_key: str = Field(default=os.getenv("BASE_LLM_API_KEY", "sk-bf-59dc5727-7e10-4d29-a28e-cf7d655f595c"))
        base_llm_model: str = Field(default=os.getenv("BASE_LLM_MODEL", "vllm-8099/minimax-m2n"))

        sandbox_url: str = Field(default=os.getenv("SANDBOX_URL", "http://sandbox:8081"))
        sandbox_api_key: str = Field(default=os.getenv("SANDBOX_API_KEY", ""))

        max_rows: int = Field(default=_env_int("PIPELINE_MAX_ROWS", 200000), ge=1)
        preview_rows: int = Field(default=_env_int("PIPELINE_PREVIEW_ROWS", 200000), ge=1)
        max_cell_chars: int = Field(default=_env_int("PIPELINE_MAX_CELL_CHARS", 200), ge=10)
        code_timeout_s: int = Field(default=_env_int("PIPELINE_CODE_TIMEOUT_S", 120), ge=1)
        max_stdout_chars: int = Field(default=_env_int("PIPELINE_MAX_STDOUT_CHARS", 8000), ge=1000)

        session_cache_ttl_s: int = Field(default=_env_int("PIPELINE_SESSION_CACHE_TTL_S", 1800), ge=60)
        wait_tick_s: int = Field(default=_env_int("PIPELINE_WAIT_TICK_S", 5), ge=0)
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
            base_url=self.valves.base_llm_base_url, api_key=self.valves.base_llm_api_key or "DUMMY_KEY"
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
        router_cfg = ShortcutRouterConfig(
            catalog_path=self.valves.shortcut_catalog_path,
            index_path=self.valves.shortcut_index_path,
            meta_path=self.valves.shortcut_meta_path,
            top_k=self.valves.shortcut_top_k,
            threshold=float(self.valves.shortcut_threshold),
            margin=float(self.valves.shortcut_margin),
            vllm_base_url=self.valves.vllm_base_url,
            vllm_embed_model=self.valves.vllm_embed_model,
            vllm_api_key=self.valves.vllm_api_key,
            vllm_timeout_s=self.valves.vllm_timeout_s,
            enabled=bool(self.valves.shortcut_enabled),
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

    def _sandbox_get_profile(self, df_id: str) -> Optional[dict]:
        url = f"{self.valves.sandbox_url.rstrip('/')}/v1/dataframe/{df_id}/profile"
        try:
            resp = requests.get(url, headers=self._sandbox_headers(), timeout=10)
            if resp.status_code != 200:
                return None
            payload = resp.json()
        except Exception:
            return None
        return payload.get("profile") or {}

    def _fetch_file_meta(self, file_id: str, file_obj: Optional[dict]) -> dict:
        if isinstance(file_obj, dict):
            if (
                file_obj.get("filename")
                or file_obj.get("content_type")
                or file_obj.get("name")
                or file_obj.get("path")
            ):
                return file_obj
        url = f"{self.valves.webui_base_url.rstrip('/')}/api/v1/files/{file_id}"
        resp = requests.get(url, headers=self._webui_headers(), timeout=DEF_TIMEOUT_S)
        resp.raise_for_status()
        if resp.headers.get("content-type", "").startswith("application/json"):
            return resp.json()
        return {"raw": resp.text}

    def _fetch_file_bytes(self, file_id: str, meta: dict, file_obj: Optional[dict]) -> bytes:
        if isinstance(file_obj, dict):
            b64 = file_obj.get("data_b64") or file_obj.get("data") or file_obj.get("content_b64")
            if b64:
                return base64.b64decode(b64)

            url = file_obj.get("url") or file_obj.get("content_url")
            if url:
                if url.startswith("/"):
                    url = f"{self.valves.webui_base_url.rstrip('/')}{url}"
                resp = requests.get(url, headers=self._webui_headers(), timeout=DEF_TIMEOUT_S)
                resp.raise_for_status()
                return resp.content

            path = file_obj.get("path") or file_obj.get("filepath") or file_obj.get("file_path")
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    return f.read()

        base = self.valves.webui_base_url.rstrip("/")
        url = f"{base}/api/v1/files/{file_id}/content"
        resp = requests.get(url, headers=self._webui_headers(), timeout=DEF_TIMEOUT_S)
        resp.raise_for_status()
        return resp.content

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
        skill_injected = _SPREADSHEET_SKILL_PROMPT_MARKER in (system or "")
        logging.info(
            "event=llm_json_skill_injection active=%s prompt_hash=%s",
            skill_injected,
            hashlib.sha256((system or "").encode("utf-8")).hexdigest()[:16],
        )
        logging.info(
            "event=llm_json_request system_preview=%s user_preview=%s",
            _safe_trunc(system, 800),
            _safe_trunc(user, 1200),
        )
        resp = self._llm.chat.completions.create(
            model=self.valves.base_llm_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        logging.info("event=llm_json_response preview=%s", _safe_trunc(text, 1200))
        return _parse_json_dict_from_llm(text)

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
        payload_obj: Dict[str, Any] = {"question": question, "df_profile": profile}
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
            "df_profile": profile,
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
            "df_profile": profile,
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
            "df_profile": profile,
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
            "Use exact column names from df_profile only.",
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
                "df_profile": profile,
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
            "dtypes": (profile or {}).get("dtypes") or {},
            "df_profile": profile or {},
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
            "Use semantic similarity from question, alias, column names, dtypes, and preview values. "
            "If uncertain, return empty column and confidence 0."
        )
        system = self._with_spreadsheet_skill_prompt(system, question, profile, focus="column")
        payload = {
            "question": question,
            "alias": alias,
            "role": role,
            "columns": columns[:200],
            "dtypes": (profile or {}).get("dtypes") or {},
            "preview": ((profile or {}).get("preview") or [])[:20],
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
            "dtypes": dtypes,
            "df_profile": profile or {},
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
            "dtypes": dtypes,
            "preview": ((profile or {}).get("preview") or [])[:20],
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
        top_n: Optional[int] = None
        if isinstance(top_n_raw, (int, float)):
            top_n = max(1, int(top_n_raw))
        elif isinstance(top_n_raw, str) and top_n_raw.strip().isdigit():
            top_n = max(1, int(top_n_raw.strip()))
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
                r"|\b(?:id|sku|код|артикул)\s*[:=]?\s*\d+\b",
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

    def _lookup_status_pattern(self, value: Any) -> Optional[str]:
        s = str(value or "").strip().lower()
        if not s:
            return None
        # Positive availability intent.
        if re.search(r"(на\s+склад|в\s+наявн|наявн|в\s+запас\w*|запас\w*|залишк\w*|in\s*stock|available|inventory|warehouse|доступн|резерв|закінч\w*)", s):
            return r"(?:в\s*наявн|наявн|в\s*запас\w*|запас\w*|залишк\w*|in\s*stock|available|inventory|warehouse|доступн|на\s*склад|резерв\w*|закінч\w*)"
        # Negative availability intent.
        if re.search(r"(нема|відсутн|out\s*of\s*stock|unavailable|not\s*available)", s):
            return r"(?:нема|відсутн|out\s*of\s*stock|unavailable|not\s*available)"
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

    def _normalize_lookup_filters(
        self,
        question: str,
        filters: List[Dict[str, Any]],
        dtypes: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        explicit_eq = self._lookup_has_explicit_eq_cue(question)
        out: List[Dict[str, Any]] = []
        for f in filters:
            cur = dict(f)
            col = str(cur.get("column") or "").strip()
            op = str(cur.get("op") or "").strip().lower()
            val = cur.get("value")
            dtype = str((dtypes or {}).get(col, "")).lower()
            is_numeric = dtype.startswith(("int", "float", "uint"))
            is_status_like = self._lookup_is_status_like_column(col)

            if (
                op == "eq"
                and not explicit_eq
                and is_status_like
                and self._lookup_filter_value_is_entity_like(val)
            ):
                cur["op"] = "contains"
                logging.info(
                    "event=lookup_filter_operator_adjust column=%s from=eq to=contains reason=status_semantic value_preview=%s",
                    col,
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
                cur["op"] = "contains"
                logging.info(
                    "event=lookup_filter_operator_adjust column=%s from=eq to=contains value_preview=%s",
                    col,
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

    def _llm_pick_lookup_slots(self, question: str, profile: dict) -> Dict[str, Any]:
        columns = [str(c) for c in (profile or {}).get("columns") or []]
        if not columns:
            return {}
        dtypes = (profile or {}).get("dtypes") or {}
        system = (
            "Map the query to table lookup slots. "
            "Return ONLY JSON with keys: mode, filters, output_columns, limit. "
            "mode must be 'lookup' or 'other'. "
            "filters must be a list of objects with keys: column, op, value. "
            "column must be exact from columns. op must be one of: eq, ne, gt, ge, lt, le, contains. "
            "output_columns must be a list of exact column names from columns. "
            "limit must be integer or null. "
            "CRITICAL FILTER RULES: "
            "Use 'eq' for exact IDs/status/boolean flags and explicit exact-match asks. "
            "Use 'contains' for brand/model/product names, features/materials, colors, categories/types. "
            "When in doubt between 'eq' and 'contains' for text values, prefer 'contains'. "
            "If query likely needs search across multiple text columns, set mode='other'. "
            "Use mode='lookup' when query asks to find/show rows by conditions (e.g., where price equals X). "
            "For single-value questions like 'яка модель ...', set limit=1."
        )
        system = self._with_spreadsheet_skill_prompt(system, question, profile, focus="column")
        payload = {
            "question": question,
            "columns": columns[:200],
            "dtypes": dtypes,
            "preview": ((profile or {}).get("preview") or [])[:20],
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
                if op not in {"eq", "ne", "gt", "ge", "lt", "le", "contains"}:
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

        limit_raw = (parsed or {}).get("limit")
        if isinstance(limit_raw, (int, float)):
            out["limit"] = max(1, int(limit_raw))
        elif isinstance(limit_raw, str) and limit_raw.strip().isdigit():
            out["limit"] = max(1, int(limit_raw.strip()))

        if mode == "lookup" and self._lookup_requires_multicol_fallback(question, filters, columns, dtypes):
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
        if fallback_reason:
            hints["reason"] = fallback_reason
        return hints

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
        output_columns = [str(c) for c in ((slots or {}).get("output_columns") or []) if str(c) in columns]
        if not output_columns:
            mentioned = _find_columns_in_text(question, columns)
            output_columns = [c for c in mentioned if c not in [str(f.get("column")) for f in filters]]
        if not output_columns:
            dtypes = (profile or {}).get("dtypes") or {}
            text_cols = [c for c in columns if not str(dtypes.get(c, "")).lower().startswith(("int", "float", "uint"))]
            output_columns = text_cols[:2] if text_cols else columns[:1]

        limit = (slots or {}).get("limit")
        lines: List[str] = [
            "_work = df.copy(deep=False)",
        ]
        for i, f in enumerate(filters):
            col = str(f.get("column"))
            op = str(f.get("op")).lower()
            val = f.get("value")
            mask_name = f"_m{i}"
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
            elif op == "contains":
                sval = "" if val is None else str(val)
                status_pat = self._lookup_status_pattern(val) if self._lookup_is_status_like_column(col) else None
                if status_pat:
                    lines.append(f"_pat{i} = {status_pat!r}")
                    lines.append(f"{mask_name} = _work[_c{i}].astype(str).str.contains(_pat{i}, case=False, regex=True, na=False)")
                else:
                    lines.append(f"{mask_name} = _work[_c{i}].astype(str).str.contains({sval!r}, case=False, na=False)")
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
                    status_pat = self._lookup_status_pattern(val) if self._lookup_is_status_like_column(col) else None
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

        lines.append(f"_out_cols = {output_columns!r}")
        lines.append("_out_cols = [c for c in _out_cols if c in _work.columns]")
        if isinstance(limit, int) and limit > 0:
            lines.append(f"_work = _work.head({limit})")
        lines.append("if _out_cols:")
        lines.append("    _out = _work[_out_cols]")
        lines.append("else:")
        lines.append("    _out = _work")
        lines.append("if len(_out) == 0:")
        lines.append("    result = []")
        lines.append("elif len(_out.columns) == 1 and len(_out) == 1:")
        lines.append("    result = _out.iloc[0, 0]")
        lines.append("else:")
        lines.append("    result = _out")

        plan = "Виконати пошук рядків за умовами та повернути релевантні колонки."
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
        if top_n_slot is None:
            logging.info("event=ranking_shortcut_skip reason=missing_top_n mode=%s", mode)
            return None

        columns = [str(c) for c in ((profile or {}).get("columns") or [])]
        dtypes = (profile or {}).get("dtypes") or {}
        if mode == "row_ranking":
            if not _has_ranking_cues(question) and _looks_like_value_filter_query(question):
                logging.info("event=ranking_shortcut_skip reason=filter_like_query mode=row")
                return None
            top_n = int(top_n_slot)
            top_n = max(1, top_n)
        else:
            top_n = int(top_n_slot)
            top_n = max(1, top_n)
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
            for c in ("ID", *entity_cols, metric_col):
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
        lines.append("    _res = _work.groupby(_group_col).size().reset_index(name='count')")
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

    def _llm_extract_subset_terms(self, question: str, profile: dict) -> List[str]:
        columns = [str(c) for c in (profile or {}).get("columns") or []]
        if not columns:
            return []
        system = (
            "Extract only subset-filter terms from user query for dataframe filtering. "
            "Return ONLY JSON: {\"terms\": [..]}. "
            "Include product/entity/category/feature terms only. "
            "Exclude metric words and aggregation words like max/min/avg/sum/count/price/ціна/кількість. "
            "If useful, include 1-2 short aliases in other likely languages (uk/ru/en) that may appear in table values. "
            "Prefer 1-6 concise terms."
        )
        system = self._with_spreadsheet_skill_prompt(system, question, profile, focus="column")
        payload = {
            "question": question,
            "columns": columns[:200],
            "preview": ((profile or {}).get("preview") or [])[:20],
        }
        try:
            parsed = self._llm_json(system, json.dumps(payload, ensure_ascii=False))
        except Exception:
            return []
        out: List[str] = []
        raw = (parsed or {}).get("terms")
        if isinstance(raw, list):
            for t in raw:
                s = str(t).strip()
                if len(s) < 2 or len(s) > 40:
                    continue
                if re.fullmatch(r"[\d\s.,:%\-]+", s):
                    continue
                if s not in out:
                    out.append(s)
        logging.info("event=subset_terms_llm terms=%s", _safe_trunc(out, 240))
        return out[:6]

    def _llm_pick_subset_metric_slots(self, question: str, profile: dict) -> Dict[str, Any]:
        columns = [str(c) for c in (profile or {}).get("columns") or []]
        if not columns:
            return {}
        system = (
            "Map subset filtering query to aggregation slots. "
            "Return ONLY JSON with keys: agg, metric_col, availability_mode, availability_col. "
            "agg must be one of: count, sum, mean, min, max, median. "
            "metric_col and availability_col must be exact names from columns or empty string. "
            "availability_mode must be one of: in, out, any, none. "
            "Use agg='sum' when user asks total quantity/amount in subset. "
            "Use agg='count' only when user asks number of matching rows/items."
        )
        system = self._with_spreadsheet_skill_prompt(system, question, profile, focus="column")
        payload = {
            "question": question,
            "columns": columns[:200],
            "dtypes": (profile or {}).get("dtypes") or {},
            "preview": ((profile or {}).get("preview") or [])[:20],
        }
        try:
            parsed = self._llm_json(system, json.dumps(payload, ensure_ascii=False))
        except Exception:
            return {}
        if not isinstance(parsed, dict):
            return {}
        out: Dict[str, Any] = {}
        agg = str((parsed or {}).get("agg") or "").strip().lower()
        metric_col = str((parsed or {}).get("metric_col") or "").strip()
        availability_mode = str((parsed or {}).get("availability_mode") or "").strip().lower()
        availability_col = str((parsed or {}).get("availability_col") or "").strip()
        if agg in {"count", "sum", "mean", "min", "max", "median"}:
            out["agg"] = agg
        if metric_col in columns:
            out["metric_col"] = metric_col
        if availability_mode in {"in", "out", "any", "none"}:
            out["availability_mode"] = availability_mode
        if availability_col in columns:
            out["availability_col"] = availability_col
        logging.info("event=subset_slots_llm slots=%s", _safe_trunc(out, 260))
        return out

    def _build_subset_keyword_metric_shortcut(
        self,
        question: str,
        profile: dict,
        preferred_col: Optional[str] = None,
    ) -> Optional[Tuple[str, str]]:
        terms_hint = self._llm_extract_subset_terms(question, profile)
        slots_hint = self._llm_pick_subset_metric_slots(question, profile)
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
            if _should_reject_router_hit_for_read(has_edit, candidate_code, candidate_meta, question):
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
        if inferred_op == "read" and not is_meta and not has_edit and metrics:
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
        # Entity-like tokens (e.g., NVIDIA, RTX4090) usually indicate filtered subset.
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
            "dtypes": (profile or {}).get("dtypes") or {},
            "preview": ((profile or {}).get("preview") or [])[:20],
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
            "dtypes": (profile or {}).get("dtypes") or {},
            "df_profile": profile or {},
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
        if isinstance(top_n_raw, (int, float)):
            out["top_n"] = max(1, int(top_n_raw))
        elif isinstance(top_n_raw, str) and top_n_raw.strip().isdigit():
            out["top_n"] = max(1, int(top_n_raw.strip()))
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

        lines: List[str] = []
        if isinstance(data, dict):
            items = list(data.items())
            try:
                items.sort(key=lambda kv: float(kv[1]) if kv[1] is not None else float("-inf"), reverse=True)
            except Exception:
                pass
            rows = items[: max(1, top_n)]
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
                top_rows = data[: max(1, top_n)]
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
        if isinstance(data, dict):
            rows = list(data.items())[: max(1, top_n)]
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
            top_rows = data[: max(1, top_n)]
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

        list_answer = self._format_scalar_list_from_result(result_text)
        if list_answer:
            return list_answer

        table = self._format_table_from_result(result_text)
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
            return self._format_top_pairs_from_result(result_text, top_n=100)
        if not (wants_pairs and wants_counts):
            return None
        return self._format_top_pairs_from_result(result_text, top_n=15)

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
            parts.append(f"ID {item_id}")
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
        def _log_return(mode: str, text: str) -> None:
            logging.info("event=final_answer_return mode=%s preview=%s", mode, _safe_trunc(text, 300))

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
        if run_status == "ok" and not (error or "").strip() and not (result_text or "").strip():
            msg = "За умовою запиту не знайдено значень."
            _log_return("empty_result", msg)
            return msg
        deterministic = self._deterministic_answer(question, result_text, profile)
        if deterministic:
            logging.info("event=final_answer mode=deterministic preview=%s", _safe_trunc(deterministic, 300))
            _log_return("deterministic", deterministic)
            return deterministic
        system = self._prompts.get("final_answer_system", DEFAULT_FINAL_ANSWER_SYSTEM)
        payload = {
            "question": question,
            "df_profile": profile,
            "plan": plan,
            "analysis_code": code,
            "exec_status": run_status,
            "stdout": stdout,
            "result_text": result_text,
            "result_meta": result_meta,
            "error": error,
        }
        logging.info(
            "event=llm_final_request question_preview=%s payload_preview=%s",
            _safe_trunc(question, 400),
            _safe_trunc(json.dumps(payload, ensure_ascii=False), 1200),
        )
        resp = self._llm.chat.completions.create(
            model=self.valves.base_llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.2,
        )
        answer = (resp.choices[0].message.content or "").strip()
        logging.info("event=llm_final_response preview=%s", _safe_trunc(answer, 1200))
        if result_text and re.search(r"\d+", result_text or ""):
            real_numbers = set(re.findall(r"\d+", result_text))
            llm_numbers = set(re.findall(r"\d+", answer))
            if llm_numbers and not (llm_numbers & real_numbers):
                logging.warning(
                    "event=final_answer mode=llm_hallucinated real=%s llm=%s", real_numbers, llm_numbers
                )
                rewrite_system = self._prompts.get("final_rewrite_system", DEFAULT_FINAL_REWRITE_SYSTEM)
                rewrite_payload = {"question": question, "result_text": result_text}
                try:
                    rewrite_resp = self._llm.chat.completions.create(
                        model=self.valves.base_llm_model,
                        messages=[
                            {"role": "system", "content": rewrite_system},
                            {"role": "user", "content": json.dumps(rewrite_payload, ensure_ascii=False)},
                        ],
                        temperature=0.1,
                    )
                    rewrite = (rewrite_resp.choices[0].message.content or "").strip()
                    if rewrite:
                        rewrite_numbers = set(re.findall(r"\d+", rewrite))
                        if not rewrite_numbers or (real_numbers & rewrite_numbers):
                            logging.info("event=final_answer mode=rewrite preview=%s", _safe_trunc(rewrite, 300))
                            _log_return("rewrite", rewrite)
                            return rewrite
                except Exception as exc:
                    logging.warning("event=final_rewrite_failed err=%s", str(exc))
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
        url = f"{self.valves.sandbox_url.rstrip('/')}/v1/dataframe/load"
        payload = {
            "file_id": file_id,
            "filename": _guess_filename(meta),
            "content_type": (meta or {}).get("content_type") or (meta or {}).get("mime"),
            "data_b64": base64.b64encode(data).decode("ascii"),
            "max_rows": self.valves.max_rows,
            "preview_rows": self.valves.preview_rows,
        }
        resp = requests.post(url, headers=self._sandbox_headers(), json=payload, timeout=DEF_TIMEOUT_S)
        resp.raise_for_status()
        return resp.json()

    def _sandbox_run(self, df_id: str, code: str) -> Dict[str, Any]:
        url = f"{self.valves.sandbox_url.rstrip('/')}/v1/dataframe/run"
        payload = {
            "df_id": df_id,
            "code": code,
            "timeout_s": self.valves.code_timeout_s,
            "preview_rows": self.valves.preview_rows,
            "max_cell_chars": self.valves.max_cell_chars,
            "max_stdout_chars": self.valves.max_stdout_chars,
        }
        resp = requests.post(url, headers=self._sandbox_headers(), json=payload, timeout=DEF_TIMEOUT_S)
        resp.raise_for_status()
        return resp.json()

    def _prepare_analysis_code_for_question(
        self,
        question: str,
        profile: dict,
        has_edit: bool,
    ) -> Dict[str, Any]:
        events: List[Tuple[str, Dict[str, Any]]] = []
        events.append(("codegen", {"question": _safe_trunc(question, 200)}))

        edit_expected = False
        op: Optional[str] = None
        commit_df: Optional[bool] = None
        plan = ""

        router_hit, shortcut, router_meta, planner_hints = self._select_router_or_shortcut(question, profile, has_edit)
        lookup_hints = planner_hints.get("lookup_hints") if isinstance(planner_hints, dict) else None
        if router_hit:
            analysis_code, router_meta = router_hit
            plan = f"retrieval_intent:{router_meta.get('intent_id')}"
            events.append(("codegen_shortcut", {"intent_id": router_meta.get("intent_id")}))
        elif shortcut:
            analysis_code, plan = shortcut
            events.append(("codegen_shortcut", {}))
        else:
            analysis_code, plan, op, commit_df = self._plan_code(question, profile, lookup_hints=lookup_hints)
            if not (analysis_code or "").strip():
                events.append(("codegen_empty", {}))
                return {
                    "ok": False,
                    "events": events,
                    "status": "codegen_empty",
                    "message_sync": "Я не зміг згенерувати код для цього запиту. Спробуйте сформулювати інакше.",
                    "message_stream": "Не вдалося згенерувати код аналізу. Спробуйте інше формулювання.",
                }

        analysis_code, count_err = _enforce_count_code(question, analysis_code)
        if count_err:
            return {
                "ok": False,
                "events": events,
                "status": "invalid_code",
                "message_sync": f"Неможливо виконати запит: {count_err}",
                "message_stream": f"Неможливо виконати: {count_err}",
            }
        analysis_code = _enforce_entity_nunique_code(question, analysis_code, profile)

        analysis_code, edit_expected, finalize_err = _finalize_code_for_sandbox(
            question, analysis_code, op, commit_df, df_profile=profile
        )
        need_retry_missing_result = bool(
            finalize_err and "missing_result_assignment" in finalize_err and not shortcut and not router_hit
        )
        need_retry_missing_filter = bool(finalize_err and "missing_subset_filter" in finalize_err)
        need_retry_read_mutation = bool(
            finalize_err and "read-запиту змінює таблицю" in finalize_err and not shortcut and not router_hit
        )
        if need_retry_missing_result or need_retry_missing_filter or need_retry_read_mutation:
            if need_retry_missing_filter:
                retry_reason = "missing_subset_filter"
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
                events.append(("codegen_empty", {}))
                return {
                    "ok": False,
                    "events": events,
                    "status": "codegen_empty",
                    "message_sync": "Я не зміг згенерувати код для цього запиту. Спробуйте сформулювати інакше.",
                    "message_stream": "Не вдалося згенерувати код аналізу. Спробуйте інше формулювання.",
                }
            analysis_code, count_err = _enforce_count_code(question, analysis_code)
            if count_err:
                return {
                    "ok": False,
                    "events": events,
                    "status": "invalid_code",
                    "message_sync": f"Неможливо виконати запит: {count_err}",
                    "message_stream": f"Неможливо виконати: {count_err}",
                }
            analysis_code = _enforce_entity_nunique_code(question, analysis_code, profile)
            analysis_code, edit_expected, finalize_err = _finalize_code_for_sandbox(
                question, analysis_code, op, commit_df, df_profile=profile
            )

        if finalize_err:
            if "missing_result_assignment" in finalize_err:
                status = "invalid_missing_result"
            elif "missing_subset_filter" in finalize_err:
                status = "invalid_subset_filter"
            else:
                status = "invalid_read_mutation"
            return {
                "ok": False,
                "events": events,
                "status": status,
                "message_sync": f"Неможливо виконати запит: {finalize_err}",
                "message_stream": f"Неможливо виконати: {finalize_err}",
            }

        if _has_forbidden_import_nodes(analysis_code):
            return {
                "ok": False,
                "events": events,
                "status": "invalid_import",
                "message_sync": "Неможливо виконати запит: згенерований код містить заборонений import.",
                "message_stream": "Неможливо виконати: згенерований код містить заборонений import.",
            }

        analysis_code, plan = self._resolve_shortcut_placeholders(analysis_code, plan, question, profile)
        analysis_code = textwrap.dedent(analysis_code or "").strip() + "\n"
        if "df_profile" in (analysis_code or ""):
            analysis_code = f"df_profile = {profile!r}\n" + analysis_code
        analysis_code = _normalize_generated_code(analysis_code)
        logging.info("event=analysis_code preview=%s", _safe_trunc(analysis_code, 4000))
        if _has_forbidden_import_nodes(analysis_code):
            return {
                "ok": False,
                "events": events,
                "status": "invalid_import",
                "message_sync": "Неможливо виконати запит: згенерований код містить заборонений import.",
                "message_stream": "Неможливо виконати: згенерований код містить заборонений import.",
            }

        return {
            "ok": True,
            "events": events,
            "analysis_code": analysis_code,
            "plan": plan,
            "op": op,
            "commit_df": commit_df,
            "edit_expected": edit_expected,
        }

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
        events: List[Tuple[str, Dict[str, Any]]] = []
        events.append(("sandbox_run", {"df_id": df_id}))
        run_resp = self._sandbox_run(df_id, analysis_code)

        run_status = run_resp.get("status", "")
        run_error = run_resp.get("error", "") or ""
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
                            retry_code = f"df_profile = {profile!r}\n" + retry_code
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

        return {
            "events": events,
            "run_resp": run_resp,
            "analysis_code": analysis_code,
            "plan": plan,
            "edit_expected": edit_expected,
        }

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
            return {
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
                return {
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

        return {
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
            if self.valves.debug:
                logging.info("event=query_selection_tail payload=%s", _safe_trunc(_query_selection_debug(messages), 1200))
            if not question:
                raw_user_message = (user_message or "").strip()
                reason = "meta_without_user_query" if _is_meta_task_text(raw_user_message) else "no_user_query_found"
                logging.info("event=skip_meta_task_sync reason=%s", reason)
                self._emit(event_emitter, "final_answer", {"status": "skipped_no_user_query"})
                return ""
            if _is_search_query_meta_task(question):
                logging.info("event=skip_meta_task_sync reason=search_query_meta_task")
                self._emit(event_emitter, "final_answer", {"status": "skipped_meta_task"})
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

            self._emit(event_emitter, "file_id", {"file_id": file_id})
            if not file_id:
                self._emit(event_emitter, "no_file", {"body_keys": list((body or {}).keys())})
                self._debug_body(body, messages)
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
                    return "Не вдалося завантажити таблицю в пісочницю (sandbox)."
                self._apply_dynamic_limits(profile)
                self._session_set(session_key, file_id, df_id, profile)

            edit_expected = False
            op = None
            commit_df = None
            
            has_edit = _has_edit_triggers(question)
            prep = self._prepare_analysis_code_for_question(question, profile, has_edit)
            for ev, payload in prep.get("events", []):
                self._emit(event_emitter, ev, payload)
            if not prep.get("ok"):
                status = str(prep.get("status") or "")
                if status in {"invalid_missing_result", "invalid_subset_filter", "invalid_read_mutation", "invalid_import"}:
                    self._emit(event_emitter, "final_answer", {"status": status})
                return str(prep.get("message_sync") or "Неможливо виконати запит.")

            analysis_code = str(prep.get("analysis_code") or "")
            plan = str(prep.get("plan") or "")
            op = prep.get("op")
            commit_df = prep.get("commit_df")
            edit_expected = bool(prep.get("edit_expected"))

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
                return str(post.get("message_sync") or "Не вдалося виконати аналіз.")

            mutation_flags = dict(post.get("mutation_flags") or {})
            self._emit(event_emitter, "final_answer", {"status": run_status})
            wait = self._start_wait(event_emitter, "final_answer")
            try:
                return self._final_answer(
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
            finally:
                self._stop_wait(wait)

        except Exception as exc:
            logging.error("event=pipe_sync_error error=%s", str(exc))
            tb = _safe_trunc(traceback.format_exc(), 2000)
            try:
                self._emit(event_emitter, "error", {"error": str(exc)})
            except Exception:
                pass
            return f"Сталася помилка в пайплайні: {type(exc).__name__}: {exc}\n\n{tb}"
        finally:
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
            if self.valves.debug:
                logging.info("event=query_selection_tail payload=%s", _safe_trunc(_query_selection_debug(messages), 1200))
            if not question:
                raw_user_message = (user_message or "").strip()
                reason = "meta_without_user_query" if _is_meta_task_text(raw_user_message) else "no_user_query_found"
                logging.info("event=skip_meta_task_stream reason=%s", reason)
                emit("final_answer", {"status": "skipped_no_user_query"})
                yield from drain()
                return
            if _is_search_query_meta_task(question):
                logging.info("event=skip_meta_task_stream reason=search_query_meta_task")
                emit("final_answer", {"status": "skipped_meta_task"})
                yield from drain()
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

            emit("file_id", {"file_id": file_id})
            yield from drain()

            if not file_id:
                emit("no_file", {"body_keys": list((body or {}).keys())})
                yield from drain()
                self._debug_body(body, messages)
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
                    yield "Не вдалося завантажити файл у sandbox."
                    return
                
                self._apply_dynamic_limits(profile)
                self._session_set(session_key, file_id, df_id, profile)

            edit_expected = False
            op = None
            commit_df = None
            
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
                yield str(prep.get("message_stream") or "Неможливо виконати запит.")
                return

            analysis_code = str(prep.get("analysis_code") or "")
            plan = str(prep.get("plan") or "")
            op = prep.get("op")
            commit_df = prep.get("commit_df")
            edit_expected = bool(prep.get("edit_expected"))

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
                yield str(post.get("message_stream") or "Помилка виконання у sandbox.")
                return

            mutation_flags = dict(post.get("mutation_flags") or {})
            emit("final_answer", {"status": run_status})
            yield from drain()
            
            yield self._final_answer(
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

        except Exception as exc:
            logging.error("event=pipe_stream_error error=%s", str(exc))
            tb = _safe_trunc(traceback.format_exc(), 2000)
            emit("error", {"error": str(exc)})
            yield from drain()
            yield f"Pipeline error: {type(exc).__name__}: {exc}\n\n{tb}"
        finally:
            _REQUEST_ID_CTX.reset(request_token)
            _TRACE_ID_CTX.reset(trace_token)


pipeline = Pipeline()
