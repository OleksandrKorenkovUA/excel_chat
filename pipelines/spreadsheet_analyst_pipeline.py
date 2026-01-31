import ast
import asyncio
import base64
import inspect
import json
import logging
import os
import re
import textwrap
import threading
import time
import traceback
from typing import Any, ClassVar, Dict, Generator, Iterator, List, Optional, Tuple, Union
from pipelines.shortcut_router.shortcut_router import ShortcutRouter, ShortcutRouterConfig
import requests
from openai import OpenAI
from pydantic import BaseModel, Field




PIPELINES_DIR = os.path.dirname(__file__)

DEF_TIMEOUT_S = int(os.getenv("PIPELINE_HTTP_TIMEOUT_S", "120"))
_LOCAL_PROMPTS = os.path.join(os.path.dirname(__file__), "prompts.txt")
PROMPTS_PATH = _LOCAL_PROMPTS if os.path.exists(_LOCAL_PROMPTS) else os.path.join(PIPELINES_DIR, "prompts.txt")

DEFAULT_PLAN_CODE_SYSTEM = (
    "You write pandas code to answer questions about a DataFrame named df. "
    "Return ONLY valid JSON with keys: analysis_code, short_plan, op, commit_df. "
    "op must be 'edit' if the user asks to modify data (delete, add, rename, update values), otherwise 'read'. "
    "commit_df must be true only when the DataFrame is actually modified, otherwise false. "
    "CRITICAL RULE FOR COUNTS: If user asks for 'count products', 'how many items', or 'quantity of brands' -> YOU MUST USE len(df) or .size(). "
    "NEVER calculate sum() of a column named 'Quantity', 'Amount' or 'Кількість' unless the user EXPLICITLY asks for 'total sum' or 'total volume'. "
    "Example: 'Count products by brand' -> df.groupby('Brand').size(). "
    "If op == 'edit', your code MUST assign the updated DataFrame back to variable df (e.g. df = df.drop(...)) "
    "and MUST set COMMIT_DF = True (or set df_changed = True). "
    "If op == 'read', DO NOT set COMMIT_DF. "
    "Only set COMMIT_DF = True when df is actually modified (df assignment, df.loc/at assignment, inplace=True, or df_changed = True). "
    "If op == 'read', your code MUST NOT modify df (no column creation, no inplace ops). "
    "Compute structural facts (rows/cols/columns) from df, not from df_profile. "
    "analysis_code must be pure Python statements, no imports, no file or network access, no plotting. "
    "Put the final answer into variable result. Keep it concise."
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

SHORTCUT_COL_PLACEHOLDER = "__SHORTCUT_COL__"


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


def _is_meta_task_text(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    if s.startswith("### Task:"):
        return True
    lower = s.lower()
    if lower.startswith("analyze the chat history"):
        return True
    if lower.startswith("respond to the user query using the provided context"):
        return True
    return False


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
    return ""


def _last_user_message(messages: List[dict]) -> str:
    for msg in reversed(messages or []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, str) and part.strip():
                    parts.append(part.strip())
                elif isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
            content = "\n".join(parts).strip()
        if not isinstance(content, str):
            continue
        s = content.strip()
        if not s:
            continue
        if _is_meta_task_text(s):
            extracted = _extract_user_query_from_meta(s)
            return extracted
        return s
    return ""


def _iter_file_objs(body: dict, messages: List[dict]) -> List[dict]:
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


def _pick_file_ref(body: dict, messages: List[dict]) -> Tuple[Optional[str], Optional[dict]]:
    for obj in reversed(_iter_file_objs(body, messages)):
        fid = obj.get("id") or obj.get("file_id")
        if fid:
            return fid, obj
    fid = (body or {}).get("file_id")
    if fid:
        return fid, {"id": fid}
    return None, None


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


def _strip_commit_flag(code: str) -> str:
    if not code:
        return code
    lines = []
    for line in code.splitlines():
        if re.match(r"^\s*COMMIT_DF\s*=\s*True\s*$", line):
            continue
        lines.append(line)
    return "\n".join(lines) + ("\n" if code.endswith("\n") else "")

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

def _is_count_intent(question: str) -> bool:
    q = (question or "").lower()
    return bool(re.search(r"\b(скільк\w*|кільк\w*|count|how\s+many|qty|quantity)\b", q))

def _is_sum_intent(question: str) -> bool:
    q = (question or "").lower()
    return bool(re.search(r"\b(sum|сума|total|загальн\w*|обсяг|volume|units|залишк\w*)\b", q))

def _has_df_assignment(code: str) -> bool:
    return bool(re.search(r"(^|\n)\s*df\s*=", code or "")) or bool(
        re.search(r"df\.(loc|iloc|at|iat)\[.+?\]\s*=", code or "")
    ) or bool(re.search(r"df\[[^\]]+\]\s*=", code or ""))

def _has_inplace_op(code: str) -> bool:
    return bool(re.search(r"inplace\s*=\s*True", code or ""))

def _has_df_changed_flag(code: str) -> bool:
    return bool(re.search(r"(^|\n)\s*df_changed\s*=\s*True\s*$", code or "", re.M))

def _infer_op_from_question(question: str) -> str:
    q = (question or "").lower()
    if re.search(
        r"\b(видал|додай|додати|встав|переймен|очист|заміни|замін|update|set|rename|delete|drop|insert|add|clear|fill|sort|фільтр|відфільтр|сортуй|переміст)\b",
        q,
    ):
        return "edit"
    return "read"

def _should_commit_from_code(code: str) -> bool:
    return _has_df_assignment(code) or _has_inplace_op(code) or _has_df_changed_flag(code)

def _finalize_code_for_sandbox(
    question: str, analysis_code: str, op: Optional[str] = None, commit_df: Optional[bool] = None
) -> Tuple[str, bool]:
    code = _normalize_generated_code(analysis_code or "")
    op_norm = (op or "").strip().lower()
    if op_norm not in ("read", "edit"):
        op_norm = _infer_op_from_question(question)

    if op_norm == "edit":
        code, edit_err = _validate_edit_code(code)
        if edit_err:
            logging.warning("event=edit_degraded reason=%s", edit_err)
            op_norm = "read"

    has_change = _should_commit_from_code(code)
    if commit_df is None:
        commit_requested = op_norm == "edit"
    else:
        commit_requested = bool(commit_df)

    if op_norm == "read":
        commit_requested = False

    if commit_requested and not has_change:
        logging.info("event=commit_stripped reason=no_df_change")
        commit_requested = False
        op_norm = "read"

    if commit_requested and "COMMIT_DF" not in code:
        code = code.rstrip() + "\nCOMMIT_DF = True\n"
    if not commit_requested:
        code = _strip_commit_flag(code)

    if op_norm == "read":
        if not re.match(r"^\s*df\s*=\s*df\.copy", code):
            code = "df = df.copy(deep=False)\n" + code

    return code, commit_requested

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

def _validate_edit_code(code: str) -> Tuple[str, Optional[str]]:
    if _has_df_assignment(code) or _has_inplace_op(code):
        return code, None
    repaired = _repair_edit_code(code)
    if _has_df_assignment(repaired) or _has_inplace_op(repaired):
        return repaired, None
    return code, "Edit code does not assign back to df and has no inplace/index assignment."

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


def _find_column_in_text(text: str, columns: List[str]) -> Optional[str]:
    if not text or not columns:
        return None
    lower = text.lower()

    col = _match_column_by_index(text, columns)
    if col:
        return col
    best = None
    for name in columns:
        if not isinstance(name, str):
            continue
        n = name.strip()
        if not n:
            continue
        if n.lower() in lower:
            if not best or len(n) > len(best):
                best = n
    if best:
        return best

    _STOP_ROOTS = {"кіль", "скіл", "count", "coun", "each", "per"}

    def _root_tokens(s: str) -> List[str]:
        parts = re.split(r"[^a-zа-яіїєґ0-9]+", (s or "").lower())
        roots = [p[:4] for p in parts if len(p) >= 3 and p[:4] not in _STOP_ROOTS]
        return roots

    q_roots = set(_root_tokens(lower))
    for name in columns:
        if not isinstance(name, str):
            continue
        roots = _root_tokens(name)
        if roots and all(r in q_roots for r in roots):
            return name
    return None


def _parse_row_index(text: str) -> Optional[int]:
    m = re.search(r"(?:рядок|рядка|row)[^\d]*(\d+)", text, re.I)
    if not m:
        return None
    idx = int(m.group(1))
    return idx if idx > 0 else None


def _parse_set_value(text: str) -> Optional[str]:
    m = re.search(r"(?:на|=)\s*([^\n]+)$", text, re.I)
    if not m:
        return None
    return m.group(1).strip().strip("\"'")


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
    if "де" not in text.lower():
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
    hits: List[Tuple[int, str]] = []
    for col in columns:
        if not isinstance(col, str):
            continue
        name = col.strip()
        if not name:
            continue
        m = re.search(re.escape(name), text, re.I)
        if m:
            hits.append((m.start(), col))
    hits.sort(key=lambda x: x[0])
    return [col for _, col in hits]


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


def _template_shortcut_code(question: str, profile: dict) -> Optional[Tuple[str, str]]:
    q = (question or "").strip()
    q_low = q.lower()
    if q.startswith("### Task:") and "<user_query>" not in q_low and "user query:" not in q_low:
        return None
    columns = (profile or {}).get("columns") or []
    if not columns:
        return None

    code_lines: List[str] = []
    plan = ""

    if re.search(r"\b(покажи|показати|show)\b.*\b(клітин|комір|cell)\b", q_low):
        row_idx = _parse_row_index(q)
        col = _find_column_in_text(q, columns)
        if row_idx and col:
            code_lines.append(f"result = df.at[{row_idx - 1}, {col!r}]")
            plan = f"Показати значення клітинки в рядку {row_idx}, колонці {col}."
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
        code_lines.append("df = df.rename(columns=lambda x: str(x).lower())")
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
        if len(nums) >= 2 and ("і" in q_low or "," in q or "та" in q_low):
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
                code_lines.append(f"_col = pd.to_numeric(df[{col!r}], errors='coerce')")
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
        code_lines.append("df[num_cols] = df[num_cols].apply(lambda s: s.fillna(s.mean()))")
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
            code_lines.append(f"df[{col!r}] = pd.to_numeric(df[{col!r}], errors='coerce')")
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
            code_lines.append(f"df[{col!r}] = pd.to_numeric(df[{col!r}], errors='coerce').round({n})")
            code_lines.append("COMMIT_DF = True")
            code_lines.append("result = {'status': 'updated'}")
            plan = f"Округлити {col} до {n} знаків."
            return "\n".join(code_lines) + "\n", plan

    if re.search(r"\b(обмеж|clip)\b.*\b(не менше|lower)\b", q_low):
        col = _find_column_in_text(q, columns)
        if col:
            code_lines.append("df = df.copy()")
            code_lines.append(f"df[{col!r}] = pd.to_numeric(df[{col!r}], errors='coerce').clip(lower=0)")
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
    q = (question or "").strip()
    q_low = q.lower()
    if q.startswith("### Task:") and "<user_query>" not in q_low and "user query:" not in q_low:
        return None
    if re.search(r"\b(undo|відміни|скасуй|відкот|поверни)\b", q_low):
        code = "\n".join(
            [
                "result = {'status': 'undo'}",
                "UNDO = True",
            ]
        ) + "\n"
        return code, "Відкотити останню зміну таблиці."

    columns = (profile or {}).get("columns") or []
    if not columns:
        return None

    is_add_row = bool(re.search(r"\b(додай|додати|add)\b.*\b(рядок|row)\b", q_low))
    is_del_row = bool(re.search(r"\b(видали|видалити|delete)\b.*\b(рядок|row)\b", q_low))
    is_add_col = bool(re.search(r"\b(додай|додати|add)\b.*\b(стовпец|стовпець|колонк|column)\b", q_low))
    is_del_col = bool(re.search(r"\b(видали|видалити|delete)\b.*\b(стовпец|стовпець|колонк|column)\b", q_low))
    is_edit_cell = bool(re.search(r"\b(зміни|змініть|редагуй|поміняй)\b.*\b(клітин|комір|cell)\b", q_low))

    code_lines: List[str] = []
    plan = ""
    if is_add_col:
        col_name = _find_column_in_text(q, columns) or "new_column"
        if col_name in columns:
            col_name = f"{col_name}_new"
        raw_value = _parse_set_value(q)
        value = _parse_literal(raw_value) if raw_value is not None else None
        code_lines.append("df = df.copy()")
        if value is None or value == "":
            code_lines.append(f"df[{col_name!r}] = None")
        else:
            code_lines.append(f"df[{col_name!r}] = {value!r}")
        plan = f"Додати стовпець {col_name}."
    elif is_del_col:
        col_name = _find_column_in_text(q, columns)
        if not col_name:
            return None
        code_lines.append("df = df.copy()")
        code_lines.append(f"df = df.drop(columns=[{col_name!r}])")
        plan = f"Видалити стовпець {col_name}."
    elif is_add_row:
        m = re.findall(r"([A-Za-zА-Яа-я0-9_ ]+?)\s*[:=]\s*([^,;]+)", q)
        row = {}
        for key, val in m:
            col = _find_column_in_text(key, columns)
            if col:
                row[col] = _parse_literal(val)
        code_lines.append("df = df.copy()")
        if row:
            code_lines.append(f"_row = {row!r}")
            code_lines.append("df = pd.concat([df, pd.DataFrame([_row])], ignore_index=True)")
        else:
            code_lines.append("df = pd.concat([df, pd.DataFrame([{}])], ignore_index=True)")
        plan = "Додати рядок."
    elif is_del_row:
        row_idx = _parse_row_index(q)
        cond = _parse_condition(q, columns)
        code_lines.append("df = df.copy()")
        code_lines.append("df = df.reset_index(drop=True)")
        if row_idx:
            code_lines.append(f"df = df.drop(index=[{row_idx - 1}])")
            plan = f"Видалити рядок {row_idx}."
        elif cond:
            col, op, value, num = cond
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
            plan = f"Видалити рядки за умовою по {col}."
        else:
            return None
    elif is_edit_cell:
        row_idx = _parse_row_index(q)
        col = _find_column_in_text(q, columns)
        raw_value = _parse_set_value(q)
        value = _parse_literal(raw_value) if raw_value is not None else None
        if not row_idx or not col or value is None:
            return None
        code_lines.append("df = df.copy()")
        code_lines.append("df = df.reset_index(drop=True)")
        code_lines.append(f"df.at[{row_idx - 1}, {col!r}] = {value!r}")
        plan = f"Змінити значення в рядку {row_idx}, колонці {col}."
    else:
        return None

    code_lines.append("COMMIT_DF = True")
    code_lines.append("result = {'status': 'updated'}")
    return "\n".join(code_lines) + "\n", plan


def _choose_column_from_question(question: str, profile: dict) -> Optional[str]:
    cols = (profile or {}).get("columns") or []
    if not cols:
        return None
    q = (question or "").lower()
    for col in cols:
        if isinstance(col, str) and col.lower() in q:
            return col
    price_markers = ("price", "цiн", "ціна", "uah", "грн")
    if any(marker in q for marker in price_markers):
        for col in cols:
            col_l = str(col).lower()
            if any(marker in col_l for marker in price_markers):
                return col
    dtypes = (profile or {}).get("dtypes") or {}
    for col in cols:
        dtype = str(dtypes.get(str(col), "")).lower()
        if dtype.startswith(("int", "float", "uint")):
            return col
    return cols[0]


def _stats_shortcut_code(question: str, profile: dict) -> Optional[Tuple[str, str]]:
    q = (question or "").lower()
    if (question or "").lstrip().startswith("### Task:") and "<user_query>" not in q and "user query:" not in q:
        return None
    wants_min = bool(re.search(r"\b(min|мінімал|мінімум)\b", q))
    wants_max = bool(re.search(r"\b(max|максимал|максимум)\b", q))
    wants_mean = bool(re.search(r"\b(mean|average|avg|середн)\b", q))
    wants_sum = bool(re.search(r"\b(sum|сума)\b", q))
    wants_count = bool(re.search(r"\b(count|кільк|кількість)\b", q))
    wants_median = bool(re.search(r"\b(median|медіан)\b", q))
    if not (wants_min or wants_max or wants_mean or wants_sum or wants_count or wants_median):
        return None
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

        webui_base_url: str = Field(default=os.getenv("WEBUI_BASE_URL", "http://host.docker.internal:3000"))
        webui_api_key: str = Field(default=os.getenv("WEBUI_API_KEY", ""))

        base_llm_base_url: str = Field(
            default=os.getenv("BASE_LLM_BASE_URL", "http://alph-gpu.silly.billy:8031/v1")
        )
        base_llm_api_key: str = Field(default=os.getenv("BASE_LLM_API_KEY", ""))
        base_llm_model: str = Field(default=os.getenv("BASE_LLM_MODEL", "chat-model"))

        sandbox_url: str = Field(default=os.getenv("SANDBOX_URL", "http://sandbox:8081"))
        sandbox_api_key: str = Field(default=os.getenv("SANDBOX_API_KEY", ""))

        max_rows: int = Field(default=_env_int("PIPELINE_MAX_ROWS", 200000), ge=1)
        preview_rows: int = Field(default=_env_int("PIPELINE_PREVIEW_ROWS", 200000), ge=1)
        max_cell_chars: int = Field(default=_env_int("PIPELINE_MAX_CELL_CHARS", 200), ge=10)
        code_timeout_s: int = Field(default=_env_int("PIPELINE_CODE_TIMEOUT_S", 120), ge=1)
        max_stdout_chars: int = Field(default=_env_int("PIPELINE_MAX_STDOUT_CHARS", 8000), ge=1000)

        session_cache_ttl_s: int = Field(default=_env_int("PIPELINE_SESSION_CACHE_TTL_S", 1800), ge=60)
        wait_tick_s: int = Field(default=_env_int("PIPELINE_WAIT_TICK_S", 5), ge=0)

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
        self._session_cache: Dict[str, Dict[str, Any]] = {}
        self._prompts = _read_prompts(PROMPTS_PATH)
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
        self._session_cache[key] = {
            "file_id": file_id,
            "df_id": df_id,
            "profile": profile,
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

    def _llm_json(self, system: str, user: str) -> dict:
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
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise ValueError("LLM did not return JSON")
        return json.loads(m.group(0))

    def _plan_code(self, question: str, profile: dict) -> Tuple[str, str, str, Optional[bool]]:
        system = self._prompts.get("plan_code_system", DEFAULT_PLAN_CODE_SYSTEM)
        payload = json.dumps({"question": question, "df_profile": profile}, ensure_ascii=False)
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

    def _resolve_shortcut_placeholders(
        self, analysis_code: str, plan: str, question: str, profile: dict
    ) -> Tuple[str, str]:
        if SHORTCUT_COL_PLACEHOLDER not in (analysis_code or "") and SHORTCUT_COL_PLACEHOLDER not in (plan or ""):
            return analysis_code, plan
        columns = (profile or {}).get("columns") or []
        col = _find_column_in_text(question, columns)
        if not col:
            col = self._llm_pick_column_for_shortcut(question, profile)
        if not col:
            return analysis_code, plan
        return (analysis_code or "").replace(SHORTCUT_COL_PLACEHOLDER, col), (plan or "").replace(
            SHORTCUT_COL_PLACEHOLDER, col
        )

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

    def _deterministic_answer(self, question: str, result_text: str, profile: Optional[dict]) -> Optional[str]:
        q = (question or "").lower()
        if re.search(r"\b(рядк\w*|стовпц\w*|columns?|rows?)\b", q):
            structural = self._format_structure_from_result(result_text)
            if structural:
                return structural
        wants_availability = bool(
            re.search(r"\b(наявн|в\s+наявності|доступн|availability|available|in\s+stock|out\s+of\s+stock)\b", q)
        )
        wants_pairs = bool(re.search(r"\b(кожн\w*|each|per|по\s+кожн\w*|для\s+кожн\w*)\b", q))
        wants_counts = bool(re.search(r"\b(скільк\w*|кільк\w*|count|к-?ст[ьі])\b", q))
        is_grouping = bool(
            re.search(
                r"\b(бренд\w*|категорі\w*|brand|category|"
                r"груп\w*|розбив\w*|розподіл\w*|структур\w*|"
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
        if is_grouping and (wants_counts or wants_pairs):
            return self._format_top_pairs_from_result(result_text, top_n=100)
        if not (wants_pairs and wants_counts):
            return None
        return self._format_top_pairs_from_result(result_text, top_n=15)

    def _final_answer(
        self,
        question: str,
        profile: dict,
        plan: str,
        code: str,
        run_status: str,
        stdout: str,
        result_text: str,
        result_meta: dict,
        error: str,
    ) -> str:
        deterministic = self._deterministic_answer(question, result_text, profile)
        if deterministic:
            logging.info("event=final_answer mode=deterministic preview=%s", _safe_trunc(deterministic, 300))
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
                            return rewrite
                except Exception as exc:
                    logging.warning("event=final_rewrite_failed err=%s", str(exc))
                return "Не можу безпечно сформувати відповідь. Спробуйте уточнити запит."
        logging.info("event=final_answer mode=llm preview=%s", _safe_trunc(answer, 300))
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
        try:
            self._emit(event_emitter, "start", {"model_id": model_id, "has_messages": bool(messages)})

            question = (user_message or "").strip() or _last_user_message(messages)
            if _is_meta_task_text(question):
                extracted = _extract_user_query_from_meta(question)
                if extracted:
                    question = extracted
                else:
                    question = _last_user_message(messages) or question
            session_key = _session_key(body)
            session = self._session_get(session_key)

            file_id, file_obj = _pick_file_ref(body, messages)
            if not file_id and session:
                file_id = session.get("file_id")
                file_obj = None

            self._emit(event_emitter, "file_id", {"file_id": file_id})
            if not file_id:
                self._emit(event_emitter, "no_file", {"body_keys": list((body or {}).keys())})
                self._debug_body(body, messages)
                return "Attach a CSV/XLSX file and ask a question."

            if session and session.get("file_id") == file_id and session.get("df_id"):
                df_id = session["df_id"]
                cached_profile = session.get("profile") or {}
                fresh_profile = self._sandbox_get_profile(df_id)
                if fresh_profile:
                    profile = fresh_profile
                    if fresh_profile != cached_profile:
                        self._session_set(session_key, file_id, df_id, fresh_profile)
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
                    return "Failed to load the table in the sandbox."
                self._apply_dynamic_limits(profile)
                self._session_set(session_key, file_id, df_id, profile)

            edit_expected = False
            op = None
            commit_df = None
            router_hit = self._shortcut_router.shortcut_to_sandbox_code(question, profile)
            shortcut = None
            router_meta: Dict[str, Any] = {}
            if router_hit:
                analysis_code, router_meta = router_hit
                plan = f"retrieval_intent:{router_meta.get('intent_id')}"
                logging.info("event=shortcut_router status=ok meta=%s", _safe_trunc(router_meta, 800))
            else:
                logging.info("event=shortcut_router status=miss question=%s", _safe_trunc(question, 200))
                shortcut = _template_shortcut_code(question, profile)
                if not shortcut:
                    shortcut = _edit_shortcut_code(question, profile)
                if not shortcut:
                    shortcut = _stats_shortcut_code(question, profile)

            self._emit(event_emitter, "codegen", {"question": _safe_trunc(question, 200)})
            if router_hit:
                logging.info("event=shortcut status=ok meta=%s", _safe_trunc(router_meta, 800))
                self._emit(
                    event_emitter,
                    "codegen_shortcut",
                    {"question": _safe_trunc(question, 200), "intent_id": router_meta.get("intent_id")},
                )
            elif shortcut:
                self._emit(event_emitter, "codegen_shortcut", {"question": _safe_trunc(question, 200)})
                analysis_code, plan = shortcut
            else:
                wait = self._start_wait(event_emitter, "codegen")
                try:
                    analysis_code, plan, op, commit_df = self._plan_code(question, profile)
                finally:
                    self._stop_wait(wait)
                if not (analysis_code or "").strip():
                    self._emit(event_emitter, "codegen_empty", {})
                    return "I could not generate analysis code for this question. Try rephrasing."
            analysis_code, count_err = _enforce_count_code(question, analysis_code)
            if count_err:
                return f"Неможливо виконати запит: {count_err}"
            analysis_code, edit_expected = _finalize_code_for_sandbox(question, analysis_code, op, commit_df)

            analysis_code, plan = self._resolve_shortcut_placeholders(analysis_code, plan, question, profile)
            analysis_code = textwrap.dedent(analysis_code or "").strip() + "\n"
            if "df_profile" in (analysis_code or ""):
                analysis_code = f"df_profile = {profile!r}\n" + analysis_code
            analysis_code = _normalize_generated_code(analysis_code)
            logging.info("event=analysis_code preview=%s", _safe_trunc(analysis_code, 4000))

            self._emit(event_emitter, "sandbox_run", {"df_id": df_id})
            wait = self._start_wait(event_emitter, "sandbox_run")
            try:
                run_resp = self._sandbox_run(df_id, analysis_code)
            finally:
                self._stop_wait(wait)
            run_status = run_resp.get("status", "")
            if run_status != "ok":
                self._emit(event_emitter, "final_answer", {"status": run_status})
                error = run_resp.get("error", "") or "Sandbox execution failed."
                return f"Не вдалося виконати аналіз у sandbox (статус: {run_status}). {error}"
            if edit_expected and ("COMMIT_DF" in (analysis_code or "")) and not run_resp.get("committed"):
                self._emit(event_emitter, "final_answer", {"status": "commit_failed"})
                return "Edit не було закомічено в sandbox. Повторіть запит із явним COMMIT_DF = True та присвоєнням у df."
            new_profile = run_resp.get("profile") or {}
            if new_profile:
                profile = new_profile
                self._session_set(session_key, file_id, df_id, profile)
            self._emit(event_emitter, "final_answer", {"status": run_status})
            wait = self._start_wait(event_emitter, "final_answer")
            try:
                return self._final_answer(
                    question=question,
                    profile=profile,
                    plan=plan,
                    code=analysis_code,
                    run_status=run_status,
                    stdout=run_resp.get("stdout", ""),
                    result_text=run_resp.get("result_text", ""),
                    result_meta=run_resp.get("result_meta", {}) or {},
                    error=run_resp.get("error", ""),
                )
            finally:
                self._stop_wait(wait)
        except Exception as exc:
            tb = _safe_trunc(traceback.format_exc(), 2000)
            try:
                self._emit(event_emitter, "error", {"error": str(exc)})
            except Exception:
                pass
            return f"Pipeline error: {type(exc).__name__}: {exc}\n\n{tb}"

    def _pipe_stream(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Generator[str, None, None]:
        status_queue: List[str] = []

        def emit(event: str, payload: Dict[str, Any]) -> None:
            self._emit(None, event, payload, status_queue=status_queue)

        def drain() -> Iterator[str]:
            while status_queue:
                yield status_queue.pop(0)

        try:
            emit("start", {"model_id": model_id, "has_messages": bool(messages)})
            yield from drain()

            question = (user_message or "").strip() or _last_user_message(messages)
            if _is_meta_task_text(question):
                extracted = _extract_user_query_from_meta(question)
                if extracted:
                    question = extracted
                else:
                    question = _last_user_message(messages) or question
            session_key = _session_key(body)
            session = self._session_get(session_key)

            file_id, file_obj = _pick_file_ref(body, messages)
            if not file_id and session:
                file_id = session.get("file_id")
                file_obj = None

            emit("file_id", {"file_id": file_id})
            yield from drain()
            if not file_id:
                emit("no_file", {"body_keys": list((body or {}).keys())})
                yield from drain()
                self._debug_body(body, messages)
                yield "Attach a CSV/XLSX file and ask a question."
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
                    yield "Failed to load the table in the sandbox."
                    return
                self._apply_dynamic_limits(profile)
                self._session_set(session_key, file_id, df_id, profile)

            edit_expected = False
            op = None
            commit_df = None
            router_hit = self._shortcut_router.shortcut_to_sandbox_code(question, profile)
            shortcut = None
            router_meta: Dict[str, Any] = {}
            if router_hit:
                analysis_code, router_meta = router_hit
                plan = f"retrieval_intent:{router_meta.get('intent_id')}"
                logging.info("event=shortcut_router status=ok meta=%s", _safe_trunc(router_meta, 800))
            else:
                logging.info("event=shortcut_router status=miss question=%s", _safe_trunc(question, 200))
                shortcut = _template_shortcut_code(question, profile)
                if not shortcut:
                    shortcut = _edit_shortcut_code(question, profile)
                if not shortcut:
                    shortcut = _stats_shortcut_code(question, profile)

            emit("codegen", {"question": _safe_trunc(question, 200)})
            yield from drain()
            if router_hit:
                logging.info("event=shortcut status=ok meta=%s", _safe_trunc(router_meta, 800))
                emit(
                    "codegen_shortcut",
                    {"question": _safe_trunc(question, 200), "intent_id": router_meta.get("intent_id")},
                )
                yield from drain()
            elif shortcut:
                emit("codegen_shortcut", {"question": _safe_trunc(question, 200)})
                yield from drain()
                analysis_code, plan = shortcut
            else:
                analysis_code, plan, op, commit_df = self._plan_code(question, profile)
                if not (analysis_code or "").strip():
                    emit("codegen_empty", {})
                    yield from drain()
                    yield "I could not generate analysis code for this question. Try rephrasing."
                    return

            analysis_code, count_err = _enforce_count_code(question, analysis_code)
            if count_err:
                emit("final_answer", {"status": "invalid_code"})
                yield from drain()
                yield f"Неможливо виконати запит: {count_err}"
                return
            analysis_code, edit_expected = _finalize_code_for_sandbox(question, analysis_code, op, commit_df)

            analysis_code, plan = self._resolve_shortcut_placeholders(analysis_code, plan, question, profile)
            analysis_code = textwrap.dedent(analysis_code or "").strip() + "\n"
            if "df_profile" in (analysis_code or ""):
                analysis_code = f"df_profile = {profile!r}\n" + analysis_code
            analysis_code = _normalize_generated_code(analysis_code)
            logging.info("event=analysis_code preview=%s", _safe_trunc(analysis_code, 4000))

            emit("sandbox_run", {"df_id": df_id})
            yield from drain()
            run_resp = self._sandbox_run(df_id, analysis_code)
            run_status = run_resp.get("status", "")
            if run_status != "ok":
                emit("final_answer", {"status": run_status})
                yield from drain()
                error = run_resp.get("error", "") or "Sandbox execution failed."
                yield f"Не вдалося виконати аналіз у sandbox (статус: {run_status}). {error}"
                return
            if edit_expected and ("COMMIT_DF" in (analysis_code or "")) and not run_resp.get("committed"):
                emit("final_answer", {"status": "commit_failed"})
                yield from drain()
                yield "Edit не було закомічено в sandbox. Повторіть запит із явним COMMIT_DF = True та присвоєнням у df."
                return
            new_profile = run_resp.get("profile") or {}
            if new_profile:
                profile = new_profile
                self._session_set(session_key, file_id, df_id, profile)
            emit("final_answer", {"status": run_status})
            yield from drain()
            yield self._final_answer(
                question=question,
                profile=profile,
                plan=plan,
                code=analysis_code,
                run_status=run_status,
                stdout=run_resp.get("stdout", ""),
                result_text=run_resp.get("result_text", ""),
                result_meta=run_resp.get("result_meta", {}) or {},
                error=run_resp.get("error", ""),
            )
        except Exception as exc:
            tb = _safe_trunc(traceback.format_exc(), 2000)
            emit("error", {"error": str(exc)})
            yield from drain()
            yield f"Pipeline error: {type(exc).__name__}: {exc}\n\n{tb}"


pipeline = Pipeline()
