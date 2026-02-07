"""
title: Send File + Query to External API (Pipe)
author: you
version: 0.1.1
license: MIT
description: Downloads uploaded file bytes from OpenWebUI and sends them + user query to an external API via multipart/form-data.
requirements: httpx,pydantic
"""
import asyncio
import base64
import contextlib
import inspect
import json
import logging
import os
import re
import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_META_TASK_HINTS = (
    "<user_query>",
    "<chat_history>",
    "respond to the user query using the provided context",
    "analyze the chat history to determine the necessity of generating search queries",
    "prioritize generating 1-3 broad and relevant search queries",
)

_STATUS_TEXT_HINTS = (
    "Стартую обробку запиту.",
    "Завантажую файл:",
    "Файл отримано:",
    "Надсилаю запит у сервіс pipelines.",
    "Отримую відповідь зі стріму.",
    "Стрім завершено.",
    "Отримано відповідь від сервісу pipelines",
    "Помилка сервісу pipelines",
    "Надсилаю запит у резервний API.",
    "Отримано відповідь від резервного API",
    "Очікую відповідь",
    "Готово. Відповідь сформована",
    "Генерую план та код аналізу.",
    "Використовую швидкий режим для генерації коду.",
    "Виконую аналіз у sandbox.",
    "Отримую метадані файлу.",
    "Завантажую таблицю в sandbox.",
)


def _env_int(name, default):
    val = os.getenv(name)
    if val is None or not str(val).strip():
        return default
    try:
        return int(val)
    except Exception:
        return default


def _env_bool(name, default):
    val = os.getenv(name)
    if val is None or not str(val).strip():
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")


def _is_system_task_text(text):
    t = (text or "").strip()
    if not t:
        return False
    low = t.lower()
    if t.startswith("### Task:"):
        return True
    return any(h in low for h in _META_TASK_HINTS)


def _normalize_query_text(text):
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not s.strip():
        return ""
    lines = []
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


def _message_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
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


def _last_user_text(messages):
    for m in reversed(messages or []):
        if m.get("role") != "user":
            continue
        raw = _message_text(m.get("content", ""))
        if not isinstance(raw, str):
            continue
        t = raw.strip()
        if not t:
            continue
        if _is_system_task_text(t):
            extracted = _extract_user_query(t)
            if extracted:
                return _normalize_query_text(extracted)
            return ""
        return _normalize_query_text(t)
    return ""


def _extract_user_query(text):
    t = (text or "").strip()
    if not t:
        return ""
    if not _is_system_task_text(t):
        return _normalize_query_text(t)

    m = re.search(r"<user_query>\s*(.+?)\s*</user_query>", t, re.S | re.I)
    if m:
        return _normalize_query_text(m.group(1))
    m = re.search(r"###\s*User Query:\s*(.+?)(?:\n###|$)", t, re.S | re.I)
    if m:
        return _normalize_query_text(m.group(1))
    m = re.search(r"###\s*User Query\s*\n+\s*(.+?)(?:\n###|\n<|$)", t, re.S | re.I)
    if m:
        return _normalize_query_text(m.group(1))
    m = re.search(r"\bUser Query:\s*(.+?)(?:\n###|$)", t, re.S | re.I)
    if m:
        return _normalize_query_text(m.group(1))
    m = re.search(r"\bЗапит\s+користувача:\s*(.+?)(?:\n###|$)", t, re.S | re.I)
    if m:
        return _normalize_query_text(m.group(1))
    m = re.search(r"(?:^|\n)\s*(?:user|користувач)\s*[:>-]\s*(.+)", t, re.I)
    if m:
        return _normalize_query_text(m.group(1))
    m = re.search(r"(?:\"|')?user_query(?:\"|')?\s*[:=]\s*['\"](.+?)['\"]", t, re.S | re.I)
    if m:
        return _normalize_query_text(m.group(1))

    chat_block = re.search(r"<chat_history>\s*(.+?)\s*</chat_history>", t, re.S | re.I)
    if chat_block:
        chat = (chat_block.group(1) or "").strip()
        user_lines = re.findall(r"(?:^|\n)\s*(?:user|користувач)\s*[:>-]\s*(.+)", chat, re.I)
        if user_lines:
            return _normalize_query_text(user_lines[-1])

    return ""


def _safe_trunc(text, limit):
    s = str(text)
    if len(s) <= limit:
        return s
    return s[:limit] + "...(truncated)"


_STATUS_MARKER_PREFIX = "[[PIPELINE_STATUS:"
_STATUS_MARKER_SUFFIX = "]]"


def _extract_status_markers(buffer):
    out = []
    markers = []
    while True:
        start = buffer.find(_STATUS_MARKER_PREFIX)
        if start < 0:
            out.append(buffer)
            buffer = ""
            break
        out.append(buffer[:start])
        end = buffer.find(_STATUS_MARKER_SUFFIX, start + len(_STATUS_MARKER_PREFIX))
        if end < 0:
            buffer = buffer[start:]
            break
        payload = buffer[start + len(_STATUS_MARKER_PREFIX):end]
        markers.append(payload)
        buffer = buffer[end + len(_STATUS_MARKER_SUFFIX):]
    return ("".join(out), buffer, markers)


def _status_message(event, payload):
    if event == "start":
        return "Стартую обробку запиту."
    if event == "file_fetch_start":
        name = (payload or {}).get("filename") or (payload or {}).get("file_id") or "файл"
        return f"Завантажую файл: {name}"
    if event == "file_fetch_done":
        name = (payload or {}).get("filename") or (payload or {}).get("file_id") or "файл"
        size = (payload or {}).get("size_bytes")
        if size is not None:
            return f"Файл отримано: {name} ({size} байт)"
        return f"Файл отримано: {name}"
    if event == "pipeline_request":
        return "Надсилаю запит у сервіс pipelines."
    if event == "stream_start":
        return "Отримую відповідь зі стріму."
    if event == "stream_done":
        tokens = (payload or {}).get("tokens")
        if tokens is not None:
            return f"Стрім завершено. Отримано ~{tokens} токенів."
        return "Стрім завершено."
    if event == "stream_progress":
        tokens = (payload or {}).get("tokens")
        if tokens is not None:
            return f"Отримано ~{tokens} токенів."
        return "Отримую відповідь зі стріму."
    if event == "pipeline_response":
        status = (payload or {}).get("status")
        if status:
            return f"Отримано відповідь від сервісу pipelines (HTTP {status})."
        return "Отримано відповідь від сервісу pipelines."
    if event == "pipeline_error":
        err = (payload or {}).get("error")
        if err:
            return f"Помилка сервісу pipelines, переходжу на резервний API: {err}"
        return "Помилка сервісу pipelines, переходжу на резервний API."
    if event == "target_request":
        return "Надсилаю запит у резервний API."
    if event == "target_response":
        status = (payload or {}).get("status")
        if status:
            return f"Отримано відповідь від резервного API (HTTP {status})."
        return "Отримано відповідь від резервного API."
    if event == "wait":
        label = (payload or {}).get("label") or ""
        seconds = (payload or {}).get("seconds")
        if label == "pipeline":
            base = "Очікую відповідь від сервісу pipelines."
        elif label == "target":
            base = "Очікую відповідь від резервного API."
        else:
            base = "Очікую відповідь."
        if isinstance(seconds, int) and seconds > 0:
            return f"{base} Минуло ~{seconds}с."
        return base
    if event == "error":
        err = (payload or {}).get("error")
        if err:
            return f"Помилка: {err}"
    return _safe_trunc(f"{event}: {payload}", 200)


def _emit_status(event_emitter, description, done=False, hidden=False):
    if not event_emitter:
        return
    payload = {"type": "status", "data": {"description": description, "done": done, "hidden": hidden}}
    try:
        if callable(event_emitter):
            res = event_emitter(payload)
            if inspect.isawaitable(res):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(res)
                else:
                    loop.create_task(res)
            return
        if hasattr(event_emitter, "emit"):
            event_emitter.emit(
                "status",
                {"description": description, "message": description, "done": done, "hidden": hidden},
            )
    except Exception:
        pass


def _emit(event_emitter, event, payload):
    msg = _status_message(event, payload)
    done = event in ("pipeline_response", "target_response", "error")
    _emit_status(event_emitter, msg, done=done)


def _emit_message(event_emitter, content):
    text = (content or "").strip()
    if not event_emitter or not text:
        return
    payload = {"type": "message", "data": {"content": text}}
    try:
        if callable(event_emitter):
            res = event_emitter(payload)
            if inspect.isawaitable(res):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(res)
                else:
                    loop.create_task(res)
            return
        if hasattr(event_emitter, "emit"):
            event_emitter.emit("message", {"content": text})
    except Exception:
        pass


def _emit_replace(event_emitter, content):
    text = (content or "").strip()
    if not event_emitter or not text:
        return
    payload = {"type": "replace", "data": {"content": text}}
    try:
        if callable(event_emitter):
            res = event_emitter(payload)
            if inspect.isawaitable(res):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(res)
                else:
                    loop.create_task(res)
            return
        if hasattr(event_emitter, "emit"):
            event_emitter.emit("replace", {"content": text})
    except Exception:
        pass


def _publish_answer(event_emitter, content, emit_message_event=False):
    text = (content or "").strip()
    if not text:
        return ""
    if emit_message_event:
        _emit_message(event_emitter, text)
    return text


def _choice_delta_text(choice):
    delta = choice.get("delta") or {}
    if not isinstance(delta, dict):
        return ""
    text = _message_text(delta.get("content"))
    if text:
        return text
    alt = delta.get("text")
    return alt.strip() if isinstance(alt, str) else ""


def _choice_message_text(choice):
    message = choice.get("message") or {}
    if isinstance(message, dict):
        text = _message_text(message.get("content"))
        if text:
            return text
        alt = message.get("text")
        if isinstance(alt, str) and alt.strip():
            return alt.strip()
    direct = choice.get("text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    return ""


def _is_progress_description(text):
    s = (text or "").strip()
    if not s:
        return False
    if s in {"Querying", "Retrieved 1 source", "Retrieved 2 sources"}:
        return True
    return any(s.startswith(hint) for hint in _STATUS_TEXT_HINTS)


def _extract_status_from_payload(payload):
    if not isinstance(payload, dict):
        return ""
    typ = str(payload.get("type") or "").strip().lower()
    data = payload.get("data")
    if typ == "status" and isinstance(data, dict):
        desc = data.get("description") or data.get("message")
        return desc.strip() if isinstance(desc, str) and desc.strip() else ""
    event = str(payload.get("event") or "").strip().lower()
    if event == "status":
        if isinstance(data, dict):
            desc = data.get("description") or data.get("message")
            if isinstance(desc, str) and desc.strip():
                return desc.strip()
        desc = payload.get("description") or payload.get("message")
        return desc.strip() if isinstance(desc, str) and desc.strip() else ""
    return ""


def _extract_text_from_payload(payload):
    if not isinstance(payload, dict):
        return ""
    if _extract_status_from_payload(payload):
        # Status payloads are progress updates, not assistant final text.
        return ""
    # OpenAI-like shape
    choices = payload.get("choices") or []
    if isinstance(choices, list) and choices:
        choice = choices[0] or {}
        text = _choice_delta_text(choice)
        if text:
            return text
        text = _choice_message_text(choice)
        if text:
            return text
    # Generic gateway shapes
    text = _message_text(payload.get("content"))
    if text:
        return text
    message = payload.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    if isinstance(message, dict):
        text = _message_text(message.get("content"))
        if text:
            return text
        alt = message.get("text")
        if isinstance(alt, str) and alt.strip():
            return alt.strip()
    direct_text = payload.get("text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()
    data = payload.get("data")
    if isinstance(data, dict):
        nested_choices = data.get("choices") or []
        if isinstance(nested_choices, list) and nested_choices:
            choice = nested_choices[0] or {}
            text = _choice_delta_text(choice)
            if text:
                return text
            text = _choice_message_text(choice)
            if text:
                return text
        text = _message_text(data.get("content"))
        if text:
            return text
        nested_message = data.get("message")
        if isinstance(nested_message, dict):
            text = _message_text(nested_message.get("content"))
            if text:
                return text
            alt = nested_message.get("text")
            if isinstance(alt, str) and alt.strip():
                return alt.strip()
        elif isinstance(nested_message, str) and nested_message.strip():
            return nested_message.strip()
        alt = data.get("text") or data.get("message") or data.get("response") or data.get("answer")
        if isinstance(alt, str) and alt.strip():
            return alt.strip()
    return ""


def _looks_like_status_only_text(text):
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return False
    for line in lines:
        if line in {"Querying", "Retrieved 1 source", "Retrieved 2 sources"}:
            continue
        if any(line.startswith(hint) for hint in _STATUS_TEXT_HINTS):
            continue
        return False
    return True


def _start_wait_ticker(event_emitter, label, interval_s):
    if not event_emitter or interval_s <= 0:
        return None

    async def _ticker():
        elapsed = 0
        while True:
            await asyncio.sleep(interval_s)
            elapsed += interval_s
            _emit(event_emitter, "wait", {"label": label, "seconds": elapsed})

    return asyncio.create_task(_ticker())


async def _stop_wait_ticker(task):
    if not task:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def _extract_files(body, extra_files=None):
    out = []
    body = body or {}
    for f in (extra_files or []):
        if isinstance(f, dict):
            fid = f.get("id") or f.get("file_id") or (f.get("file") or {}).get("id")
            if fid:
                out.append(f)
    for k in ("files", "attachments"):
        for f in body.get(k) or []:
            if isinstance(f, dict) and (f.get("id") or f.get("file_id")):
                out.append(f)
    for m in body.get("messages") or []:
        for k in ("files", "attachments"):
            for f in m.get(k) or []:
                if isinstance(f, dict) and (f.get("id") or f.get("file_id")):
                    out.append(f)
        c = m.get("content")
        if isinstance(c, list):
            for x in c:
                if isinstance(x, dict) and (x.get("type") in ("file", "input_file") or x.get("kind") == "file"):
                    if x.get("id") or x.get("file_id"):
                        out.append(x)
    fid = body.get("file_id")
    if fid:
        out.append({"id": fid})
    seen = set()
    uniq = []
    for f in out:
        k = f.get("id") or f.get("file_id") or (f.get("file") or {}).get("id")
        if k in seen:
            continue
        seen.add(k)
        uniq.append(f)
    return uniq


def _pick_token(__user__, valves):
    for k in ("token", "api_key", "access_token", "jwt"):
        v = (__user__ or {}).get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return (valves.WEBUI_API_KEY or "").strip()


def _auth_headers_from_request(__request__):
    if not __request__:
        return {}
    try:
        auth = __request__.headers.get("authorization") or __request__.headers.get("Authorization")
        if auth:
            return {"Authorization": auth}
    except Exception:
        pass
    return {}


def _pipelines_headers(valves):
    h = {}
    key = (valves.PIPELINES_API_KEY or "").strip()
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def _guess_mime(name):
    n = (name or "").lower()
    if n.endswith(".csv"):
        return "text/csv"
    if n.endswith(".tsv"):
        return "text/tab-separated-values"
    if n.endswith(".xlsx"):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if n.endswith(".xls"):
        return "application/vnd.ms-excel"
    return "application/octet-stream"


def _name_from_cd(cd):
    if not cd:
        return ""
    m = re.search(r'filename\*?=(?:UTF-8\'\')?("?)([^";]+)\1', cd, re.I)
    return m.group(2) if m else ""


class Pipe:
    class Valves(BaseModel):
        WEBUI_BASE_URL: str = Field(default=os.getenv("WEBUI_BASE_URL", "http://openwebui:8080"))
        WEBUI_API_KEY: str = Field(default=os.getenv("WEBUI_API_KEY", ""))
        PIPELINES_BASE_URL: str = Field(default=os.getenv("PIPELINES_BASE_URL", "http://pipelines:9099/v1"))
        PIPELINES_API_KEY: str = Field(default=os.getenv("PIPELINES_API_KEY", "0p3n-w3bu!"))
        PIPELINES_MODEL: str = Field(default=os.getenv("PIPELINES_MODEL", "spreadsheet_analyst_pipeline"))
        TARGET_API_URL: str = Field(default=os.getenv("TARGET_API_URL", "http://host.docker.internal:8088/analyze"))
        TARGET_API_KEY: str = Field(default=os.getenv("TARGET_API_KEY", ""))
        FILE_FIELD: str = Field(default=os.getenv("FILE_FIELD", "file"))
        QUERY_FIELD: str = Field(default=os.getenv("QUERY_FIELD", "query"))
        TIMEOUT_S: int = Field(default=_env_int("TIMEOUT_S", 120))
        PICK_FIRST_FILE: bool = Field(default=_env_bool("PICK_FIRST_FILE", True))
        WAIT_TICK_S: int = Field(default=_env_int("WAIT_TICK_S", 5))
        PIPE_SUPPRESS_PIPE_STATUS: bool = Field(default=_env_bool("PIPE_SUPPRESS_PIPE_STATUS", True))

    def __init__(self):
        self.valves = self.Valves()

    async def pipe(self, body: dict, __user__: dict = None, __files__: list = None, __request__=None,
                   __event_emitter__=None):
        event_emitter = __event_emitter__ or (body or {}).get("event_emitter") or (body or {}).get("__event_emitter__")
        suppress_pipe_status = bool(event_emitter) and self.valves.PIPE_SUPPRESS_PIPE_STATUS

        def emit_pipe(event, payload, force=False):
            if suppress_pipe_status and not force and event in (
            "start", "pipeline_request", "stream_start", "stream_progress", "stream_done", "pipeline_response", "wait"):
                return
            _emit(event_emitter, event, payload)

        def emit_marker_status(description, done=False, hidden=False):
            if suppress_pipe_status:
                return
            if _is_progress_description(description):
                _emit_status(event_emitter, description, done=done, hidden=hidden)

        explicit_query = _extract_user_query((body or {}).get("user_message") or "")
        raw_query = explicit_query or _last_user_text((body or {}).get("messages") or [])
        query = _extract_user_query(raw_query)
        logger.info(
            "query_selection raw_preview=%s selected_preview=%s selected_empty=%s",
            _safe_trunc(raw_query, 200),
            _safe_trunc(query, 200),
            not bool(query),
        )
        files = _extract_files(body or {}, __files__ or [])
        emit_pipe("start", {"has_query": bool(query), "files_count": len(files)})
        if not query:
            logger.info("skip_meta_task reason=no_user_query_found")
            return None

        webui = (self.valves.WEBUI_BASE_URL or "").rstrip("/")
        if not webui:
            return "WEBUI_BASE_URL порожній."
        token = _pick_token(__user__ or {}, self.valves)
        auth_headers = _auth_headers_from_request(__request__)

        async with httpx.AsyncClient(timeout=self.valves.TIMEOUT_S) as client:
            file_payloads = []
            fallback_files = []
            if files:
                if self.valves.PICK_FIRST_FILE:
                    files = files[:1]
                for f in files:
                    fid = f.get("id") or f.get("file_id") or (f.get("file") or {}).get("id")
                    fname = f.get("name") or f.get("filename") or (f.get("file") or {}).get("name") or fid
                    emit_pipe("file_fetch_start", {"file_id": fid, "filename": fname}, force=True)
                    h = dict(auth_headers)
                    if not h and token:
                        h["Authorization"] = f"Bearer {token}"
                    r = await client.get(f"{webui}/api/v1/files/{fid}/content", headers=h)
                    r.raise_for_status()
                    raw = r.content
                    cd_name = _name_from_cd(r.headers.get("content-disposition", ""))
                    if cd_name:
                        fname = cd_name
                    emit_pipe("file_fetch_done", {"file_id": fid, "filename": fname, "size_bytes": len(raw)},
                              force=True)
                    file_payloads.append({
                        "id": fid,
                        "filename": fname,
                        "content_type": _guess_mime(fname),
                        "data_b64": base64.b64encode(raw).decode("ascii"),
                    })
                    fallback_files.append((fname, raw, _guess_mime(fname)))

            payload = dict(body or {})
            if query:
                payload["user_message"] = query
            if not isinstance(payload.get("messages"), list) or not payload.get("messages") or query != raw_query:
                payload["messages"] = [{"role": "user", "content": query or ""}]
            payload["files"] = file_payloads
            payload["stream"] = True
            payload["model"] = self.valves.PIPELINES_MODEL

            url = f"{self.valves.PIPELINES_BASE_URL.rstrip('/')}/chat/completions"
            try:
                emit_pipe("pipeline_request",
                          {"url": url, "model": payload.get("model"), "files": len(file_payloads)})
                wait_emitter = None if suppress_pipe_status else event_emitter
                wait_task = _start_wait_ticker(wait_emitter, "pipeline", self.valves.WAIT_TICK_S)
                try:
                    async with client.stream("POST", url, headers=_pipelines_headers(self.valves), json=payload) as resp:
                        resp.raise_for_status()
                        content_type = resp.headers.get("content-type", "")
                        last_status_desc = ""
                        if "text/event-stream" not in content_type:
                            data = await resp.json()
                            logger.info("pipelines_response status=%s body=%s", resp.status_code,
                                        _safe_trunc(data, 800))
                            emit_pipe("pipeline_response", {"status": resp.status_code})
                            try:
                                content = (data["choices"][0]["message"]["content"] or "")
                                clean, _, markers = _extract_status_markers(content)
                                for raw in markers:
                                    try:
                                        payload = json.loads(raw)
                                    except Exception:
                                        continue
                                    desc = payload.get("description")
                                    done = bool(payload.get("done"))
                                    hidden = bool(payload.get("hidden"))
                                    if isinstance(desc, str) and desc.strip():
                                        last_status_desc = desc.strip()
                                        emit_marker_status(desc, done=done, hidden=hidden)
                                logger.info(
                                    "wrapper_stream_complete parts_count=%s final_text_preview=%s",
                                    1 if clean else 0,
                                    _safe_trunc(clean, 300),
                                )
                                if clean.strip():
                                    final_text = clean.strip()
                                    if event_emitter:
                                        _emit_replace(event_emitter, final_text)
                                    return final_text
                                fallback_answer = last_status_desc or "Запит виконано."
                                logger.info(
                                    "wrapper_stream_empty_using_fallback preview=%s",
                                    _safe_trunc(fallback_answer, 200),
                                )
                                if event_emitter:
                                    _emit_replace(event_emitter, fallback_answer)
                                return fallback_answer
                            except Exception:
                                fallback = json.dumps(data, ensure_ascii=False)
                                if event_emitter:
                                    _emit_replace(event_emitter, fallback)
                                return fallback

                        emit_pipe("stream_start", {})
                        parts = []
                        token_count = 0
                        last_emit_tokens = 0
                        status_buffer = ""
                        seen_markers = set()

                        def _update_tokens(text):
                            nonlocal token_count, last_emit_tokens
                            if not text:
                                return
                            token_count += len(re.findall(r"\S+", text))
                            if token_count - last_emit_tokens >= 25:
                                last_emit_tokens = token_count
                                emit_pipe("stream_progress", {"tokens": token_count})

                        def _handle_stream_text(text):
                            nonlocal status_buffer, last_status_desc
                            if not text:
                                return ""
                            combined = status_buffer + text
                            clean, status_buffer, markers = _extract_status_markers(combined)
                            for raw in markers:
                                if raw in seen_markers:
                                    continue
                                seen_markers.add(raw)
                                try:
                                    data = json.loads(raw)
                                except Exception:
                                    continue
                                desc = data.get("description")
                                done = bool(data.get("done"))
                                hidden = bool(data.get("hidden"))
                                if isinstance(desc, str) and desc.strip():
                                    last_status_desc = desc.strip()
                                    emit_marker_status(desc, done=done, hidden=hidden)
                            return clean

                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            if line.startswith("data:"):
                                line = line[5:].strip()
                            if not line:
                                continue
                            if line == "[DONE]":
                                break
                            try:
                                data = json.loads(line)
                            except Exception:
                                clean = _handle_stream_text(line)
                                if clean:
                                    parts.append(clean)
                                    _update_tokens(clean)
                                continue
                            choices = (data or {}).get("choices") or []
                            if not choices:
                                continue
                            choice = choices[0] or {}
                            delta_text = _choice_delta_text(choice)
                            if delta_text:
                                clean = _handle_stream_text(delta_text)
                                if clean:
                                    parts.append(clean)
                                    _update_tokens(clean)
                                continue
                            message_text = _choice_message_text(choice)
                            if message_text:
                                clean = _handle_stream_text(message_text)
                                if clean:
                                    _update_tokens(clean)
                                    parts = [clean]
                                else:
                                    parts = [""]
                                break
                        if token_count and token_count != last_emit_tokens:
                            emit_pipe("stream_progress", {"tokens": token_count})
                        emit_pipe("stream_done", {"tokens": token_count})
                        emit_pipe("pipeline_response", {"status": resp.status_code})
                        final_text = "".join(parts)
                        logger.info(
                            "wrapper_stream_complete parts_count=%d final_text_preview=%s",
                            len(parts),
                            _safe_trunc(final_text, 300),
                        )
                        clean, _, markers = _extract_status_markers(final_text)
                        for raw in markers:
                            if raw in seen_markers:
                                continue
                            seen_markers.add(raw)
                            try:
                                data = json.loads(raw)
                            except Exception:
                                continue
                            desc = data.get("description")
                            done = bool(data.get("done"))
                            hidden = bool(data.get("hidden"))
                            if isinstance(desc, str) and desc.strip():
                                last_status_desc = desc.strip()
                                emit_marker_status(desc, done=done, hidden=hidden)
                        if clean.strip():
                            final_text = clean.strip()
                            if event_emitter:
                                _emit_replace(event_emitter, final_text)
                            return final_text
                        fallback_answer = last_status_desc or "Запит виконано."
                        logger.info("wrapper_stream_empty_using_fallback preview=%s", _safe_trunc(fallback_answer, 200))
                        if event_emitter:
                            _emit_replace(event_emitter, fallback_answer)
                        return fallback_answer
                finally:
                    await _stop_wait_ticker(wait_task)
            except Exception as exc:
                logger.warning("pipelines_error error=%s", exc)
                emit_pipe("pipeline_error", {"error": str(exc)}, force=True)
                if not self.valves.TARGET_API_URL:
                    raise
                h2 = {}
                if (self.valves.TARGET_API_KEY or "").strip():
                    h2["Authorization"] = f"Bearer {(self.valves.TARGET_API_KEY or '').strip()}"
                data = {self.valves.QUERY_FIELD: query or ""}
                emit_pipe("target_request", {"url": self.valves.TARGET_API_URL, "files": len(fallback_files)},
                          force=True)
                wait_task = _start_wait_ticker(event_emitter, "target", self.valves.WAIT_TICK_S)
                try:
                    resp = await client.post(self.valves.TARGET_API_URL, headers=h2, data=data,
                                             files=[(self.valves.FILE_FIELD, f) for f in fallback_files] or None)
                finally:
                    await _stop_wait_ticker(wait_task)
                resp.raise_for_status()
                emit_pipe("target_response", {"status": resp.status_code}, force=True)
                if event_emitter:
                    _emit_replace(event_emitter, resp.text)
                return resp.text
