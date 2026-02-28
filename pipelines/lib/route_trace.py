import contextvars
import hashlib
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

_BASE_SECRET_KEY_REGEX = r"(api[_-]?key|authorization|auth|token|secret|password|passwd|bearer|cookie|session|private[_-]?key)"
_EXTRA_KEY_REGEX = os.getenv("ROUTE_TRACE_REDACT_KEY_REGEX", "").strip()
if _EXTRA_KEY_REGEX:
    try:
        _SECRET_KEY_RE = re.compile(f"(?:{_BASE_SECRET_KEY_REGEX})|(?:{_EXTRA_KEY_REGEX})", re.I)
    except re.error:
        _SECRET_KEY_RE = re.compile(_BASE_SECRET_KEY_REGEX, re.I)
else:
    _SECRET_KEY_RE = re.compile(_BASE_SECRET_KEY_REGEX, re.I)
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._\-+/=]+", re.I)
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9]{12,}\b")
_LONG_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_\-]{24,}\b")
_EXTRA_VALUE_REGEX = os.getenv("ROUTE_TRACE_REDACT_VALUE_REGEX", "").strip()
if _EXTRA_VALUE_REGEX:
    try:
        _EXTRA_VALUE_RE = re.compile(_EXTRA_VALUE_REGEX, re.I)
    except re.error:
        _EXTRA_VALUE_RE = None
else:
    _EXTRA_VALUE_RE = None

_SUMMARY_INCLUDE_PREVIEW = os.getenv("ROUTE_TRACE_SUMMARY_INCLUDE_PREVIEW", "false").lower() in ("1", "true", "yes", "on")
try:
    _SINK_TIMEOUT_S = float(os.getenv("ROUTE_TRACE_SINK_TIMEOUT_S", "0.8"))
except Exception:
    _SINK_TIMEOUT_S = 0.8

_ACTIVE_ROUTE_TRACER: contextvars.ContextVar[Optional["RouteTracer"]] = contextvars.ContextVar(
    "active_route_tracer", default=None
)


def set_active_route_tracer(tracer: Optional["RouteTracer"]) -> contextvars.Token:
    return _ACTIVE_ROUTE_TRACER.set(tracer)


def reset_active_route_tracer(token: contextvars.Token) -> None:
    _ACTIVE_ROUTE_TRACER.reset(token)


def current_route_tracer() -> Optional["RouteTracer"]:
    return _ACTIVE_ROUTE_TRACER.get()


def _safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    except Exception:
        return str(value)


def _safe_preview(text: str, limit: int) -> str:
    s = str(text or "")
    if len(s) <= limit:
        return s
    return s[:limit] + "...(truncated)"


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\S+", text))


def _redact_string(text: str) -> str:
    s = str(text or "")
    if not s:
        return s
    s = _BEARER_RE.sub("Bearer [REDACTED]", s)
    s = _OPENAI_KEY_RE.sub("sk-[REDACTED]", s)
    if _EXTRA_VALUE_RE is not None:
        s = _EXTRA_VALUE_RE.sub("[REDACTED]", s)
    # Keep natural language readable while masking suspicious long token-like values.
    s = _LONG_TOKEN_RE.sub(lambda m: "[REDACTED_TOKEN]" if any(c.isdigit() for c in m.group(0)) else m.group(0), s)
    return s


def redact_payload(value: Any, max_depth: int = 6) -> Any:
    def _walk(obj: Any, depth: int) -> Any:
        if depth <= 0:
            return "[MAX_DEPTH]"
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                key = str(k)
                if _SECRET_KEY_RE.search(key):
                    out[key] = "[REDACTED]"
                else:
                    out[key] = _walk(v, depth - 1)
            return out
        if isinstance(obj, list):
            return [_walk(v, depth - 1) for v in obj]
        if isinstance(obj, tuple):
            return [_walk(v, depth - 1) for v in obj]
        if isinstance(obj, bytes):
            return f"[bytes:{len(obj)}]"
        if isinstance(obj, str):
            return _redact_string(obj)
        return obj

    return _walk(value, max_depth)


def summarize_payload(payload: Any, max_preview_chars: int = 400) -> Dict[str, Any]:
    value = redact_payload(payload)
    summary: Dict[str, Any] = {"kind": type(value).__name__, "size_bytes": None}

    if value is None:
        summary["kind"] = "none"
        return summary

    if isinstance(value, bytes):
        summary["kind"] = "bytes"
        summary["size_bytes"] = len(value)
        return summary

    if isinstance(value, str):
        summary["kind"] = "text"
        summary["chars"] = len(value)
        summary["tokens_estimate"] = _estimate_tokens(value)
        summary["size_bytes"] = len(value.encode("utf-8", errors="ignore"))
        if _SUMMARY_INCLUDE_PREVIEW:
            summary["preview"] = _safe_preview(value, max_preview_chars)
        return summary

    if isinstance(value, dict):
        raw = _safe_json_dumps(value)
        summary["kind"] = "json_object"
        summary["size_bytes"] = len(raw.encode("utf-8", errors="ignore"))
        keys = list(value.keys())
        summary["keys"] = [str(k) for k in keys[:50]]
        summary["keys_count"] = len(keys)
        if {"rows", "cols", "columns", "dtypes"}.issubset(set(value.keys())):
            summary["dataframe_profile"] = {
                "rows": value.get("rows"),
                "cols": value.get("cols"),
                "columns_count": len(value.get("columns") or []),
                "dtypes_count": len(value.get("dtypes") or {}),
            }
        if "data_b64" in value and isinstance(value.get("data_b64"), str):
            summary["base64_chars"] = len(value.get("data_b64") or "")
        if _SUMMARY_INCLUDE_PREVIEW:
            summary["preview"] = _safe_preview(raw, max_preview_chars)
        return summary

    if isinstance(value, list):
        raw = _safe_json_dumps(value)
        summary["kind"] = "json_array"
        summary["items"] = len(value)
        summary["item_types"] = sorted({type(v).__name__ for v in value[:30]})
        summary["size_bytes"] = len(raw.encode("utf-8", errors="ignore"))
        if _SUMMARY_INCLUDE_PREVIEW:
            summary["preview"] = _safe_preview(raw, max_preview_chars)
        return summary

    raw = _safe_json_dumps(value)
    summary["kind"] = "scalar"
    summary["size_bytes"] = len(raw.encode("utf-8", errors="ignore"))
    if _SUMMARY_INCLUDE_PREVIEW:
        summary["preview"] = _safe_preview(raw, max_preview_chars)
    return summary


@dataclass
class StageEvent:
    stage_id: str
    stage_index: int
    stage_key: str
    stage_name: str
    purpose: str
    started_at_ts: float
    ended_at_ts: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "in_progress"
    input_summary: Dict[str, Any] = field(default_factory=dict)
    processing_summary: str = ""
    output_summary: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    details: Dict[str, Any] = field(default_factory=dict)


class RouteTracer:
    def __init__(
        self,
        *,
        request_id: str,
        trace_id: Optional[str] = None,
        sink_url: str = "",
        sink_api_key: str = "",
        max_payload_chars: int = 4000,
        persist_path: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.request_id = str(request_id or "").strip() or uuid.uuid4().hex
        self.trace_id = str(trace_id or "").strip() or uuid.uuid4().hex
        self.sink_url = str(sink_url or "").strip()
        self.sink_api_key = str(sink_api_key or "").strip()
        self.max_payload_chars = max(256, int(max_payload_chars or 4000))
        self.persist_path = str(persist_path or "").strip()
        self.meta = dict(meta or {})

        self._lock = threading.Lock()
        self._started_monotonic = time.monotonic()
        self._started_at_ts = time.time()
        self._ended_at_ts: Optional[float] = None
        self._status = "in_progress"
        self._next_stage_index = 1
        self._open: Dict[str, float] = {}
        self._stages: List[StageEvent] = []

    def start_stage(
        self,
        *,
        stage_key: str,
        stage_name: str,
        purpose: str,
        input_payload: Any = None,
        processing_summary: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        with self._lock:
            stage_index = self._next_stage_index
            self._next_stage_index += 1
            stage_id = f"{stage_index}:{stage_key}"
            event = StageEvent(
                stage_id=stage_id,
                stage_index=stage_index,
                stage_key=str(stage_key),
                stage_name=str(stage_name),
                purpose=str(purpose),
                started_at_ts=time.time(),
                input_summary=summarize_payload(input_payload, max_preview_chars=self.max_payload_chars),
                processing_summary=_safe_preview(processing_summary, self.max_payload_chars),
                details=redact_payload(details or {}),
            )
            self._stages.append(event)
            self._open[stage_id] = time.monotonic()
            return stage_id

    def end_stage(
        self,
        stage_id: str,
        *,
        status: str = "ok",
        output_payload: Any = None,
        processing_summary: Optional[str] = None,
        error: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        publish: bool = True,
    ) -> None:
        with self._lock:
            event = next((it for it in self._stages if it.stage_id == stage_id), None)
            if event is None:
                return
            start = self._open.pop(stage_id, None)
            event.ended_at_ts = time.time()
            if start is not None:
                event.duration_ms = round((time.monotonic() - start) * 1000.0, 3)
            event.status = status
            event.output_summary = summarize_payload(output_payload, max_preview_chars=self.max_payload_chars)
            if processing_summary is not None:
                event.processing_summary = _safe_preview(processing_summary, self.max_payload_chars)
            if error:
                event.error = redact_payload(error)
            if details:
                merged = dict(event.details or {})
                merged.update(redact_payload(details))
                event.details = merged

        if publish:
            self.publish_snapshot(final=False)

    def update_stage(self, stage_id: str, *, processing_summary: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            event = next((it for it in self._stages if it.stage_id == stage_id), None)
            if event is None:
                return
            if processing_summary is not None:
                event.processing_summary = _safe_preview(processing_summary, self.max_payload_chars)
            if details:
                merged = dict(event.details or {})
                merged.update(redact_payload(details))
                event.details = merged

    def record_stage(
        self,
        *,
        stage_key: str,
        stage_name: str,
        purpose: str,
        input_payload: Any = None,
        output_payload: Any = None,
        processing_summary: str = "",
        status: str = "ok",
        details: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        sid = self.start_stage(
            stage_key=stage_key,
            stage_name=stage_name,
            purpose=purpose,
            input_payload=input_payload,
            processing_summary=processing_summary,
            details=details,
        )
        self.end_stage(
            sid,
            status=status,
            output_payload=output_payload,
            processing_summary=processing_summary,
            details=details,
            error=error,
        )

    def _stage_dicts(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for st in self._stages:
            out.append(
                {
                    "stage_id": st.stage_id,
                    "stage_index": st.stage_index,
                    "stage_key": st.stage_key,
                    "stage_name": st.stage_name,
                    "purpose": st.purpose,
                    "started_at_ts": st.started_at_ts,
                    "ended_at_ts": st.ended_at_ts,
                    "duration_ms": st.duration_ms,
                    "status": st.status,
                    "input_summary": st.input_summary,
                    "processing_summary": st.processing_summary,
                    "output_summary": st.output_summary,
                    "error": st.error,
                    "details": st.details,
                }
            )
        return out

    def to_dict(self, final: bool = False) -> Dict[str, Any]:
        with self._lock:
            ended = self._ended_at_ts or time.time()
            total_latency_ms = round((ended - self._started_at_ts) * 1000.0, 3)
            payload = {
                "trace_id": self.trace_id,
                "request_id": self.request_id,
                "status": self._status,
                "started_at_ts": self._started_at_ts,
                "ended_at_ts": self._ended_at_ts,
                "total_latency_ms": total_latency_ms,
                "meta": redact_payload(self.meta),
                "stages": self._stage_dicts(),
                "updated_at_ts": time.time(),
                "final": bool(final),
            }
        return payload

    def publish_snapshot(self, final: bool = False) -> None:
        payload = self.to_dict(final=final)
        if self.sink_url:
            headers: Dict[str, str] = {"Content-Type": "application/json"}
            if self.sink_api_key:
                headers["Authorization"] = f"Bearer {self.sink_api_key}"
            try:
                requests.post(self.sink_url, headers=headers, json=payload, timeout=max(0.2, _SINK_TIMEOUT_S))
            except Exception:
                pass

        if final and self.persist_path:
            try:
                os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
                with open(self.persist_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False))
                    f.write("\n")
            except Exception:
                pass

    def finalize(self, status: str = "ok") -> None:
        with self._lock:
            self._status = status
            self._ended_at_ts = time.time()
        # Do not block request completion on trace sink/network latency.
        threading.Thread(target=self.publish_snapshot, kwargs={"final": True}, daemon=True).start()

    def dashboard_url(self, public_base_url: str) -> str:
        base = str(public_base_url or "").strip()
        if not base:
            return ""
        sep = "&" if "?" in base else "?"
        return f"{base}{sep}trace_id={self.trace_id}&request_id={self.request_id}"

    def trace_fingerprint(self) -> str:
        raw = _safe_json_dumps(self.to_dict(final=False))
        return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]
