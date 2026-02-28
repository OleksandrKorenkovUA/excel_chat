import base64
import copy
import io
import json
import os
import re
import time
import uuid
import traceback
import ast
import contextlib
import multiprocessing
import resource
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


app = FastAPI()


def _df_append_compat(self: pd.DataFrame, other: Any, ignore_index: bool = False, **kwargs: Any) -> pd.DataFrame:
    if other is None:
        return self.copy()
    if isinstance(other, pd.Series):
        other_df = other.to_frame().T
    elif isinstance(other, pd.DataFrame):
        other_df = other
    elif isinstance(other, dict):
        other_df = pd.DataFrame([other])
    elif isinstance(other, list):
        if other and all(isinstance(x, (pd.DataFrame, pd.Series)) for x in other):
            frames = [self] + [x.to_frame().T if isinstance(x, pd.Series) else x for x in other]
            return pd.concat(frames, ignore_index=ignore_index)
        other_df = pd.DataFrame(other)
    else:
        other_df = pd.DataFrame([other])
    return pd.concat([self, other_df], ignore_index=ignore_index)


if not hasattr(pd.DataFrame, "append"):
    pd.DataFrame.append = _df_append_compat

SANDBOX_API_KEY = os.getenv("SANDBOX_API_KEY", "")
DEF_MAX_ROWS = int(os.getenv("MAX_ROWS", "200000"))
DEF_PREVIEW_ROWS = int(os.getenv("PREVIEW_ROWS", str(DEF_MAX_ROWS)))
DEF_MAX_CELL_CHARS = int(os.getenv("MAX_CELL_CHARS", "200"))
DEF_MAX_STDOUT_CHARS = int(os.getenv("MAX_STDOUT_CHARS", "8000"))
DEF_MAX_RESULT_CHARS = int(os.getenv("MAX_RESULT_CHARS", "20000"))
DF_CACHE_TTL_S = int(os.getenv("DF_CACHE_TTL_S", "1800"))
MAX_DF_CACHE_ITEMS = int(os.getenv("MAX_DF_CACHE_ITEMS", "32"))
MAX_DF_HISTORY = int(os.getenv("MAX_DF_HISTORY", "5"))
CPU_TIME_S = int(os.getenv("CPU_TIME_S", "120"))
MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", "1024"))
ROUTE_TRACE_MAX_ITEMS = int(os.getenv("ROUTE_TRACE_MAX_ITEMS", "200"))
ROUTE_TRACE_PERSIST_PATH = os.getenv("ROUTE_TRACE_PERSIST_PATH", "")
ROUTE_TRACE_READ_AUTH = os.getenv("ROUTE_TRACE_READ_REQUIRE_AUTH", "false").lower() in ("1", "true", "yes", "on")
ROUTE_TRACE_API_KEY = os.getenv("ROUTE_TRACE_API_KEY", "")
ROUTE_TRACE_REDACT_KEY_REGEX = os.getenv("ROUTE_TRACE_REDACT_KEY_REGEX", "").strip()
ROUTE_TRACE_REDACT_VALUE_REGEX = os.getenv("ROUTE_TRACE_REDACT_VALUE_REGEX", "").strip()


class LoadRequest(BaseModel):
    file_id: Optional[str] = None
    filename: Optional[str] = None
    content_type: Optional[str] = None
    data_b64: str
    max_rows: Optional[int] = Field(default=None, ge=1)
    preview_rows: Optional[int] = Field(default=None, ge=1)


class RunRequest(BaseModel):
    df_id: str
    code: str
    timeout_s: Optional[int] = Field(default=None, ge=1)
    preview_rows: Optional[int] = Field(default=None, ge=1)
    max_cell_chars: Optional[int] = Field(default=None, ge=10)
    max_stdout_chars: Optional[int] = Field(default=None, ge=1000)
    max_result_chars: Optional[int] = Field(default=None, ge=1000)


class RouteTraceUpsertRequest(BaseModel):
    trace_id: str
    request_id: str
    status: Optional[str] = None
    started_at_ts: Optional[float] = None
    ended_at_ts: Optional[float] = None
    total_latency_ms: Optional[float] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    stages: List[Dict[str, Any]] = Field(default_factory=list)
    updated_at_ts: Optional[float] = None
    final: bool = False


DF_STORE: Dict[str, Dict[str, Any]] = {}
FILE_ID_INDEX: Dict[str, str] = {}
TRACE_STORE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
TRACE_REQUEST_INDEX: Dict[str, str] = {}
TRACE_LOCK = threading.Lock()


def _require_auth(request: Request) -> None:
    if not SANDBOX_API_KEY:
        return
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {SANDBOX_API_KEY}":
        raise HTTPException(status_code=401, detail="unauthorized")


def _safe_trunc(text: str, limit: int) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def _require_trace_write_auth(request: Request) -> None:
    expected = ROUTE_TRACE_API_KEY or SANDBOX_API_KEY
    if not expected:
        return
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="unauthorized")


def _require_trace_read_auth(request: Request) -> None:
    if not ROUTE_TRACE_READ_AUTH:
        return
    _require_trace_write_auth(request)


_TRACE_BASE_SECRET_KEY_REGEX = r"(api[_-]?key|authorization|auth|token|secret|password|passwd|cookie|session|private[_-]?key)"
if ROUTE_TRACE_REDACT_KEY_REGEX:
    try:
        _TRACE_SECRET_KEY_RE = re.compile(
            f"(?:{_TRACE_BASE_SECRET_KEY_REGEX})|(?:{ROUTE_TRACE_REDACT_KEY_REGEX})",
            re.I,
        )
    except re.error:
        _TRACE_SECRET_KEY_RE = re.compile(_TRACE_BASE_SECRET_KEY_REGEX, re.I)
else:
    _TRACE_SECRET_KEY_RE = re.compile(_TRACE_BASE_SECRET_KEY_REGEX, re.I)
_TRACE_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._\-/+=]+", re.I)
_TRACE_OPENAI_RE = re.compile(r"\bsk-[A-Za-z0-9]{12,}\b")
if ROUTE_TRACE_REDACT_VALUE_REGEX:
    try:
        _TRACE_EXTRA_VALUE_RE = re.compile(ROUTE_TRACE_REDACT_VALUE_REGEX, re.I)
    except re.error:
        _TRACE_EXTRA_VALUE_RE = None
else:
    _TRACE_EXTRA_VALUE_RE = None


def _trace_redact_string(text: str) -> str:
    s = str(text or "")
    if not s:
        return s
    s = _TRACE_BEARER_RE.sub("Bearer [REDACTED]", s)
    s = _TRACE_OPENAI_RE.sub("sk-[REDACTED]", s)
    if _TRACE_EXTRA_VALUE_RE is not None:
        s = _TRACE_EXTRA_VALUE_RE.sub("[REDACTED]", s)
    return s


def _trace_redact(value: Any, depth: int = 6) -> Any:
    if depth <= 0:
        return "[MAX_DEPTH]"
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            if _TRACE_SECRET_KEY_RE.search(key):
                out[key] = "[REDACTED]"
            else:
                out[key] = _trace_redact(v, depth - 1)
        return out
    if isinstance(value, list):
        return [_trace_redact(v, depth - 1) for v in value]
    if isinstance(value, tuple):
        return [_trace_redact(v, depth - 1) for v in value]
    if isinstance(value, bytes):
        return f"[bytes:{len(value)}]"
    if isinstance(value, str):
        return _trace_redact_string(value)
    return value


def _trace_prune_locked() -> None:
    while len(TRACE_STORE) > ROUTE_TRACE_MAX_ITEMS:
        old_trace_id, old_payload = TRACE_STORE.popitem(last=False)
        request_id = str((old_payload or {}).get("request_id") or "").strip()
        if request_id and TRACE_REQUEST_INDEX.get(request_id) == old_trace_id:
            TRACE_REQUEST_INDEX.pop(request_id, None)


def _trace_persist_if_enabled(payload: Dict[str, Any]) -> None:
    if not ROUTE_TRACE_PERSIST_PATH:
        return
    try:
        os.makedirs(os.path.dirname(ROUTE_TRACE_PERSIST_PATH) or ".", exist_ok=True)
        with open(ROUTE_TRACE_PERSIST_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")
    except Exception:
        return


def _trace_normalize_stage(stage: Dict[str, Any], index_hint: int) -> Dict[str, Any]:
    now = time.time()
    normalized = _trace_redact(stage if isinstance(stage, dict) else {})
    stage_index = normalized.get("stage_index")
    if not isinstance(stage_index, int) or stage_index <= 0:
        stage_index = max(1, int(index_hint))
    stage_key = str(normalized.get("stage_key") or f"stage_{stage_index}")
    stage_id = str(normalized.get("stage_id") or f"{stage_index}:{stage_key}")
    stage_name = str(normalized.get("stage_name") or stage_key)
    purpose = str(normalized.get("purpose") or "")
    started = normalized.get("started_at_ts")
    started_at_ts = float(started) if isinstance(started, (int, float)) else now
    ended = normalized.get("ended_at_ts")
    ended_at_ts = float(ended) if isinstance(ended, (int, float)) else None
    duration = normalized.get("duration_ms")
    duration_ms = float(duration) if isinstance(duration, (int, float)) else None
    if duration_ms is None and ended_at_ts is not None:
        duration_ms = max(0.0, round((ended_at_ts - started_at_ts) * 1000.0, 3))
    status = str(normalized.get("status") or "in_progress")
    if status not in {"in_progress", "ok", "warn", "error"}:
        status = "warn"
    input_summary = normalized.get("input_summary") if isinstance(normalized.get("input_summary"), dict) else {}
    output_summary = normalized.get("output_summary") if isinstance(normalized.get("output_summary"), dict) else {}
    processing_summary = str(normalized.get("processing_summary") or "")
    error = normalized.get("error") if isinstance(normalized.get("error"), dict) else None
    details = normalized.get("details") if isinstance(normalized.get("details"), dict) else {}
    return {
        "stage_id": stage_id,
        "stage_index": stage_index,
        "stage_key": stage_key,
        "stage_name": stage_name,
        "purpose": purpose,
        "started_at_ts": started_at_ts,
        "ended_at_ts": ended_at_ts,
        "duration_ms": duration_ms,
        "status": status,
        "input_summary": input_summary,
        "processing_summary": processing_summary,
        "output_summary": output_summary,
        "error": error,
        "details": details,
    }


def _trace_sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    trace_id = str(data.get("trace_id") or "").strip()
    request_id = str(data.get("request_id") or "").strip()
    if not trace_id:
        raise HTTPException(status_code=400, detail="trace_id_required")
    if not request_id:
        request_id = trace_id
    stages_raw = data.get("stages")
    stages = []
    if isinstance(stages_raw, list):
        for idx, stage in enumerate(stages_raw, start=1):
            if isinstance(stage, dict):
                stages.append(_trace_normalize_stage(stage, idx))
    status = str(data.get("status") or "in_progress")
    if status not in {"in_progress", "ok", "warn", "error"}:
        status = "warn"
    started = data.get("started_at_ts")
    started_at_ts = float(started) if isinstance(started, (int, float)) else time.time()
    ended = data.get("ended_at_ts")
    ended_at_ts = float(ended) if isinstance(ended, (int, float)) else None
    total_latency = data.get("total_latency_ms")
    if isinstance(total_latency, (int, float)):
        total_latency_ms = float(total_latency)
    elif ended_at_ts is not None:
        total_latency_ms = max(0.0, round((ended_at_ts - started_at_ts) * 1000.0, 3))
    else:
        total_latency_ms = None
    meta_raw = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    meta = _trace_redact(meta_raw)
    updated = data.get("updated_at_ts")
    updated_at_ts = float(updated) if isinstance(updated, (int, float)) else time.time()
    final = bool(data.get("final"))
    return {
        "trace_id": trace_id,
        "request_id": request_id,
        "status": status,
        "started_at_ts": started_at_ts,
        "ended_at_ts": ended_at_ts,
        "total_latency_ms": total_latency_ms,
        "meta": meta,
        "stages": stages,
        "updated_at_ts": updated_at_ts,
        "final": final,
    }


def _trace_upsert(payload: Dict[str, Any]) -> Dict[str, Any]:
    redacted = _trace_sanitize_payload(payload)
    trace_id = str(redacted.get("trace_id") or "").strip()
    request_id = str(redacted.get("request_id") or "").strip()

    with TRACE_LOCK:
        existing = TRACE_STORE.get(trace_id) or {}
        merged = dict(existing)
        merged.update(redacted)
        merged["trace_id"] = trace_id
        merged["request_id"] = request_id
        merged["updated_at_ts"] = time.time()
        TRACE_STORE[trace_id] = merged
        TRACE_STORE.move_to_end(trace_id)
        if request_id:
            TRACE_REQUEST_INDEX[request_id] = trace_id
        _trace_prune_locked()

    if bool(redacted.get("final")):
        _trace_persist_if_enabled(copy.deepcopy(merged))
    return merged


def _trace_get_by_id(trace_id: str) -> Optional[Dict[str, Any]]:
    key = str(trace_id or "").strip()
    if not key:
        return None
    with TRACE_LOCK:
        row = TRACE_STORE.get(key)
        if not row:
            return None
        return copy.deepcopy(row)


def _trace_get_latest(request_id: str = "") -> Optional[Dict[str, Any]]:
    request_id = str(request_id or "").strip()
    with TRACE_LOCK:
        if request_id:
            trace_id = TRACE_REQUEST_INDEX.get(request_id)
            if trace_id and trace_id in TRACE_STORE:
                return copy.deepcopy(TRACE_STORE[trace_id])
            for row in reversed(list(TRACE_STORE.values())):
                if str((row or {}).get("request_id") or "").strip() == request_id:
                    return copy.deepcopy(row)
            return None
        if not TRACE_STORE:
            return None
        last_key = next(reversed(TRACE_STORE))
        return copy.deepcopy(TRACE_STORE[last_key])


def _trace_list(limit: int = 100) -> List[Dict[str, Any]]:
    with TRACE_LOCK:
        rows = list(TRACE_STORE.values())[-max(1, limit) :]
    out: List[Dict[str, Any]] = []
    for row in reversed(rows):
        stages = row.get("stages") if isinstance(row.get("stages"), list) else []
        out.append(
            {
                "trace_id": row.get("trace_id"),
                "request_id": row.get("request_id"),
                "status": row.get("status"),
                "total_latency_ms": row.get("total_latency_ms"),
                "stage_count": len(stages),
                "updated_at_ts": row.get("updated_at_ts"),
                "started_at_ts": row.get("started_at_ts"),
                "ended_at_ts": row.get("ended_at_ts"),
                "meta": row.get("meta") if isinstance(row.get("meta"), dict) else {},
            }
        )
    return out


ROUTE_DASHBOARD_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Data Route Dashboard</title>
  <style>
    :root {
      --bg: #f5f3ee;
      --panel: #fffdf6;
      --line: #d7ceb9;
      --text: #2f2a1f;
      --muted: #786f5b;
      --ok: #1f7a1f;
      --warn: #a26700;
      --err: #b00020;
      --accent: #0e6b5b;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font: 14px/1.45 "IBM Plex Sans", "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }
    .topbar { position: sticky; top: 0; z-index: 5; background: linear-gradient(135deg, #fffdf6, #ece5d3); border-bottom: 1px solid var(--line); padding: 10px 14px; display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    input, select, button { border: 1px solid var(--line); background: #fff; color: var(--text); border-radius: 8px; padding: 6px 10px; }
    button { cursor: pointer; }
    .wrap { max-width: 1200px; margin: 0 auto; padding: 14px; }
    .summary { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 12px; margin-bottom: 12px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }
    .kv { background: #f9f6ee; border: 1px solid var(--line); border-radius: 8px; padding: 8px; }
    .k { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
    .v { margin-top: 2px; font-weight: 600; word-break: break-word; }
    .stage { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; margin-bottom: 10px; overflow: hidden; }
    .stage summary { list-style: none; cursor: pointer; padding: 12px; display: flex; justify-content: space-between; gap: 12px; align-items: center; background: #fff9ea; }
    .stage summary::-webkit-details-marker { display: none; }
    .stage .body { padding: 10px 12px 14px; border-top: 1px solid var(--line); }
    .pill { border-radius: 999px; padding: 2px 10px; font-size: 12px; font-weight: 700; border: 1px solid transparent; }
    .ok { color: var(--ok); border-color: #93c594; background: #edf8ed; }
    .warn { color: var(--warn); border-color: #efcc94; background: #fff5e5; }
    .error { color: var(--err); border-color: #ef9aa8; background: #ffeef1; }
    pre { margin: 0; background: #f9f6ee; border: 1px solid var(--line); border-radius: 8px; padding: 8px; overflow: auto; max-height: 320px; white-space: pre-wrap; word-break: break-word; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
    .muted { color: var(--muted); }
    .toolrow { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
    .modal { position: fixed; inset: 0; background: rgba(22, 19, 14, .62); display: none; align-items: center; justify-content: center; padding: 20px; }
    .modal.show { display: flex; }
    .modal-card { width: min(1000px, 95vw); max-height: 88vh; overflow: auto; background: #fffdf6; border: 1px solid var(--line); border-radius: 12px; padding: 14px; }
    @media (max-width: 900px) { .row { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="topbar">
    <label>trace_id <input id="traceId" placeholder="trace id" /></label>
    <label>request_id <input id="requestId" placeholder="request id" /></label>
    <label>stage filter <input id="stageFilter" placeholder="name/status" /></label>
    <label>status <select id="statusFilter"><option value="">all</option><option>ok</option><option>warn</option><option>error</option></select></label>
    <label><input id="rawToggle" type="checkbox" /> show raw details</label>
    <button id="loadBtn">Load</button>
    <button id="latestBtn">Latest</button>
    <button id="downloadBtn">Download JSON</button>
  </div>
  <div class="wrap">
    <div class="summary" id="summaryBox"><span class="muted">No trace loaded</span></div>
    <div id="stageList"></div>
  </div>
  <div class="modal" id="modal">
    <div class="modal-card">
      <div class="toolrow"><button onclick="closeModal()">Close</button><button onclick="copyFrom('modalPre')">Copy</button></div>
      <pre id="modalPre"></pre>
    </div>
  </div>
<script>
let traceData = null;

function qs(name) { return new URL(window.location.href).searchParams.get(name) || ""; }
function fmtTs(ts) { if (!ts) return ""; try { return new Date(ts * 1000).toLocaleString(); } catch (e) { return String(ts); } }
function fmtMs(v) { if (v === null || v === undefined) return ""; return Number(v).toFixed(2) + " ms"; }
function j(v) { return JSON.stringify(v ?? {}, null, 2); }
function esc(s) { return String(s ?? "").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;"); }

function copyText(text) {
  navigator.clipboard.writeText(text || "").catch(() => {});
}

function copyFrom(id) {
  const el = document.getElementById(id);
  copyText(el ? el.textContent : "");
}

function openModal(content) {
  document.getElementById("modalPre").textContent = typeof content === "string" ? content : j(content);
  document.getElementById("modal").classList.add("show");
}

function closeModal() {
  document.getElementById("modal").classList.remove("show");
}

function statusClass(status) {
  if (status === "ok") return "ok";
  if (status === "warn") return "warn";
  if (status === "error") return "error";
  return "";
}

function cardForStage(stage) {
  const details = stage.details || {};
  const llm = details.llm || null;
  const sandbox = details.sandbox || null;
  const vector = details.vector_search || null;
  const attemptLabel = (stage._attempt_total || 1) > 1 ? ` (attempt ${stage._attempt}/${stage._attempt_total})` : "";
  const rawEnabled = document.getElementById("rawToggle").checked;
  const stageId = `stage_${(stage.stage_index || 0)}_${Math.random().toString(36).slice(2,8)}`;
  return `
    <details class="stage" open>
      <summary>
        <div>
          <div><strong>Stage ${stage.stage_index}: ${esc(stage.stage_name || stage.stage_key)}${esc(attemptLabel)}</strong></div>
          <div class="muted">${esc(stage.purpose || "")}</div>
        </div>
        <div style="text-align:right">
          <div><span class="pill ${statusClass(stage.status)}">${esc(stage.status || "")}</span></div>
          <div class="muted">${fmtMs(stage.duration_ms)}</div>
        </div>
      </summary>
      <div class="body">
        <div class="grid">
          <div class="kv"><div class="k">Started</div><div class="v">${esc(fmtTs(stage.started_at_ts))}</div></div>
          <div class="kv"><div class="k">Ended</div><div class="v">${esc(fmtTs(stage.ended_at_ts))}</div></div>
          <div class="kv"><div class="k">Duration</div><div class="v">${esc(fmtMs(stage.duration_ms))}</div></div>
          <div class="kv"><div class="k">Stage Key</div><div class="v">${esc(stage.stage_key || "")}</div></div>
        </div>
        <div class="row">
          <div>
            <div class="k">Input Summary</div>
            <pre>${esc(j(stage.input_summary || {}))}</pre>
          </div>
          <div>
            <div class="k">Output Summary</div>
            <pre>${esc(j(stage.output_summary || {}))}</pre>
          </div>
        </div>
        <div style="margin-top:10px">
          <div class="k">Processing Summary</div>
          <pre>${esc(stage.processing_summary || "")}</pre>
        </div>
        ${stage.error ? `<div style="margin-top:10px"><div class="k">Error</div><pre>${esc(j(stage.error))}</pre></div>` : ""}
        <div class="toolrow">
          ${llm ? `<button onclick="openModal(${JSON.stringify(j(llm))})">LLM Drill-down</button>` : ""}
          ${llm && llm.messages ? `<button onclick="copyText(${JSON.stringify(j(llm.messages))})">Copy LLM Prompt</button>` : ""}
          ${llm && llm.raw_response ? `<button onclick="copyText(${JSON.stringify(String(llm.raw_response))})">Copy LLM Response</button>` : ""}
          ${sandbox ? `<button onclick="openModal(${JSON.stringify(j(sandbox))})">Sandbox Drill-down</button>` : ""}
          ${sandbox && sandbox.code ? `<button onclick="copyText(${JSON.stringify(String(sandbox.code))})">Copy Sandbox Code</button>` : ""}
          ${vector ? `<button onclick="openModal(${JSON.stringify(j(vector))})">Vector Drill-down</button>` : ""}
          <button onclick="copyText(${JSON.stringify(j(stage))})">Copy Stage JSON</button>
        </div>
        ${rawEnabled ? `<div style="margin-top:10px"><div class="k">Raw Stage Details</div><pre id="${stageId}">${esc(j(stage.details || {}))}</pre></div>` : ""}
      </div>
    </details>
  `;
}

function renderTrace(trace) {
  traceData = trace;
  const stagesRaw = Array.isArray(trace.stages) ? trace.stages : [];
  const totals = {};
  for (const s of stagesRaw) {
    const key = String((s && s.stage_key) || "");
    totals[key] = (totals[key] || 0) + 1;
  }
  const seen = {};
  const stages = stagesRaw.map((s) => {
    const key = String((s && s.stage_key) || "");
    seen[key] = (seen[key] || 0) + 1;
    return { ...s, _attempt: seen[key], _attempt_total: totals[key] || 1 };
  });
  const stageFilter = (document.getElementById("stageFilter").value || "").toLowerCase().trim();
  const statusFilter = (document.getElementById("statusFilter").value || "").toLowerCase().trim();
  const filtered = stages.filter((s) => {
    if (statusFilter && String(s.status || "").toLowerCase() !== statusFilter) return false;
    if (!stageFilter) return true;
    const hay = [s.stage_name, s.stage_key, s.status, s.processing_summary, s.purpose].join(" ").toLowerCase();
    return hay.includes(stageFilter);
  });
  const summary = `
    <div class="grid">
      <div class="kv"><div class="k">Trace ID</div><div class="v">${esc(trace.trace_id || "")}</div></div>
      <div class="kv"><div class="k">Request ID</div><div class="v">${esc(trace.request_id || "")}</div></div>
      <div class="kv"><div class="k">Status</div><div class="v"><span class="pill ${statusClass(trace.status)}">${esc(trace.status || "")}</span></div></div>
      <div class="kv"><div class="k">Total Latency</div><div class="v">${esc(fmtMs(trace.total_latency_ms))}</div></div>
      <div class="kv"><div class="k">Started</div><div class="v">${esc(fmtTs(trace.started_at_ts))}</div></div>
      <div class="kv"><div class="k">Ended</div><div class="v">${esc(fmtTs(trace.ended_at_ts))}</div></div>
      <div class="kv"><div class="k">Stages</div><div class="v">${filtered.length} / ${stages.length}</div></div>
      <div class="kv"><div class="k">Meta</div><div class="v">${esc(j(trace.meta || {}))}</div></div>
    </div>
  `;
  document.getElementById("summaryBox").innerHTML = summary;
  document.getElementById("stageList").innerHTML = filtered.map(cardForStage).join("") || '<div class="muted">No stages after filtering</div>';
}

async function loadTrace() {
  const traceId = (document.getElementById("traceId").value || "").trim();
  const requestId = (document.getElementById("requestId").value || "").trim();
  let url = "";
  if (traceId) {
    url = `/v1/traces/${encodeURIComponent(traceId)}`;
  } else if (requestId) {
    url = `/v1/traces/latest?request_id=${encodeURIComponent(requestId)}`;
  } else {
    url = "/v1/traces/latest";
  }
  const resp = await fetch(url, { cache: "no-store" });
  if (!resp.ok) {
    document.getElementById("summaryBox").innerHTML = `<span class="muted">Trace load failed (HTTP ${resp.status})</span>`;
    document.getElementById("stageList").innerHTML = "";
    return;
  }
  const data = await resp.json();
  renderTrace(data);
}

async function loadLatestList() {
  const resp = await fetch("/v1/traces?limit=1", { cache: "no-store" });
  if (!resp.ok) return;
  const list = await resp.json();
  if (!Array.isArray(list) || !list.length) return;
  const traceId = list[0].trace_id || "";
  if (!traceId) return;
  document.getElementById("traceId").value = traceId;
  await loadTrace();
}

function downloadJson() {
  if (!traceData) return;
  const blob = new Blob([JSON.stringify(traceData, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${traceData.trace_id || "route_trace"}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

document.getElementById("loadBtn").onclick = () => loadTrace().catch(() => {});
document.getElementById("latestBtn").onclick = () => loadLatestList().catch(() => {});
document.getElementById("downloadBtn").onclick = downloadJson;
document.getElementById("stageFilter").oninput = () => { if (traceData) renderTrace(traceData); };
document.getElementById("statusFilter").onchange = () => { if (traceData) renderTrace(traceData); };
document.getElementById("rawToggle").onchange = () => { if (traceData) renderTrace(traceData); };
document.getElementById("modal").onclick = (e) => { if (e.target.id === "modal") closeModal(); };

document.getElementById("traceId").value = qs("trace_id");
document.getElementById("requestId").value = qs("request_id");
if (document.getElementById("traceId").value || document.getElementById("requestId").value) {
  loadTrace().catch(() => {});
} else {
  loadLatestList().catch(() => {});
}
</script>
</body>
</html>
"""


def _cleanup_store() -> None:
    now = time.time()
    expired = [
        key
        for key, entry in DF_STORE.items()
        if now - entry.get("ts", 0) > DF_CACHE_TTL_S
    ]
    for key in expired:
        df_entry = DF_STORE.pop(key, None)
        if df_entry and df_entry.get("file_id"):
            FILE_ID_INDEX.pop(df_entry.get("file_id"), None)
    while len(DF_STORE) > MAX_DF_CACHE_ITEMS:
        oldest_key = min(DF_STORE.items(), key=lambda item: item[1].get("ts", 0))[0]
        df_entry = DF_STORE.pop(oldest_key, None)
        if df_entry and df_entry.get("file_id"):
            FILE_ID_INDEX.pop(df_entry.get("file_id"), None)


def _guess_ext(filename: Optional[str], content_type: Optional[str]) -> str:
    if filename and "." in filename:
        return filename.rsplit(".", 1)[-1].lower()
    if content_type:
        ct = content_type.lower()
        if "csv" in ct:
            return "csv"
        if "tsv" in ct:
            return "tsv"
        if "excel" in ct or "spreadsheet" in ct:
            return "xlsx"
        if "parquet" in ct:
            return "parquet"
        if "json" in ct:
            return "json"
    return ""


def _load_dataframe(data: bytes, filename: str, content_type: str, max_rows: int) -> pd.DataFrame:
    ext = _guess_ext(filename, content_type)
    if ext in {"xlsx", "xls"}:
        return pd.read_excel(io.BytesIO(data), engine="openpyxl")
    if ext in {"parquet", "pq"}:
        return pd.read_parquet(io.BytesIO(data))
    if ext in {"json", "jsonl"}:
        return pd.read_json(io.BytesIO(data), lines=True)
    if ext in {"tsv"}:
        return pd.read_csv(io.BytesIO(data), sep="\t", nrows=max_rows, low_memory=False)
    return pd.read_csv(io.BytesIO(data), nrows=max_rows, low_memory=False)


def _profile_dataframe(df: pd.DataFrame, preview_rows: int, max_cell_chars: int) -> dict:
    preview_df = df.head(preview_rows).copy()
    for col in preview_df.columns:
        if preview_df[col].dtype == object:
            preview_df[col] = preview_df[col].astype(str).map(lambda v: _safe_trunc(v, max_cell_chars))
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns[:200]],
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "nulls_top": {str(k): int(v) for k, v in df.isna().sum().head(200).items()},
        "preview": preview_df.to_dict(orient="records"),
    }


def _ast_guard(code: str) -> None:
    tree = ast.parse(code)
    forbidden_nodes = (
        ast.Import,
        ast.ImportFrom,
        ast.Global,
        ast.Nonlocal,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.Raise,
        ast.Lambda,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Delete,
    )
    forbidden_calls = {
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "__import__",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "help",
    }
    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            raise ValueError(f"forbidden_node:{type(node).__name__}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise ValueError("forbidden_dunder_attr")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("forbidden_dunder_name")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
                raise ValueError(f"forbidden_call:{node.func.id}")
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "pd":
                    # Block pandas I/O APIs, but allow safe conversion helpers.
                    safe_pd_to = {"to_numeric", "to_datetime", "to_timedelta"}
                    if node.func.attr.startswith("read_"):
                        raise ValueError("forbidden_pandas_io")
                    if node.func.attr.startswith("to_") and node.func.attr not in safe_pd_to:
                        raise ValueError("forbidden_pandas_io")
                if node.func.value.id == "df" and node.func.attr.startswith("to_"):
                    raise ValueError("forbidden_dataframe_io")


def _result_meta(result: Any) -> dict:
    if isinstance(result, pd.DataFrame):
        return {
            "rows": int(result.shape[0]),
            "cols": int(result.shape[1]),
            "columns": [str(c) for c in result.columns[:200]],
        }
    if isinstance(result, pd.Series):
        return {
            "length": int(result.shape[0]),
            "name": str(result.name),
        }
    return {}


def _to_jsonable_cell(value: Any, max_len: int = 120) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, str):
        return _safe_trunc(value, max_len)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _safe_trunc(str(value), max_len)


def _mutation_summary(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    max_changes: int = 8,
    max_cells_for_full_diff: int = 2_000_000,
) -> dict:
    rows_before = int(df_before.shape[0])
    rows_after = int(df_after.shape[0])
    cols_before = [str(c) for c in df_before.columns]
    cols_after = [str(c) for c in df_after.columns]
    summary = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "added_rows": max(0, rows_after - rows_before),
        "removed_rows": max(0, rows_before - rows_after),
        "added_columns": [c for c in cols_after if c not in cols_before],
        "removed_columns": [c for c in cols_before if c not in cols_after],
        "changed_cells_count": 0,
        "changed_cells": [],
    }

    cols_after_set = set(cols_after)
    common_cols = [c for c in cols_before if c in cols_after_set]
    common_rows = min(rows_before, rows_after)
    if common_rows <= 0 or not common_cols:
        return summary
    if common_rows * len(common_cols) > max_cells_for_full_diff:
        return summary

    try:
        left = df_before.iloc[:common_rows][common_cols]
        right = df_after.iloc[:common_rows][common_cols]
        neq = ~(left.eq(right) | (left.isna() & right.isna()))
        matrix = neq.to_numpy()
        changed_count = int(matrix.sum())
        summary["changed_cells_count"] = changed_count
        if changed_count <= 0:
            return summary
        coords = np.argwhere(matrix)
        out = []
        for r, c in coords[:max_changes]:
            col = common_cols[int(c)]
            out.append(
                {
                    "row": int(r) + 1,
                    "column": str(col),
                    "old_value": _to_jsonable_cell(left.iat[int(r), int(c)]),
                    "new_value": _to_jsonable_cell(right.iat[int(r), int(c)]),
                }
            )
        summary["changed_cells"] = out
    except Exception:
        return summary
    return summary


def _render_result(result: Any, preview_rows: int, max_cell_chars: int, max_result_chars: int) -> str:
    if result is None:
        return ""
    if isinstance(result, pd.DataFrame):
        preview = result.head(preview_rows).copy()
        for col in preview.columns:
            if preview[col].dtype == object:
                preview[col] = preview[col].astype(str).map(lambda v: _safe_trunc(v, max_cell_chars))
        try:
            text = preview.to_markdown(index=False)
        except Exception:
            text = preview.to_string(index=False)
        return _safe_trunc(text, max_result_chars)
    if isinstance(result, pd.Series):
        text = result.head(preview_rows).to_string()
        return _safe_trunc(text, max_result_chars)
    if isinstance(result, (np.ndarray, pd.Index, pd.api.extensions.ExtensionArray)):
        try:
            data = result.tolist() if hasattr(result, "tolist") else list(result)
            if isinstance(data, list) and len(data) > preview_rows:
                data = data[:preview_rows]
            text = json.dumps(data, ensure_ascii=False, default=str)
            return _safe_trunc(text, max_result_chars)
        except Exception:
            return _safe_trunc(str(result), max_result_chars)
    if isinstance(result, (dict, list, tuple)):
        text = json.dumps(result, ensure_ascii=False, default=str)
        return _safe_trunc(text, max_result_chars)
    return _safe_trunc(result, max_result_chars)


def _run_code(
    code: str,
    df: pd.DataFrame,
    timeout_s: int,
    preview_rows: int,
    max_cell_chars: int,
    max_stdout_chars: int,
    max_result_chars: int,
) -> Tuple[str, str, str, dict, str, dict, Optional[pd.DataFrame], bool, bool, bool, bool]:
    _ast_guard(code)

    def worker(queue: multiprocessing.Queue) -> None:
        stdout = io.StringIO()
        df_before = df.copy(deep=True)
        safe_builtins = {
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "sorted": sorted,
            "range": range,
            "enumerate": enumerate,
            "abs": abs,
            "round": round,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "float": float,
            "int": int,
            "str": str,
            "bool": bool,
            "zip": zip,
            "print": print,
            # Needed for internal lazy imports used by pandas/numpy runtime paths.
            # User code still cannot call __import__ directly due AST guard.
            "__import__": __import__,
        }
        env: Dict[str, Any] = {
            "df": df,
            "pd": pd,
            "np": np,
            "re": re,
            "__builtins__": safe_builtins,
        }
        try:
            if MAX_MEMORY_MB > 0:
                mem_bytes = MAX_MEMORY_MB * 1024 * 1024
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
                except Exception:
                    pass
            if CPU_TIME_S > 0:
                cpu_limit = min(CPU_TIME_S, timeout_s) if timeout_s else CPU_TIME_S
                try:
                    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
                except Exception:
                    pass
            with contextlib.redirect_stdout(stdout):
                exec(compile(code, "<analysis>", "exec"), env, env)
            result = env.get("result")
            result_text = _render_result(result, preview_rows, max_cell_chars, max_result_chars)
            result_meta = _result_meta(result)
            commit_flag = bool(env.get("COMMIT_DF"))
            undo_flag = bool(env.get("UNDO"))
            df_after = env.get("df")
            auto_commit_flag = False
            structure_changed = False
            if isinstance(df_after, pd.DataFrame):
                try:
                    structure_changed = (
                        df_after.shape[0] != df_before.shape[0]
                        or df_after.shape[1] != df_before.shape[1]
                    )
                    auto_commit_flag = (
                        df_after.shape != df_before.shape
                        or not df_after.columns.equals(df_before.columns)
                        or not df_after.dtypes.equals(df_before.dtypes)
                        or not df_after.equals(df_before)
                    )
                except Exception:
                    auto_commit_flag = False
                    structure_changed = False

            if structure_changed:
                auto_commit_flag = True

            effective_commit = bool(commit_flag or auto_commit_flag)
            df_out = df_after if effective_commit and isinstance(df_after, pd.DataFrame) else None
            mutation_summary = {}
            if effective_commit and isinstance(df_after, pd.DataFrame):
                mutation_summary = _mutation_summary(df_before, df_after)
            queue.put(
                (
                    "ok",
                    stdout.getvalue(),
                    result_text,
                    result_meta,
                    "",
                    mutation_summary,
                    df_out,
                    commit_flag,
                    undo_flag,
                    auto_commit_flag,
                    structure_changed,
                )
            )
        except Exception as exc:
            queue.put(
                ("err", stdout.getvalue(), "", {}, f"{type(exc).__name__}: {exc}", {}, None, False, False, False, False)
            )

    if hasattr(multiprocessing, "get_context"):
        try:
            ctx = multiprocessing.get_context("fork")
        except Exception:
            ctx = multiprocessing.get_context("spawn")
    else:
        ctx = multiprocessing
    queue: multiprocessing.Queue = ctx.Queue()
    proc = ctx.Process(target=worker, args=(queue,))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(1)
        return ("timeout", "", "", {}, f"Timeout after {timeout_s}s", {}, None, False, False, False, False)
    if queue.empty():
        return ("err", "", "", {}, "NoResult", {}, None, False, False, False, False)
    status, stdout, result_text, result_meta, err, mutation_summary, df_out, commit_flag, undo_flag, auto_commit_flag, structure_changed = queue.get()
    stdout = _safe_trunc(stdout, max_stdout_chars)
    return (
        status,
        stdout,
        result_text,
        result_meta,
        err,
        mutation_summary,
        df_out,
        commit_flag,
        undo_flag,
        auto_commit_flag,
        structure_changed,
    )


@app.post("/v1/traces/upsert")
def trace_upsert(req: RouteTraceUpsertRequest, request: Request) -> dict:
    _require_trace_write_auth(request)
    payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    stored = _trace_upsert(payload)
    return {
        "status": "ok",
        "trace_id": stored.get("trace_id"),
        "request_id": stored.get("request_id"),
        "updated_at_ts": stored.get("updated_at_ts"),
    }


@app.get("/v1/traces")
def trace_list(request: Request, limit: int = Query(default=50, ge=1, le=500)) -> List[Dict[str, Any]]:
    _require_trace_read_auth(request)
    return _trace_list(limit=limit)


@app.get("/v1/traces/latest")
def trace_latest(request: Request, request_id: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    _require_trace_read_auth(request)
    row = _trace_get_latest(request_id=request_id or "")
    if not row:
        raise HTTPException(status_code=404, detail="trace_not_found")
    return row


@app.get("/v1/traces/dashboard", response_class=HTMLResponse)
def trace_dashboard(request: Request) -> HTMLResponse:
    _require_trace_read_auth(request)
    return HTMLResponse(content=ROUTE_DASHBOARD_HTML)


@app.get("/v1/traces/{trace_id}")
def trace_get(trace_id: str, request: Request) -> Dict[str, Any]:
    _require_trace_read_auth(request)
    row = _trace_get_by_id(trace_id)
    if not row:
        raise HTTPException(status_code=404, detail="trace_not_found")
    return row


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/dataframe/load")
def load_dataframe(req: LoadRequest, request: Request) -> dict:
    _require_auth(request)
    _cleanup_store()

    if req.file_id and req.file_id in FILE_ID_INDEX:
        df_id = FILE_ID_INDEX.get(req.file_id)
        entry = DF_STORE.get(df_id)
        if entry:
            entry["ts"] = time.time()
            return {"df_id": df_id, "profile": entry.get("profile")}

    try:
        data = base64.b64decode(req.data_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid_base64")

    max_rows = req.max_rows or DEF_MAX_ROWS
    preview_rows = req.preview_rows or DEF_PREVIEW_ROWS
    try:
        df = _load_dataframe(data, req.filename or "", req.content_type or "", max_rows)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"read_error:{type(exc).__name__}:{exc}")

    if len(df) > max_rows:
        df = df.head(max_rows)

    profile = _profile_dataframe(df, preview_rows, DEF_MAX_CELL_CHARS)
    df_id = str(uuid.uuid4())
    DF_STORE[df_id] = {
        "df": df,
        "profile": profile,
        "ts": time.time(),
        "file_id": req.file_id,
        "history": [],
    }
    if req.file_id:
        FILE_ID_INDEX[req.file_id] = df_id

    return {"df_id": df_id, "profile": profile}


@app.get("/v1/dataframe/{df_id}/profile")
def get_profile(df_id: str, request: Request) -> dict:
    _require_auth(request)
    _cleanup_store()

    entry = DF_STORE.get(df_id)
    if not entry:
        raise HTTPException(status_code=404, detail="df_not_found")
    entry["ts"] = time.time()
    return {"df_id": df_id, "profile": entry.get("profile"), "ts": entry.get("ts")}


@app.post("/v1/dataframe/run")
def run_code(req: RunRequest, request: Request) -> dict:
    _require_auth(request)
    _cleanup_store()

    entry = DF_STORE.get(req.df_id)
    if not entry:
        raise HTTPException(status_code=404, detail="df_not_found")

    timeout_s = req.timeout_s or CPU_TIME_S
    preview_rows = req.preview_rows or DEF_PREVIEW_ROWS
    max_cell_chars = req.max_cell_chars or DEF_MAX_CELL_CHARS
    max_stdout_chars = req.max_stdout_chars or DEF_MAX_STDOUT_CHARS
    max_result_chars = req.max_result_chars or DEF_MAX_RESULT_CHARS

    try:
        status, stdout, result_text, result_meta, err, mutation_summary, df_out, commit_flag, undo_flag, auto_commit_flag, structure_changed = _run_code(
            req.code,
            entry["df"],
            timeout_s,
            preview_rows,
            max_cell_chars,
            max_stdout_chars,
            max_result_chars,
        )
    except Exception as exc:
        trace = _safe_trunc(traceback.format_exc(), 1000)
        return {
            "status": "err",
            "stdout": "",
            "result_text": "",
            "result_meta": {},
            "error": f"{type(exc).__name__}: {exc} | {trace}",
            "profile": entry.get("profile"),
            "committed": False,
            "auto_committed": False,
            "structure_changed": False,
        }

    was_committed = False

    if status == "ok":
        if undo_flag:
            history = entry.get("history") or []
            if not history:
                return {
                    "status": "err",
                    "stdout": stdout,
                    "result_text": result_text,
                    "result_meta": result_meta,
                    "error": "undo_empty",
                    "profile": entry.get("profile"),
                    "committed": False,
                    "auto_committed": False,
                    "structure_changed": False,
                }
            entry["df"] = history.pop()
            entry["profile"] = _profile_dataframe(entry["df"], preview_rows, max_cell_chars)
            was_committed = True
        elif (commit_flag or auto_commit_flag or structure_changed) and isinstance(df_out, pd.DataFrame):
            history = entry.get("history") or []
            history.append(entry["df"].copy(deep=True))
            if MAX_DF_HISTORY > 0 and len(history) > MAX_DF_HISTORY:
                history = history[-MAX_DF_HISTORY:]
            entry["history"] = history
            entry["df"] = df_out
            entry["profile"] = _profile_dataframe(entry["df"], preview_rows, max_cell_chars)
            was_committed = True
    
    entry["ts"] = time.time()
    return {
        "status": status,
        "stdout": stdout,
        "result_text": result_text,
        "result_meta": result_meta,
        "mutation_summary": mutation_summary,
        "error": err,
        "profile": entry.get("profile"),
        "committed": was_committed,
        "auto_committed": bool(auto_commit_flag and was_committed),
        "structure_changed": bool(structure_changed),
    }
