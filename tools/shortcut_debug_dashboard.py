#!/usr/bin/env python3
import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse


DEFAULT_TRACE_PATH = os.getenv(
    "SHORTCUT_DEBUG_TRACE_PATH",
    "pipelines/shortcut_router/learning/debug_trace.jsonl",
)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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
                    rows.append(obj)
    except FileNotFoundError:
        return []
    except Exception:
        return []
    return rows


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Shortcut Debug Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root { --bg:#0f172a; --panel:#111827; --line:#334155; --text:#e5e7eb; --muted:#94a3b8; --accent:#22d3ee; }
    body { margin:0; font:14px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background:var(--bg); color:var(--text); }
    .wrap { display:grid; grid-template-columns: 1.2fr 1fr; height:100vh; }
    .pane { border-right:1px solid var(--line); overflow:auto; }
    .pane:last-child { border-right:none; }
    .toolbar { position:sticky; top:0; background:var(--panel); padding:10px; border-bottom:1px solid var(--line); z-index:2; }
    .toolbar input, .toolbar select { background:#0b1220; color:var(--text); border:1px solid var(--line); padding:6px; }
    .toolbar button { background:#0b1220; color:var(--accent); border:1px solid var(--accent); padding:6px 10px; cursor:pointer; }
    table { width:100%; border-collapse:collapse; }
    th, td { padding:8px; border-bottom:1px solid #1f2937; vertical-align:top; }
    tr:hover { background:#0b1220; cursor:pointer; }
    tr.sel { background:#082f49; }
    .muted { color:var(--muted); }
    .detail { padding:10px; }
    pre { background:#020617; border:1px solid #1f2937; padding:8px; white-space:pre-wrap; word-break:break-word; }
    h3 { margin:8px 0; color:var(--accent); }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="pane">
      <div class="toolbar">
        <label>Limit</label>
        <select id="limit">
          <option>50</option><option selected>150</option><option>300</option><option>800</option>
        </select>
        <label>Trace ID</label>
        <input id="trace" placeholder="exact trace_id" />
        <label>Filter</label>
        <input id="q" placeholder="request_id / intent / text" />
        <button id="refresh">Refresh</button>
        <button id="open-latest">Open Latest Request</button>
        <label><input type="checkbox" id="auto" checked /> auto</label>
        <span id="meta" class="muted"></span>
      </div>
      <table>
        <thead>
          <tr><th>time</th><th>request</th><th>intent</th><th>status</th><th>llm_tok</th><th>llm_ms</th><th>ret_top</th><th>judge</th><th>query</th></tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
    <div class="pane">
      <div class="detail" id="detail"><span class="muted">Select a row</span></div>
    </div>
  </div>
<script>
let data = [];
let selected = null;

function fmtTs(ts) {
  if (!ts) return "";
  try { return new Date(ts * 1000).toLocaleString(); } catch (e) { return String(ts); }
}

function num(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function esc(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function j(v) { return JSON.stringify(v ?? {}, null, 2); }

function llmUsage(row) {
  let calls = num(row.llm_calls_count);
  let prompt = num(row.llm_prompt_tokens);
  let completion = num(row.llm_completion_tokens);
  let total = num(row.llm_total_tokens);
  let latency = num(row.llm_total_latency_ms);
  const items = Array.isArray(row.llm_calls) ? row.llm_calls : [];
  if (calls === null) calls = items.length;
  if (prompt === null) {
    let s = 0, ok = false;
    for (const it of items) {
      const v = num(it?.prompt_tokens);
      if (v !== null) { s += v; ok = true; }
    }
    prompt = ok ? s : null;
  }
  if (completion === null) {
    let s = 0, ok = false;
    for (const it of items) {
      const v = num(it?.completion_tokens);
      if (v !== null) { s += v; ok = true; }
    }
    completion = ok ? s : null;
  }
  if (total === null) {
    let s = 0, ok = false;
    for (const it of items) {
      const v = num(it?.total_tokens);
      if (v !== null) { s += v; ok = true; }
    }
    total = ok ? s : null;
  }
  if (latency === null) {
    let s = 0, ok = false;
    for (const it of items) {
      const v = num(it?.latency_ms);
      if (v !== null) { s += v; ok = true; }
    }
    latency = ok ? Number(s.toFixed(3)) : null;
  }
  return { calls, prompt, completion, total, latency };
}

function retrievalTopScore(row) {
  const direct = num(row.retrieval_top_score);
  if (direct !== null) return direct;
  const list = Array.isArray(row.retrieval_candidates) ? row.retrieval_candidates : [];
  if (!list.length) return null;
  return num(list[0]?.score);
}

function fmtTokens(v) {
  const n = num(v);
  return n === null ? "" : String(n);
}

function fmtMs(v) {
  const n = num(v);
  return n === null ? "" : `${n.toFixed(1)} ms`;
}

function fmtScore(v) {
  const n = num(v);
  return n === null ? "" : n.toFixed(4);
}

function drawDetail(row) {
  const box = document.getElementById("detail");
  if (!row) { box.innerHTML = '<span class="muted">Select a row</span>'; return; }
  const usage = llmUsage(row);
  const retrieval = Array.isArray(row.retrieval_candidates) ? row.retrieval_candidates : [];
  const retrievalSummary = {
    query_used: row.retrieval_query_used || "",
    normalized_query: row.normalized_query || "",
    threshold: row.retrieval_threshold,
    margin: row.retrieval_margin,
    candidate_count: row.retrieval_candidate_count,
    top_score: row.retrieval_top_score,
    second_score: row.retrieval_second_score,
    selector_mode: row.selector_mode || "",
    selector_confidence: row.selector_confidence,
    selected_intent: row.intent_id || "",
    selected_score: row.score
  };
  box.innerHTML = `
    <h3>Request</h3><pre>${esc(row.request_id || "")} / ${esc(row.trace_id || "")}
run_status: ${esc(row.run_status || "")} | selector: ${esc(row.selector_mode || "")} | intent: ${esc(row.intent_id || "")}
selected_score: ${fmtScore(row.score)}</pre>
    <h3>LLM Usage</h3><pre>calls: ${usage.calls ?? ""}
prompt_tokens: ${usage.prompt ?? ""}
completion_tokens: ${usage.completion ?? ""}
total_tokens: ${usage.total ?? ""}
total_latency_ms: ${usage.latency ?? ""}</pre>
    <h3>LLM Calls</h3><pre>${esc(j(row.llm_calls || []))}</pre>
    <h3>Retrieval Summary</h3><pre>${esc(j(retrievalSummary))}</pre>
    <h3>Retrieval Candidates</h3><pre>${esc(j(retrieval))}</pre>
    <h3>Judge</h3><pre>score: ${row.llm_judge_score ?? ""} | status: ${esc(row.llm_judge_status || "")} | reason: ${esc(row.llm_judge_reason || "")}</pre>
    <h3>User Query</h3><pre>${esc(row.question || "")}</pre>
    <h3>QueryIR</h3><pre>${esc(j(row.query_ir))}</pre>
    <h3>QueryIR Summary</h3><pre>${esc(j(row.query_ir_summary))}</pre>
    <h3>Slots</h3><pre>${esc(j(row.slots))}</pre>
    <h3>Generated Code</h3><pre>${esc(row.analysis_code || "")}</pre>
    <h3>Final Answer</h3><pre>${esc(row.final_answer || "")}</pre>
    <h3>Result Text</h3><pre>${esc(row.result_text || "")}</pre>
  `;
}

function drawRows() {
  const tbody = document.getElementById("rows");
  const trace = (document.getElementById("trace").value || "").trim();
  const q = (document.getElementById("q").value || "").toLowerCase();
  const list = data.filter(r => {
    if (trace && String(r.trace_id || "") !== trace) return false;
    if (!q) return true;
    const hay = [
      r.request_id, r.intent_id, r.status, r.run_status, r.question,
      r.llm_judge_status, r.llm_judge_score, r.llm_judge_reason,
      r.llm_total_tokens, r.llm_total_latency_ms, r.score, r.retrieval_top_score,
      JSON.stringify(r.slots || {}), JSON.stringify(r.query_ir || {}),
      JSON.stringify(r.retrieval_candidates || []), JSON.stringify(r.llm_calls || [])
    ].join(" ").toLowerCase();
    return hay.includes(q);
  });
  tbody.innerHTML = "";
  for (const row of list) {
    const usage = llmUsage(row);
    const topScore = retrievalTopScore(row);
    const tr = document.createElement("tr");
    if (selected && selected.request_id === row.request_id && selected.ts === row.ts) tr.className = "sel";
    tr.innerHTML = `
      <td>${fmtTs(row.ts)}</td>
      <td>${esc(row.request_id || "")}</td>
      <td>${esc(row.intent_id || "")}</td>
      <td>${esc(row.run_status || "")}</td>
      <td>${fmtTokens(usage.total)}</td>
      <td>${fmtMs(usage.latency)}</td>
      <td>${fmtScore(topScore)}</td>
      <td>${row.llm_judge_score ?? ""}</td>
      <td>${esc((row.question || "").slice(0, 120))}</td>
    `;
    tr.onclick = () => { selected = row; drawRows(); drawDetail(row); };
    tbody.appendChild(tr);
  }
  document.getElementById("meta").textContent = `rows: ${list.length}/${data.length}`;
  if (selected && !list.find(r => r.request_id === selected.request_id && r.ts === selected.ts)) {
    selected = null;
    drawDetail(null);
  }
}

function openLatestRequest() {
  const trace = (document.getElementById("trace").value || "").trim();
  const list = data.filter(r => !trace || String(r.trace_id || "") === trace);
  if (!list.length) {
    selected = null;
    drawRows();
    drawDetail(null);
    return;
  }
  selected = list[0];
  drawRows();
  drawDetail(selected);
}

async function loadData() {
  const limit = Number(document.getElementById("limit").value || 150);
  const trace = encodeURIComponent((document.getElementById("trace").value || "").trim());
  const resp = await fetch(`/api/traces?limit=${limit}&trace_id=${trace}`, { cache: "no-store" });
  data = await resp.json();
  drawRows();
  if (!selected && data.length) {
    openLatestRequest();
  }
}

document.getElementById("refresh").onclick = loadData;
document.getElementById("q").oninput = drawRows;
document.getElementById("trace").oninput = loadData;
document.getElementById("limit").onchange = loadData;
document.getElementById("open-latest").onclick = openLatestRequest;

setInterval(() => {
  if (document.getElementById("auto").checked) loadData().catch(() => {});
}, 2000);

loadData().catch(() => {});
</script>
</body>
</html>
"""


def make_handler(trace_path: str, default_limit: int):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _html(self, payload: str) -> None:
            body = payload.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                return self._html(HTML)
            if parsed.path == "/api/health":
                return self._json({"ok": True, "ts": int(time.time()), "trace_path": trace_path})
            if parsed.path == "/api/traces":
                qs = parse_qs(parsed.query or "")
                try:
                    limit = max(1, min(5000, int((qs.get("limit") or [default_limit])[0])))
                except Exception:
                    limit = default_limit
                trace_id = str((qs.get("trace_id") or [""])[0] or "").strip()
                rows = read_jsonl(trace_path)
                if trace_id:
                    rows = [r for r in rows if str(r.get("trace_id") or "") == trace_id]
                rows = rows[-limit:]
                rows.reverse()
                return self._json(rows)
            return self._json({"error": "not_found"}, status=404)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Shortcut Router debug dashboard")
    parser.add_argument("--trace-path", default=DEFAULT_TRACE_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--limit", type=int, default=150)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.trace_path) or ".", exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.trace_path, args.limit))
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Trace path: {args.trace_path}")
    server.serve_forever()


if __name__ == "__main__":
    main()
