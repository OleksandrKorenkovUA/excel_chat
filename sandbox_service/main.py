import base64
import io
import json
import os
import time
import uuid
import traceback
import ast
import contextlib
import multiprocessing
import resource
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
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


DF_STORE: Dict[str, Dict[str, Any]] = {}
FILE_ID_INDEX: Dict[str, str] = {}


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
                if node.func.value.id == "pd" and node.func.attr.startswith(("read_", "to_")):
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
) -> Tuple[str, str, str, dict, str, Optional[pd.DataFrame], bool, bool]:
    _ast_guard(code)

    def worker(queue: multiprocessing.Queue) -> None:
        stdout = io.StringIO()
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
        }
        env: Dict[str, Any] = {
            "df": df,
            "pd": pd,
            "np": np,
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
            df_out = None
            if commit_flag:
                df_out = env.get("df")
            queue.put(("ok", stdout.getvalue(), result_text, result_meta, "", df_out, commit_flag, undo_flag))
        except Exception as exc:
            queue.put(("err", stdout.getvalue(), "", {}, f"{type(exc).__name__}: {exc}", None, False, False))

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
        return ("timeout", "", "", {}, f"Timeout after {timeout_s}s", None, False, False)
    if queue.empty():
        return ("err", "", "", {}, "NoResult", None, False, False)
    status, stdout, result_text, result_meta, err, df_out, commit_flag, undo_flag = queue.get()
    stdout = _safe_trunc(stdout, max_stdout_chars)
    return status, stdout, result_text, result_meta, err, df_out, commit_flag, undo_flag


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
        status, stdout, result_text, result_meta, err, df_out, commit_flag, undo_flag = _run_code(
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
        }

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
                }
            entry["df"] = history.pop()
            entry["profile"] = _profile_dataframe(entry["df"], preview_rows, max_cell_chars)
        elif commit_flag and isinstance(df_out, pd.DataFrame):
            history = entry.get("history") or []
            history.append(entry["df"].copy(deep=True))
            if MAX_DF_HISTORY > 0 and len(history) > MAX_DF_HISTORY:
                history = history[-MAX_DF_HISTORY:]
            entry["history"] = history
            entry["df"] = df_out
            entry["profile"] = _profile_dataframe(entry["df"], preview_rows, max_cell_chars)
    entry["ts"] = time.time()
    return {
        "status": status,
        "stdout": stdout,
        "result_text": result_text,
        "result_meta": result_meta,
        "error": err,
        "profile": entry.get("profile"),
    }
