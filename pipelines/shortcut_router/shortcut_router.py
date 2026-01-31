import logging
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

try:  # pragma: no cover - runtime guard
    import faiss  # type: ignore
except Exception:  # pragma: no cover - runtime guard
    faiss = None


class ShortcutRouterConfig:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class ShortcutRouter:
    def __init__(self, config: Optional[ShortcutRouterConfig] = None, llm_json: Any = None) -> None:
        self.config = config or ShortcutRouterConfig()
        self._llm_json = llm_json
        self._disabled_reason: Optional[str] = None
        self._loaded = False
        self._index = None
        self._meta: Dict[str, Any] = {}
        self._catalog: Dict[str, Any] = {}
        self._intents: Dict[str, Dict[str, Any]] = {}

    @property
    def disabled_reason(self) -> Optional[str]:
        return self._disabled_reason

    def shortcut_to_sandbox_code(self, query: str, profile: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not self._ensure_loaded():
            logging.info(
                "event=shortcut_router_call status=disabled reason=%s query_preview=%s",
                self._disabled_reason,
                (query or "").strip()[:500],
            )
            return None

        q = (query or "").strip()
        if not q:
            return None

        try:
            hit = self._retrieve(q)
        except Exception as exc:
            logging.warning("event=shortcut_router_retrieval status=error error=%s", exc)
            return None

        if not hit:
            logging.info("event=shortcut_router_retrieval status=miss query_preview=%s", q[:300])
            return None

        intent_id, score, example = hit
        intent = self._intents.get(intent_id)
        if not intent:
            logging.info("event=shortcut_router_retrieval status=unknown_intent id=%s score=%.4f", intent_id, score)
            return None

        columns = [str(c) for c in (profile or {}).get("columns") or []]
        slots = self._fill_slots(intent, q, columns, profile)
        if slots is None:
            return None

        code = self._compile_plan(intent, slots, profile)
        if not code:
            return None

        meta = {"intent_id": intent_id, "score": score, "example": example, "slots": slots}
        return code, meta

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        if self._disabled_reason is not None:
            return False
        if not getattr(self.config, "enabled", True):
            self._disabled_reason = "disabled"
            return False
        if faiss is None:
            self._disabled_reason = "faiss_missing"
            return False
        index_path = getattr(self.config, "index_path", "")
        meta_path = getattr(self.config, "meta_path", "")
        catalog_path = getattr(self.config, "catalog_path", "")
        if not (index_path and os.path.exists(index_path)):
            self._disabled_reason = "missing_index"
            return False
        if not (meta_path and os.path.exists(meta_path)):
            self._disabled_reason = "missing_meta"
            return False
        if not (catalog_path and os.path.exists(catalog_path)):
            self._disabled_reason = "missing_catalog"
            return False

        try:
            self._index = faiss.read_index(index_path)
        except Exception as exc:  # pragma: no cover - runtime guard
            self._disabled_reason = f"index_load_failed:{exc}"
            return False
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                self._meta = json.load(f)
            with open(catalog_path, "r", encoding="utf-8") as f:
                self._catalog = json.load(f)
        except Exception as exc:  # pragma: no cover - runtime guard
            self._disabled_reason = f"meta_load_failed:{exc}"
            return False

        intents = self._catalog.get("intents") or []
        self._intents = {str(i.get("id")): i for i in intents if i.get("id")}
        if not (self._meta.get("rows") or []):
            self._disabled_reason = "empty_meta_rows"
            return False
        self._loaded = True
        return True

    def _retrieve(self, query: str) -> Optional[Tuple[str, float, str]]:
        rows = self._meta.get("rows") or []
        vec = self._embed_query(query)
        if vec is None:
            return None
        scores, idxs = self._index.search(vec, int(getattr(self.config, "top_k", 5)))
        if scores.size == 0:
            return None
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()
        candidates: List[Tuple[int, float]] = [
            (int(i), float(s)) for i, s in zip(idxs, scores) if i >= 0 and i < len(rows)
        ]
        if not candidates:
            return None

        best_idx, best_score = candidates[0]
        threshold = float(getattr(self.config, "threshold", 0.35))
        margin = float(getattr(self.config, "margin", 0.05))
        second_score = candidates[1][1] if len(candidates) > 1 else float("-inf")
        if best_score < threshold or (best_score - second_score) < margin:
            logging.info(
                "event=shortcut_router_retrieval status=below_threshold score=%.4f second=%.4f threshold=%.2f margin=%.2f",
                best_score,
                second_score,
                threshold,
                margin,
            )
            return None

        row = rows[best_idx] or {}
        intent_id = str(row.get("intent_id") or "")
        example = str(row.get("text") or "")
        logging.info(
            "event=shortcut_router_retrieval status=hit intent_id=%s score=%.4f example=%s",
            intent_id,
            best_score,
            example[:300],
        )
        return intent_id, best_score, example

    def _embed_query(self, query: str) -> Optional[np.ndarray]:
        base_url = getattr(self.config, "vllm_base_url", "") or ""
        model = getattr(self.config, "vllm_embed_model", "") or ""
        api_key = getattr(self.config, "vllm_api_key", "") or ""
        timeout_s = int(getattr(self.config, "vllm_timeout_s", 30) or 30)
        if not base_url or not model:
            logging.info("event=shortcut_router_embed status=missing_config")
            return None
        url = f"{base_url.rstrip('/')}/embeddings"
        headers: Dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {"model": model, "input": [query]}
        logging.info("event=shortcut_router_embed_request model=%s query_preview=%s", model, query[:300])
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        vecs = [item.get("embedding") for item in (data.get("data") or [])]
        if not vecs:
            return None
        arr = np.array(vecs, dtype=np.float32)
        if arr.ndim != 2:
            return None
        faiss.normalize_L2(arr)
        return arr

    def _fill_slots(
        self, intent: Dict[str, Any], query: str, columns: List[str], profile: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        slots = {}
        spec = intent.get("slots") or {}
        for name, cfg in spec.items():
            if name in slots:
                continue
            val = self._slot_from_text(name, cfg, query, columns)
            if val is None and "default" in cfg:
                val = cfg.get("default")
            if val is None and cfg.get("required"):
                # Try LLM fill as last resort
                if self._llm_json:
                    val = self._slot_from_llm(intent, name, cfg, query, columns, profile)
            if val is None and cfg.get("required"):
                return None
            if val is not None:
                slots[name] = val
        return slots

    def _slot_from_llm(
        self,
        intent: Dict[str, Any],
        name: str,
        cfg: Dict[str, Any],
        query: str,
        columns: List[str],
        profile: Dict[str, Any],
    ) -> Optional[Any]:
        try:
            payload = {
                "intent_id": intent.get("id"),
                "slot_name": name,
                "slot_type": cfg.get("type"),
                "slot_values": cfg.get("values") or [],
                "query": query,
                "columns": columns,
                "df_profile": profile,
            }
            system = (
                "Fill the requested slot value based on the query and columns. "
                "Return ONLY JSON: {\"value\": <value or null>}."
            )
            logging.info("event=shortcut_router_llm_slot_request slot=%s", name)
            res = self._llm_json(system, json.dumps(payload, ensure_ascii=False))
            logging.info("event=shortcut_router_llm_slot_response slot=%s", name)
            return res.get("value")
        except Exception:
            return None

    def _slot_from_text(
        self, name: str, cfg: Dict[str, Any], query: str, columns: List[str]
    ) -> Optional[Any]:
        slot_type = cfg.get("type")
        q = query or ""
        q_low = q.lower()

        if slot_type == "column":
            return self._best_column_match(q, columns)
        if slot_type in ("columns", "list[column]"):
            cols = self._all_column_matches(q, columns)
            return cols or None
        if slot_type == "row_indices":
            nums = [int(n) for n in re.findall(r"\b(\d+)\b", q)]
            nums = [n for n in nums if n > 0]
            return nums or None
        if slot_type == "int":
            nums = [int(n) for n in re.findall(r"\b(\d+)\b", q)]
            return nums[0] if nums else None
        if slot_type == "float":
            m = re.search(r"(-?\d+(?:[.,]\d+)?)", q)
            if not m:
                return None
            return float(m.group(1).replace(",", "."))
        if slot_type == "bool":
            if re.search(r"\b(true|yes|так|1)\b", q_low):
                return True
            if re.search(r"\b(false|no|ні|0)\b", q_low):
                return False
            return None
        if slot_type == "str":
            m = re.search(r"['\"]([^'\"]+)['\"]", q)
            if m:
                return m.group(1)
            return None
        if slot_type == "enum":
            values = cfg.get("values") or []
            return self._match_enum(values, q_low)
        return None

    def _match_enum(self, values: List[str], q_low: str) -> Optional[str]:
        if not values:
            return None
        enums = {v.lower(): v for v in values}
        for key, val in enums.items():
            if key in q_low:
                return val
        if "mean" in enums and re.search(r"\b(average|avg|середн|mean)\b", q_low):
            return enums["mean"]
        if "sum" in enums and re.search(r"\b(sum|сума|total)\b", q_low):
            return enums["sum"]
        if "min" in enums and re.search(r"\b(min|мін)\b", q_low):
            return enums["min"]
        if "max" in enums and re.search(r"\b(max|макс)\b", q_low):
            return enums["max"]
        if "median" in enums and re.search(r"\b(median|медіан)\b", q_low):
            return enums["median"]
        if "asc" in enums and re.search(r"\b(asc|зрост|ascending)\b", q_low):
            return enums["asc"]
        if "desc" in enums and re.search(r"\b(desc|спад|descending)\b", q_low):
            return enums["desc"]
        if "head" in enums and re.search(r"\b(head|перш|почат)\b", q_low):
            return enums["head"]
        if "tail" in enums and re.search(r"\b(tail|останні|кінец)\b", q_low):
            return enums["tail"]
        return None

    def _normalize_text(self, text: str) -> str:
        t = re.sub(r"[\W_]+", " ", text.lower())
        return re.sub(r"\s+", " ", t).strip()

    def _best_column_match(self, query: str, columns: List[str]) -> Optional[str]:
        q_norm = self._normalize_text(query)
        if not q_norm or not columns:
            return None
        best = None
        best_score = 0.0
        q_tokens = set(q_norm.split())
        for col in columns:
            c_norm = self._normalize_text(col)
            if not c_norm:
                continue
            if c_norm in q_norm:
                return col
            c_tokens = c_norm.split()
            score = 0.0
            for ct in c_tokens:
                for qt in q_tokens:
                    if len(ct) >= 3 and qt.startswith(ct[:3]):
                        score += 1.0
                        break
            if score > best_score:
                best_score = score
                best = col
        return best

    def _all_column_matches(self, query: str, columns: List[str]) -> List[str]:
        q_norm = self._normalize_text(query)
        if not q_norm:
            return []
        out = []
        for col in columns:
            c_norm = self._normalize_text(col)
            if c_norm and c_norm in q_norm:
                out.append(col)
        if out:
            return out
        best = self._best_column_match(query, columns)
        return [best] if best else []

    def _compile_plan(self, intent: Dict[str, Any], slots: Dict[str, Any], profile: Dict[str, Any]) -> Optional[str]:
        plan = intent.get("plan") or []
        if not plan:
            return None
        code_lines: List[str] = []
        for step in plan:
            op = step.get("op")
            args = step.get("args") or {}
            if op == "stats_shape":
                code_lines.append("result = {'rows': int(df.shape[0]), 'cols': int(df.shape[1])}")
            elif op == "head_tail":
                n = int(slots.get("n") or 5)
                side = str(slots.get("side") or "head")
                if side == "tail":
                    code_lines.append(f"result = df.tail({n})")
                else:
                    code_lines.append(f"result = df.head({n})")
            elif op == "stats_nulls":
                col = slots.get("column")
                top_n = int(slots.get("top_n") or 0)
                if col:
                    code_lines.append(f"result = int(df[{col!r}].isna().sum())")
                else:
                    code_lines.append("result = df.isna().sum().to_dict()")
                    if top_n > 0:
                        code_lines.append("result = dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True)[:%d])" % top_n)
            elif op == "stats_nunique":
                col = slots.get("column")
                if col:
                    code_lines.append(f"result = int(df[{col!r}].nunique())")
                else:
                    code_lines.append("result = df.nunique().to_dict()")
            elif op == "stats_aggregation":
                col = slots.get("column")
                metric = slots.get("metric") or "sum"
                if not col:
                    return None
                code_lines.append(f"result = getattr(df[{col!r}], {metric!r})()")
            elif op == "groupby_count":
                group_col = slots.get("group_col")
                out_col = slots.get("out_col") or "count"
                sort = bool(slots.get("sort", True))
                top_n = int(slots.get("top_n") or 0)
                if not group_col:
                    return None
                code_lines.append(
                    f"result = df.groupby({group_col!r}).size().reset_index(name={out_col!r})"
                )
                if sort:
                    code_lines.append(f"result = result.sort_values({out_col!r}, ascending=False)")
                if top_n > 0:
                    code_lines.append(f"result = result.head({top_n})")
            elif op == "groupby_agg":
                group_col = slots.get("group_col")
                target_col = slots.get("target_col")
                agg = slots.get("agg") or "sum"
                if not group_col or not target_col:
                    return None
                code_lines.append(
                    f"result = df.groupby({group_col!r})[{target_col!r}].agg({agg!r}).reset_index()"
                )
            elif op == "filter_equals":
                col = slots.get("column")
                val = slots.get("value")
                case_insensitive = bool(slots.get("case_insensitive", False))
                if not col:
                    return None
                if case_insensitive:
                    code_lines.append(
                        f"df = df[df[{col!r}].astype(str).str.lower() == str({val!r}).lower()]"
                    )
                else:
                    code_lines.append(f"df = df[df[{col!r}] == {val!r}]")
                code_lines.append("COMMIT_DF = True")
            elif op == "filter_comparison":
                col = slots.get("column")
                op_sym = slots.get("operator") or ">"
                val = slots.get("value")
                if not col:
                    return None
                code_lines.append(f"df = df[df[{col!r}] {op_sym} {val!r}]")
                code_lines.append("COMMIT_DF = True")
            elif op == "filter_contains":
                col = slots.get("column")
                val = slots.get("value")
                if not col:
                    return None
                code_lines.append(
                    f"result = df[df[{col!r}].astype(str).str.contains({val!r}, case=False, na=False)]"
                )
            elif op == "filter_range_numeric":
                col = slots.get("column")
                min_v = slots.get("min")
                max_v = slots.get("max")
                if not col:
                    return None
                conds = []
                if min_v is not None:
                    conds.append(f"(df[{col!r}] >= {min_v!r})")
                if max_v is not None:
                    conds.append(f"(df[{col!r}] <= {max_v!r})")
                if not conds:
                    return None
                code_lines.append(f"result = df[{ ' & '.join(conds) }]")
            elif op == "sort_values":
                col = slots.get("column")
                order = str(slots.get("order") or "asc")
                if not col:
                    return None
                asc = "False" if order == "desc" else "True"
                code_lines.append(f"df = df.sort_values(by={col!r}, ascending={asc})")
                code_lines.append("COMMIT_DF = True")
            elif op == "select_columns":
                cols = slots.get("columns") or []
                if not cols:
                    return None
                code_lines.append(f"df = df[{cols!r}].copy()")
                code_lines.append("COMMIT_DF = True")
            elif op == "drop_columns":
                cols = slots.get("columns") or []
                if not cols:
                    return None
                code_lines.append(f"df = df.drop(columns={cols!r})")
                code_lines.append("COMMIT_DF = True")
            elif op == "rename_column":
                old = slots.get("old_name")
                new = slots.get("new_name")
                if not old or not new:
                    return None
                code_lines.append(f"df = df.rename(columns={{ {old!r}: {new!r} }})")
                code_lines.append("COMMIT_DF = True")
            elif op == "drop_rows_by_position":
                idxs = slots.get("indices") or slots.get("row_indices") or []
                if not idxs:
                    return None
                code_lines.append(f"_idx_1b = {idxs!r}")
                code_lines.append("_idx_0b = [i - 1 for i in _idx_1b if i > 0]")
                code_lines.append("df = df.reset_index(drop=True)")
                code_lines.append("df = df.drop(index=_idx_0b)")
                code_lines.append("COMMIT_DF = True")
            elif op == "drop_duplicates":
                subset = slots.get("subset")
                if subset:
                    code_lines.append(f"df = df.drop_duplicates(subset={subset!r})")
                else:
                    code_lines.append("df = df.drop_duplicates()")
                code_lines.append("COMMIT_DF = True")
            elif op == "stats_describe":
                cols = slots.get("columns")
                if cols:
                    code_lines.append(f"result = df[{cols!r}].describe(include='all')")
                else:
                    code_lines.append("result = df.describe(include='all')")
            elif op == "export_csv":
                return None
            elif op == "return_df_preview":
                rows = int(args.get("rows") or 20)
                code_lines.append(f"result = df.head({rows})")
            else:
                return None
        return "\n".join(code_lines).strip() + "\n"
