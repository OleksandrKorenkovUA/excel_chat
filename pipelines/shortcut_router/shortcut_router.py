import logging
import json
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

        complexity = self._assess_query_complexity(q, profile)
        if complexity > 0.7:
            logging.info(
                "event=shortcut_router_call status=skipped reason=complex_query score=%.2f query_preview=%s",
                complexity,
                q[:300],
            )
            return None

        if self._has_filter_context(q, profile):
            logging.info(
                "event=shortcut_router_call status=skipped reason=filter_context query_preview=%s",
                q[:300],
            )
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
        intent, preset_slots = self._resolve_intent_and_slots(intent, q, columns, profile)
        intent_id = str(intent.get("id") or intent_id)
        slots = self._fill_slots(intent, q, columns, profile, preset_slots=preset_slots)
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
        self,
        intent: Dict[str, Any],
        query: str,
        columns: List[str],
        profile: Dict[str, Any],
        preset_slots: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        slots: Dict[str, Any] = dict(preset_slots or {})
        spec = intent.get("slots") or {}
        intent_id = str(intent.get("id") or "")

        # Special handling for contains-style intents:
        # resolve filter value first, then choose filter column from value/profile.
        if intent_id in {"filter_contains", "filtered_metric_aggregation"}:
            value_slot = "value" if intent_id == "filter_contains" else "filter_value"
            filter_col_slot = "column" if intent_id == "filter_contains" else "filter_col"

            value_cfg = spec.get(value_slot) or {}
            if value_slot not in slots and value_cfg:
                value = self._slot_from_text(value_slot, value_cfg, query, columns)
                if value is None:
                    value = self._extract_filter_value_from_query(query)
                if value is None and "default" in value_cfg:
                    value = value_cfg.get("default")
                if value is None and value_cfg.get("required") and self._llm_json:
                    value = self._slot_from_llm(intent, value_slot, value_cfg, query, columns, profile)
                if value is not None:
                    slots[value_slot] = value

            if filter_col_slot not in slots:
                filter_value = slots.get(value_slot)
                filter_col = self._best_filter_column_for_value(
                    query=query,
                    value=filter_value,
                    columns=columns,
                    profile=profile,
                    intent_id=intent_id,
                )
                if filter_col:
                    slots[filter_col_slot] = filter_col

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
                # For stats_aggregation we can still recover a numeric column in post-processing.
                if intent_id == "stats_aggregation" and name == "column":
                    continue
                if intent_id == "filtered_metric_aggregation" and name == "target_col":
                    continue
                return None
            if val is not None:
                slots[name] = val
        if intent_id == "stats_aggregation":
            col = slots.get("column")
            dtypes = (profile or {}).get("dtypes") or {}
            if not isinstance(col, str) or col not in columns:
                col = None
            if col and not self._is_numeric_dtype(dtypes.get(col, "")):
                # Retry with LLM for column slot, but enforce numeric column.
                cfg = (intent.get("slots") or {}).get("column") or {}
                llm_col = self._slot_from_llm(intent, "column", cfg, query, columns, profile) if self._llm_json else None
                if isinstance(llm_col, str) and llm_col in columns and self._is_numeric_dtype(dtypes.get(llm_col, "")):
                    col = llm_col
                else:
                    col = self._best_numeric_column_for_query(query, columns, profile)
            if not col:
                return None
            slots["column"] = col
        if intent_id == "filtered_metric_aggregation":
            dtypes = (profile or {}).get("dtypes") or {}
            filter_col = slots.get("filter_col")
            filter_value = slots.get("filter_value")
            if not isinstance(filter_col, str) or filter_col not in columns:
                filter_col = None
            if filter_col and self._is_numeric_dtype(dtypes.get(filter_col, "")):
                filter_col = None
            if not filter_col:
                filter_col = self._best_filter_column_for_value(
                    query=query,
                    value=filter_value,
                    columns=columns,
                    profile=profile,
                    intent_id=intent_id,
                )
            if not filter_col:
                return None
            slots["filter_col"] = filter_col

            target_col = slots.get("target_col")
            if not isinstance(target_col, str) or target_col not in columns:
                target_col = None
            if target_col and not self._is_numeric_dtype(dtypes.get(target_col, "")):
                target_col = None
            if not target_col:
                cfg = (intent.get("slots") or {}).get("target_col") or {}
                llm_col = self._slot_from_llm(intent, "target_col", cfg, query, columns, profile) if self._llm_json else None
                if isinstance(llm_col, str) and llm_col in columns and self._is_numeric_dtype(dtypes.get(llm_col, "")):
                    target_col = llm_col
                else:
                    target_col = self._best_numeric_column_for_query(query, columns, profile)
            if not target_col:
                return None
            slots["target_col"] = target_col
        return slots

    def _resolve_intent_and_slots(
        self,
        intent: Dict[str, Any],
        query: str,
        columns: List[str],
        profile: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        intent_id = str(intent.get("id") or "")
        if intent_id not in {"groupby_count", "groupby_agg"}:
            return intent, {}
        llm_slots = self._llm_groupby_slots(query, columns, profile)
        if not llm_slots:
            return intent, {}

        if self._is_group_revenue_product_query(query):
            resolved_intent = self._intents.get("groupby_agg") or intent
            preset = self._build_group_revenue_slots(query, columns, profile, llm_slots)
            if preset:
                logging.info(
                    "event=shortcut_router_groupby_redirect reason=revenue_product slots=%s",
                    json.dumps(preset, ensure_ascii=False),
                )
                return resolved_intent, preset

        agg = str(llm_slots.get("agg") or "").strip().lower()
        if self._is_per_item_normalization_query(query) and not self._has_explicit_grouping_cue(query):
            stats_intent = self._intents.get("stats_aggregation")
            target_col = llm_slots.get("target_col")
            if not (isinstance(target_col, str) and target_col in columns):
                target_col = self._best_numeric_column_for_query(query, columns, profile)
            metric = agg if agg in {"sum", "mean", "min", "max", "median"} else "mean"
            if isinstance(stats_intent, dict) and target_col:
                preset = {"column": target_col, "metric": metric}
                logging.info(
                    "event=shortcut_router_groupby_redirect reason=per_item_normalization metric=%s column=%s",
                    metric,
                    target_col,
                )
                return stats_intent, preset
        if self._is_group_total_quantity_query(query):
            qty_col = self._best_numeric_column_for_query("кількість qty quantity units stock", columns, profile)
            if qty_col:
                agg = "sum"
                llm_slots["agg"] = "sum"
                llm_slots["target_col"] = qty_col
        resolved_intent = intent
        if agg and agg != "count" and "groupby_agg" in self._intents:
            resolved_intent = self._intents["groupby_agg"]
        elif agg == "count" and "groupby_count" in self._intents:
            resolved_intent = self._intents["groupby_count"]

        preset: Dict[str, Any] = {}
        group_col = llm_slots.get("group_col")
        target_col = llm_slots.get("target_col")
        top_n = llm_slots.get("top_n")
        if isinstance(group_col, str) and group_col in columns:
            preset["group_col"] = group_col
        if isinstance(target_col, str) and target_col in columns:
            preset["target_col"] = target_col
        if agg in {"sum", "mean", "min", "max"}:
            preset["agg"] = agg
        if isinstance(top_n, int) and top_n > 0:
            preset["top_n"] = top_n
        if "top_n" not in preset:
            parsed_top = self._extract_top_n(query)
            if parsed_top and parsed_top > 0:
                preset["top_n"] = parsed_top
        logging.info(
            "event=shortcut_router_groupby_resolve intent_from=%s intent_to=%s slots=%s",
            intent_id,
            str(resolved_intent.get("id") or ""),
            json.dumps(preset, ensure_ascii=False),
        )
        return resolved_intent, preset

    def _is_group_revenue_product_query(self, query: str) -> bool:
        q = (query or "").lower()
        if not q:
            return False
        has_group = bool(
            re.search(
                r"\b(group\s*by|by\s+\w+|по\s+\w+|за\s+\w+)\b"
                r"|(?:по|за)\s+(?:категор\w*|бренд\w*|модел\w*|тип\w*|груп\w*)",
                q,
                re.I,
            )
        )
        if not has_group:
            return False
        has_revenue_like = bool(
            re.search(r"\b(вируч\w*|revenue|дохід\w*|оборот\w*|sales|gmv|варт\w*)\b", q, re.I)
        )
        has_mul = bool(("×" in q) or ("*" in q) or re.search(r"\b(x|mul|помнож|добут)\w*\b", q, re.I))
        has_price = bool(re.search(r"\b(ціна|price|cost|amount)\w*\b", q, re.I))
        has_qty = bool(re.search(r"\b(кільк\w*|qty|quantity|units?|штук|одиниц\w*)\b", q, re.I))
        return has_group and (has_revenue_like or (has_mul and has_price and has_qty))

    def _pick_price_like_column(self, columns: List[str], profile: Dict[str, Any]) -> Optional[str]:
        dtypes = (profile or {}).get("dtypes") or {}
        numeric_cols = [c for c in columns if self._is_numeric_dtype(dtypes.get(c, ""))]
        for c in numeric_cols:
            if re.search(r"(цін|price|варт|cost|amount|revenue|вируч)", str(c).lower()):
                return c
        return None

    def _pick_qty_like_column(self, columns: List[str], profile: Dict[str, Any]) -> Optional[str]:
        dtypes = (profile or {}).get("dtypes") or {}
        numeric_cols = [c for c in columns if self._is_numeric_dtype(dtypes.get(c, ""))]
        for c in numeric_cols:
            if re.search(r"(кільк|qty|quantity|units|stock|залишк|штук)", str(c).lower()):
                return c
        return None

    def _pick_group_like_column(self, query: str, columns: List[str], profile: Dict[str, Any]) -> Optional[str]:
        # Prefer explicit semantic dimensions commonly used for grouped reports.
        priority = ("Категорія", "Бренд", "Модель", "Статус")
        for p in priority:
            if p in columns and re.search(re.escape(p.lower()[:5]), (query or "").lower()):
                return p
        for p in priority:
            if p in columns:
                return p
        return self._best_column_match(query, self._text_columns(columns, profile))

    def _build_group_revenue_slots(
        self,
        query: str,
        columns: List[str],
        profile: Dict[str, Any],
        llm_slots: Dict[str, Any],
    ) -> Dict[str, Any]:
        group_col = llm_slots.get("group_col") if isinstance(llm_slots.get("group_col"), str) else None
        if not group_col or group_col not in columns:
            group_col = self._pick_group_like_column(query, columns, profile)
        price_col = self._pick_price_like_column(columns, profile)
        qty_col = self._pick_qty_like_column(columns, profile)
        if not group_col or not price_col or not qty_col:
            return {}
        return {
            "group_col": group_col,
            "target_col": price_col,  # keep compatibility with groupby_agg intent contract
            "agg": "sum",
            "mul_left_col": price_col,
            "mul_right_col": qty_col,
            "out_col": "Виручка_UAH",
        }

    def _has_explicit_grouping_cue(self, query: str) -> bool:
        q = (query or "").lower()
        if self._is_per_item_normalization_query(q):
            # "per item / на один товар" is normalization, not group-by.
            return False
        return bool(
            re.search(
                r"\b(group\s*by|by\s+\w+)\b"
                r"|\bper\s+(?:category|categories|brand|brands|model|models|type|types|status|month|year|day)\b"
                r"|(?:по|за)\s+(?:категор\w*|бренд\w*|модел\w*|тип\w*|груп\w*)",
                q,
                re.I,
            )
        )

    def _is_per_item_normalization_query(self, query: str) -> bool:
        q = (query or "").lower()
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
            "category",
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

    def _is_group_total_quantity_query(self, query: str) -> bool:
        q = (query or "").lower()
        if not q:
            return False
        has_group = bool(
            re.search(
                r"\b(group\s*by|per\s+\w+|by\s+\w+|по\s+\w+|за\s+\w+)\b"
                r"|(?:по|за)\s+(?:категор\w*|бренд\w*|модел\w*|тип\w*|груп\w*)",
                q,
                re.I,
            )
        )
        if not has_group:
            return False
        has_qty = bool(re.search(r"\b(кільк\w*|штук|одиниц\w*|qty|quantity|units?)\b", q, re.I))
        if not has_qty:
            return False
        has_totalish = bool(re.search(r"\b(загальн\w*|total|sum|сума|на\s+склад\w*|наявн\w*|залишк\w*)\b", q, re.I))
        return has_totalish

    def _llm_groupby_slots(self, query: str, columns: List[str], profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self._llm_json or not columns:
            return None
        system = (
            "You map spreadsheet group-by questions to execution slots. "
            "Return ONLY JSON with keys: group_col, target_col, agg, top_n. "
            "group_col and target_col must be exact names from columns or empty string. "
            "agg must be one of: count, sum, mean, min, max. "
            "Use agg='count' when user asks to count records per group. "
            "Use agg='sum' when user asks total quantity/amount per group."
        )
        payload = {
            "query": query,
            "columns": columns[:200],
            "dtypes": (profile or {}).get("dtypes") or {},
            "rows": (profile or {}).get("rows"),
            "preview": (profile or {}).get("preview") or [],
        }
        try:
            res = self._llm_json(system, json.dumps(payload, ensure_ascii=False))
        except Exception:
            return None
        if not isinstance(res, dict):
            return None
        out: Dict[str, Any] = {}
        group_col = res.get("group_col")
        target_col = res.get("target_col")
        agg = str(res.get("agg") or "").strip().lower()
        top_n = res.get("top_n")
        if isinstance(group_col, str):
            out["group_col"] = group_col.strip()
        if isinstance(target_col, str):
            out["target_col"] = target_col.strip()
        if agg in {"count", "sum", "mean", "min", "max"}:
            out["agg"] = agg
        if isinstance(top_n, (int, float)):
            out["top_n"] = int(top_n)
        elif isinstance(top_n, str) and top_n.strip().isdigit():
            out["top_n"] = int(top_n.strip())
        return out or None

    def _extract_top_n(self, query: str) -> int:
        q = (query or "").lower()
        m = re.search(r"(?:top|топ)\s*[-–]?\s*(\d+)", q, re.I)
        if m:
            return int(m.group(1))
        m = re.search(r"\b(\d+)\b", q)
        if m and re.search(r"(?:top|топ)", q, re.I):
            return int(m.group(1))
        return 0

    def _question_has_metric_cue(self, question: str) -> bool:
        q_low = (question or "").lower()
        return bool(
            re.search(
                r"\b("
                r"max(?:imum)?|мін(?:імум|імаль\w*)?|minimum|min|"
                r"mean|average|avg|median|sum|total|count|"
                r"макс\w*|середн\w*|сума|підсум\w*|"
                r"кільк\w*|скільк\w*"
                r")\b",
                q_low,
                re.I,
            )
        )

    def _question_has_filter_cue(self, question: str) -> bool:
        q_low = (question or "").lower()
        return bool(
            re.search(
                r"\b(серед|among|within|where|де|тільки|only|лише|having)\b"
                r"|(?:що|які)\s+мають"
                r"|(?:with|for)\s+[a-zа-яіїєґ0-9_]{2,}",
                q_low,
                re.I,
            )
        )

    def _question_mentions_preview_value(self, question: str, profile: Dict[str, Any]) -> bool:
        q_low = (question or "").lower()
        if not q_low:
            return False
        q_tokens = [
            tok for tok in re.split(r"[^a-zа-яіїєґ0-9]+", q_low) if len(tok) >= 3
        ]
        if not q_tokens:
            return False
        q_roots = {tok[:5] for tok in q_tokens}
        stop_roots = {
            "скіл",
            "кіль",
            "макс",
            "мін",
            "сере",
            "сума",
            "poun",
            "coun",
            "tabl",
            "рядк",
            "коло",
            "това",
            "дани",
        }
        preview = (profile or {}).get("preview") or []
        for row in preview[:60]:
            if not isinstance(row, dict):
                continue
            for val in row.values():
                if val is None:
                    continue
                sval = str(val).strip().lower()
                if len(sval) < 3 or len(sval) > 60:
                    continue
                if re.fullmatch(r"[\d\s.,:%\-]+", sval):
                    continue
                parts = re.split(r"[^a-zа-яіїєґ0-9]+", sval)
                for part in parts:
                    if len(part) < 3:
                        continue
                    root = part[:5]
                    if root in stop_roots:
                        continue
                    if root in q_roots:
                        return True
        return False

    def _has_filter_context(self, question: str, profile: Dict[str, Any]) -> bool:
        """
        Detects metric questions asked within a filtered subset:
        e.g. "max price among mice", "avg amount for red items".
        """
        if not self._question_has_metric_cue(question):
            return False
        if self._question_has_filter_cue(question):
            return True
        return self._question_mentions_preview_value(question, profile)

    def _assess_query_complexity(self, question: str, profile: Dict[str, Any]) -> float:
        q_low = (question or "").lower()
        score = 0.0

        if self._question_has_filter_cue(question):
            score += 0.3
        if self._question_mentions_preview_value(question, profile):
            score += 0.2
        if re.search(r"\b(та|і|and|or|або)\b", q_low, re.I):
            score += 0.2

        columns = [str(c).lower() for c in ((profile or {}).get("columns") or [])]
        mentioned_cols = 0
        for col in columns:
            col = col.strip()
            if len(col) < 3:
                continue
            if col in q_low:
                mentioned_cols += 1
        if mentioned_cols >= 2:
            score += 0.2

        has_grouping = bool(re.search(r"\b(топ|top|кожн\w*|по\s+\w+|group\w*)\b", q_low, re.I))
        has_metric = self._question_has_metric_cue(question)
        has_filter = self._question_has_filter_cue(question) or self._question_mentions_preview_value(question, profile)
        if has_grouping and has_metric and has_filter:
            score += 0.3

        return min(1.0, score)

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
        if "mean" in enums and re.search(r"\b(average|avg|mean|середн\w*)\b", q_low):
            return enums["mean"]
        if "sum" in enums and re.search(r"\b(sum|сума|total)\b", q_low):
            return enums["sum"]
        if "min" in enums and re.search(r"\b(min(?:imum)?|мін\w*)\b", q_low):
            return enums["min"]
        if "max" in enums and re.search(r"\b(max(?:imum)?|макс\w*)\b", q_low):
            return enums["max"]
        if "median" in enums and re.search(r"\b(median|медіан\w*)\b", q_low):
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

    def _is_numeric_dtype(self, dtype: Any) -> bool:
        return str(dtype or "").lower().startswith(("int", "float", "uint"))

    def _text_columns(self, columns: List[str], profile: Dict[str, Any]) -> List[str]:
        dtypes = (profile or {}).get("dtypes") or {}
        text_cols = [c for c in columns if not self._is_numeric_dtype(dtypes.get(c, ""))]
        return text_cols or list(columns)

    def _extract_filter_value_from_query(self, query: str) -> Optional[str]:
        q = (query or "").strip()
        if not q:
            return None
        m = re.search(r"['\"]([^'\"]+)['\"]", q)
        if m:
            val = (m.group(1) or "").strip()
            return val or None

        patterns = [
            r"(?:містить|contains)\s+(.+)$",
            r"(?:дорівнює|equals?|=)\s+(.+)$",
        ]
        for pat in patterns:
            m = re.search(pat, q, re.I)
            if not m:
                continue
            val = (m.group(1) or "").strip()
            if not val:
                continue
            val = re.split(r"\s+(?:зі|з|де|where|та|і|and|with|у|в)\b", val, maxsplit=1, flags=re.I)[0]
            val = val.strip().strip("\"'.,;:?!()[]{}")
            if val:
                return val

        # Universal fallback: take the last semantically meaningful token.
        tokens_raw = [t for t in re.split(r"[^A-Za-zА-Яа-яІіЇїЄєҐґ0-9]+", q) if t]
        if not tokens_raw:
            return None
        stop_roots = {
            "скіл",
            "кіль",
            "count",
            "coun",
            "sum",
            "сума",
            "mean",
            "aver",
            "avg",
            "medi",
            "меді",
            "max",
            "макс",
            "min",
            "мін",
            "ціна",
            "цін",
            "варт",
            "price",
            "cost",
            "total",
            "това",
            "item",
            "prod",
            "стат",
            "stat",
            "наяв",
            "avail",
            "stock",
            "скла",
            "where",
            "де",
            "with",
            "for",
            "and",
            "та",
            "і",
        }
        for tok_raw in reversed(tokens_raw):
            tok = tok_raw.lower()
            if len(tok) < 3:
                continue
            if re.fullmatch(r"\d+", tok):
                continue
            if tok[:4] in stop_roots:
                continue
            return tok_raw
        return None

    def _best_filter_column_from_preview(
        self,
        value: Any,
        columns: List[str],
        profile: Dict[str, Any],
    ) -> Optional[str]:
        preview = (profile or {}).get("preview") or []
        if not isinstance(preview, list) or not preview:
            return None
        val = self._normalize_text(str(value or ""))
        if not val:
            return None
        val_tokens = [t for t in val.split() if len(t) >= 2]
        if not val_tokens:
            return None

        best_col: Optional[str] = None
        best_score = 0.0
        for col in columns:
            hits_exact = 0
            hits_fuzzy = 0
            non_empty = 0
            for row in preview[:200]:
                if not isinstance(row, dict):
                    continue
                cell = row.get(col)
                if cell is None:
                    continue
                cell_norm = self._normalize_text(str(cell))
                if not cell_norm:
                    continue
                non_empty += 1
                if val in cell_norm:
                    hits_exact += 1
                    continue
                for tok in val_tokens:
                    root_len = 3 if re.search(r"[а-яіїєґ]", tok, re.I) else 4
                    root = tok[: max(2, min(root_len, len(tok)))]
                    if root and root in cell_norm:
                        hits_fuzzy += 1
                        break

            if non_empty == 0:
                continue
            support = hits_exact + hits_fuzzy
            if support <= 0:
                continue
            density = ((3.0 * hits_exact) + hits_fuzzy) / float(non_empty)
            confidence = min(1.0, support / 3.0)
            score = density + confidence
            if score > best_score:
                best_score = score
                best_col = col
        if best_score <= 0:
            return None
        return best_col

    def _llm_pick_filter_column_for_value(
        self,
        query: str,
        value: Any,
        columns: List[str],
        profile: Dict[str, Any],
        intent_id: str,
    ) -> Optional[str]:
        if not self._llm_json or not columns:
            return None
        value_text = str(value or "").strip()
        if not value_text:
            return None
        system = (
            "Pick the single best column to apply a text contains filter. "
            "Return ONLY JSON: {\"column\": \"<exact column name from list or empty>\"}. "
            "Use query meaning, dtypes and preview values. "
            "Choose the column where filter_value most likely appears in cell values."
        )
        payload = {
            "intent_id": intent_id,
            "query": query,
            "filter_value": value_text,
            "columns": columns[:200],
            "dtypes": (profile or {}).get("dtypes") or {},
            "preview": (profile or {}).get("preview") or [],
        }
        try:
            parsed = self._llm_json(system, json.dumps(payload, ensure_ascii=False))
        except Exception:
            return None
        col = str((parsed or {}).get("column") or "").strip()
        if col in columns:
            return col
        return None

    def _best_filter_column_for_value(
        self,
        query: str,
        value: Any,
        columns: List[str],
        profile: Dict[str, Any],
        intent_id: str = "filter_contains",
    ) -> Optional[str]:
        if not columns:
            return None
        text_cols = self._text_columns(columns, profile)
        value_text = str(value or "").strip()

        if value_text:
            try:
                m = re.search(re.escape(value_text), query or "", re.I)
            except Exception:
                m = None
            if m:
                left_ctx = (query or "")[max(0, m.start() - 80) : m.start()]
                ctx_col = self._best_column_match(left_ctx, text_cols)
                if ctx_col:
                    logging.info(
                        "event=shortcut_router_filter_column_resolve source=value_context intent_id=%s column=%s value=%s",
                        intent_id,
                        ctx_col,
                        value_text[:80],
                    )
                    return ctx_col

            preview_col = self._best_filter_column_from_preview(value_text, text_cols, profile)
            if preview_col:
                logging.info(
                    "event=shortcut_router_filter_column_resolve source=preview intent_id=%s column=%s value=%s",
                    intent_id,
                    preview_col,
                    value_text[:80],
                )
                return preview_col

            llm_col = self._llm_pick_filter_column_for_value(
                query=query,
                value=value_text,
                columns=text_cols,
                profile=profile,
                intent_id=intent_id,
            )
            if llm_col:
                logging.info(
                    "event=shortcut_router_filter_column_resolve source=llm intent_id=%s column=%s value=%s",
                    intent_id,
                    llm_col,
                    value_text[:80],
                )
                return llm_col

        # Value-first fallback (less sensitive to metric words in full query).
        if value_text:
            col = self._best_column_match(value_text, text_cols)
            if col:
                logging.info(
                    "event=shortcut_router_filter_column_resolve source=value_text intent_id=%s column=%s",
                    intent_id,
                    col,
                )
                return col

        col = self._best_column_match(query, text_cols)
        if col:
            logging.info(
                "event=shortcut_router_filter_column_resolve source=query_text intent_id=%s column=%s",
                intent_id,
                col,
            )
            return col
        return None

    def _best_numeric_column_for_query(self, query: str, columns: List[str], profile: Dict[str, Any]) -> Optional[str]:
        dtypes = (profile or {}).get("dtypes") or {}
        numeric_cols = [c for c in columns if self._is_numeric_dtype(dtypes.get(c, ""))]
        if not numeric_cols:
            return None
        q_low = (query or "").lower()
        price_pref = [
            c for c in numeric_cols
            if re.search(r"(цін|price|варт|cost|amount|total|revenue|вируч)", str(c).lower())
        ]
        qty_pref = [
            c for c in numeric_cols
            if re.search(r"(кільк|qty|quantity|units|stock|залишк)", str(c).lower())
        ]
        if re.search(r"\b(середн|average|avg|mean|min|max|median|сума|sum|варт|price|cost|вируч)\b", q_low):
            if price_pref:
                return price_pref[0]
        if re.search(r"\b(кільк|qty|quantity|units|stock|залишк)\b", q_low):
            if qty_pref:
                return qty_pref[0]
        non_id = [
            c for c in numeric_cols
            if not re.search(r"(?:^|_)(id|sku|код|артикул)(?:$|_)", str(c).lower())
        ]
        if non_id:
            return non_id[0]
        return numeric_cols[0]

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
                    code_lines.append("result = df.isna().sum()")
                    if top_n > 0:
                        code_lines.append(f"result = result.sort_values(ascending=False).head({top_n})")
                    code_lines.append("result = result.to_dict()")
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
                code_lines.append(f"_col_name = {col!r}")
                code_lines.append(f"_metric = {metric!r}")
                code_lines.append("if _col_name not in df.columns:")
                code_lines.append("    result = None")
                code_lines.append("else:")
                code_lines.append("    _col = df[_col_name]")
                code_lines.append("    _dtype = str(_col.dtype).lower()")
                code_lines.append("    if _dtype.startswith(('int', 'float', 'uint')):")
                code_lines.append("        _num = _col.astype(float)")
                code_lines.append("    else:")
                code_lines.append("        _raw = _col.astype(str)")
                code_lines.append(
                    r"        _clean = _raw.str.replace(r'[\s\xa0]', '', regex=True).str.replace(r'[^0-9,.\-]', '', regex=True).str.replace(',', '.')"
                )
                code_lines.append(r"        _mask = _clean.str.match(r'^-?(\d+(\.\d*)?|\.\d+)$')")
                code_lines.append("        _num = _clean.where(_mask, np.nan).astype(float)")
                code_lines.append("    if _metric == 'mean':")
                code_lines.append("        _v = _num.mean()")
                code_lines.append("    elif _metric == 'min':")
                code_lines.append("        _v = _num.min()")
                code_lines.append("    elif _metric == 'max':")
                code_lines.append("        _v = _num.max()")
                code_lines.append("    elif _metric == 'median':")
                code_lines.append("        _v = _num.median()")
                code_lines.append("    else:")
                code_lines.append("        _v = _num.sum()")
                code_lines.append("    result = None if pd.isna(_v) else float(_v)")
            elif op == "groupby_count":
                group_col = slots.get("group_col")
                out_col = slots.get("out_col") or "count"
                sort_mode = str(slots.get("sort") or "desc").strip().lower()
                top_n = int(slots.get("top_n") or 0)
                if not group_col:
                    return None
                code_lines.append(
                    f"result = df.groupby({group_col!r}).size().reset_index(name={out_col!r})"
                )
                if sort_mode != "none":
                    asc = "True" if sort_mode == "asc" else "False"
                    code_lines.append(f"result = result.sort_values({out_col!r}, ascending={asc})")
                if top_n > 0:
                    code_lines.append(f"result = result.head({top_n})")
            elif op == "groupby_agg":
                group_col = slots.get("group_col")
                target_col = slots.get("target_col")
                agg = slots.get("agg") or "sum"
                top_n = int(slots.get("top_n") or 0)
                out_col = slots.get("out_col")
                mul_left_col = slots.get("mul_left_col")
                mul_right_col = slots.get("mul_right_col")
                if not out_col:
                    out_col = str(target_col) if target_col else "value"
                if not group_col or not target_col:
                    return None
                if (
                    isinstance(mul_left_col, str)
                    and isinstance(mul_right_col, str)
                    and agg == "sum"
                ):
                    code_lines.append(f"_group_col = {group_col!r}")
                    code_lines.append(f"_left_col = {mul_left_col!r}")
                    code_lines.append(f"_right_col = {mul_right_col!r}")
                    code_lines.append("_ok = (_group_col in df.columns) and (_left_col in df.columns) and (_right_col in df.columns)")
                    code_lines.append("if not _ok:")
                    code_lines.append("    result = []")
                    code_lines.append("else:")
                    code_lines.append("    _left = pd.to_numeric(df[_left_col], errors='coerce')")
                    code_lines.append("    _right = pd.to_numeric(df[_right_col], errors='coerce')")
                    code_lines.append("    _work = df.copy(deep=False)")
                    code_lines.append("    _work['_metric'] = _left * _right")
                    code_lines.append("    _work = _work.loc[_work['_metric'].notna()].copy()")
                    code_lines.append(
                        f"    result = _work.groupby(_group_col)['_metric'].sum().reset_index(name={out_col!r})"
                    )
                else:
                    code_lines.append(
                        f"result = df.groupby({group_col!r})[{target_col!r}].agg({agg!r}).reset_index(name={out_col!r})"
                    )
                code_lines.append(f"result = result.sort_values({out_col!r}, ascending=False)")
                if top_n > 0:
                    code_lines.append(f"result = result.head({top_n})")
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
            elif op == "filter_contains_aggregation":
                filter_col = slots.get("filter_col")
                filter_value = slots.get("filter_value")
                target_col = slots.get("target_col")
                metric = slots.get("metric") or "max"
                if not filter_col or not target_col:
                    return None
                code_lines.append(f"_filter_col = {filter_col!r}")
                code_lines.append(f"_filter_value = {filter_value!r}")
                code_lines.append(f"_target_col = {target_col!r}")
                code_lines.append(f"_metric = {metric!r}")
                code_lines.append("if _filter_col not in df.columns or _target_col not in df.columns:")
                code_lines.append("    result = None")
                code_lines.append("else:")
                code_lines.append("    _work = df[df[_filter_col].astype(str).str.contains(str(_filter_value), case=False, na=False)].copy()")
                code_lines.append("    _raw = _work[_target_col].astype(str)")
                code_lines.append(
                    r"    _clean = _raw.str.replace(r'[\s\xa0]', '', regex=True).str.replace(r'[^0-9,.\-]', '', regex=True).str.replace(',', '.')"
                )
                code_lines.append(r"    _mask = _clean.str.match(r'^-?(\d+(\.\d*)?|\.\d+)$')")
                code_lines.append("    _num = _clean.where(_mask, np.nan).astype(float)")
                code_lines.append("    if _metric == 'mean':")
                code_lines.append("        _v = _num.mean()")
                code_lines.append("    elif _metric == 'min':")
                code_lines.append("        _v = _num.min()")
                code_lines.append("    elif _metric == 'max':")
                code_lines.append("        _v = _num.max()")
                code_lines.append("    elif _metric == 'median':")
                code_lines.append("        _v = _num.median()")
                code_lines.append("    else:")
                code_lines.append("        _v = _num.sum()")
                code_lines.append("    result = None if pd.isna(_v) else float(_v)")
            elif op == "keyword_search_count":
                keyword = slots.get("keyword")
                if keyword is None:
                    return None
                code_lines.append(f"_kw = str({keyword!r}).strip()")
                code_lines.append("if not _kw:")
                code_lines.append("    result = 0")
                code_lines.append("else:")
                code_lines.append("    _text_cols = [c for c in df.columns if not str(df[c].dtype).lower().startswith(('int', 'float', 'uint'))]")
                code_lines.append("    if _text_cols:")
                code_lines.append("        _text = df[_text_cols].fillna('').astype(str).apply(' '.join, axis=1)")
                code_lines.append("    else:")
                code_lines.append("        _text = df.fillna('').astype(str).apply(' '.join, axis=1)")
                code_lines.append("    result = int(_text.str.contains(_kw, case=False, na=False).sum())")
            elif op == "keyword_search_rows":
                keyword = slots.get("keyword")
                top_n = int(slots.get("top_n") or 20)
                if keyword is None:
                    return None
                code_lines.append(f"_kw = str({keyword!r}).strip()")
                code_lines.append(f"_top_n = {top_n}")
                code_lines.append("if not _kw:")
                code_lines.append("    result = []")
                code_lines.append("else:")
                code_lines.append("    _text_cols = [c for c in df.columns if not str(df[c].dtype).lower().startswith(('int', 'float', 'uint'))]")
                code_lines.append("    if _text_cols:")
                code_lines.append("        _text = df[_text_cols].fillna('').astype(str).apply(' '.join, axis=1)")
                code_lines.append("    else:")
                code_lines.append("        _text = df.fillna('').astype(str).apply(' '.join, axis=1)")
                code_lines.append("    _mask = _text.str.contains(_kw, case=False, na=False)")
                code_lines.append("    result = df.loc[_mask].head(_top_n)")
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
                code_lines.append("_invalid_idx = [i for i in _idx_1b if i <= 0]")
                code_lines.append("if _invalid_idx: raise ValueError('row indices must be positive 1-based integers')")
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
                include_index = bool(slots.get("include_index", False))
                code_lines.append(f"result = df.to_csv(index={include_index})")
            elif op == "return_df_preview":
                rows = int(args.get("rows") or 20)
                code_lines.append(f"result = df.head({rows})")
            else:
                return None
        return "\n".join(code_lines).strip() + "\n"
