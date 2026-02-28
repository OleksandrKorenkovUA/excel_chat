import logging
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
from pipelines.lib.query_signals import has_router_filter_context_cue, has_router_metric_cue
from pipelines.lib.route_trace import current_route_tracer

try:  # pragma: no cover - runtime guard
    import faiss  # type: ignore
except Exception:  # pragma: no cover - runtime guard
    faiss = None

_MULTI_FILTER_ALLOWED_OPERATORS = {"equals", "contains", "gt", "lt", "gte", "lte"}
_LLM_INTENT_OVERRIDE_MIN_TOP_SCORE = 0.45
_LLM_INTENT_OVERRIDE_MIN_GAP = 0.12
_LLM_INTENT_OVERRIDE_MIN_LEAD = 0.12


def _safe_trunc(value: Any, limit: int = 300) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


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

    def _bool_config(self, name: str, env_name: str, default: bool) -> bool:
        raw = getattr(self.config, name, None)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        env_raw = os.getenv(env_name, "1" if default else "0")
        return str(env_raw).strip().lower() in {"1", "true", "yes", "on"}

    def _int_config(self, name: str, env_name: str, default: int) -> int:
        raw = getattr(self.config, name, None)
        if raw is None:
            raw = os.getenv(env_name, str(default))
        try:
            val = int(raw)
        except Exception:
            val = int(default)
        return max(0, val)

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

        normalized_query = self._llm_normalize_query_for_retrieval(q, profile)
        if normalized_query:
            logging.info(
                "event=shortcut_router_query_normalized status=ok original_preview=%s normalized_preview=%s",
                q[:200],
                normalized_query[:200],
            )
        else:
            logging.info(
                "event=shortcut_router_query_normalized status=skip query_preview=%s",
                q[:200],
            )

        query_ir = self._extract_query_ir(q, normalized_query, profile)

        candidates: List[Dict[str, Any]] = []
        retrieval_query_used = q
        try:
            candidates = self._retrieve_candidates(q)
        except Exception as exc:
            logging.warning("event=shortcut_router_retrieval status=error error=%s", exc)
            return None

        if not candidates and normalized_query and normalized_query != q:
            logging.info(
                "event=shortcut_router_retrieval status=retry_with_normalized reason=miss original_preview=%s normalized_preview=%s",
                q[:200],
                normalized_query[:200],
            )
            try:
                normalized_candidates = self._retrieve_candidates(normalized_query)
            except Exception as exc:
                logging.warning("event=shortcut_router_retrieval status=error_normalized error=%s", exc)
                normalized_candidates = []
            if normalized_candidates:
                candidates = normalized_candidates
                retrieval_query_used = normalized_query

        if not candidates:
            logging.info("event=shortcut_router_retrieval status=miss query_preview=%s", q[:300])
            return None

        pick = self._pick_intent_from_candidates(retrieval_query_used, profile, candidates)
        if not pick and normalized_query and normalized_query not in {q, retrieval_query_used}:
            logging.info(
                "event=shortcut_router_retrieval status=retry_with_normalized reason=below_threshold original_preview=%s normalized_preview=%s",
                q[:200],
                normalized_query[:200],
            )
            try:
                normalized_candidates = self._retrieve_candidates(normalized_query)
            except Exception as exc:
                logging.warning("event=shortcut_router_retrieval status=error_normalized error=%s", exc)
                normalized_candidates = []
            if normalized_candidates:
                maybe_pick = self._pick_intent_from_candidates(normalized_query, profile, normalized_candidates)
                if maybe_pick:
                    pick = maybe_pick
                    candidates = normalized_candidates
                    retrieval_query_used = normalized_query

        if not pick:
            logging.info("event=shortcut_router_retrieval status=below_threshold query_preview=%s", q[:300])
            return None

        retrieval_top_k = max(1, int(getattr(self.config, "top_k", 5) or 5))
        retrieval_threshold = float(getattr(self.config, "threshold", 0.35))
        retrieval_margin = float(getattr(self.config, "margin", 0.05))
        retrieval_candidates = self._compact_retrieval_candidates(candidates, limit=retrieval_top_k)
        retrieval_top_score = float(candidates[0].get("score") or 0.0) if candidates else None
        retrieval_second_score = float(candidates[1].get("score") or 0.0) if len(candidates) > 1 else None

        columns = [str(c) for c in (profile or {}).get("columns") or []]
        attempts = self._build_intent_attempts(pick, candidates)
        selected_code: Optional[str] = None
        selected_meta: Optional[Dict[str, Any]] = None
        selected_intent_id = str(pick.get("intent_id") or "").strip()

        for attempt_idx, attempt in enumerate(attempts):
            attempt_intent_id = str(attempt.get("intent_id") or "").strip()
            attempt_score = float(attempt.get("retrieval_score") or 0.0)
            attempt_mode = str(attempt.get("selector_mode") or "retrieval")
            logging.info(
                "event=shortcut_router_candidate_attempt idx=%d intent_id=%s score=%.4f mode=%s",
                attempt_idx,
                attempt_intent_id,
                attempt_score,
                attempt_mode,
            )
            built = self._try_compile_intent_attempt(
                attempt=attempt,
                query=q,
                columns=columns,
                profile=profile,
                query_ir=query_ir,
            )
            if not built:
                continue
            selected_code = built["code"]
            selected_meta = {
                "intent_id": built["intent_id"],
                "score": attempt_score,
                "example": str(attempt.get("example") or ""),
                "slots": built["slots"],
                "selector_confidence": float(attempt.get("confidence") or 0.0),
                "selector_mode": attempt_mode if attempt_idx == 0 else f"{attempt_mode}_fallback",
                "retrieval_query_used": retrieval_query_used,
                "normalized_query": normalized_query or "",
                "query_ir": query_ir or {},
                "fallback_attempts": attempt_idx,
                "query_ir_summary": self._summarize_query_ir(query_ir),
                "retrieval_threshold": retrieval_threshold,
                "retrieval_margin": retrieval_margin,
                "retrieval_candidate_count": len(candidates),
                "retrieval_candidates": retrieval_candidates,
                "retrieval_top_score": retrieval_top_score,
                "retrieval_second_score": retrieval_second_score,
            }
            if attempt_idx > 0:
                selected_meta["initial_intent_id"] = selected_intent_id
            break

        if selected_code is None or selected_meta is None:
            logging.info(
                "event=shortcut_router status=miss reason=no_candidate_compiled attempted=%s",
                _safe_trunc([str(a.get("intent_id") or "") for a in attempts], 320),
            )
            return None
        return selected_code, selected_meta

    def _compact_retrieval_candidates(self, candidates: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx, cand in enumerate(candidates[: max(1, int(limit))]):
            out.append(
                {
                    "rank": idx + 1,
                    "intent_id": str(cand.get("intent_id") or ""),
                    "score": float(cand.get("score") or 0.0),
                    "example": _safe_trunc(str(cand.get("example") or ""), 220),
                    "doc_ref": str(cand.get("doc_ref") or ""),
                }
            )
        return out

    def _build_intent_attempts(
        self,
        primary_pick: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        attempts: List[Dict[str, Any]] = []
        seen: set[str] = set()

        def _append_attempt(item: Dict[str, Any], fallback_mode: str) -> None:
            intent_id = str(item.get("intent_id") or "").strip()
            if not intent_id or intent_id in seen:
                return
            seen.add(intent_id)
            attempts.append(
                {
                    "intent_id": intent_id,
                    "confidence": float(item.get("confidence") or 0.0),
                    "retrieval_score": float(item.get("retrieval_score") or item.get("score") or 0.0),
                    "example": str(item.get("example") or ""),
                    "selector_mode": str(item.get("selector_mode") or fallback_mode),
                }
            )

        _append_attempt(primary_pick or {}, "primary")
        for cand in candidates:
            _append_attempt(cand or {}, "retrieval")
        return attempts

    def _try_compile_intent_attempt(
        self,
        attempt: Dict[str, Any],
        query: str,
        columns: List[str],
        profile: Dict[str, Any],
        query_ir: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        intent_id = str(attempt.get("intent_id") or "").strip()
        intent = self._intents.get(intent_id)
        if not intent:
            logging.info(
                "event=shortcut_router_candidate_attempt status=reject reason=unknown_intent intent_id=%s",
                intent_id,
            )
            return None
        if bool(getattr(self.config, "query_ir_require_resolved_hard_columns", True)):
            unresolved_hard = self._query_ir_unresolved_hard_conditions(query_ir)
            if unresolved_hard:
                logging.info(
                    "event=shortcut_router_candidate_attempt status=reject reason=hard_condition_unresolved_column intent_id=%s unresolved=%s",
                    intent_id,
                    _safe_trunc(json.dumps(unresolved_hard[:4], ensure_ascii=False), 700),
                )
                return None
        resolved_intent, preset_slots = self._resolve_intent_and_slots(intent, query, columns, profile)
        resolved_intent_id = str(resolved_intent.get("id") or intent_id)
        slots = self._fill_slots(resolved_intent, query, columns, profile, preset_slots=preset_slots)
        if slots is None:
            logging.info(
                "event=shortcut_router_candidate_attempt status=reject reason=missing_or_unresolved intent_id=%s",
                resolved_intent_id,
            )
            return None
        slot_issues = self._validate_slots(resolved_intent, slots, columns, profile)
        if slot_issues:
            logging.info(
                "event=shortcut_router_candidate_attempt status=reject reason=slot_validation intent_id=%s issues=%s",
                resolved_intent_id,
                json.dumps(slot_issues[:10], ensure_ascii=False),
            )
            return None
        is_compatible, incompat_reason = self._intent_slots_compatible_with_query(
            intent_id=resolved_intent_id,
            slots=slots,
            query=query,
            query_ir=query_ir,
        )
        if not is_compatible:
            logging.info(
                "event=shortcut_router_candidate_attempt status=reject reason=%s intent_id=%s",
                incompat_reason,
                resolved_intent_id,
            )
            return None

        verifier_ok, verifier_reason = self._llm_verify_attempt_against_ir(
            query=query,
            intent_id=resolved_intent_id,
            slots=slots,
            query_ir=query_ir,
            columns=columns,
            profile=profile,
        )
        if not verifier_ok:
            logging.info(
                "event=shortcut_router_candidate_attempt status=reject reason=%s intent_id=%s",
                verifier_reason,
                resolved_intent_id,
            )
            return None

        code = self._compile_plan(resolved_intent, slots, profile)
        if not code:
            logging.info(
                "event=shortcut_router_candidate_attempt status=reject reason=compile_failed intent_id=%s",
                resolved_intent_id,
            )
            return None
        logging.info(
            "event=shortcut_router_candidate_attempt status=accept intent_id=%s",
            resolved_intent_id,
        )
        return {
            "intent_id": resolved_intent_id,
            "slots": slots,
            "code": code,
        }

    def _intent_slots_compatible_with_query(
        self,
        intent_id: str,
        slots: Dict[str, Any],
        query: str,
        query_ir: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        caps = self._intent_capabilities(intent_id)
        filter_ops = self._intent_filter_ops(caps)
        metric_types = self._intent_metric_types(caps)
        prefilter_before_groupby = self._intent_prefilter_before_groupby(caps)
        plan_ops = set(self._intent_plan_ops(intent_id))
        is_count_query = self._is_count_query(query)

        # Backward-compatible defaults for legacy intents without capabilities metadata.
        if not filter_ops and intent_id in {"filter_comparison", "filter_count"}:
            filter_ops = {"gt", "lt", "gte", "lte", "equals"}
        if not filter_ops and intent_id == "filter_multi_conditions":
            filter_ops = set(_MULTI_FILTER_ALLOWED_OPERATORS)
        if prefilter_before_groupby is None and intent_id in {"groupby_count", "groupby_agg"}:
            prefilter_before_groupby = True
        if prefilter_before_groupby and not filter_ops and intent_id in {"groupby_count", "groupby_agg"}:
            filter_ops = {"equals", "contains", "startswith", "endswith"}

        if is_count_query:
            has_preview = "return_df_preview" in plan_ops
            is_count_like_intent = bool(
                plan_ops.intersection({"filter_count", "keyword_search_count", "groupby_count"})
            )
            if intent_id == "filter_comparison":
                return False, "constraint_count_query_requires_count_intent"
            if has_preview and not is_count_like_intent:
                return False, "constraint_count_query_preview_only_intent"

        startswith_value = self._extract_startswith_value(query)
        if startswith_value:
            if "startswith" not in filter_ops:
                return False, "constraint_startswith_unsupported_by_capability"
            if str(slots.get("filter_op") or "").strip().lower() != "startswith":
                return False, "constraint_startswith_missing_filter_op"
            if not str(slots.get("filter_value") or "").strip():
                return False, "constraint_startswith_missing_filter_value"

        inferred_cmp_op = self._infer_comparison_operator_from_query(query)
        numeric_value = self._first_numeric_from_text(query)
        if inferred_cmp_op in {"<", ">"} and numeric_value is not None:
            supports_numeric_comparison = bool(filter_ops.intersection({"gt", "lt", "gte", "lte"}))
            if not supports_numeric_comparison:
                return False, "constraint_numeric_comparison_unsupported_by_capability"
            if not self._slots_preserve_numeric_comparison(slots, inferred_cmp_op):
                return False, "constraint_numeric_comparison_not_preserved"

        is_revenue_product_query = self._is_group_revenue_product_query(query)
        should_enforce_revenue = is_revenue_product_query and bool(
            metric_types or intent_id in {"groupby_count", "groupby_agg"}
        )
        if should_enforce_revenue:
            supports_product = "product" in metric_types
            if not supports_product and (not caps and intent_id == "groupby_agg"):
                supports_product = True
            if not supports_product:
                return False, "constraint_revenue_product_unsupported_by_capability"
            if not (slots.get("mul_left_col") and slots.get("mul_right_col")):
                return False, "constraint_revenue_missing_product_slots"

        if bool(getattr(self.config, "query_ir_require_hard_coverage", True)):
            hard_conditions = self._query_ir_conditions(query_ir, "hard_conditions")
            if not self._slots_cover_hard_conditions(slots, hard_conditions, query):
                return False, "constraint_hard_conditions_not_preserved"

        if bool(getattr(self.config, "query_ir_block_soft_promotion", True)):
            soft_conditions = self._query_ir_conditions(query_ir, "soft_conditions")
            if not self._slots_keep_soft_conditions_non_binding(slots, soft_conditions, query, query_ir):
                return False, "constraint_soft_conditions_promoted_to_hard"

        return True, "ok"

    def _intent_plan_ops(self, intent_id: str) -> List[str]:
        intent = self._intents.get(str(intent_id) or "") or {}
        out: List[str] = []
        for step in (intent.get("plan") or []):
            if not isinstance(step, dict):
                continue
            op = str(step.get("op") or "").strip()
            if op:
                out.append(op)
        return out

    def _is_count_query(self, query: str) -> bool:
        q = (query or "").lower()
        if not q:
            return False
        has_count_cue = bool(
            re.search(
                r"\b(скільк\w*|кільк\w*|count|number\s+of|how\s+many|всього)\b",
                q,
                re.I,
            )
        )
        if not has_count_cue:
            return False
        has_non_count_metric = bool(
            re.search(
                r"\b(sum|сума|total|mean|average|avg|середн\w*|median|медіан\w*|"
                r"min(?:imum)?|мін\w*|max(?:imum)?|макс\w*|revenue|вируч\w*|price|ціна|варт\w*)\b",
                q,
                re.I,
            )
        )
        return not has_non_count_metric

    def _intent_capabilities(self, intent_id: str) -> Dict[str, Any]:
        intent = self._intents.get(str(intent_id) or "") or {}
        caps = intent.get("capabilities")
        return caps if isinstance(caps, dict) else {}

    def _intent_filter_ops(self, caps: Dict[str, Any]) -> set[str]:
        raw_ops: Any = caps.get("filters")
        if isinstance(raw_ops, dict):
            raw_ops = raw_ops.get("operators") or raw_ops.get("ops") or []
        if not isinstance(raw_ops, list):
            raw_ops = []
        out: set[str] = set()
        for raw in raw_ops:
            op = str(raw or "").strip().lower()
            if not op:
                continue
            if op in {"startswith", "endswith"}:
                out.add(op)
                continue
            normalized = self._normalize_condition_operator(op)
            if normalized:
                out.add(normalized)
        return out

    def _intent_metric_types(self, caps: Dict[str, Any]) -> set[str]:
        raw = caps.get("metric_types")
        if not isinstance(raw, list):
            return set()
        return {str(v).strip().lower() for v in raw if str(v or "").strip()}

    def _intent_prefilter_before_groupby(self, caps: Dict[str, Any]) -> Optional[bool]:
        val = caps.get("prefilter_before_groupby")
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            v = val.strip().lower()
            if v in {"true", "1", "yes"}:
                return True
            if v in {"false", "0", "no"}:
                return False
        return None

    def _slots_preserve_numeric_comparison(self, slots: Dict[str, Any], inferred_cmp_op: str) -> bool:
        expected_symbols = {"<": {"<", "<="}, ">": {">", ">="}}.get(inferred_cmp_op, set())
        expected_ops = {"<": {"lt", "lte"}, ">": {"gt", "gte"}}.get(inferred_cmp_op, set())

        direct_raw = slots.get("operator")
        if direct_raw is not None:
            symbol = self._comparison_operator_symbol(direct_raw)
            if symbol in expected_symbols:
                return True
            op_name = self._normalize_condition_operator(direct_raw)
            if op_name in expected_ops:
                return True

        filter_raw = slots.get("filter_op")
        if filter_raw is not None:
            op_name = self._normalize_condition_operator(filter_raw)
            if op_name in expected_ops:
                return True

        conditions = slots.get("conditions") or []
        if isinstance(conditions, list):
            for cond in conditions:
                if not isinstance(cond, dict):
                    continue
                op_name = self._normalize_condition_operator(cond.get("operator") or cond.get("op"))
                if op_name in expected_ops:
                    return True
        return False

    def _extract_query_ir(self, query: str, normalized_query: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        if not bool(getattr(self.config, "query_ir_enabled", True)):
            logging.info("event=shortcut_router_query_ir status=disabled")
            return {}
        q = str(query or "").strip()
        if not q:
            return {}
        columns = [str(c) for c in ((profile or {}).get("columns") or [])]
        llm_ir: Dict[str, Any] = {}
        if bool(getattr(self.config, "query_ir_llm_enabled", True)):
            llm_ir = self._llm_extract_query_ir(q, columns, profile)

        hard_conditions = self._normalize_query_ir_conditions(llm_ir.get("hard_conditions"), columns)
        soft_conditions = self._normalize_query_ir_conditions(llm_ir.get("soft_conditions"), columns)

        fallback_conditions = self._extract_multi_conditions_llm(q, columns, profile, min_conditions=1) or []
        for cond in fallback_conditions:
            norm = self._normalize_query_ir_conditions([cond], columns)
            if not norm:
                continue
            c = norm[0]
            if self._is_condition_hard_from_query(q, c):
                if not self._has_condition_signature(hard_conditions, c):
                    hard_conditions.append(c)
            else:
                if not self._has_condition_signature(soft_conditions, c):
                    soft_conditions.append(c)

        inferred_cmp = self._infer_comparison_operator_from_query(q)
        numeric_value = self._first_numeric_from_text(q)
        if inferred_cmp in {"<", ">"} and numeric_value is not None:
            op = "lt" if inferred_cmp == "<" else "gt"
            inferred_cond = {"column": "", "operator": op, "value": float(numeric_value), "source": "rule_numeric"}
            if not self._has_condition_signature(hard_conditions, inferred_cond):
                hard_conditions.append(inferred_cond)

        startswith_value = self._extract_startswith_value(q)
        if startswith_value:
            inferred_cond = {
                "column": "",
                "operator": "startswith",
                "value": startswith_value,
                "source": "rule_startswith",
            }
            if not self._has_condition_signature(hard_conditions, inferred_cond):
                hard_conditions.append(inferred_cond)

        hard_conditions = self._resolve_query_ir_missing_columns(
            query=q,
            conditions=hard_conditions,
            columns=columns,
            profile=profile,
            bucket="hard",
        )

        explicit_multi = bool(llm_ir.get("explicit_multi_filter")) if isinstance(llm_ir, dict) else False
        if not explicit_multi:
            explicit_multi = self._has_explicit_multi_filter_cue(q)
        ir = {
            "hard_conditions": hard_conditions,
            "soft_conditions": soft_conditions,
            "explicit_multi_filter": explicit_multi,
            "normalized_query": normalized_query or "",
        }
        logging.info("event=shortcut_router_query_ir status=ok summary=%s", _safe_trunc(self._summarize_query_ir(ir), 500))
        logging.info(
            "event=shortcut_router_query_ir_full payload=%s",
            _safe_trunc(json.dumps(ir, ensure_ascii=False), 2000),
        )
        return ir

    def _llm_extract_query_ir(self, query: str, columns: List[str], profile: Dict[str, Any]) -> Dict[str, Any]:
        if not self._llm_json:
            return {}
        system = (
            "Extract canonical spreadsheet QueryIR. "
            "Return ONLY JSON object with keys: hard_conditions, soft_conditions, explicit_multi_filter. "
            "hard_conditions/soft_conditions must be arrays of objects: "
            "{\"column\":\"<exact column name or empty>\",\"operator\":\"equals|contains|startswith|endswith|gt|lt|gte|lte\",\"value\":\"<value>\",\"source\":\"llm\"}. "
            "Put only explicitly requested constraints into hard_conditions. "
            "Put inferred hints into soft_conditions."
        )
        payload = {
            "query": query,
            "columns": columns[:200],
            "dtypes": (profile or {}).get("dtypes") or {},
            "preview": ((profile or {}).get("preview") or [])[:20],
        }
        try:
            parsed = self._llm_json(system, json.dumps(payload, ensure_ascii=False))
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _normalize_query_ir_conditions(self, raw_conditions: Any, columns: List[str]) -> List[Dict[str, Any]]:
        if not isinstance(raw_conditions, list):
            return []
        out: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str]] = set()
        for cond in raw_conditions:
            if not isinstance(cond, dict):
                continue
            raw_col = str(cond.get("column") or "").strip()
            col = raw_col if raw_col in columns else ""
            op = str(cond.get("operator") or cond.get("op") or "").strip().lower()
            if op not in {"startswith", "endswith"}:
                op = self._normalize_condition_operator(op)
            if op not in {"equals", "contains", "startswith", "endswith", "gt", "lt", "gte", "lte"}:
                continue
            value = cond.get("value")
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                # Empty string values are non-constraints and can cause false subset guards.
                continue
            signature = (col, op, self._normalize_text(str(value)))
            if signature in seen:
                continue
            seen.add(signature)
            out.append(
                {
                    "column": col,
                    "operator": op,
                    "value": value,
                    "source": str(cond.get("source") or "llm"),
                }
            )
        return out

    def _resolve_query_ir_missing_columns(
        self,
        query: str,
        conditions: List[Dict[str, Any]],
        columns: List[str],
        profile: Dict[str, Any],
        bucket: str = "hard",
    ) -> List[Dict[str, Any]]:
        if not conditions:
            return []
        out: List[Dict[str, Any]] = []
        for cond in conditions:
            if not isinstance(cond, dict):
                continue
            col = str(cond.get("column") or "").strip()
            if col in columns:
                out.append(cond)
                continue
            op = str(cond.get("operator") or "").strip().lower()
            value = cond.get("value")
            resolved_col = self._resolve_query_ir_condition_column(
                query=query,
                operator=op,
                value=value,
                columns=columns,
                profile=profile,
            )
            if resolved_col:
                updated = dict(cond)
                updated["column"] = resolved_col
                source = str(updated.get("source") or "").strip()
                updated["source"] = f"{source}+resolved" if source else "resolved"
                out.append(updated)
                logging.info(
                    "event=shortcut_router_query_ir_column_resolve status=resolved bucket=%s operator=%s value=%s column=%s",
                    bucket,
                    op,
                    _safe_trunc(value, 120),
                    resolved_col,
                )
                continue
            out.append(cond)
            logging.info(
                "event=shortcut_router_query_ir_column_resolve status=unresolved bucket=%s operator=%s value=%s",
                bucket,
                op,
                _safe_trunc(value, 120),
            )
        return out

    def _resolve_query_ir_condition_column(
        self,
        query: str,
        operator: str,
        value: Any,
        columns: List[str],
        profile: Dict[str, Any],
    ) -> Optional[str]:
        if not columns:
            return None
        op = str(operator or "").strip().lower()
        dtypes = (profile or {}).get("dtypes") or {}
        numeric_cols = [c for c in columns if self._is_numeric_dtype(dtypes.get(c, ""))]

        if op in {"gt", "lt", "gte", "lte"}:
            direct = self._best_column_match(query, numeric_cols)
            if direct:
                return direct
            if len(numeric_cols) == 1:
                return numeric_cols[0]
            return self._best_numeric_column_for_query(query, columns, profile)

        if op in {"equals", "contains", "startswith", "endswith"}:
            if self._safe_float(value) is not None and numeric_cols:
                direct = self._best_column_match(query, numeric_cols)
                if direct:
                    return direct
                if len(numeric_cols) == 1:
                    return numeric_cols[0]
            return self._best_filter_column_for_value(
                query=query,
                value=value,
                columns=columns,
                profile=profile,
                intent_id="query_ir",
            )
        return None

    def _has_condition_signature(self, conditions: List[Dict[str, Any]], condition: Dict[str, Any]) -> bool:
        sig = (
            str(condition.get("column") or "").strip(),
            str(condition.get("operator") or "").strip().lower(),
            self._normalize_text(str(condition.get("value") or "")),
        )
        for c in conditions:
            c_sig = (
                str(c.get("column") or "").strip(),
                str(c.get("operator") or "").strip().lower(),
                self._normalize_text(str(c.get("value") or "")),
            )
            if c_sig == sig:
                return True
        return False

    def _is_condition_hard_from_query(self, query: str, condition: Dict[str, Any]) -> bool:
        op = str(condition.get("operator") or "").strip().lower()
        if op in {"gt", "lt", "gte", "lte"}:
            return True
        if op in {"startswith", "endswith"}:
            return True
        return self._condition_value_grounded_in_query(query, condition.get("value"))

    def _condition_value_grounded_in_query(self, query: str, value: Any) -> bool:
        q_norm = self._normalize_text(query or "")
        v_norm = self._normalize_text(str(value or ""))
        if not q_norm or not v_norm:
            return False
        if v_norm in q_norm:
            return True
        v_tokens = [t for t in v_norm.split() if len(t) >= 3]
        if not v_tokens:
            return False
        q_tokens = [t for t in q_norm.split() if len(t) >= 3]
        q_roots = {t[:5] for t in q_tokens}
        overlap = sum(1 for t in v_tokens if t[:5] in q_roots)
        return overlap >= max(1, len(v_tokens))

    def _condition_value_explicitly_mentioned(self, query: str, value: Any) -> bool:
        q_norm = self._normalize_text(query or "")
        v_norm = self._normalize_text(str(value or ""))
        if not q_norm or not v_norm:
            return False
        if v_norm in q_norm:
            return True
        q_tokens = set(q_norm.split())
        v_tokens = [t for t in v_norm.split() if t]
        if not v_tokens:
            return False
        return all(t in q_tokens for t in v_tokens)

    def _query_ir_conditions(self, query_ir: Optional[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
        if not isinstance(query_ir, dict):
            return []
        raw = query_ir.get(key)
        if not isinstance(raw, list):
            return []
        out: List[Dict[str, Any]] = []
        for cond in raw:
            if not isinstance(cond, dict):
                continue
            op = str(cond.get("operator") or "").strip().lower()
            if op not in {"startswith", "endswith"}:
                op = self._normalize_condition_operator(op)
            if op not in {"equals", "contains", "startswith", "endswith", "gt", "lt", "gte", "lte"}:
                continue
            out.append(
                {
                    "column": str(cond.get("column") or "").strip(),
                    "operator": op,
                    "value": cond.get("value"),
                    "source": str(cond.get("source") or ""),
                }
            )
        return out

    def _slots_cover_hard_conditions(
        self,
        slots: Dict[str, Any],
        hard_conditions: List[Dict[str, Any]],
        query: str,
    ) -> bool:
        if not hard_conditions:
            return True
        slot_conditions = self._slot_conditions(slots)
        inferred_cmp = self._infer_comparison_operator_from_query(query)
        for cond in hard_conditions:
            op = str(cond.get("operator") or "").strip().lower()
            col = str(cond.get("column") or "").strip()
            value = cond.get("value")
            if not col and bool(getattr(self.config, "query_ir_require_resolved_hard_columns", True)):
                return False
            if op in {"gt", "lt", "gte", "lte"}:
                if not self._slots_preserve_numeric_comparison(slots, "<" if op in {"lt", "lte"} else ">"):
                    return False
                if col:
                    has_col = any(str(c.get("column") or "").strip() == col for c in slot_conditions)
                    if not has_col and str(slots.get("column") or "").strip() != col:
                        return False
                if value is not None:
                    slot_num = self._safe_float(slots.get("value"))
                    cond_num = self._safe_float(value)
                    if slot_num is not None and cond_num is not None and abs(slot_num - cond_num) > 1e-9:
                        # Allow query-direction-preserving operators with nearby values only when
                        # exact numeric value was not explicitly represented in slots.
                        return False
                continue
            if op in {"startswith", "endswith"}:
                matched = False
                for sc in slot_conditions:
                    sc_op = str(sc.get("operator") or "").strip().lower()
                    if sc_op != op:
                        continue
                    if col and str(sc.get("column") or "").strip() != col:
                        continue
                    if self._normalize_text(str(sc.get("value") or "")) == self._normalize_text(str(value or "")):
                        matched = True
                        break
                if not matched:
                    return False
                continue
            matched = False
            for sc in slot_conditions:
                sc_op = str(sc.get("operator") or "").strip().lower()
                if not self._ops_compatible_for_hard_coverage(op, sc_op, value, sc.get("value")):
                    continue
                if col and str(sc.get("column") or "").strip() != col:
                    continue
                if self._normalize_text(str(sc.get("value") or "")) == self._normalize_text(str(value or "")):
                    matched = True
                    break
            if not matched and inferred_cmp in {"<", ">"} and op in {"lt", "gt", "lte", "gte"}:
                matched = self._slots_preserve_numeric_comparison(slots, inferred_cmp)
            if not matched:
                return False
        return True

    def _ops_compatible_for_hard_coverage(
        self,
        hard_op: str,
        slot_op: str,
        hard_value: Any,
        slot_value: Any,
    ) -> bool:
        hard_norm = str(hard_op or "").strip().lower()
        slot_norm = str(slot_op or "").strip().lower()
        if hard_norm == slot_norm:
            return True
        # Tolerate equals<->contains mismatches for textual entity values.
        # This prevents false rejects when slot fillers use contains for category/brand names.
        if {hard_norm, slot_norm} == {"equals", "contains"}:
            if self._safe_float(hard_value) is not None or self._safe_float(slot_value) is not None:
                return False
            hard_text = self._normalize_text(str(hard_value or ""))
            slot_text = self._normalize_text(str(slot_value or ""))
            if not hard_text or not slot_text:
                return False
            bool_like = {"true", "false", "yes", "no", "1", "0"}
            if hard_text in bool_like or slot_text in bool_like:
                return False
            return hard_text in slot_text or slot_text in hard_text
        return False

    def _slot_conditions(self, slots: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        raw = slots.get("conditions")
        if isinstance(raw, list):
            for cond in raw:
                if not isinstance(cond, dict):
                    continue
                op = str(cond.get("operator") or cond.get("op") or "").strip().lower()
                if op not in {"startswith", "endswith"}:
                    op = self._normalize_condition_operator(op)
                if not op:
                    continue
                out.append(
                    {
                        "column": str(cond.get("column") or "").strip(),
                        "operator": op,
                        "value": cond.get("value"),
                        "source": str(cond.get("source") or ""),
                    }
                )
        filter_op = str(slots.get("filter_op") or "").strip().lower()
        if filter_op:
            normalized = filter_op if filter_op in {"startswith", "endswith"} else self._normalize_condition_operator(filter_op)
            if normalized:
                out.append(
                    {
                        "column": str(slots.get("filter_col") or "").strip(),
                        "operator": normalized,
                        "value": slots.get("filter_value"),
                        "source": str(slots.get("filter_source") or ""),
                    }
                )
        op = str(slots.get("operator") or "").strip()
        if op:
            normalized = self._normalize_condition_operator(op)
            if normalized:
                out.append(
                    {
                        "column": str(slots.get("column") or "").strip(),
                        "operator": normalized,
                        "value": slots.get("value"),
                        "source": str(slots.get("source") or ""),
                    }
                )
        filter_col = str(slots.get("filter_col") or "").strip()
        filter_value = slots.get("filter_value")
        if filter_col and filter_value is not None:
            inferred_filter_op = str(slots.get("filter_op") or "").strip().lower()
            normalized = (
                inferred_filter_op
                if inferred_filter_op in {"startswith", "endswith"}
                else self._normalize_condition_operator(inferred_filter_op)
            )
            if not normalized:
                normalized = "contains"
            filter_sig = (
                filter_col,
                normalized,
                self._normalize_text(str(filter_value or "")),
            )
            has_filter_sig = any(
                (
                    str(c.get("column") or "").strip(),
                    str(c.get("operator") or "").strip().lower(),
                    self._normalize_text(str(c.get("value") or "")),
                )
                == filter_sig
                for c in out
            )
            if not has_filter_sig:
                out.append(
                    {
                        "column": filter_col,
                        "operator": normalized,
                        "value": filter_value,
                        "source": str(slots.get("filter_source") or "slot_inferred"),
                    }
                )
        return out

    def _slots_keep_soft_conditions_non_binding(
        self,
        slots: Dict[str, Any],
        soft_conditions: List[Dict[str, Any]],
        query: str,
        query_ir: Optional[Dict[str, Any]],
    ) -> bool:
        if not soft_conditions:
            return True
        if isinstance(query_ir, dict) and bool(query_ir.get("explicit_multi_filter")):
            return True
        slot_conditions = self._slot_conditions(slots)
        soft_sigs = {
            (
                str(c.get("column") or "").strip(),
                str(c.get("operator") or "").strip().lower(),
                self._normalize_text(str(c.get("value") or "")),
            )
            for c in soft_conditions
        }
        for sc in slot_conditions:
            source = str(sc.get("source") or "").strip().lower()
            if not source.startswith("augmented"):
                continue
            sig = (
                str(sc.get("column") or "").strip(),
                str(sc.get("operator") or "").strip().lower(),
                self._normalize_text(str(sc.get("value") or "")),
            )
            if sig in soft_sigs and not self._condition_value_explicitly_mentioned(query, sc.get("value")):
                return False
        return True

    def _summarize_query_ir(self, query_ir: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(query_ir, dict):
            return {"enabled": False}
        hard = self._query_ir_conditions(query_ir, "hard_conditions")
        soft = self._query_ir_conditions(query_ir, "soft_conditions")
        unresolved_hard = [c for c in hard if not str(c.get("column") or "").strip()]
        return {
            "enabled": True,
            "hard_count": len(hard),
            "soft_count": len(soft),
            "hard_unresolved_count": len(unresolved_hard),
            "explicit_multi_filter": bool(query_ir.get("explicit_multi_filter")),
            "hard_preview": hard[:3],
            "hard_unresolved_preview": unresolved_hard[:3],
            "soft_preview": soft[:3],
        }

    def _query_ir_unresolved_hard_conditions(self, query_ir: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        hard = self._query_ir_conditions(query_ir, "hard_conditions")
        return [c for c in hard if not str(c.get("column") or "").strip()]

    def _llm_verify_attempt_against_ir(
        self,
        query: str,
        intent_id: str,
        slots: Dict[str, Any],
        query_ir: Optional[Dict[str, Any]],
        columns: List[str],
        profile: Dict[str, Any],
    ) -> Tuple[bool, str]:
        if not bool(getattr(self.config, "query_ir_llm_verify_enabled", False)):
            return True, "llm_verifier_disabled"
        if not self._llm_json:
            return True, "llm_verifier_unavailable"
        ir_summary = self._summarize_query_ir(query_ir)
        hard = ir_summary.get("hard_preview") or []
        soft = ir_summary.get("soft_preview") or []
        if not hard and not soft:
            return True, "llm_verifier_no_constraints"
        system = (
            "Validate whether chosen spreadsheet intent and slots satisfy user intent constraints. "
            "Return ONLY JSON: {\"ok\": true|false, \"reason\": \"<short_code>\"}. "
            "Reject when hard constraints are missing or when inferred soft constraints become mandatory filters without explicit user request."
        )
        payload = {
            "query": query,
            "intent_id": intent_id,
            "slots": slots,
            "hard_conditions": self._query_ir_conditions(query_ir, "hard_conditions"),
            "soft_conditions": self._query_ir_conditions(query_ir, "soft_conditions"),
            "explicit_multi_filter": bool((query_ir or {}).get("explicit_multi_filter")) if isinstance(query_ir, dict) else False,
            "columns": columns[:200],
            "dtypes": (profile or {}).get("dtypes") or {},
        }
        try:
            parsed = self._llm_json(system, json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            fail_open = bool(getattr(self.config, "query_ir_llm_verify_fail_open", True))
            return (True, "llm_verifier_error_fail_open") if fail_open else (False, f"llm_verifier_error:{exc}")
        if not isinstance(parsed, dict):
            fail_open = bool(getattr(self.config, "query_ir_llm_verify_fail_open", True))
            return (True, "llm_verifier_invalid_fail_open") if fail_open else (False, "llm_verifier_invalid")
        ok = bool(parsed.get("ok", True))
        reason = str(parsed.get("reason") or ("llm_verifier_ok" if ok else "llm_verifier_reject")).strip()
        if ok:
            return True, reason or "llm_verifier_ok"
        return False, reason or "llm_verifier_reject"

    def _pick_intent_from_candidates(
        self,
        query: str,
        profile: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        hit = self._retrieve_from_candidates(candidates)
        if hit and candidates:
            top_score = float(candidates[0].get("score") or 0.0)
            second_score = float(candidates[1].get("score") or float("-inf")) if len(candidates) > 1 else float("-inf")
            threshold = float(getattr(self.config, "threshold", 0.35))
            margin = float(getattr(self.config, "margin", 0.05))
            confident_top = top_score >= max(0.90, threshold + 0.20)
            clear_lead = (top_score - second_score) >= max(0.20, margin * 3.0)
            if confident_top and clear_lead:
                intent_id, selected_score, selected_example = hit
                logging.info(
                    "event=shortcut_router_llm_select status=skipped reason=confident_retrieval "
                    "intent_id=%s score=%.4f lead=%.4f",
                    intent_id,
                    selected_score,
                    (top_score - second_score),
                )
                return {
                    "intent_id": intent_id,
                    "confidence": 0.0,
                    "retrieval_score": selected_score,
                    "example": selected_example,
                    "selector_mode": "retrieval_confident",
                }
        selected = self._select_intent_with_llm(query, profile, candidates)
        if selected:
            llm_intent_id = str(selected.get("intent_id") or "").strip()
            llm_score = float(selected.get("retrieval_score") or 0.0)
            llm_conf = float(selected.get("confidence") or 0.0)
            llm_example = str(selected.get("example") or "")

            # Universal stability guard:
            # when top retrieval candidate is very strong and LLM tries to switch
            # to a much weaker candidate, trust retrieval ranking.
            top = candidates[0] if candidates else {}
            top_intent_id = str((top or {}).get("intent_id") or "").strip()
            top_score = float((top or {}).get("score") or 0.0)
            second_score = float(candidates[1].get("score") or float("-inf")) if len(candidates) > 1 else float("-inf")
            threshold = float(getattr(self.config, "threshold", 0.35))
            margin = float(getattr(self.config, "margin", 0.05))
            score_gap_guard = max(_LLM_INTENT_OVERRIDE_MIN_GAP, margin * 2.0)
            lead_guard = max(_LLM_INTENT_OVERRIDE_MIN_LEAD, margin * 2.0)
            strong_top = top_score >= max(_LLM_INTENT_OVERRIDE_MIN_TOP_SCORE, threshold + 0.10)
            clear_top_lead = (top_score - second_score) >= lead_guard
            switched_from_top = bool(top_intent_id and llm_intent_id and top_intent_id != llm_intent_id)
            large_gap = (top_score - llm_score) >= score_gap_guard

            if switched_from_top and (strong_top or clear_top_lead) and large_gap and hit:
                intent_id, selected_score, selected_example = hit
                logging.info(
                    "event=shortcut_router_llm_select status=overridden_by_retrieval "
                    "top_intent_id=%s top_score=%.4f second_score=%.4f lead=%.4f llm_intent_id=%s llm_score=%.4f "
                    "gap=%.4f gap_guard=%.4f lead_guard=%.4f",
                    top_intent_id,
                    top_score,
                    second_score,
                    (top_score - second_score),
                    llm_intent_id,
                    llm_score,
                    (top_score - llm_score),
                    score_gap_guard,
                    lead_guard,
                )
                return {
                    "intent_id": intent_id,
                    "confidence": 0.0,
                    "retrieval_score": selected_score,
                    "example": selected_example,
                    "selector_mode": "retrieval_guarded",
                }

            return {
                "intent_id": llm_intent_id,
                "confidence": llm_conf,
                "retrieval_score": llm_score,
                "example": llm_example,
                "selector_mode": "llm",
            }
        if not hit:
            return None
        intent_id, selected_score, selected_example = hit
        return {
            "intent_id": intent_id,
            "confidence": 0.0,
            "retrieval_score": selected_score,
            "example": selected_example,
            "selector_mode": "retrieval",
        }

    def _llm_normalize_query_for_retrieval(self, query: str, profile: Dict[str, Any]) -> str:
        q = (query or "").strip()
        if not q or not self._llm_json:
            return ""
        # For short/simple non-metric queries normalization often adds noise and latency.
        if len(re.findall(r"\S+", q)) <= 8 and not has_router_metric_cue(q):
            return ""
        columns = [str(c) for c in ((profile or {}).get("columns") or [])]
        system = (
            "Normalize spreadsheet user query for intent retrieval. "
            "Keep original language and meaning. "
            "Expand short metric hints into explicit analytical wording when helpful. "
            "Do NOT add constraints absent in the query. "
            "Return ONLY JSON: {\"normalized_query\": \"<string>\", \"confidence\": 0..1}."
        )
        payload = {
            "query": q,
            "columns": columns[:200],
            "dtypes": (profile or {}).get("dtypes") or {},
        }
        try:
            parsed = self._llm_json(system, json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            logging.info("event=shortcut_router_query_normalized status=error error=%s", str(exc))
            return ""
        if not isinstance(parsed, dict):
            return ""
        normalized = str((parsed or {}).get("normalized_query") or "").strip()
        if not normalized:
            return ""
        if normalized.lower() == q.lower():
            return ""
        if len(normalized) > 600:
            normalized = normalized[:600].strip()
        # Skip noisy rewrites that are too short compared to original user query.
        if len(normalized) < max(12, int(len(q) * 0.4)):
            return ""
        return normalized

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

    def _retrieve_candidates(self, query: str) -> List[Dict[str, Any]]:
        tracer = current_route_tracer()
        stage_id = ""
        if tracer:
            stage_id = tracer.start_stage(
                stage_key="vector_search",
                stage_name="Vector Retrieval",
                purpose="Embed user query and retrieve top shortcut intents from FAISS index.",
                input_payload={
                    "query": query,
                    "top_k": int(getattr(self.config, "top_k", 5)),
                    "threshold": float(getattr(self.config, "threshold", 0.35)),
                    "margin": float(getattr(self.config, "margin", 0.05)),
                    "embedding_model": str(getattr(self.config, "vllm_embed_model", "") or ""),
                },
                processing_summary="Generate embedding and query retrieval index for nearest intent examples.",
            )
        rows = self._meta.get("rows") or []
        vec = self._embed_query(query)
        if vec is None:
            if tracer and stage_id:
                tracer.end_stage(
                    stage_id,
                    status="warn",
                    output_payload={"candidates": [], "reason": "embedding_unavailable"},
                    processing_summary="Embedding not available; retrieval skipped.",
                )
            return []
        scores, idxs = self._index.search(vec, int(getattr(self.config, "top_k", 5)))
        if scores.size == 0:
            if tracer and stage_id:
                tracer.end_stage(
                    stage_id,
                    status="warn",
                    output_payload={"candidates": [], "reason": "empty_scores"},
                    processing_summary="FAISS search returned empty score matrix.",
                )
            return []
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()
        raw_candidates: List[Tuple[int, float]] = [
            (int(i), float(s)) for i, s in zip(idxs, scores) if i >= 0 and i < len(rows)
        ]
        if not raw_candidates:
            if tracer and stage_id:
                tracer.end_stage(
                    stage_id,
                    status="warn",
                    output_payload={"candidates": [], "reason": "no_valid_indices"},
                    processing_summary="No valid retrieval index references matched metadata rows.",
                )
            return []

        best_by_intent: Dict[str, Dict[str, Any]] = {}
        for row_idx, score in raw_candidates:
            row = rows[row_idx] or {}
            intent_id = str(row.get("intent_id") or "").strip()
            if not intent_id:
                continue
            prev = best_by_intent.get(intent_id)
            if prev is not None and float(prev.get("score") or 0.0) >= score:
                continue
            best_by_intent[intent_id] = {
                "intent_id": intent_id,
                "score": score,
                "example": str(row.get("text") or ""),
                "doc_ref": str(row.get("doc_ref") or row.get("source") or intent_id),
                "row_index": row_idx,
            }
        candidates = sorted(best_by_intent.values(), key=lambda x: float(x.get("score") or 0.0), reverse=True)
        logging.info(
            "event=shortcut_router_retrieval status=candidates count=%d top=%s",
            len(candidates),
            _safe_trunc(candidates[:3], 320),
        )
        if tracer and stage_id:
            tracer.end_stage(
                stage_id,
                status="ok" if candidates else "warn",
                output_payload={
                    "candidate_count": len(candidates),
                    "candidates": candidates[: int(getattr(self.config, "top_k", 5))],
                },
                processing_summary="Vector retrieval complete with top candidate shortlist.",
                details={
                    "vector_search": {
                        "query": query,
                        "embedding_model": str(getattr(self.config, "vllm_embed_model", "") or ""),
                        "embedding_dim": int(vec.shape[1]) if hasattr(vec, "shape") and len(vec.shape) > 1 else None,
                        "top_k": int(getattr(self.config, "top_k", 5)),
                        "results": candidates[: int(getattr(self.config, "top_k", 5))],
                    }
                },
            )
        return candidates

    def _retrieve_from_candidates(self, candidates: List[Dict[str, Any]]) -> Optional[Tuple[str, float, str]]:
        if not candidates:
            return None
        best = candidates[0]
        best_score = float(best.get("score") or 0.0)
        second_score = float(candidates[1].get("score") or float("-inf")) if len(candidates) > 1 else float("-inf")
        threshold = float(getattr(self.config, "threshold", 0.35))
        margin = float(getattr(self.config, "margin", 0.05))
        if best_score < threshold or (best_score - second_score) < margin:
            logging.info(
                "event=shortcut_router_retrieval status=below_threshold score=%.4f second=%.4f threshold=%.2f margin=%.2f",
                best_score,
                second_score,
                threshold,
                margin,
            )
            return None
        intent_id = str(best.get("intent_id") or "")
        example = str(best.get("example") or "")
        logging.info(
            "event=shortcut_router_retrieval status=hit intent_id=%s score=%.4f example=%s",
            intent_id,
            best_score,
            example[:300],
        )
        return intent_id, best_score, example

    def _select_intent_with_llm(
        self,
        query: str,
        profile: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not self._llm_json:
            return None
        max_candidates = int(getattr(self.config, "llm_intent_max_candidates", 8) or 8)
        min_conf = float(getattr(self.config, "llm_intent_min_confidence", 0.45) or 0.45)
        shortlist = candidates[: max(1, max_candidates)]
        catalog = self._intent_catalog_for_llm(shortlist)
        if not catalog:
            return None
        system = (
            "STRICT JSON MODE.\n"
            "Return exactly one JSON object:\n"
            "{\"intent_id\":\"<id or NONE>\",\"confidence\":0.0}\n\n"
            "Rules:\n"
            "- choose only from provided intent ids\n"
            "- do not explain\n"
            "- do not add extra keys\n"
            "- if uncertain, use {\"intent_id\":\"NONE\",\"confidence\":0.0}\n\n"
            "Prefer precise intent-function match over broad guesses. "
            "Prefer the highest retrieval_score candidate unless it is clearly incompatible with user intent."
        )
        payload = {
            "query": query,
            "columns": [str(c) for c in ((profile or {}).get("columns") or [])][:200],
            "dtypes": (profile or {}).get("dtypes") or {},
            "intent_catalog": catalog,
        }
        try:
            parsed = self._llm_json(system, json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            logging.info("event=shortcut_router_llm_select status=error error=%s", str(exc))
            return None
        if not isinstance(parsed, dict):
            logging.info("event=shortcut_router_llm_select status=invalid_response")
            return None
        raw_intent = str(parsed.get("intent_id") or "").strip()
        conf = self._safe_float(parsed.get("confidence"))
        if conf is None:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        if not raw_intent or raw_intent.upper() == "NONE":
            logging.info("event=shortcut_router_llm_select status=none confidence=%.2f", conf)
            return None
        best = next((c for c in shortlist if str(c.get("intent_id") or "") == raw_intent), None)
        if not best:
            logging.info(
                "event=shortcut_router_llm_select status=unknown_intent intent_id=%s confidence=%.2f",
                raw_intent,
                conf,
            )
            return None
        if conf < min_conf:
            logging.info(
                "event=shortcut_router_llm_select status=low_confidence intent_id=%s confidence=%.2f threshold=%.2f",
                raw_intent,
                conf,
                min_conf,
            )
            return None
        logging.info(
            "event=shortcut_router_llm_select status=accepted intent_id=%s confidence=%.2f retrieval_score=%.4f",
            raw_intent,
            conf,
            float(best.get("score") or 0.0),
        )
        return {
            "intent_id": raw_intent,
            "confidence": conf,
            "retrieval_score": float(best.get("score") or 0.0),
            "example": str(best.get("example") or ""),
        }

    def _intent_catalog_for_llm(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for c in candidates:
            intent_id = str(c.get("intent_id") or "")
            intent = self._intents.get(intent_id) or {}
            if not intent:
                continue
            slots = intent.get("slots") or {}
            required_slots = [str(k) for k, v in slots.items() if isinstance(v, dict) and bool(v.get("required"))]
            optional_slots = [str(k) for k in slots.keys() if str(k) not in set(required_slots)]
            plan = intent.get("plan") or []
            ops = [str(step.get("op") or "") for step in plan if isinstance(step, dict) and step.get("op")]
            out.append(
                {
                    "intent_id": intent_id,
                    "description": str(intent.get("description") or intent.get("name") or ""),
                    "ops": ops[:6],
                    "required_slots": required_slots[:12],
                    "optional_slots": optional_slots[:12],
                    "retrieval_score": float(c.get("score") or 0.0),
                }
            )
        return out

    def _safe_float(self, value: Any) -> Optional[float]:
        parsed = self._parse_numeric_value(value)
        if parsed is None:
            return None
        try:
            return float(parsed)
        except Exception:
            return None

    def _parse_numeric_value(self, value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            try:
                return float(value)
            except Exception:
                return None
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            s = s.replace("\u00a0", " ").replace("−", "-").replace("–", "-")
            s = re.sub(r"[^\d,.\-'\s]", "", s)
            s = re.sub(r"\s+", "", s)
            s = s.replace("'", "")
            if not s or not re.search(r"\d", s):
                return None

            # Dot and comma at once: detect decimal separator by the right-most symbol.
            if "," in s and "." in s:
                if s.rfind(",") > s.rfind("."):
                    s = s.replace(".", "")
                    s = s.replace(",", ".")
                else:
                    s = s.replace(",", "")
            elif "," in s:
                # 3,000 -> 3000 ; 12,50 -> 12.50
                if re.fullmatch(r"-?\d{1,3}(,\d{3})+", s):
                    s = s.replace(",", "")
                elif re.fullmatch(r"-?\d+,\d+", s):
                    left, right = s.split(",", 1)
                    if len(right) == 3 and len(left.lstrip("-")) <= 3:
                        s = left + right
                    else:
                        s = left + "." + right
                else:
                    s = s.replace(",", "")
            elif "." in s:
                # 3.000 -> 3000
                if re.fullmatch(r"-?\d{1,3}(\.\d{3})+", s):
                    s = s.replace(".", "")

            try:
                return float(s)
            except Exception:
                return None
        return None

    def _first_numeric_from_text(self, text: str) -> Optional[float]:
        q = str(text or "")
        for m in re.finditer(r"-?\d[\d\s\u00a0,.'-]*", q):
            candidate = (m.group(0) or "").strip()
            if not candidate:
                continue
            val = self._parse_numeric_value(candidate)
            if val is not None:
                return val
        return None

    def _comparison_operator_symbol(self, raw_op: Any) -> str:
        op = str(raw_op or "").strip()
        if op in {">", "<", ">=", "<=", "="}:
            return op
        normalized = self._normalize_condition_operator(raw_op)
        if normalized == "gt":
            return ">"
        if normalized == "lt":
            return "<"
        if normalized == "gte":
            return ">="
        if normalized == "lte":
            return "<="
        return "="

    def _infer_comparison_operator_from_query(self, query: str) -> Optional[str]:
        q = (query or "").lower()
        if not q:
            return None
        if re.search(
            r"(<|<=|\b(дешев\w*|менш\w*|нижч\w*|до\b|under|below|less\s+than|cheaper)\b)",
            q,
            re.I,
        ):
            return "<"
        if re.search(
            r"(>|>=|\b(дорожч\w*|більш\w*|вищ\w*|понад|over|above|greater\s+than|more\s+than)\b)",
            q,
            re.I,
        ):
            return ">"
        return None

    def _normalize_top_n(self, value: Any) -> Optional[int]:
        if value is None or isinstance(value, bool):
            return None
        try:
            n = int(value)
        except Exception:
            return None
        return n if n > 0 else None

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

        if intent_id == "filter_multi_conditions" and "conditions" not in slots:
            conditions = self._extract_multi_conditions_llm(query, columns, profile, min_conditions=1) or []
            if self._should_augment_multi_conditions(query, conditions):
                conditions = self._augment_multi_conditions_from_query(
                    query=query,
                    columns=columns,
                    profile=profile,
                    conditions=conditions,
                    min_conditions=2,
                )
            if not conditions or len(conditions) < 2:
                return None
            slots["conditions"] = conditions

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
            if intent_id in {"groupby_count", "groupby_agg"} and name == "out_col":
                explicit_alias_cue = bool(
                    re.search(
                        r"\b(as|alias|назв[аи]\w*|іменуй\w*|column\s+name|назва\s+стовпц\w*)\b",
                        query or "",
                        re.I,
                    )
                )
                val = self._slot_from_text(name, cfg, query, columns) if explicit_alias_cue else None
                if val is None and "default" in cfg:
                    val = cfg.get("default")
                if val is not None:
                    slots[name] = val
                continue
            if intent_id == "filter_equals" and name == "top_n":
                parsed_top = self._normalize_top_n(self._extract_top_n(query))
                if parsed_top is not None:
                    slots[name] = parsed_top
                    continue
                if "default" in cfg:
                    slots[name] = cfg.get("default")
                    continue
                if cfg.get("required"):
                    return None
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
                if intent_id in {"filter_comparison", "filter_count"} and name in {"column", "operator", "value"}:
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
        if intent_id in {"filter_comparison", "filter_count"}:
            dtypes = (profile or {}).get("dtypes") or {}
            llm_conditions = self._extract_multi_conditions_llm(
                query=query,
                columns=columns,
                profile=profile,
                min_conditions=1,
            )
            if llm_conditions:
                for cond in llm_conditions:
                    if not isinstance(cond, dict):
                        continue
                    col = str(cond.get("column") or "").strip()
                    op_name = self._normalize_condition_operator(cond.get("operator") or cond.get("op"))
                    if op_name not in {"gt", "lt", "gte", "lte", "equals"}:
                        continue
                    val = self._safe_float(cond.get("value"))
                    if val is None:
                        continue
                    if col in columns and self._is_numeric_dtype(dtypes.get(col, "")):
                        slots["column"] = col
                        slots["operator"] = self._comparison_operator_symbol(op_name)
                        slots["value"] = val
                        break

            col = slots.get("column")
            if not (isinstance(col, str) and col in columns and self._is_numeric_dtype(dtypes.get(col, ""))):
                col = self._best_numeric_column_for_query(query, columns, profile)
            if not col:
                return None
            slots["column"] = col

            raw_op = slots.get("operator")
            op_sym = self._comparison_operator_symbol(raw_op)
            if op_sym == "=":
                inferred_op = self._infer_comparison_operator_from_query(query)
                if inferred_op in {">", "<"} and str(raw_op or "").strip() not in {"=", "==", "eq", "equals"}:
                    op_sym = inferred_op
            if op_sym not in {">", "<", ">=", "<=", "="}:
                return None
            slots["operator"] = op_sym

            value_num = self._safe_float(slots.get("value"))
            if value_num is None:
                value_num = self._first_numeric_from_text(query)
            if value_num is None:
                return None
            slots["value"] = float(value_num)
        return slots

    def _validate_slots(
        self,
        intent: Dict[str, Any],
        slots: Dict[str, Any],
        columns: List[str],
        profile: Dict[str, Any],
    ) -> List[str]:
        issues: List[str] = []
        dtypes = (profile or {}).get("dtypes") or {}
        spec = intent.get("slots") or {}
        intent_id = str(intent.get("id") or "")

        for name, cfg in spec.items():
            cfg = cfg if isinstance(cfg, dict) else {}
            required = bool(cfg.get("required"))
            slot_type = str(cfg.get("type") or "").strip().lower()
            enum_values = [str(v) for v in (cfg.get("values") or [])]
            val = slots.get(name)
            if required and val is None:
                issues.append(f"missing_required:{name}")
                continue
            if val is None:
                continue
            if slot_type == "column":
                if not (isinstance(val, str) and val in columns):
                    issues.append(f"invalid_column:{name}")
            elif slot_type in {"columns", "list[column]"}:
                if not (isinstance(val, list) and val and all(isinstance(c, str) and c in columns for c in val)):
                    issues.append(f"invalid_columns:{name}")
            elif slot_type == "row_indices":
                if not (isinstance(val, list) and all(isinstance(x, int) and x > 0 for x in val)):
                    issues.append(f"invalid_row_indices:{name}")
            elif slot_type == "int":
                if not isinstance(val, int):
                    issues.append(f"invalid_int:{name}")
            elif slot_type == "float":
                if self._safe_float(val) is None:
                    issues.append(f"invalid_float:{name}")
            elif slot_type == "bool":
                if not isinstance(val, bool):
                    issues.append(f"invalid_bool:{name}")
            elif slot_type == "enum":
                if not (isinstance(val, str) and val in enum_values):
                    issues.append(f"invalid_enum:{name}")
            elif slot_type == "str":
                if not (isinstance(val, str) and val.strip()):
                    issues.append(f"invalid_str:{name}")
            elif slot_type == "json":
                if not isinstance(val, (dict, list)):
                    issues.append(f"invalid_json:{name}")

            # Universal guard: slots that look like column references must point to real columns.
            if isinstance(val, str) and ("col" in name.lower()) and (name.lower() not in {"color", "out_col"}):
                if val not in columns:
                    issues.append(f"unknown_column_ref:{name}")

        # Universal numeric guards for intents that aggregate numeric metrics.
        numeric_col_like_keys = ("target_col", "metric_col", "column")
        needs_numeric = False
        for step in (intent.get("plan") or []):
            op = str((step or {}).get("op") or "").strip().lower()
            if op in {"stats_aggregation", "groupby_agg", "filtered_metric_aggregation", "row_ranking"}:
                needs_numeric = True
                break
        if needs_numeric:
            for key in numeric_col_like_keys:
                val = slots.get(key)
                if isinstance(val, str) and val in columns:
                    if not self._is_numeric_dtype(dtypes.get(val, "")):
                        # For groupby_count numeric target may be absent; skip strict check there.
                        agg = str(slots.get("agg") or "").strip().lower()
                        if key == "target_col" and agg == "count":
                            continue
                        issues.append(f"non_numeric_metric_column:{key}")

        if intent_id == "groupby_agg":
            group_col = slots.get("group_col")
            target_col = slots.get("target_col")
            agg = str(slots.get("agg") or "").strip().lower()
            if not (isinstance(group_col, str) and group_col in columns):
                issues.append("invalid_group_col")
            if agg != "count":
                if not (isinstance(target_col, str) and target_col in columns):
                    issues.append("invalid_target_col")
        if intent_id in {"filter_comparison", "filter_count"}:
            comp_col = slots.get("column")
            comp_op = str(slots.get("operator") or "").strip()
            comp_val = slots.get("value")
            if not (isinstance(comp_col, str) and comp_col in columns):
                issues.append("invalid_comparison_column")
            elif not self._is_numeric_dtype(dtypes.get(comp_col, "")):
                issues.append("non_numeric_comparison_column")
            if comp_op not in {">", "<", ">=", "<=", "="}:
                issues.append("invalid_comparison_operator")
            if self._safe_float(comp_val) is None:
                issues.append("invalid_comparison_value")
        if intent_id == "filter_multi_conditions":
            conditions = slots.get("conditions")
            if not isinstance(conditions, list) or len(conditions) < 2:
                issues.append("invalid_conditions")
            else:
                for i, cond in enumerate(conditions):
                    if not isinstance(cond, dict):
                        issues.append(f"invalid_condition:{i}")
                        continue
                    col = str(cond.get("column") or "").strip()
                    op = str(cond.get("operator") or "").strip().lower()
                    if col not in columns:
                        issues.append(f"invalid_condition_column:{i}")
                    if op not in _MULTI_FILTER_ALLOWED_OPERATORS:
                        issues.append(f"invalid_condition_operator:{i}")
                    if cond.get("value") is None:
                        issues.append(f"invalid_condition_value:{i}")
        return issues

    def _normalize_condition_operator(self, raw_op: Any) -> str:
        op = str(raw_op or "").strip().lower()
        mapping = {
            "=": "equals",
            "==": "equals",
            "eq": "equals",
            "equals": "equals",
            "exact": "equals",
            "exactly": "equals",
            "contains": "contains",
            "like": "contains",
            "gt": "gt",
            ">": "gt",
            "lt": "lt",
            "<": "lt",
            "gte": "gte",
            ">=": "gte",
            "ge": "gte",
            "lte": "lte",
            "<=": "lte",
            "le": "lte",
        }
        return mapping.get(op, "")

    def _extract_multi_conditions_llm(
        self,
        query: str,
        columns: List[str],
        profile: Dict[str, Any],
        min_conditions: int = 2,
    ) -> Optional[List[Dict[str, Any]]]:
        if not self._llm_json or not columns:
            return None
        system = (
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
            "Never echo the user query.\n\n"
            "Schema:\n"
            "{\"conditions\":[{\"column\":\"<exact column>\",\"operator\":\"equals|contains|gt|lt|gte|lte\",\"value\":\"<value>\",\"case_insensitive\":true}]}\n\n"
            "Valid:\n"
            "{\"conditions\":[]}\n\n"
            "Now produce output.\n"
            "Use ONLY provided column names."
        )
        payload = {
            "query": query,
            "columns": columns[:200],
            "dtypes": (profile or {}).get("dtypes") or {},
            "preview": ((profile or {}).get("preview") or [])[:30],
        }
        try:
            parsed = self._llm_json(system, json.dumps(payload, ensure_ascii=False))
        except Exception:
            return None
        raw_conditions: Any = None
        if isinstance(parsed, dict):
            raw_conditions = parsed.get("conditions")
            if raw_conditions is None and isinstance(parsed.get("filters"), list):
                raw_conditions = parsed.get("filters")
        elif isinstance(parsed, list):
            raw_conditions = parsed
        if not isinstance(raw_conditions, list) or not raw_conditions:
            return None

        valid: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str]] = set()
        for cond in raw_conditions:
            if not isinstance(cond, dict):
                continue
            col = str(cond.get("column") or "").strip()
            if col not in columns:
                continue
            raw_op = str(cond.get("operator") or cond.get("op") or "").strip().lower()
            if "|" in raw_op:
                continue
            op = self._normalize_condition_operator(raw_op)
            if not op and raw_op:
                continue
            if not op:
                op = "equals"
            if op not in _MULTI_FILTER_ALLOWED_OPERATORS:
                continue
            value = cond.get("value")
            if value is None:
                continue
            if isinstance(value, str):
                value_text = value.strip()
                if not value_text:
                    continue
                low_value = value_text.lower()
                if low_value in {"<value>", "value", "exact value"}:
                    continue
                if "<" in value_text and ">" in value_text:
                    continue
            ci = bool(cond.get("case_insensitive", True))
            key = (col, op, str(value).strip().lower())
            if key in seen:
                continue
            seen.add(key)
            valid.append(
                {
                    "column": col,
                    "operator": op,
                    "value": value,
                    "case_insensitive": ci,
                    "source": "llm_extract",
                }
            )
        required = max(1, int(min_conditions or 1))
        return valid if len(valid) >= required else None

    def _query_semantic_roots(self, query: str) -> set[str]:
        q_norm = self._normalize_text(query or "")
        if not q_norm:
            return set()
        stop_roots = {
            "знайд",
            "пока",
            "всі",
            "усі",
            "all",
            "find",
            "show",
            "item",
            "prod",
            "това",
            "рядк",
            "запи",
            "where",
            "де",
            "яки",
            "яка",
            "яке",
            "with",
            "для",
            "and",
            "or",
            "та",
            "або",
            "і",
        }
        roots: set[str] = set()
        for token in q_norm.split():
            if len(token) < 3:
                continue
            if re.fullmatch(r"\d+", token):
                continue
            root = token[:5]
            if root in stop_roots:
                continue
            roots.add(root)
        return roots

    def _augment_multi_conditions_from_query(
        self,
        query: str,
        columns: List[str],
        profile: Dict[str, Any],
        conditions: List[Dict[str, Any]],
        min_conditions: int = 2,
    ) -> List[Dict[str, Any]]:
        valid: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str]] = set()
        for cond in conditions or []:
            if not isinstance(cond, dict):
                continue
            col = str(cond.get("column") or "").strip()
            op = self._normalize_condition_operator(cond.get("operator") or cond.get("op"))
            value = cond.get("value")
            if not col or col not in columns or not op or value is None:
                continue
            key = (col, op, str(value).strip().lower())
            if key in seen:
                continue
            seen.add(key)
            valid.append(
                {
                    "column": col,
                    "operator": op,
                    "value": value,
                    "case_insensitive": bool(cond.get("case_insensitive", True)),
                    "source": str(cond.get("source") or "llm_extract"),
                }
            )

        required = max(1, int(min_conditions or 1))
        if len(valid) >= required:
            return valid

        q_norm = self._normalize_text(query or "")
        q_roots = self._query_semantic_roots(query)
        if not q_norm or not q_roots:
            return valid

        preview = (profile or {}).get("preview") or []
        if not isinstance(preview, list) or not preview:
            return valid

        existing_cols = {str(c.get("column") or "").strip() for c in valid if isinstance(c, dict)}
        text_cols = self._text_columns(columns, profile)
        scored_candidates: List[Tuple[float, str, str]] = []

        for col in text_cols:
            if col in existing_cols:
                continue
            best_score = 0.0
            best_value = ""
            per_value_score: Dict[str, float] = {}
            for row in preview[:250]:
                if not isinstance(row, dict):
                    continue
                raw_value = row.get(col)
                if raw_value is None:
                    continue
                value = str(raw_value).strip()
                if not value:
                    continue
                value_norm = self._normalize_text(value)
                if not value_norm:
                    continue
                if re.fullmatch(r"[\d\s.,:%\-]+", value_norm):
                    continue

                score = 0.0
                if value_norm in q_norm:
                    score += 3.0
                value_roots = {tok[:5] for tok in value_norm.split() if len(tok) >= 3}
                overlap = q_roots.intersection(value_roots)
                if overlap:
                    score += float(len(overlap))
                if score <= 0.0:
                    continue

                prev_score = per_value_score.get(value, 0.0)
                if score > prev_score:
                    per_value_score[value] = score
                if per_value_score[value] > best_score:
                    best_score = per_value_score[value]
                    best_value = value

            if best_score > 0.0 and best_value:
                scored_candidates.append((best_score, col, best_value))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        for score, col, value in scored_candidates:
            if len(valid) >= required:
                break
            key = (col, "equals", str(value).strip().lower())
            if key in seen:
                continue
            valid.append(
                {
                    "column": col,
                    "operator": "equals",
                    "value": value,
                    "case_insensitive": True,
                    "source": "augmented_preview",
                }
            )
            seen.add(key)
            logging.info(
                "event=shortcut_router_multi_conditions status=augmented source=preview col=%s value=%s score=%.2f",
                col,
                _safe_trunc(value, 80),
                score,
            )

        return valid

    def _has_explicit_multi_filter_cue(self, query: str) -> bool:
        q = str(query or "").lower()
        if not q:
            return False
        return bool(
            re.search(
                r"\b(and|or|та|і|й|або)\b"
                r"|[,;]\s*(?:and|or|та|і|й|або)\b"
                r"|\b(одночасно|разом|simultaneously|both)\b",
                q,
                re.I,
            )
        )

    def _conditions_contain_numeric_comparison(self, conditions: List[Dict[str, Any]]) -> bool:
        for cond in conditions or []:
            if not isinstance(cond, dict):
                continue
            op = self._normalize_condition_operator(cond.get("operator") or cond.get("op"))
            if op in {"gt", "lt", "gte", "lte"}:
                return True
        return False

    def _should_augment_multi_conditions(self, query: str, conditions: List[Dict[str, Any]]) -> bool:
        # Augmentation is useful when LLM missed an implicit text constraint
        # (e.g. "ноутбуки Apple"), but dangerous for single numeric filters
        # (e.g. "менше 5 штук"), where preview-derived text conditions overconstrain results.
        if len(conditions or []) >= 2:
            return False
        if self._conditions_contain_numeric_comparison(conditions) and not self._has_explicit_multi_filter_cue(query):
            return False
        return True

    def _resolve_intent_and_slots(
        self,
        intent: Dict[str, Any],
        query: str,
        columns: List[str],
        profile: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        intent_id = str(intent.get("id") or "")
        if intent_id in {"filter_equals", "filter_contains", "filter_comparison"}:
            multi_intent = self._intents.get("filter_multi_conditions")
            if isinstance(multi_intent, dict):
                conditions = self._extract_multi_conditions_llm(query, columns, profile, min_conditions=1) or []
                if self._should_augment_multi_conditions(query, conditions):
                    conditions = self._augment_multi_conditions_from_query(
                        query=query,
                        columns=columns,
                        profile=profile,
                        conditions=conditions,
                        min_conditions=2,
                    )
                if conditions and len(conditions) >= 2:
                    logging.info(
                        "event=shortcut_router_filter_redirect reason=multi_conditions intent_from=%s intent_to=%s conditions=%s",
                        intent_id,
                        str(multi_intent.get("id") or ""),
                        _safe_trunc(conditions, 500),
                    )
                    return multi_intent, {"conditions": conditions}
        if intent_id == "filter_comparison" and self._is_count_query(query):
            count_intent = self._intents.get("filter_count")
            if isinstance(count_intent, dict):
                logging.info(
                    "event=shortcut_router_filter_redirect reason=count_query intent_from=%s intent_to=%s",
                    intent_id,
                    str(count_intent.get("id") or ""),
                )
                return count_intent, {}
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
            llm_target_col = llm_slots.get("target_col") if isinstance(llm_slots.get("target_col"), str) else ""
            llm_target_col = llm_target_col.strip()
            can_override_target = (
                not llm_target_col
                or self._is_qty_like_column_name(llm_target_col)
                or self._is_id_like_column_name(llm_target_col)
            )
            qty_col = self._best_numeric_column_for_query("кількість qty quantity units stock", columns, profile)
            if qty_col and can_override_target:
                agg = "sum"
                llm_slots["agg"] = "sum"
                llm_slots["target_col"] = qty_col
            elif qty_col and llm_target_col:
                logging.info(
                    "event=shortcut_router_groupby_quantity_override status=skip reason=llm_target_locked target_col=%s qty_candidate=%s",
                    llm_target_col,
                    qty_col,
                )
        resolved_intent = intent
        if agg and agg != "count" and "groupby_agg" in self._intents:
            resolved_intent = self._intents["groupby_agg"]
        elif agg == "count" and "groupby_count" in self._intents:
            resolved_intent = self._intents["groupby_count"]

        preset: Dict[str, Any] = {}
        group_col = llm_slots.get("group_col")
        target_col = llm_slots.get("target_col")
        top_n = self._normalize_top_n(llm_slots.get("top_n"))
        if isinstance(group_col, str) and group_col in columns:
            preset["group_col"] = group_col
        if isinstance(target_col, str) and target_col in columns:
            preset["target_col"] = target_col
        startswith_value = self._extract_startswith_value(query)
        if startswith_value and isinstance(group_col, str) and group_col in columns:
            preset["filter_col"] = group_col
            preset["filter_op"] = "startswith"
            preset["filter_value"] = startswith_value
        if agg in {"sum", "mean", "min", "max"}:
            preset["agg"] = agg
        if isinstance(top_n, int):
            preset["top_n"] = top_n
        if "top_n" not in preset:
            parsed_top = self._normalize_top_n(self._extract_top_n(query))
            if isinstance(parsed_top, int):
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
        has_totalish = bool(re.search(r"\b(загальн\w*|total|sum|сума|підсум\w*)\b", q, re.I))
        has_all_items = bool(
            re.search(
                r"\b(вс[ії]\w*\s+товар\w*|ус[ії]\w*\s+товар\w*|all\s+items?|all\s+products?)\b",
                q,
                re.I,
            )
        )
        return has_group and (
            has_revenue_like
            or (has_mul and has_price and has_qty)
            or (has_totalish and has_price and has_all_items)
        )

    def _pick_price_like_column(self, columns: List[str], profile: Dict[str, Any]) -> Optional[str]:
        dtypes = (profile or {}).get("dtypes") or {}
        numeric_cols = [c for c in columns if self._is_numeric_dtype(dtypes.get(c, ""))]
        for c in numeric_cols:
            if self._is_price_like_column_name(str(c)):
                return c
        return None

    def _pick_qty_like_column(self, columns: List[str], profile: Dict[str, Any]) -> Optional[str]:
        dtypes = (profile or {}).get("dtypes") or {}
        numeric_cols = [c for c in columns if self._is_numeric_dtype(dtypes.get(c, ""))]
        for c in numeric_cols:
            if self._is_qty_like_column_name(str(c)):
                return c
        return None

    def _is_price_like_column_name(self, column: str) -> bool:
        c = str(column or "").strip().lower()
        if not c:
            return False
        return bool(
            re.search(
                r"(цін|price|cost|amount|revenue|sales|gmv|вируч|дохід|оборот|варт|income|profit|margin)",
                c,
                re.I,
            )
        )

    def _is_qty_like_column_name(self, column: str) -> bool:
        c = str(column or "").strip().lower()
        if not c:
            return False
        if self._is_price_like_column_name(c):
            return False
        return bool(re.search(r"(кільк|qty|quantity|units?|stock|залишк|штук|count|volume)", c, re.I))

    def _query_has_money_cue(self, query: str) -> bool:
        q = (query or "").lower()
        if not q:
            return False
        if any(sym in q for sym in ("₴", "$", "€", "£", "¥")):
            return True
        return bool(
            re.search(
                r"\b(цін\w*|price\w*|cost\w*|amount\w*|revenue\w*|sales\w*|gmv|"
                r"вируч\w*|дохід\w*|оборот\w*|варт\w*|грн\w*|uah|usd|eur)\b",
                q,
                re.I,
            )
        )

    def _is_id_like_column_name(self, column: str) -> bool:
        c = str(column or "").strip().lower()
        if not c:
            return False
        return bool(re.search(r"(?:^|_)(id|sku|код|артикул)(?:$|_)", c))

    def _unique_output_column_name(self, base: str, columns: List[str]) -> str:
        existing = {str(c) for c in (columns or [])}
        candidate = str(base or "metric_sum").strip() or "metric_sum"
        if candidate not in existing:
            return candidate
        idx = 2
        while f"{candidate}_{idx}" in existing:
            idx += 1
        return f"{candidate}_{idx}"

    def _pick_group_like_column(self, query: str, columns: List[str], profile: Dict[str, Any]) -> Optional[str]:
        text_cols = self._text_columns(columns, profile)
        if not text_cols:
            return None

        explicit = self._best_column_match(query, text_cols)
        if explicit:
            return explicit

        # Prefer semantic dimension columns across schema variants.
        dimension_re = re.compile(
            r"(categor|segment|group|brand|model|type|class|family|region|country|city|dept|status|"
            r"категор|сегмент|груп|бренд|модел|тип|клас|статус|регіон|місто|країн)",
            re.I,
        )
        dimension_cols = [c for c in text_cols if dimension_re.search(str(c))]
        if dimension_cols:
            non_id = [c for c in dimension_cols if not self._is_id_like_column_name(c)]
            return non_id[0] if non_id else dimension_cols[0]

        non_id = [c for c in text_cols if not self._is_id_like_column_name(c)]
        if non_id:
            return non_id[0]
        return text_cols[0]

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
        top_n = self._normalize_top_n(llm_slots.get("top_n"))
        if top_n is None:
            top_n = self._normalize_top_n(self._extract_top_n(query))
        out_base = "revenue_sum" if re.search(r"(вируч|revenue|sales|дохід|оборот)", (query or "").lower(), re.I) else "metric_sum"
        out_col = self._unique_output_column_name(out_base, columns)
        slots: Dict[str, Any] = {
            "group_col": group_col,
            "target_col": price_col,  # keep compatibility with groupby_agg intent contract
            "agg": "sum",
            "mul_left_col": price_col,
            "mul_right_col": qty_col,
            "out_col": out_col,
        }
        if isinstance(top_n, int):
            slots["top_n"] = top_n
        return slots

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
        has_stockish = bool(
            re.search(r"\b(на\s+склад\w*|в\s+наявн\w*|наявн\w*|in\s*stock|available|inventory|warehouse|залишк\w*)\b", q, re.I)
        )
        if not has_qty and not has_stockish:
            return False
        has_totalish = bool(re.search(r"\b(загальн\w*|total|sum|сума|на\s+склад\w*|наявн\w*|залишк\w*)\b", q, re.I))
        if not has_totalish:
            return False
        money_like = self._query_has_money_cue(q)
        return not money_like

    def _query_has_startswith_cue(self, query: str) -> bool:
        q = (query or "").lower()
        if not q:
            return False
        return bool(
            re.search(
                r"\b(starts?\s+with|begin(?:s|ning)?\s+with|почина\w*\s+на|на\s+літер\w*|з\s+літер\w*)\b",
                q,
                re.I,
            )
        )

    def _extract_startswith_value(self, query: str) -> Optional[str]:
        q = str(query or "").strip()
        if not q or not self._query_has_startswith_cue(q):
            return None
        quoted = re.findall(r"[\"'«“”`]\s*([^\"'«“”`]+?)\s*[\"'»“”`]", q)
        for token in quoted:
            s = str(token or "").strip()
            if s:
                return s
        m = re.search(
            r"(?:на\s+літер\w*|з\s+літер\w*|starts?\s+with|begin(?:s|ning)?\s+with)\s+([A-Za-zА-Яа-яІіЇїЄєҐґ0-9._\-]+)",
            q,
            re.I,
        )
        if m:
            s = str(m.group(1) or "").strip()
            if s:
                return s
        return None

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
        top_n = self._normalize_top_n(res.get("top_n"))
        if isinstance(group_col, str):
            out["group_col"] = group_col.strip()
        if isinstance(target_col, str):
            out["target_col"] = target_col.strip()
        if agg in {"count", "sum", "mean", "min", "max"}:
            out["agg"] = agg
        if isinstance(top_n, int):
            out["top_n"] = top_n
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
        return has_router_metric_cue(question)

    def _question_has_filter_cue(self, question: str) -> bool:
        return has_router_filter_context_cue(question)

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
                "STRICT JSON MODE.\n"
                "Return exactly one JSON object: {\"value\": <value or null>}\n"
                "No prose. No markdown. No explanation. No extra keys.\n"
                "If no safe value exists, return {\"value\": null}"
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
            return self._first_numeric_from_text(q)
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
        preview_source = "df"
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
                target_col = slots.get("target_col")
                out_col = slots.get("out_col") or "count"
                sort_mode = str(slots.get("sort") or "desc").strip().lower()
                top_n = int(slots.get("top_n") or 0)
                filter_col = slots.get("filter_col")
                filter_op = str(slots.get("filter_op") or "").strip().lower()
                filter_value = slots.get("filter_value")
                if not group_col:
                    return None
                if isinstance(filter_col, str) and filter_col and filter_value is not None:
                    code_lines.append("_src = df")
                    if filter_op == "startswith":
                        code_lines.append(
                            f"_src = _src[_src[{filter_col!r}].astype(str).str.lower().str.startswith(str({filter_value!r}).lower(), na=False)]"
                        )
                    elif filter_op == "endswith":
                        code_lines.append(
                            f"_src = _src[_src[{filter_col!r}].astype(str).str.lower().str.endswith(str({filter_value!r}).lower(), na=False)]"
                        )
                    elif filter_op == "contains":
                        code_lines.append(
                            f"_src = _src[_src[{filter_col!r}].astype(str).str.contains(str({filter_value!r}), case=False, na=False)]"
                        )
                    else:
                        code_lines.append(
                            f"_src = _src[_src[{filter_col!r}].astype(str).str.lower() == str({filter_value!r}).lower()]"
                        )
                else:
                    code_lines.append("_src = df")
                if isinstance(target_col, str) and target_col:
                    code_lines.append(f"_target_col = {target_col!r}")
                    code_lines.append("if _target_col in _src.columns:")
                    code_lines.append(
                        f"    result = _src.groupby({group_col!r})[_target_col].nunique(dropna=True).reset_index(name={out_col!r})"
                    )
                    code_lines.append("else:")
                    code_lines.append(
                        f"    result = _src.groupby({group_col!r}).size().reset_index(name={out_col!r})"
                    )
                else:
                    code_lines.append(
                        f"result = _src.groupby({group_col!r}).size().reset_index(name={out_col!r})"
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
                filter_col = slots.get("filter_col")
                filter_op = str(slots.get("filter_op") or "").strip().lower()
                filter_value = slots.get("filter_value")
                if not out_col:
                    out_col = str(target_col) if target_col else "value"
                if not group_col or not target_col:
                    return None
                code_lines.append("_src = df")
                if isinstance(filter_col, str) and filter_col and filter_value is not None:
                    if filter_op == "startswith":
                        code_lines.append(
                            f"_src = _src[_src[{filter_col!r}].astype(str).str.lower().str.startswith(str({filter_value!r}).lower(), na=False)]"
                        )
                    elif filter_op == "endswith":
                        code_lines.append(
                            f"_src = _src[_src[{filter_col!r}].astype(str).str.lower().str.endswith(str({filter_value!r}).lower(), na=False)]"
                        )
                    elif filter_op == "contains":
                        code_lines.append(
                            f"_src = _src[_src[{filter_col!r}].astype(str).str.contains(str({filter_value!r}), case=False, na=False)]"
                        )
                    else:
                        code_lines.append(
                            f"_src = _src[_src[{filter_col!r}].astype(str).str.lower() == str({filter_value!r}).lower()]"
                        )
                if (
                    isinstance(mul_left_col, str)
                    and isinstance(mul_right_col, str)
                    and agg == "sum"
                ):
                    code_lines.append(f"_group_col = {group_col!r}")
                    code_lines.append(f"_left_col = {mul_left_col!r}")
                    code_lines.append(f"_right_col = {mul_right_col!r}")
                    code_lines.append("_ok = (_group_col in _src.columns) and (_left_col in _src.columns) and (_right_col in _src.columns)")
                    code_lines.append("if not _ok:")
                    code_lines.append("    result = []")
                    code_lines.append("else:")
                    code_lines.append("    _left = pd.to_numeric(_src[_left_col], errors='coerce')")
                    code_lines.append("    _right = pd.to_numeric(_src[_right_col], errors='coerce')")
                    code_lines.append("    _work = _src.copy(deep=False)")
                    code_lines.append("    _work['_metric'] = _left * _right")
                    code_lines.append("    _work = _work.loc[_work['_metric'].notna()].copy()")
                    code_lines.append(
                        f"    result = _work.groupby(_group_col)['_metric'].sum().reset_index(name={out_col!r})"
                    )
                else:
                    code_lines.append(
                        f"result = _src.groupby({group_col!r})[{target_col!r}].agg({agg!r}).reset_index(name={out_col!r})"
                    )
                code_lines.append(f"result = result.sort_values({out_col!r}, ascending=False)")
                if top_n > 0:
                    code_lines.append(f"result = result.head({top_n})")
            elif op == "filter_equals":
                col = slots.get("column")
                val = slots.get("value")
                case_insensitive = bool(slots.get("case_insensitive", False))
                top_n = int(slots.get("top_n") or 0)
                if not col:
                    return None
                if case_insensitive:
                    code_lines.append(
                        f"_work = df[df[{col!r}].astype(str).str.lower() == str({val!r}).lower()]"
                    )
                else:
                    code_lines.append(f"_work = df[df[{col!r}] == {val!r}]")
                if top_n > 0:
                    code_lines.append(f"result = _work.head({top_n})")
                else:
                    code_lines.append("result = _work")
                preview_source = "_work"
            elif op == "filter_comparison":
                col = slots.get("column")
                op_sym = slots.get("operator") or ">"
                val = slots.get("value")
                if not col:
                    return None
                code_lines.append(f"_work = df[df[{col!r}] {op_sym} {val!r}]")
                code_lines.append("result = _work")
                preview_source = "_work"
            elif op == "filter_count":
                col = slots.get("column")
                op_sym = slots.get("operator") or ">"
                val = slots.get("value")
                if not col:
                    return None
                code_lines.append(f"_num = pd.to_numeric(df[{col!r}], errors='coerce')")
                code_lines.append(f"_mask = _num {op_sym} {val!r}")
                code_lines.append("result = int(_mask.fillna(False).sum())")
            elif op == "filter_contains":
                col = slots.get("column")
                val = slots.get("value")
                if not col:
                    return None
                code_lines.append(
                    f"_work = df[df[{col!r}].astype(str).str.contains({val!r}, case=False, na=False)]"
                )
                code_lines.append("result = _work")
                preview_source = "_work"
            elif op == "filter_multi_conditions":
                conditions = slots.get("conditions") or []
                if not isinstance(conditions, list) or len(conditions) < 2:
                    return None
                cond_parts: List[str] = []
                for cond in conditions:
                    if not isinstance(cond, dict):
                        continue
                    col = str(cond.get("column") or "").strip()
                    op_name = self._normalize_condition_operator(cond.get("operator"))
                    if not op_name:
                        op_name = "equals"
                    value = cond.get("value")
                    if not col or value is None:
                        continue
                    ci = bool(cond.get("case_insensitive", True))
                    if op_name == "contains":
                        cond_parts.append(
                            f"(df[{col!r}].astype(str).str.contains({value!r}, case={str(not ci)}, na=False))"
                        )
                    elif op_name in {"gt", "lt", "gte", "lte"}:
                        num = self._safe_float(value)
                        if num is None:
                            continue
                        sign = ">" if op_name == "gt" else "<" if op_name == "lt" else ">=" if op_name == "gte" else "<="
                        cond_parts.append(
                            f"(pd.to_numeric(df[{col!r}], errors='coerce') {sign} {num!r})"
                        )
                    else:
                        if ci:
                            cond_parts.append(
                                f"(df[{col!r}].astype(str).str.lower() == str({value!r}).lower())"
                            )
                        else:
                            cond_parts.append(f"(df[{col!r}] == {value!r})")
                if len(cond_parts) < 2:
                    return None
                code_lines.append(f"_work = df[{ ' & '.join(cond_parts) }]")
                code_lines.append("result = _work")
                preview_source = "_work"
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
                respect_catalog_rows = self._bool_config(
                    "respect_catalog_preview_rows",
                    "SHORTCUT_RESPECT_CATALOG_PREVIEW_ROWS",
                    False,
                )
                default_preview_rows = self._int_config(
                    "default_preview_rows",
                    "SHORTCUT_RETURN_DF_PREVIEW_ROWS",
                    0,
                )
                source = preview_source if preview_source in {"df", "_work"} else "df"
                rows_from_query = self._normalize_top_n(slots.get("top_n"))
                rows_from_catalog = self._normalize_top_n(args.get("rows")) if respect_catalog_rows else None
                rows = rows_from_query if rows_from_query is not None else rows_from_catalog
                if rows is None:
                    rows = default_preview_rows if default_preview_rows > 0 else 0
                if rows > 0:
                    code_lines.append(f"result = {source}.head({rows})")
                else:
                    code_lines.append(f"result = {source}")
            else:
                return None
        return "\n".join(code_lines).strip() + "\n"
