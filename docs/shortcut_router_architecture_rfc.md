# Shortcut Router RFC: Canonical Query IR and Intent Compatibility

Status: Draft (Phase-ready)
Owner: Spreadsheet Pipeline Team
Scope: `pipelines/shortcut_router/shortcut_router.py`, `pipelines/catalog.json`, retrieval index lifecycle

## 1. Problem Statement

Current failures repeat in two forms:

1. The model extracts a correct condition, but router ignores or overwrites it.
2. Retrieval returns a strong candidate, but final selected intent is semantically wrong for the query constraints.

These failures are architectural, not only prompt-related.

## 2. Root Causes (Architecture-Level)

1. Multi-stage interpretation without shared state:
- normalize query
- retrieve candidates
- select intent
- resolve slots
- compile code
Each stage can reinterpret the query independently.

2. No canonical intermediate representation (IR):
- Constraints (filters/grouping/metric/order/limit) are re-derived multiple times with different heuristics.

3. Contract drift between `catalog.json` and runtime behavior:
- Runtime injects hidden slots/heuristics not explicitly modeled in intent contracts.

4. Candidate compression per intent:
- Retrieval keeps only one best example per intent, which reduces selector context quality.

5. Online learning promotion lacks correctness gate:
- Promotion is based on execution success + consistency, not answer correctness.

## 3. Target Architecture

## 3.1 Canonical Query IR (single source of truth)

Build once per request and reuse downstream.

IR schema (conceptual):

- `language`: `uk|en|other`
- `intent_family`: `filter|groupby|aggregation|lookup|edit|other`
- `filters`: list of
  - `column` (optional before column binding)
  - `operator`: `eq|contains|startswith|endswith|gt|lt|gte|lte|range`
  - `value`
  - `value_type`: `text|number|date|bool`
- `group_by`: list of dimension slots
- `metrics`: list of metric intents (`count|sum|mean|min|max|median|custom`)
- `sort`: optional (`column`,`direction`)
- `limit`: optional int
- `cues`: normalized lexical cues (e.g. startswith cue, comparison cue, revenue-product cue)

Rule: downstream stages may only refine bindings, not invent contradictory constraints.

## 3.2 Intent Compatibility Layer

Before final intent commit, validate intent against IR constraints.

Examples:
- If IR has numeric comparison filter, `groupby_count` without filter support is incompatible.
- If IR has startswith filter and grouping, selected intent must support pre-group filter.
- If IR has product metric (`price × qty`), any count-only intent is incompatible.

If incompatible:
1. try next candidate,
2. or fallback to planner path (non-shortcut),
3. never force an incompatible shortcut.

## 3.3 Deterministic Slot Merge Policy

Slot sources (priority):
1. explicit user constraints from IR
2. intent preset transforms (`_resolve_intent_and_slots`)
3. llm slot fill
4. text heuristic fallback
5. defaults

Conflict resolution:
- higher-priority source wins,
- log structured conflict record.

## 3.4 Compile layer must be purely declarative

Compiler reads resolved slots only.
No semantic reinterpretation in `_compile_plan`.
No hidden query parsing inside compile.

## 4. Intent Contract Changes in catalog.json

Each intent should explicitly declare:
- supported filter operators
- whether pre-filter before groupby is supported
- whether numeric-only columns required
- whether text-only columns required

Proposed additions per intent (schema extension):
- `capabilities.filters` (operators)
- `capabilities.prefilter_before_groupby` (bool)
- `capabilities.metric_types` (`count|numeric|product`)
- `capabilities.requires_numeric_target` (bool)

This enables compatibility checks without hardcoded intent IDs.

## 5. Migration Plan (Phased, low-risk)

## Phase 0 (Immediate safety, no behavior break)

1. Add structured decision trace object in logs:
- `ir_summary`
- `candidate_scores`
- `compatibility_rejections`
- `slot_conflicts`

2. Add invariant tests:
- intent-selection compatibility invariants
- slot merge deterministic behavior
- no contradictory compile code from same IR

## Phase 1 (Introduce IR extraction)

1. Implement `extract_query_ir(query, profile)`.
2. Wire router flow to pass IR through selection, slot fill, compile.
3. Keep old path behind feature flag.

Flags:
- `SHORTCUT_IR_ENABLED`
- `SHORTCUT_COMPAT_CHECK_ENABLED`

## Phase 2 (Intent compatibility enforcement)

1. Build intent capability map from catalog.
2. Enforce compatibility before accepting intent.
3. If rejected, fallback to next candidate / planner.

## Phase 3 (Catalog contract hardening)

1. Update intents with capabilities metadata.
2. Add CI check: every intent must declare capabilities.
3. Add migration script to enrich existing catalog entries.

## Phase 4 (Learning pipeline hardening)

1. Split learning into `staging` and `promoted`.
2. Promote only with correctness gate (golden queries / offline evaluator / manual review threshold).
3. Rebuild index from promoted examples only.

## 6. Success Metrics

Primary:
- Intent mismatch rate (selected intent incompatible with extracted constraints)
- Slot contradiction rate (extracted condition dropped/overwritten)

Secondary:
- Fallback-to-planner rate
- Wrong-answer-on-shortcut rate from regression suite
- Learning rollback events

## 7. Test Strategy

1. Golden query packs by intent family:
- filter exact/contains/comparison/range
- grouped with prefilters
- revenue product
- startswith/endswith

2. Differential tests:
- old path vs IR path output parity where expected.

3. Contract tests:
- catalog capabilities coverage
- intent compiler consumes only declared slots.

## 8. Immediate actionable backlog

1. Implement `extract_query_ir` (read-only path first).
2. Add `intent_compatible(intent, ir)`.
3. Add deterministic `merge_slots(ir, preset, llm, text, defaults)`.
4. Add compatibility logs and regression dashboards.
5. Enable `SHORTCUT_IR_ENABLED` in shadow mode, compare decisions.

---

This RFC is intentionally incremental: it reduces wrong-routing risk first, then migrates core decisioning to a single canonical representation.
