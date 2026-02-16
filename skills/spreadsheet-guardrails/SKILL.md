---
name: spreadsheet-guardrails
description: Reduce hallucinations during spreadsheet and table-task coding. Use when implementing or modifying features that read table schemas, match user column names to real columns, edit cell values, add/remove rows or columns, persist table changes, validate post-save integrity, or enforce strict generation constraints for allowed and forbidden code patterns.
---

# Spreadsheet Guardrails

Follow a deterministic workflow for every spreadsheet task.

## Core Workflow

1. Discover schema before writing logic.
2. Resolve user field names to real column identifiers using `references/column-matching.md`.
3. Refuse to guess when multiple columns are plausible matches.
4. Plan mutation type (cell edit, row edit, column edit, resize) before coding.
5. Preserve invariants during write operations using `references/table-mutation-playbooks.md`.
6. Validate persisted output against pre-mutation expectations.
7. Report assumptions and confidence in the final response.

## Mandatory Rules

- Use existing schema metadata as source of truth for column names and types.
- Prefer exact-match mapping over fuzzy matching.
- Ask for clarification when confidence is below threshold or ties exist.
- Keep operations idempotent where possible.
- Recompute derived fields only through existing formula logic, not hardcoded values.
- Fail fast with explicit errors when required columns are missing.
- Perform pre-write and post-write validation for row count, column count, and data type compatibility.

## Forbidden Behavior

- Do not invent column names, table names, ranges, or sheet identifiers.
- Do not silently map a user field to a low-confidence column.
- Do not drop formulas, validation, formatting rules, or protected ranges unless explicitly requested.
- Do not overwrite an entire table when only targeted mutation is required.
- Do not generate code that swallows write or parse errors.
- Do not commit schema-changing writes without backup or rollback path.

## Resource Loading

- Read `references/column-matching.md` for matching algorithm, thresholds, and ambiguity handling.
- Read `references/table-mutation-playbooks.md` for safe mutation patterns and persistence invariants.
- Read `references/forbidden-code-patterns.md` before generating final code for mutation flows.

## Output Contract

- Return resolved column mapping with confidence score.
- Return mutation plan before execution for destructive or schema-changing tasks.
- Return post-save verification summary with pass/fail checks.
