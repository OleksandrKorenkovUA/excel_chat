# Table Mutation Playbooks

Apply these playbooks for reliable writes and saves.

## Global Pre-Write Checklist

1. Snapshot schema: column order, types, formulas, validation rules.
2. Snapshot shape: row count and column count.
3. Validate target identifiers (sheet/table/range) exist.
4. Validate requested mutation scope (single cell vs batch vs schema change).

## Playbook A: Cell Value Update

Use for editing one or more cell values without schema change.

Steps:
1. Resolve row key and column key deterministically.
2. Validate type compatibility (`number`, `date`, `string`, `boolean`).
3. Write only targeted cells.
4. Recalculate dependent formulas through engine-native recalc.
5. Save.
6. Verify changed cell values and unchanged table shape.

Required post-checks:
- row count unchanged
- column count unchanged
- no formula loss in unaffected cells

## Playbook B: Row Count Change (Add/Delete Rows)

Use for insertion or deletion of data rows.

Steps:
1. Validate primary key uniqueness before insert.
2. Preserve table headers and named ranges.
3. Insert/delete rows using table-aware API (not raw text rewrite).
4. Reapply row-level formulas/validation for newly added rows.
5. Save.
6. Verify row count delta matches requested operation.

Required post-checks:
- expected row delta applied
- column schema unchanged
- formulas propagated correctly

## Playbook C: Column Count Change (Add/Delete/Reorder Columns)

Use only for explicit schema-change requests.

Steps:
1. Require explicit user confirmation for destructive operations.
2. Backup or transaction-wrap before mutation.
3. Apply change with schema-aware operation.
4. Update formula references and validation rules.
5. Save.
6. Verify column count and order match requested result.

Required post-checks:
- expected column delta/order applied
- formula references valid
- no orphaned named ranges

## Save Integrity Contract

Always enforce:
1. `pre_hash != post_hash` when mutation expected.
2. `pre_hash == post_hash` when dry-run or no-op.
3. `post_save_readback` equals intended written state.

## Error Handling Rules

- Abort and return explicit error when write target is missing.
- Abort and request clarification on ambiguous row selectors.
- Abort on partial write failure; do not report success.
- Return remediation hints with exact failing constraint.

## Example: Complex Request

Request: "Add 2 rows, update `status` for customer 1007, then save."

Execution pattern:
1. Resolve `status` with column-matching rules.
2. Insert 2 rows with default formulas/validation.
3. Locate customer `1007` by primary key.
4. Update only `status` cell.
5. Save and read back.
6. Return:
- inserted rows: `+2`
- updated cells: `1`
- schema change: `none`
- verification: `passed`
