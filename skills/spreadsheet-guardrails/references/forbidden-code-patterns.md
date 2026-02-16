# Forbidden Code Patterns

Block these patterns in generated code for spreadsheet mutations.

## Disallowed Patterns

- Blind fallback mapping:
```ts
const col = schema[userInput] || schema[0];
```

- Silent error swallowing:
```ts
try { await save(); } catch (_) {}
```

- Full-table overwrite for single-cell change:
```ts
sheet.values = newValues;
```

- Unvalidated destructive schema write:
```ts
table.columns.splice(idx, 1);
await workbook.save();
```

- Position-only column access when headers exist:
```ts
row[3] = value;
```

## Required Replacements

- Replace fallback mapping with confidence-scored resolver.
- Replace swallowed errors with explicit error propagation.
- Replace full overwrite with targeted range update APIs.
- Replace direct destructive write with backup + confirmation + validation.
- Replace index-only targeting with header-key resolution.
