# Column Matching Rules

Use this algorithm to map user language to real schema columns.

## Deterministic Matching Pipeline

1. Normalize user token and schema names:
- lowercase
- trim whitespace
- replace `_` and `-` with space
- collapse multiple spaces
- remove surrounding punctuation
2. Attempt exact normalized match.
3. Attempt exact alias match from project alias dictionary.
4. Attempt token-overlap and fuzzy match only if steps 2-3 fail.
5. Rank candidates by score and enforce confidence thresholds.

## Confidence Thresholds

- `>= 0.95`: accept automatically.
- `0.85 - 0.94`: accept only if top score beats second score by `>= 0.08`.
- `< 0.85`: do not map automatically; ask user to confirm.
- Any tie within `0.05`: treat as ambiguous and ask clarification.

## Required Evidence in Output

- Show original user field.
- Show resolved schema column.
- Show confidence score.
- Show why other near candidates were rejected.

## Ambiguity Handling

When ambiguous:
1. Present top 2-3 candidate columns.
2. Show short semantic difference.
3. Request explicit user confirmation.

## Hard Safety Constraints

- Never resolve by position index alone when headers exist.
- Never guess missing columns.
- Never map derived metric names to base columns without explicit rule.
- Never use fuzzy matching when schema provides explicit synonym metadata.

## Minimal Alias Dictionary Shape

Use a deterministic map format:

```json
{
  "email": ["email", "e-mail", "mail", "email address"],
  "customer_id": ["customer id", "client id", "cid"]
}
```

Keep aliases versioned with schema updates.
