import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional


def _load_catalog(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_catalog(path: str, catalog: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _find_intent(catalog: Dict[str, Any], intent_id: str) -> Optional[Dict[str, Any]]:
    for intent in catalog.get("intents") or []:
        if str(intent.get("id") or "").strip() == intent_id:
            return intent
    return None


def _find_existing_query(catalog: Dict[str, Any], query: str) -> Optional[str]:
    q_norm = query.strip().casefold()
    if not q_norm:
        return None
    for intent in catalog.get("intents") or []:
        intent_id = str(intent.get("id") or "").strip()
        for example in intent.get("examples") or []:
            if str(example or "").strip().casefold() == q_norm:
                return intent_id
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append a failed user query into catalog intent examples and optionally rebuild retrieval index."
    )
    parser.add_argument("--query", required=True, help="Failed user query text to add.")
    parser.add_argument("--intent", required=True, help="Intent id where the query should be added.")
    parser.add_argument("--catalog", default="pipelines/catalog.json", help="Path to catalog.json.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild retrieval index after catalog update.")
    parser.add_argument("--out", default="pipelines/retrieval_index", help="Retrieval index output directory.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embeddings.")
    args = parser.parse_args()

    catalog_path = args.catalog
    if not os.path.exists(catalog_path):
        print(f"Catalog not found: {catalog_path}", file=sys.stderr)
        return 2

    query = str(args.query or "").strip()
    if not query:
        print("Query is empty.", file=sys.stderr)
        return 2

    catalog = _load_catalog(catalog_path)
    existing_in = _find_existing_query(catalog, query)
    if existing_in:
        print(f"Query already exists in intent '{existing_in}'; skip update.")
    else:
        intent = _find_intent(catalog, args.intent)
        if not intent:
            ids: List[str] = [str(i.get("id") or "") for i in (catalog.get("intents") or [])]
            print(f"Intent not found: {args.intent}", file=sys.stderr)
            print(f"Known intents: {', '.join([i for i in ids if i])}", file=sys.stderr)
            return 2
        examples = intent.get("examples")
        if not isinstance(examples, list):
            examples = []
            intent["examples"] = examples
        examples.append(query)
        _save_catalog(catalog_path, catalog)
        print(f"Added query to intent '{args.intent}' in {catalog_path}")

    if args.rebuild:
        from build_index import build_index  # local import to avoid faiss dependency unless requested

        build_index(catalog_path, args.out, args.batch_size)
        print("Index rebuild completed.")
    else:
        print("Catalog updated. Index rebuild skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

