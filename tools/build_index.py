import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import httpx
import numpy as np

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover - runtime guard
    print("FAISS is required to build the retrieval index. Install faiss-cpu.", file=sys.stderr)
    sys.exit(2)


def _load_catalog(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_examples(catalog: Dict[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for intent in catalog.get("intents") or []:
        intent_id = str(intent.get("id") or "").strip()
        for text in intent.get("examples") or []:
            t = str(text or "").strip()
            if intent_id and t:
                rows.append({"intent_id": intent_id, "text": t})
    return rows


def _embedding_config(catalog: Dict[str, Any]) -> Tuple[str, str, int, str]:
    emb = catalog.get("embedding") or {}
    base_url = (
        os.getenv("EMB_BASE_URL")
        or os.getenv("VLLM_BASE_URL")
        or emb.get("base_url")
        or "http://gpu-test.silly.billy:8022/v1"
    ).rstrip("/")
    model = os.getenv("EMB_MODEL") or os.getenv("VLLM_EMBED_MODEL") or emb.get("model") or "multilingual-embeddings"
    timeout_s = int(os.getenv("VLLM_TIMEOUT_S") or emb.get("timeout_s") or 30)
    api_key = (os.getenv("EMB_API_KEY") or os.getenv("VLLM_API_KEY") or "DUMMY_KEY").strip()
    if not base_url or not model:
        raise RuntimeError("Missing embedding config: set EMB_BASE_URL and EMB_MODEL (or VLLM_*), or provide in catalog.")
    return base_url, model, timeout_s, api_key


def _embed_batch(base_url: str, model: str, timeout_s: int, api_key: str, inputs: List[str]) -> np.ndarray:
    url = f"{base_url}/embeddings"
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "input": inputs}
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    vectors = [item.get("embedding") for item in (data.get("data") or [])]
    if not vectors:
        raise RuntimeError("Embeddings response did not contain vectors.")
    arr = np.array(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise RuntimeError(f"Unexpected embedding shape: {arr.shape}")
    faiss.normalize_L2(arr)
    return arr


def build_index(catalog_path: str, out_dir: str, batch_size: int) -> None:
    t0 = time.time()
    catalog = _load_catalog(catalog_path)
    rows = _iter_examples(catalog)
    if not rows:
        raise RuntimeError("No examples found in catalog.")

    base_url, model, timeout_s, api_key = _embedding_config(catalog)
    print(f"Embedding {len(rows)} examples via {base_url}/embeddings model={model}")

    all_vecs: List[np.ndarray] = []
    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i : i + batch_size]
        inputs = [r["text"] for r in batch_rows]
        bt0 = time.time()
        vecs = _embed_batch(base_url, model, timeout_s, api_key, inputs)
        all_vecs.append(vecs)
        print(f"  batch {i:5d}-{i+len(batch_rows)-1:5d} -> shape={vecs.shape} in {time.time()-bt0:.2f}s")

    matrix = np.vstack(all_vecs).astype(np.float32)
    dim = int(matrix.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, "index.faiss")
    meta_path = os.path.join(out_dir, "meta.json")
    faiss.write_index(index, index_path)

    meta = {
        "catalog_version": catalog.get("catalog_version"),
        "dim": dim,
        "rows": rows,
        "embedding": {
            "base_url": base_url,
            "model": model,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    dt = time.time() - t0
    print(f"Wrote {index_path} and {meta_path} (dim={dim}, rows={len(rows)}) in {dt:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS retrieval index from catalog.json examples.")
    parser.add_argument("--catalog", default=os.getenv("SHORTCUT_CATALOG_PATH", "sandbox_service/catalog.json"))
    parser.add_argument("--out", default="retrieval_index")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    build_index(args.catalog, args.out, args.batch_size)


if __name__ == "__main__":
    main()
