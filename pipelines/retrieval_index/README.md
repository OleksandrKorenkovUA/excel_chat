This directory stores the local FAISS retrieval index:

- `index.faiss`
- `meta.json`

Build it from the catalog with:

```bash
python tools/build_index.py --catalog sandbox_service/catalog.json --out retrieval_index
```

The build step requires:

- a running OpenAI-compatible embeddings endpoint (vLLM) at `VLLM_BASE_URL`
- `faiss` installed in the environment

