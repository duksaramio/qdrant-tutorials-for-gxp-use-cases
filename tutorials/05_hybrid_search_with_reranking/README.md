# Tutorial 05: Hybrid Search with Late-Interaction (ColBERT) Reranking

| Time: 25–35 min | Level: Intermediate / Advanced | Infrastructure: Local Qdrant (`http://localhost:6333`) + Local Ollama (`qwen3-embedding:8b`) + Langfuse (`http://localhost:3000`) |
| :--- | :--- | :--- |

## Overview

In Life Sciences and Computer System Validation (CSV / GAMP 5), finding regulatory documents requires high recall and pinpoint precision.

This tutorial implements a 2-stage retrieval pipeline:
1. **Stage 1 (Fast Recall):** Prefetch candidates using Dense (Ollama 4096d) + BM25 Sparse search.
2. **Stage 2 (MaxSim Precision):** Rerank candidate documents using ColBERT token-level multi-vectors (`hnsw_config=HnswConfigDiff(m=0)`).
3. **Observability:** Monitor each stage's latency, intermediate candidate pools, and final rerank scores in **Langfuse** (`http://localhost:3000`).

---

## 🔍 Observability with Langfuse

- **`@observe(as_type="embedding")`**: Traces the 3 distinct embedding pipelines: Ollama dense semantic (4096d), BM25 sparse keyword, and ColBERT token-level multivector embeddings.
- **`@observe(as_type="span")`**: Traces multi-vector point uploading and schema verification.
- **`@observe(as_type="retriever")`**: Traces Stage 1 prefetching (dense & sparse hits), RRF score aggregation, and Stage 2 ColBERT MaxSim reranked hit ordering.

---

## 💻 Running the Tutorial

```bash
python tutorials/05_hybrid_search_with_reranking/hybrid_search_reranking_gxp.py
```
