# Tutorial 08: Multi-Representation Search Across Titles, Scopes, and Body Chunks for GxP & CSV

| Time: 30–45 min | Level: Intermediate / Advanced | Infrastructure: Local Qdrant (`http://localhost:6333`) + Local Ollama (`qwen3-embedding:8b`) + Langfuse (`http://localhost:3000`) |
| :--- | :--- | :--- |

## Overview

In Life Sciences and Computer System Validation (CSV / GAMP 5), a controlled document is rarely well-represented by a single embedding.

This tutorial builds a **Multi-Representation Search Pipeline** on **Local Qdrant (`http://localhost:6333`)** using **Ollama (`qwen3-embedding:8b`, 4096-dim)**, **BM25 Sparse vectors**, and **Langfuse Observability**:
1. **Schema Design:** Every document chunk is indexed with four distinct named vectors (`dense_chunk`, `dense_title`, `dense_scope`, `sparse_title`).
2. **Multi-Vector Prefetching:** Executes parallel sub-queries across all representations.
3. **Reciprocal Rank Fusion (RRF):** Merges the ranked candidate lists across dense semantic and sparse lexical signals.
4. **Grouped Retrieval (`query_points_groups`):** Automatically groups matching chunk hits by `document_id`.
5. **Observability:** Trace 4-way vector queries, RRF rankings, and parent document group hierarchies in **Langfuse** at `http://localhost:3000`.

---

## 🔍 Observability with Langfuse

- **`@observe(as_type="embedding")`**: Traces parallel generation of dense embeddings (Ollama 4096d) and sparse lexical vectors (FastEmbed BM25).
- **`@observe(as_type="span")`**: Traces chunked document multi-representation indexing.
- **`@observe(as_type="retriever")`**: Traces 4-way prefetch fusion and grouped document resolution.

---

## 💻 Running the Tutorial

```bash
python tutorials/08_multi_representation_search/multi_representation_search_gxp.py
```
