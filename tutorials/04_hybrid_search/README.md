# Tutorial 04: Dense + BM25 Sparse Hybrid Search with Reciprocal Rank Fusion (RRF)

| Time: 25–30 min | Level: Intermediate | Infrastructure: Local Qdrant (`http://localhost:6333`) + Local Ollama (`qwen3-embedding:8b`) + Langfuse (`http://localhost:3000`) |
| :--- | :--- | :--- |

## Overview

In Life Sciences and Computer System Validation (CSV / GAMP 5), searching through controlled quality records poses a unique dual challenge:
1. **Dense Semantic Embeddings** (via local Ollama `qwen3-embedding:8b`, 4096-dim) capture conceptual intent (*"unauthorized alteration of analytical records"*).
2. **Sparse Lexical Embeddings** (via BM25 with server-side IDF) pinpoint exact alphanumeric identifiers, SOP numbers, and regulatory clauses (*`SOP-QA-042`*, *`21 CFR 11.10(e)`*, *`CAPA-2023-019`*).

This tutorial demonstrates how to combine both retrieval strategies using **Qdrant Named Vectors** and **Reciprocal Rank Fusion (RRF)** on a local Qdrant instance, monitored in real time with **Langfuse Observability**.

---

## 🔍 Observability with Langfuse

- **`@observe(as_type="embedding")`**: Traces parallel generation of dense embeddings (Ollama 4096d) and sparse lexical vectors (FastEmbed BM25).
- **`@observe(as_type="span")`**: Traces hybrid document point batch ingestion.
- **`@observe(as_type="retriever")`**: Traces comparison lookups (Dense-only vs. BM25-only vs. RRF-fused) and metadata-filtered hybrid retrieval.

---

## 💻 Running the Tutorial

```bash
python tutorials/04_hybrid_search/hybrid_search_gxp.py
```
