# Tutorial 05: Hybrid Search with Late-Interaction (ColBERT) Reranking

| Time: 25–35 min | Level: Intermediate / Advanced | Infrastructure: Local Qdrant (`http://localhost:6333`) + Local Ollama (`qwen3-embedding:8b`) |
| :--- | :--- | :--- |

## Overview

In Life Sciences and Computer System Validation (CSV / GAMP 5), finding regulatory documents requires a delicate balance between **recall** (broad search) and **precision** (exact test scripts, parameter limits, and regulatory definitions):
- **Dense embeddings** (via local Ollama `qwen3-embedding:8b`, 4096-dim) capture conceptual context (*"preventing unauthorized changes to analytical data"*).
- **BM25 Sparse embeddings** capture exact alphanumeric codes (*`SOP-QA-042`*, *`21 CFR 11.10(e)`*).
- **ColBERT Late-Interaction Multi-Vectors** perform fine-grained token-level cross-matching (`MaxSim`) to rerank candidates with near-lossless precision.

This tutorial implements a high-throughput **2-stage retrieval pipeline**:
1. **Stage 1 (Fast Recall):** Prefetch candidates using Dense (Ollama) + BM25 Sparse search.
2. **Stage 2 (MaxSim Precision):** Rerank candidate documents using ColBERT token-level multi-vectors (`hnsw_config=HnswConfigDiff(m=0)`).

---

## 🏗️ 2-Stage Retrieval Architecture

```text
                      ┌───────────────────────────┐
                      │        User Query         │
                      └─────────────┬─────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
   Ollama Dense Vector                             BM25 Sparse Vector
   (qwen3-embedding:8b)                            (FastEmbed Qdrant/bm25)
   [4096 Dimensions]                               [Sparse Lexical Indices]
            │                                               │
            ▼                                               ▼
   [Prefetch 1: dense]                             [Prefetch 2: sparse]
   Cosine Semantic Similarity                       Lexical Token Match
            │                                               │
            └───────────────────────┬───────────────────────┘
                                    │
                                    ▼
                      Candidate Document Pool (Top 10)
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │  Stage 2: ColBERT Rerank  │
                      │   Late-Interaction MaxSim │
                      │  ['multi' vector with m=0]│
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      Final High-Precision GxP Hits
```

---

## 💻 Running the Tutorial

```bash
python tutorials/05_hybrid_search_with_reranking/hybrid_search_reranking_gxp.py
```
