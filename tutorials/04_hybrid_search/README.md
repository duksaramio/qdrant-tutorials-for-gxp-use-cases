# Tutorial 04: Dense + BM25 Sparse Hybrid Search with Reciprocal Rank Fusion (RRF)

| Time: 25–30 min | Level: Intermediate | Infrastructure: Local Qdrant (`http://localhost:6333`) + Local Ollama (`qwen3-embedding:8b`) |
| :--- | :--- | :--- |

## Overview

In Life Sciences and Computer System Validation (CSV / GAMP 5), searching through controlled quality records poses a unique dual challenge:
1. **Dense Semantic Embeddings** (via local Ollama `qwen3-embedding:8b`, 4096-dim) capture conceptual intent (*"unauthorized alteration of analytical records"*).
2. **Sparse Lexical Embeddings** (via BM25 with server-side IDF) pinpoint exact alphanumeric identifiers, SOP numbers, and regulatory clauses (*`SOP-QA-042`*, *`21 CFR 11.10(e)`*, *`CAPA-2023-019`*).

This tutorial demonstrates how to combine both retrieval strategies using **Qdrant Named Vectors** and **Reciprocal Rank Fusion (RRF)** on a local Qdrant instance.

---

## 🏗️ Architecture

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
   [Prefetch 1: dense_vector]                      [Prefetch 2: bm25_sparse_vector]
   Cosine Semantic Similarity                       Lexical Token / IDF Match
            │                                               │
            └───────────────────────┬───────────────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │    Reciprocal Rank        │
                      │      Fusion (RRF)         │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                       Top Relevant GxP Records
```

---

## 💻 Running the Tutorial

```bash
python tutorials/04_hybrid_search/hybrid_search_gxp.py
```
