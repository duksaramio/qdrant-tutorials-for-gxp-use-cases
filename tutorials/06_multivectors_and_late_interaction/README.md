# Tutorial 06: Multivectors and Late Interaction with HNSW Optimization (m=0)

| Time: 25–35 min | Level: Intermediate / Advanced | Infrastructure: Local Qdrant (`http://localhost:6333`) + Local Ollama (`qwen3-embedding:8b`) |
| :--- | :--- | :--- |

## Overview

In Life Sciences and Computer System Validation (CSV / GAMP 5), controlled documents (SOPs, Validation Protocols, URS, System Risk Assessments) contain intricate, multi-clause regulatory specifications.

Single-vector dense representations (like those produced by standard sentence transformers or LLM embeddings) condense the entire document into one pooled vector, averaging out granular token-level details. **ColBERT Late-Interaction Multi-Vectors** preserve a distinct 128-dimensional embedding for every single token in the document, scoring similarities using token-level **MaxSim** late interaction.

### The RAM & Scaling Challenge
Building an HNSW graph on 100+ token vectors per document causes heavy RAM overhead and slow indexing times.

### The Qdrant Solution: HNSW m=0 Optimization
1. Store a **Dense Single Vector** (via local Ollama `qwen3-embedding:8b`, 4096-dim) with default **HNSW enabled** for fast first-stage candidate retrieval.
2. Store the **ColBERT Multi-Vector** with **HNSW disabled (`hnsw_config=HnswConfigDiff(m=0)`)**, strictly used for second-stage exact MaxSim reranking.

---

## 🏗️ Architecture

```text
               ┌────────────────────────────────────────────────────────┐
               │                      User Query                        │
               │  "21 CFR Part 11 requirements for e-signatures..."     │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
         Ollama Dense Vector                             ColBERT Multi-Vector
        (qwen3-embedding:8b)                            (128-dim per query token)
         [4096 Dimensions]                                        │
                  │                                               │
                  ▼                                               │
        [Stage 1: Dense Prefetch]                                 │
        Fast ANN Candidate Recall (HNSW ON)                       │
                  │                                               │
                  ▼                                               │
        Top-10 Candidate Pool                                     │
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          ▼
                      ┌───────────────────────────────────────┐
                      │ Stage 2: ColBERT MaxSim Reranking     │
                      │ ['colbert' multivector with m=0]      │
                      └───────────────────┬───────────────────┘
                                          │
                                          ▼
                                High-Precision GxP Hits
```

---

## 💻 Running the Tutorial

```bash
python tutorials/06_multivectors_and_late_interaction/multivectors_late_interaction_gxp.py
```
