# Tutorial 06: Multivectors and Late Interaction with HNSW Optimization (m=0)

| Time: 25–35 min | Level: Intermediate / Advanced | Infrastructure: Local Qdrant (`http://localhost:6333`) + Local Ollama (`qwen3-embedding:8b`) + Langfuse (`http://localhost:3000`) |
| :--- | :--- | :--- |

## Overview

In Life Sciences and Computer System Validation (CSV / GAMP 5), controlled documents (SOPs, Validation Protocols, URS, System Risk Assessments) contain intricate, multi-clause regulatory specifications.

Single-vector dense representations condense an entire document into one pooled vector, averaging out granular token-level details. **ColBERT Late-Interaction Multi-Vectors** preserve a distinct 128-dimensional embedding for every single token in the document, scoring similarities using token-level **MaxSim** late interaction.

### The Qdrant Solution: HNSW m=0 Optimization
1. Store a **Dense Single Vector** (via local Ollama `qwen3-embedding:8b`, 4096-dim) with default **HNSW enabled** for fast first-pass candidate retrieval.
2. Store the **ColBERT Multi-Vector** with **HNSW disabled (`hnsw_config=HnswConfigDiff(m=0)`)**, strictly used for second-pass exact MaxSim reranking.
3. Observe token scores and pipeline latency live in **Langfuse** at `http://localhost:3000`.

---

## 🔍 Observability with Langfuse

- **`@observe(as_type="embedding")`**: Traces single dense vectors (Ollama 4096d) and token-level ColBERT multivectors (128d/token).
- **`@observe(as_type="span")`**: Traces optimized HNSW m=0 collection indexing.
- **`@observe(as_type="retriever")`**: Compares single-vector pooled cosine similarity against token-level MaxSim late interaction scoring and regulatory filters.

---

## 💻 Running the Tutorial

```bash
python tutorials/06_multivectors_and_late_interaction/multivectors_late_interaction_gxp.py
```
