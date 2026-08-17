# Tutorial 09: Branch-Aware Search Over Versioned GxP & CSV Document Lifecycles

| Time: 25–35 min | Level: Intermediate | Infrastructure: Local Qdrant (`http://localhost:6333`) + Local Ollama (`qwen3-embedding:8b`) + Langfuse (`http://localhost:3000`) |
| :--- | :--- | :--- |

## Overview

In Life Science Quality Management Systems (QMS / EDMS) and Computer System Validation (CSV / GAMP 5), controlled document repositories evolve across **git-style branches and versioned lifecycles**:
- **`main-effective`:** Officially approved, legally binding GxP SOPs, qualification protocols, and validated baselines.
- **`draft-cc-2024`:** Proposed draft revisions under Change Control (CC) review.
- **`site-eu-overlay`:** Regional manufacturing site overlays (incorporating local EU GMP Annex 11 / Qualified Person requirements).

This tutorial demonstrates how to index a versioned GxP corpus in **Local Qdrant (`http://localhost:6333`)** with **Ollama (`qwen3-embedding:8b`, 4096-dim)** and scope each vector query strictly to a single branch's live view, monitored via **Langfuse** at `http://localhost:3000`.

---

## 🔍 Observability with Langfuse

- **`@observe(as_type="embedding")`**: Traces dense embeddings generated for versioned document updates and semantic search queries.
- **`@observe(as_type="span")`**: Traces commit updates and supersede history tracking.
- **`@observe(as_type="retriever")`**: Traces branch exact lookups and branch-scoped vector search queries.

---

## 💻 Running the Tutorial

```bash
python tutorials/09_branch_aware_search/branch_aware_search_gxp.py
```
