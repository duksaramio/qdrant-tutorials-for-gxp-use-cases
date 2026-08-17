# Tutorial 10: Indexing Payloads of Random Shape (Dynamic Attributes) for GxP & CSV

| Time: 25–35 min | Level: Intermediate / Advanced | Infrastructure: Local Qdrant (`http://localhost:6333`) + Local Ollama (`qwen3-embedding:8b`) + Langfuse (`http://localhost:3000`) |
| :--- | :--- | :--- |

## Overview

In Life Science Quality (QMS) and Computer System Validation (CSV / GAMP 5), computerized systems across analytical QC labs, manufacturing suites, and cloud platforms generate thousands of open-ended, system-specific telemetry and qualification attributes.

Creating a separate payload index for every distinct incoming attribute causes severe index bloat and high memory overhead.

### The Entity-Attribute-Value (EAV) Solution:
1. **Reshape Dynamic Attributes** into typed EAV arrays (`attrs`, `attrs_num`, `attrs_bool`, `attrs_flat`).
2. **Fixed Indexes:** Build only 8 fixed payload indexes that cover an infinite number of dynamic attributes.
3. **Multi-Attribute Filter Queries:** Execute exact keyword matches, numerical range queries, and hybrid semantic searches against dynamic attributes.
4. **Langfuse Observability:** Monitor dynamic payload queries and filter evaluations in **Langfuse** at `http://localhost:3000`.

---

## 🔍 Observability with Langfuse

- **`@observe(as_type="embedding")`**: Traces dense embeddings (Ollama 4096d) generated for dynamic payload summaries.
- **`@observe(as_type="span")`**: Traces the dynamic attribute reshaping and point batch indexing phase.
- **`@observe(as_type="retriever")`**: Traces exact categorical filters, numerical range lookups, and hybrid semantic search queries with DR bounds.

---

## 💻 Running the Tutorial

```bash
python tutorials/10_indexing_dynamic_payloads/indexing_dynamic_payloads_gxp.py
```
