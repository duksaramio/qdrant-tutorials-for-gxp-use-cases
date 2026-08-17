# Tutorial 03: 21 CFR Part 11 & EU Annex 11 Regulatory Clause Mapping

| Time: 15–20 min | Level: Intermediate | Infrastructure: Local Qdrant (`http://localhost:6333`) + Local Ollama (`qwen3-embedding:8b`) + Langfuse (`http://localhost:3000`) |
| :--- | :--- | :--- |

## Overview

In CSV vendor assessments, regulatory audit preparation, and gap analyses, life science compliance leads must prove that technical architectural controls map to specific regulatory citations (such as **21 CFR 11.10(e)** time-stamped audit trails, **21 CFR 11.50** signature manifestations, or **EU Annex 11.7** backup verification).

This tutorial demonstrates how to:
1. Index regulatory predicate rules in **Local Qdrant** (`http://localhost:6333`).
2. Generate semantic embeddings of vendor architectural controls via **Local Ollama** (`qwen3-embedding:8b`, 4096 dimensions).
3. Trace the automated predicate rule mapping pipeline with **Langfuse Observability** at `http://localhost:3000`.

---

## 🔍 Observability with Langfuse

- **`@observe(as_type="embedding")`**: Traces Ollama embedding generation for both regulatory predicate rules and vendor technical descriptions.
- **`@observe(as_type="span")`**: Traces regulatory clause indexing and point ingestion.
- **`@observe(as_type="retriever")`**: Traces vendor specification matching, logging input feature text and top matched regulatory predicate citations with similarity scores.

---

## 💻 Running the Tutorial

```bash
python tutorials/03_regulatory_clause_mapping/part11_clause_mapping.py
```
