# Tutorial 02: Automated URS-to-OQ Requirements Traceability Matrix (RTM)

| Time: 15–20 min | Level: Beginner / Intermediate | Infrastructure: Local Qdrant (`http://localhost:6333`) + Local Ollama (`qwen3-embedding:8b`) + Langfuse (`http://localhost:3000`) |
| :--- | :--- | :--- |

## Overview

In Computer System Validation (CSV / GAMP 5), establishing and maintaining an end-to-end **Requirements Traceability Matrix (RTM)** is critical for proving that all User Requirements (URS) have corresponding Operational Qualification (OQ) verification test scripts.

This tutorial demonstrates how to automate RTM generation using:
- **Local Qdrant** at `http://localhost:6333` for vector similarity matching.
- **Local Ollama** (`qwen3-embedding:8b`, 4096 dimensions) for semantic embedding of technical specifications.
- **Langfuse Observability** at `http://localhost:3000` to trace URS query inputs, matched OQ test scripts, similarity thresholds, and verification verdicts.

---

## 🔍 Observability with Langfuse

- **`@observe(as_type="embedding")`**: Traces Ollama embedding generation for both test scripts and requirements.
- **`@observe(as_type="span")`**: Traces OQ test script batch indexing.
- **`@observe(as_type="retriever")`**: Traces individual requirement matching lookups against the OQ vector index, logging the requirement ID, text, and top matching test protocols.

---

## 💻 Running the Tutorial

```bash
python tutorials/02_urs_to_oq_traceability/traceability_search.py
```
