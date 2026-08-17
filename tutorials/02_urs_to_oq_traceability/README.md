# Tutorial 02: Automated URS-to-OQ Requirements Traceability Matrix (RTM)

| Time: 20–25 min | Level: Intermediate | Infrastructure: Local Qdrant (`http://localhost:6333`) + Local Ollama (`qwen3-embedding:8b`) |
| :--- | :--- | :--- |

## Overview

In Computer System Validation (CSV / GAMP 5), constructing and maintaining a **Requirements Traceability Matrix (RTM)** is mandatory for demonstrating that every User Requirement (URS) is tested and verified by an Operational Qualification (OQ) or Performance Qualification (PQ) test script.

Manual RTM generation for enterprise systems (e.g. MES, LIMS, QMS) with hundreds of requirements is slow, error-prone, and painful during audits.

This tutorial demonstrates how to use **Qdrant Vector Search** and local **Ollama (`qwen3-embedding:8b`)** embeddings (4096-dimensional vectors) to automate URS-to-OQ test mapping.

---

## 🎯 What You Will Learn

1. **Test Verification Indexing:** Vectorize detailed OQ test scripts (steps, modules, expected results).
2. **Automated Trace Matching:** Query User Requirement statements against the OQ vector space.
3. **Traceability Scoring:** Automatically distinguish confirmed test traces from unverified requirement gaps based on cosine similarity thresholds.

---

## 💻 Running the Tutorial

```bash
python tutorials/02_urs_to_oq_traceability/traceability_search.py
```
