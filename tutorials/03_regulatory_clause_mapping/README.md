# Tutorial 03: 21 CFR Part 11 & EU Annex 11 Regulatory Clause Mapping

| Time: 20–25 min | Level: Intermediate | Infrastructure: Local Qdrant (`http://localhost:6333`) + Local Ollama (`qwen3-embedding:8b`) |
| :--- | :--- | :--- |

## Overview

In Life Sciences and Computer System Validation (CSV), Quality and Regulatory Affairs teams frequently review vendor documentation, System Architecture Documents (SAD), and Software Requirements Specifications (SRS) to assess compliance against predicate regulations:
- **FDA 21 CFR Part 11** (Electronic Records; Electronic Signatures)
- **EU GMP Annex 11** (Computerised Systems)
- **GAMP 5 2nd Edition** (Risk-Based Approach to Compliant GxP Computerized Systems)

This tutorial demonstrates how to use **Qdrant Vector Search** and local **Ollama (`qwen3-embedding:8b`)** embeddings (4096-dimensional vectors) to automatically map vendor software features to specific regulatory clauses.

---

## 🎯 What You Will Learn

1. **Regulatory Indexing:** Vectorize predicate rules (21 CFR 11.10, 11.50, 11.70, EU Annex 11.7, 11.9).
2. **Vendor Assessment:** Query vendor technical claims against the regulatory index.
3. **Automated Citation:** Retrieve exact regulatory clauses and confidence scores for compliance gap analysis.

---

## 💻 Running the Tutorial

```bash
python tutorials/03_regulatory_clause_mapping/part11_clause_mapping.py
```
