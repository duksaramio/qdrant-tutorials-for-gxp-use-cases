# Tutorial 02: Automated Requirements-to-Test Traceability Matrix (URS $\rightarrow$ OQ)

| Time: 10 min | Level: Beginner / Intermediate | Focus: GAMP 5 / CSV Traceability |
| :--- | :--- | :--- |

## Overview
Under **GAMP 5 (2nd Edition)** and CSV regulations, life science organizations must demonstrate that every critical User Requirement Specification (URS) is tested and verified in an Operational Qualification (OQ) or Performance Qualification (PQ) test protocol.

In large enterprise systems with hundreds of requirements, building and maintaining the **Requirements Traceability Matrix (RTM)** manually leads to verification gaps and human error during regulatory audits.

This tutorial demonstrates how to use **Qdrant Vector Search** to:
1. Index OQ qualification test scripts as vector embeddings.
2. Query each URS statement semantically against test scripts.
3. Automatically generate candidate traceability links with similarity confidence scores.

## Run Tutorial

```bash
python tutorials/02_urs_to_oq_traceability/traceability_search.py
```
