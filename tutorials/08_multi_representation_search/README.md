# Tutorial 08: Multi-Representation Search Across Titles, Scopes, and Body Chunks for GxP & CSV

| Time: 30–45 min | Level: Intermediate / Advanced | Infrastructure: Local Qdrant (`http://localhost:6333`) + Local Ollama (`qwen3-embedding:8b`) |
| :--- | :--- | :--- |

## Overview

In Life Sciences and Computer System Validation (CSV / GAMP 5), a controlled document is rarely well-represented by a single embedding:
- **The Document Title** carries the formal system name, SOP code, and regulatory identity (`SOP-QA-042: Electronic Records, Signatures, and Audit Trail Review`).
- **The Executive Scope / Abstract** carries high-level regulatory frameworks (`21 CFR Part 11`, `EU Annex 11`, `GAMP 5 Category 4`).
- **The Specific Body Chunks** contain granular test scripts, acceptance criteria, or failure mode mitigations.
- **The Lexical Sparse Title** carries exact acronyms (`Empower 3 CDS`, `RTO/RPO`, `Modbus TCP/IP`).

If all representations are merged into a single dense vector, the title gets diluted, specific test conditions are averaged out, and chunk-level grounding disappears.

This tutorial builds a **Multi-Representation Search Pipeline** on **Local Qdrant (`http://localhost:6333`)** using **Ollama (`qwen3-embedding:8b`, 4096-dim)** and **BM25 Sparse vectors**:
1. **Schema Design:** Every document chunk is indexed as a point with four distinct named vectors (`dense_chunk`, `dense_title`, `dense_scope`, `sparse_title`).
2. **Multi-Vector Prefetching:** Executes parallel sub-queries across all representations.
3. **Reciprocal Rank Fusion (RRF):** Merges the ranked candidate lists across dense semantic and sparse lexical signals.
4. **Grouped Retrieval (`query_points_groups`):** Automatically groups matching chunk hits by `document_id` to present cohesive document-level search results with chunk-level grounding.

---

## 🏗️ Architecture: Multi-Representation Grouping

```text
                                  ┌───────────────────────────┐
                                  │        User Query         │
                                  └─────────────┬─────────────┘
                                                │
         ┌────────────────────────┬─────────────┴─────────────┬────────────────────────┐
         ▼                        ▼                           ▼                        ▼
  Ollama Dense Query       Ollama Dense Query          Ollama Dense Query         Sparse Query
   (qwen3-embedding:8b)     (qwen3-embedding:8b)        (qwen3-embedding:8b)     (FastEmbed BM25)
         │                        │                           │                        │
         ▼                        ▼                           ▼                        ▼
[Prefetch: dense_chunk]  [Prefetch: dense_title]     [Prefetch: dense_scope]  [Prefetch: sparse_title]
 (Body Content Search)    (Topical Naming Search)     (Regulatory Framework)   (Exact Acronym Search)
         │                        │                           │                        │
         └────────────────────────┴─────────────┬─────────────┴────────────────────────┘
                                                │
                                                ▼
                              ┌───────────────────────────────────┐
                              │    Reciprocal Rank Fusion (RRF)   │
                              └─────────────────┬─────────────────┘
                                                │
                                                ▼
                              ┌───────────────────────────────────┐
                              │ Grouping by document_id           │
                              │ (query_points_groups)             │
                              └─────────────────┬─────────────────┘
                                                │
                                                ▼
                                    Grouped Document Results
                                    [SOP-QA-042] -> Chunks 1, 2
                                    [VAL-OQ-108] -> Chunks 2, 3
```

---

## 💻 Running the Tutorial

```bash
python tutorials/08_multi_representation_search/multi_representation_search_gxp.py
```
