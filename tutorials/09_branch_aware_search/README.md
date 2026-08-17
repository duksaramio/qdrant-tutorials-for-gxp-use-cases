# Tutorial 09: Branch-Aware Search Over Versioned GxP & CSV Document Lifecycles

| Time: 25–35 min | Level: Intermediate | Infrastructure: Local Qdrant (`http://localhost:6333`) + Local Ollama (`qwen3-embedding:8b`) |
| :--- | :--- | :--- |

## Overview

In Life Science Quality Management Systems (QMS / EDMS) and Computer System Validation (CSV / GAMP 5), controlled document repositories evolve across **git-style branches and versioned lifecycles**:
- **`main-effective`:** Officially approved, legally binding GxP SOPs, qualification protocols, and validated baselines.
- **`draft-cc-2024`:** Proposed draft revisions under Change Control (CC) review.
- **`site-eu-overlay`:** Regional manufacturing site overlays (incorporating local EU GMP Annex 11 / Qualified Person requirements).

### The Cross-Branch Leakage Problem
In standard vector search, an index query leaks across versions:
- An auditor querying `main-effective` might inadvertently retrieve unapproved draft text from `draft-cc-2024`.
- A validation engineer working on `draft-cc-2024` might miss files inherited from the parent branch or see changes made on `main` *after* the fork cutoff.

This tutorial demonstrates how to index a versioned GxP corpus in **Local Qdrant (`http://localhost:6333`)** with **Ollama (`qwen3-embedding:8b`, 4096-dim)** and scope each vector query strictly to a single branch's live view:
1. Its own commits and revisions.
2. What it inherited from its ancestors up to the fork point (`seq <= fork_seq`).
3. Zero content that a later commit superseded or deleted.

---

## 🏗️ Architecture: Branch Ancestry & Cutoff Filter

```text
main-effective   ●──────●──────●──────●     seq 0-3 (v1.0 -> v2.0 -> delete -> v2.0 OQ)
                        │             │
                        │             └── site-eu-overlay ●  (forked @ seq 3)
                        │
                        └── draft-cc-2024 ●──────●           (forked @ seq 2)
```

Each point stores:
- `path`: Document identifier (`SOP-QA-042.md`).
- `branch`: Branch name (`main-effective`, `draft-cc-2024`, `site-eu-overlay`).
- `seq`: Integer commit sequence index.
- `overwritten_in`: Nested array `[{"by": "draft-cc-2024", "seq": 1}]` tracking superseded history.

---

## 💻 Running the Tutorial

```bash
python tutorials/09_branch_aware_search/branch_aware_search_gxp.py
```
