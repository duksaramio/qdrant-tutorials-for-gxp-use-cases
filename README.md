# Qdrant Vector Search Tutorials for Life Science Quality & Computer System Validation (CSV)

A practical collection of vector search and hybrid retrieval tutorials built with [Qdrant](https://qdrant.tech/), tailored specifically for **Life Sciences Quality Assurance (QA)**, **Computer System Validation (CSV / CSA)**, and **GxP Regulatory Compliance (21 CFR Part 11 / EU Annex 11 / GAMP 5)**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Why Vector & Hybrid Search in Life Science Quality & CSV?](#why-vector--hybrid-search-in-life-science-quality--csv)
- [Repository Structure](#repository-structure)
- [Tutorial Catalog](#tutorial-catalog)
- [Quickstart Guide](#quickstart-guide)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running Tutorials](#running-tutorials)
- [GxP Validation & Regulatory Considerations](#gxp-validation--regulatory-considerations)
- [License](#license)

---

## Overview

Quality and Computer System Validation engineers routinely manage vast volumes of controlled documents across diverse computerized systems (LIMS, MES, QMS, CTMS, CDS, SCADA):
- Standard Operating Procedures (SOPs)
- User Requirements Specifications (URS)
- Functional & Design Specifications (FS / DS)
- Validation Protocols & Reports (IQ / OQ / PQ / VSR)
- Deviation Investigations (DEV) & CAPAs
- Change Controls (CC / CR)
- System Risk Assessments (SRA) and Audit Findings

Standard lexical/keyword search fails when queries use colloquial phrasing or synonyms (e.g., searching for *"unauthorized digital record modification"* misses documents titled *"21 CFR Part 11 Audit Trail Review and E-Signature Controls"*). Conversely, pure dense vector search often struggles with specific alphanumeric document IDs (e.g., `SOP-QA-042`) and exact regulatory clause numbers (e.g., `21 CFR 11.10(e)`).

This repository provides hands-on tutorials showing how to build, query, filter, and fuse dense semantic search with sparse BM25 keyword search using Qdrant.

---

## 📂 Repository Structure

```text
qdrant-tutorials-for-gxp-use-cases/
├── .env.example
├── .gitignore
├── README.md
├── pyproject.toml
├── requirements.txt
├── data/
│   ├── gxp_quality_docs.json            # Controlled SOPs, CAPAs, OQs, Deviations, URS
│   ├── urs_oq_traceability_data.json    # URS statements and OQ verification test scripts
│   └── part11_compliance_clauses.json   # 21 CFR Part 11 & EU Annex 11 regulatory clauses
└── tutorials/
    ├── 01_semantic_search_101/          # Core 101: GxP document indexing, semantic queries & metadata filtering
    │   ├── README.md
    │   └── semantic_search_101_gxp.py
    ├── 02_urs_to_oq_traceability/       # Automated Requirements Traceability Matrix (RTM) search
    │   ├── README.md
    │   └── traceability_search.py
    ├── 03_regulatory_clause_mapping/    # 21 CFR Part 11 & EU Annex 11 compliance clause matching
    │   ├── README.md
    │   └── part11_clause_mapping.py
    └── 04_hybrid_search/                # Dense + BM25 Sparse Hybrid Search with RRF on Local Qdrant
        ├── README.md
        └── hybrid_search_gxp.py
```

---

## 🚀 Tutorial Catalog

| # | Tutorial | Objective | GxP Focus Area | Stack | Infrastructure |
|---|---|---|---|---|---|
| **01** | [Semantic Search 101](tutorials/01_semantic_search_101/README.md) | Spin up a Qdrant collection, upload GxP quality & CSV records, run semantic queries, and apply payload filters. | Quality Documents, CAPAs, SOPs, Deviations | Python / Qdrant Client | In-Memory / Cloud |
| **02** | [URS to OQ Traceability](tutorials/02_urs_to_oq_traceability/README.md) | Automatically match User Requirements (URS) to Operational Qualification (OQ) test scripts for RTM generation. | GAMP 5, RTM, Test Verification | Python / Qdrant Client | In-Memory / Local |
| **03** | [Regulatory Clause Mapping](tutorials/03_regulatory_clause_mapping/README.md) | Map vendor technical software controls to 21 CFR Part 11 and EU Annex 11 regulatory predicate rules. | 21 CFR Part 11, EU Annex 11, Vendor Audits | Python / Qdrant Client | In-Memory / Local |
| **04** | [Hybrid Search (Dense + BM25)](tutorials/04_hybrid_search/README.md) | Fuse dense semantic embeddings with BM25 sparse keyword vectors using Reciprocal Rank Fusion (RRF) on local Qdrant. | Exact GxP IDs, Citations & Conceptual Search | Python / FastEmbed / Qdrant | Local (`http://localhost:6333`) |

---

## ⚡ Quickstart Guide

### Prerequisites
- Python 3.10+
- Local Qdrant instance running on `http://localhost:6333`:
  ```bash
  docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
  ```

### Installation

Using `uv` (recommended):
```bash
# Clone the repository
git clone https://github.com/duksaramio/qdrant-tutorials-for-gxp-use-cases.git
cd qdrant-tutorials-for-gxp-use-cases

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Or using standard `pip`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Configuration (Optional)
```bash
cp .env.example .env
```
Default configuration targets `QDRANT_URL=http://localhost:6333`.

### Running Tutorials

```bash
# Tutorial 1: Semantic Search 101 for GxP Quality & CSV
python tutorials/01_semantic_search_101/semantic_search_101_gxp.py

# Tutorial 2: Automated URS-to-OQ Traceability Matrix Search
python tutorials/02_urs_to_oq_traceability/traceability_search.py

# Tutorial 3: 21 CFR Part 11 Regulatory Clause Mapping
python tutorials/03_regulatory_clause_mapping/part11_clause_mapping.py

# Tutorial 4: Dense + BM25 Sparse Hybrid Search with RRF (Local Qdrant)
python tutorials/04_hybrid_search/hybrid_search_gxp.py
```

---

## 🛡️ GxP Validation & Regulatory Considerations

When implementing vector and hybrid search solutions in regulated life science environments:

1. **Deterministic Embeddings**: Pin embedding model names, library versions, and model weights to guarantee reproducible vector generation across validation lifecycles.
2. **Data Integrity & ALCOA+**: Store document IDs, cryptographic hashes, version numbers, and approval timestamps in point payloads to ensure end-to-end lineage.
3. **Access Controls (21 CFR Part 11 & EU Annex 11)**: Utilize Qdrant Cloud Role-Based Access Control (RBAC) and JSON Web Tokens (JWT) to enforce segregation of duties between QA reviewers, system owners, and validation leads.
4. **Disaster Recovery & Backup Verification**: Validate automated snapshot creation (`client.create_snapshot`) and test restoration procedures periodically to satisfy CSV disaster recovery requirements.

---

## 📄 License

This project is licensed under the Apache 2.0 License.
