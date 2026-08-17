# Qdrant Vector Search Tutorials for Life Science Quality & Computer System Validation (CSV)

A practical collection of vector search, hybrid retrieval, payload engineering, and multi-representation search tutorials built with [Qdrant](https://qdrant.tech/) and [Ollama](https://ollama.ai/) (`qwen3-embedding:8b`), tailored specifically for **Life Sciences Quality Assurance (QA)**, **Computer System Validation (CSV / CSA)**, and **GxP Regulatory Compliance (21 CFR Part 11 / EU Annex 11 / GAMP 5)**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Why Vector & Hybrid Search in Life Science Quality & CSV?](#why-vector--hybrid-search-in-life-science-quality--csv)
- [Architecture & Tech Stack](#architecture--tech-stack)
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
- System Risk Assessments (SRA / FMEA) and Audit Findings

Standard lexical/keyword search fails when queries use colloquial phrasing or synonyms (e.g., searching for *"unauthorized digital record modification"* misses documents titled *"21 CFR Part 11 Audit Trail Review and E-Signature Controls"*). Conversely, pure dense vector search often struggles with specific alphanumeric document IDs (e.g., `SOP-QA-042`) and exact regulatory clause numbers (e.g., `21 CFR 11.10(e)`).

This repository provides 10 comprehensive, production-ready tutorials demonstrating how to build, query, filter, fuse (RRF), scope across branches, and index dynamic payloads using **local Qdrant** and local **Ollama `qwen3-embedding:8b` (4096 dimensions)**.

---

## 🛠️ Architecture & Tech Stack

- **Vector Database**: [Qdrant](https://qdrant.tech/) running locally at `http://localhost:6333` (v1.19+).
- **Dense Embedding Engine**: Local [Ollama](https://ollama.ai/) at `http://localhost:11434` with **`qwen3-embedding:8b`** producing 4096-dimensional high-capacity semantic vectors.
- **Sparse Lexical Engine**: FastEmbed `Qdrant/bm25` with server-side IDF modifiers for exact alphanumeric code and regulatory clause search.
- **Late-Interaction Multi-Vector Engine**: FastEmbed `colbert-ir/colbertv2.0` (128 dims/token) with `MaxSim` comparator for token-level precision reranking.
- **Python Tooling**: `qdrant-client`, `ollama`, `fastembed`, `python-dotenv`, `numpy`.

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
│   ├── part11_compliance_clauses.json   # 21 CFR Part 11 & EU Annex 11 regulatory clauses
│   ├── gxp_pdf_pages.json               # Multi-page GxP validation reports, tables & FMEA matrices
│   └── gxp_chunked_documents.json       # Multi-representation chunked GxP documents
└── tutorials/
    ├── 01_semantic_search_101/          # Core 101: GxP document indexing with Ollama qwen3-embedding:8b
    │   ├── README.md
    │   └── semantic_search_101_gxp.py
    ├── 02_urs_to_oq_traceability/       # Automated Requirements Traceability Matrix (RTM) search
    │   ├── README.md
    │   └── traceability_search.py
    ├── 03_regulatory_clause_mapping/    # 21 CFR Part 11 & EU Annex 11 compliance clause matching
    │   ├── README.md
    │   └── part11_clause_mapping.py
    ├── 04_hybrid_search/                # Dense (Ollama 4096d) + BM25 Sparse Hybrid Search with RRF
    │   ├── README.md
    │   └── hybrid_search_gxp.py
    ├── 05_hybrid_search_with_reranking/ # Hybrid Search + ColBERT Late-Interaction MaxSim Reranking
    │   ├── README.md
    │   └── hybrid_search_reranking_gxp.py
    ├── 06_multivectors_and_late_interaction/ # Token-Level Multivectors (ColBERT) with HNSW m=0 Optimization
    │   ├── README.md
    │   └── multivectors_late_interaction_gxp.py
    ├── 07_multivector_document_retrieval/    # PDF & Complex Document Retrieval with Mean-Pooled Multivectors
    │   ├── README.md
    │   └── multivector_document_retrieval_gxp.py
    ├── 08_multi_representation_search/       # Multi-Representation Search (Title + Scope + Chunk) with Grouping
    │   ├── README.md
    │   └── multi_representation_search_gxp.py
    ├── 09_branch_aware_search/               # Branch-Aware Search Over Versioned Document Lifecycles & Change Controls
    │   ├── README.md
    │   └── branch_aware_search_gxp.py
    └── 10_indexing_dynamic_payloads/         # Indexing Dynamic/Open-Ended Attributes with Typed EAV Arrays & Nested Filters
        ├── README.md
        └── indexing_dynamic_payloads_gxp.py
```

---

## 🚀 Tutorial Catalog

| # | Tutorial | Objective | GxP Focus Area | Stack | Infrastructure |
|---|---|---|---|---|---|
| **01** | [Semantic Search 101](tutorials/01_semantic_search_101/README.md) | Spin up a Qdrant collection, generate 4096d dense embeddings with Ollama `qwen3-embedding:8b`, and apply GxP payload filters. | Quality Documents, CAPAs, SOPs, Deviations | Python / Ollama / Qdrant | Local (`localhost:6333` & `11434`) |
| **02** | [URS to OQ Traceability](tutorials/02_urs_to_oq_traceability/README.md) | Automatically match User Requirements (URS) to Operational Qualification (OQ) test scripts for RTM generation. | GAMP 5, RTM, Test Verification | Python / Ollama / Qdrant | Local (`localhost:6333` & `11434`) |
| **03** | [Regulatory Clause Mapping](tutorials/03_regulatory_clause_mapping/README.md) | Map vendor technical software controls to 21 CFR Part 11 and EU Annex 11 regulatory predicate rules. | 21 CFR Part 11, EU Annex 11, Vendor Audits | Python / Ollama / Qdrant | Local (`localhost:6333` & `11434`) |
| **04** | [Hybrid Search (Dense + BM25)](tutorials/04_hybrid_search/README.md) | Fuse Ollama dense semantic embeddings (4096d) with BM25 sparse keyword vectors using Reciprocal Rank Fusion (RRF). | Exact GxP IDs, Citations & Conceptual Search | Python / Ollama / FastEmbed / Qdrant | Local (`localhost:6333` & `11434`) |
| **05** | [Hybrid Search with Reranking](tutorials/05_hybrid_search_with_reranking/README.md) | 2-stage retrieval: Prefetch with Dense (Ollama) + BM25, then rerank candidates using ColBERT late-interaction multivectors. | High-Precision Compliance & Audit Retrieval | Python / Ollama / FastEmbed / ColBERT | Local (`localhost:6333` & `11434`) |
| **06** | [Multivectors & Late Interaction](tutorials/06_multivectors_and_late_interaction/README.md) | Optimize RAM & compute with token-level ColBERT multivectors using `hnsw_config=HnswConfigDiff(m=0)` and Ollama dense prefetch. | Long Complex Protocols, Risk Assessments, URS | Python / Ollama / FastEmbed / Qdrant | Local (`localhost:6333` & `11434`) |
| **07** | [Multivector Document Retrieval](tutorials/07_multivector_document_retrieval/README.md) | Scale multi-page PDF validation document retrieval using mean-pooled multivector prefetch and MaxSim reranking. | Multi-Page Validation Reports, FMEA Tables, CoAs | Python / FastEmbed / NumPy | Local (`localhost:6333`) |
| **08** | [Multi-Representation Search](tutorials/08_multi_representation_search/README.md) | Fuse Title, Scope, and Chunk dense vectors (Ollama 4096d) via RRF and group by `document_id` for document-level presentation. | Granular Section Grounding in SOPs, Protocols & CAPAs | Python / Ollama / FastEmbed / Qdrant | Local (`localhost:6333` & `11434`) |
| **09** | [Branch-Aware Search](tutorials/09_branch_aware_search/README.md) | Index versioned GxP documents with Ollama vectors and scope queries strictly to a branch's live view (Effective baselines, Change Control drafts, Site overlays). | Document Lifecycles, Change Control Revisions, EDMS | Python / Ollama / Qdrant | Local (`localhost:6333` & `11434`) |
| **10** | [Dynamic Payload Indexing](tutorials/10_indexing_dynamic_payloads/README.md) | Reshape open-ended instrument/system attributes into typed EAV arrays to query numeric ranges and exact matches with fixed indexes. | Instrument Telemetry, System Parameters, Multi-Lab Logs | Python / Ollama / Qdrant | Local (`localhost:6333` & `11434`) |

---

## ⚡ Quickstart Guide

### Prerequisites
1. **Local Qdrant Server**:
   ```bash
   docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```
2. **Local Ollama with `qwen3-embedding:8b`**:
   ```bash
   ollama pull qwen3-embedding:8b
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

### Environment Configuration
```bash
cp .env.example .env
```
Default configuration targets:
```ini
QDRANT_URL=http://localhost:6333
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=qwen3-embedding:8b
```

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

# Tutorial 5: Hybrid Search with ColBERT Late-Interaction Reranking (Local Qdrant)
python tutorials/05_hybrid_search_with_reranking/hybrid_search_reranking_gxp.py

# Tutorial 6: Multivector Representations & Late Interaction with HNSW m=0 (Local Qdrant)
python tutorials/06_multivectors_and_late_interaction/multivectors_late_interaction_gxp.py

# Tutorial 7: Multivector Document Retrieval with Mean-Pooled Multivectors (Local Qdrant)
python tutorials/07_multivector_document_retrieval/multivector_document_retrieval_gxp.py

# Tutorial 8: Multi-Representation Search Across Titles, Scopes & Chunks with Grouping (Local Qdrant)
python tutorials/08_multi_representation_search/multi_representation_search_gxp.py

# Tutorial 9: Branch-Aware Search Over Versioned GxP Lifecycles & Change Controls (Local Qdrant)
python tutorials/09_branch_aware_search/branch_aware_search_gxp.py

# Tutorial 10: Indexing Dynamic Payloads with Typed EAV Arrays & Nested Filters (Local Qdrant)
python tutorials/10_indexing_dynamic_payloads/indexing_dynamic_payloads_gxp.py
```

---

## 🛡️ GxP Validation & Regulatory Considerations

When implementing vector and hybrid search solutions in regulated life science environments:

1. **Deterministic Embeddings**: Pin embedding model names (`qwen3-embedding:8b`), local Ollama versions, and model weights to guarantee reproducible vector generation across validation lifecycles.
2. **Data Integrity & ALCOA+**: Store document IDs, cryptographic hashes, version numbers, and approval timestamps in point payloads to ensure end-to-end lineage.
3. **Access Controls (21 CFR Part 11 & EU Annex 11)**: Utilize Qdrant Cloud Role-Based Access Control (RBAC) and JSON Web Tokens (JWT) to enforce segregation of duties between QA reviewers, system owners, and validation leads.
4. **Disaster Recovery & Backup Verification**: Validate automated snapshot creation (`client.create_snapshot`) and test restoration procedures periodically to satisfy CSV disaster recovery requirements.

---

## 📄 License

This project is licensed under the Apache 2.0 License.
