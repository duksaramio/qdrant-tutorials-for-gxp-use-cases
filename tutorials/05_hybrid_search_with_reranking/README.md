# Tutorial 05: Hybrid Search with Late-Interaction (ColBERT) Reranking for Life Science Quality & CSV

| Time: 20–30 min | Level: Intermediate | Infrastructure: Local Qdrant (`http://localhost:6333`) |
| :--- | :--- | :--- |

## Overview

In high-stakes Life Sciences regulatory compliance (FDA 21 CFR Part 11, EU Annex 11, GAMP 5), retrieval accuracy is paramount:
- **Dense embeddings** (e.g., `all-MiniLM-L6-v2`) capture conceptual context (*"preventing unauthorized changes to analytical data"*).
- **Sparse BM25 vectors** capture exact regulatory section numbers, acronyms, and SOP numbers (`21 CFR 11.10(e)`, `SOP-QA-042`, `CAPA-2023-019`).
- **Late-Interaction Rerankers** (e.g., `colbertv2.0` with **MaxSim** token alignment) compare query tokens directly against document tokens to provide nuanced, legally precise ranking over candidate documents.

This tutorial demonstrates how to build a **2-stage retrieval and reranking pipeline** on local Qdrant:
1. **Stage 1 (High Recall):** Prefetch candidates using parallel Dense + BM25 Sparse searches.
2. **Stage 2 (High Precision):** Rerank the merged candidate pool using ColBERT late-interaction multi-vectors inside Qdrant.

---

## 🏗️ Architecture

```text
               ┌────────────────────────────────────────────────────────┐
               │                      User Query                        │
               │  "regulatory compliance for immutable audit trail"     │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
         Dense Embedding (384d)                         BM25 Sparse Vector
                  │                                               │
                  ▼                                               ▼
     ┌────────────────────────┐                      ┌────────────────────────┐
     │  Dense Vector Search   │                      │   Sparse BM25 Search   │
     │  (Semantic Recall)     │                      │  (Lexical / Code Match)│
     └────────────┬───────────┘                      └────────────┬───────────┘
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          ▼
                         Prefetched Candidate Pool (Top N)
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │  ColBERT Late-Interaction Reranking   │
                      │       (MaxSim Token-Level Match)      │
                      │              [m=0 in HNSW]            │
                      └───────────────────┬───────────────────┘
                                          │
                                          ▼
                              Final High-Precision
                           Ranked GxP / CSV Documents
```

---

## 1. Setting Up Multi-Vector Collection

Create a collection configured with Dense vectors, Sparse vectors with server-side IDF, and ColBERT multivectors:

```python
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "gxp_hybrid_reranking_docs"

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
        ),
        "multi": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0),  # Disable HNSW index for reranking vectors
        ),
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
    },
)
```

> **Why `hnsw_config=models.HnswConfigDiff(m=0)`?**
> Late-interaction multivector embeddings are used exclusively in Stage 2 to rerank candidate documents retrieved in Stage 1. Disabling HNSW graph construction for `multi` eliminates index build overhead and saves memory.

---

## 2. Ingesting 3-Way Vector Embeddings

```python
dense_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
colbert_model = LateInteractionTextEmbedding(model_name="colbert-ir/colbertv2.0")

# For each document:
# dense_vec = list(dense_model.embed([text]))[0].tolist()
# sparse_vec = list(sparse_model.embed([text]))[0]
# colbert_vec = list(colbert_model.embed([text]))[0].tolist()

client.upload_points(
    collection_name=COLLECTION_NAME,
    points=[
        models.PointStruct(
            id=doc_id,
            payload=metadata,
            vector={
                "dense": dense_vec,
                "sparse": models.SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
                "multi": colbert_vec,
            },
        )
    ],
)
```

---

## 3. Query Execution & ColBERT Reranking

```python
query_text = "regulatory compliance for immutable time-stamped audit trail review"

q_dense = list(dense_model.embed([query_text]))[0].tolist()
q_sparse = list(sparse_model.embed([query_text]))[0]
q_colbert = list(colbert_model.embed([query_text]))[0].tolist()

sparse_obj = models.SparseVector(
    indices=q_sparse.indices.tolist(),
    values=q_sparse.values.tolist(),
)

# Prefetch Top-10 candidates from Dense and Sparse in parallel, then rerank with ColBERT
results = client.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[
        models.Prefetch(query=q_dense, using="dense", limit=10),
        models.Prefetch(query=sparse_obj, using="sparse", limit=10),
    ],
    query=q_colbert,
    using="multi",
    limit=3,
    with_payload=True,
)

for rank, hit in enumerate(results.points, 1):
    print(f"#{rank} [ColBERT Score: {hit.score:.4f}] {hit.payload['doc_id']}: {hit.payload['title']}")
```

---

## 4. Performance & Quality Comparison

| Retrieval Method | Precision on GxP Codes | Semantic Paraphrasing | Computational Cost | Latency Profile |
| :--- | :--- | :--- | :--- | :--- |
| **Dense Only** | Moderate | High | Low | Single-digit ms |
| **Sparse BM25 Only** | High (exact terms) | Low | Very Low | Sub-millisecond |
| **Hybrid (RRF)** | High | High | Low | Low |
| **Hybrid + ColBERT Rerank** | **Highest (MaxSim)** | **Highest** | Moderate (only over candidates) | Optimal 2-stage |

---

## 5. Running the Tutorial

```bash
python tutorials/05_hybrid_search_with_reranking/hybrid_search_reranking_gxp.py
```
