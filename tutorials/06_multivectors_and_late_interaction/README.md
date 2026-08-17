# Tutorial 06: Multivector Representations & Late Interaction for Life Science Quality & CSV

| Time: 20–30 min | Level: Intermediate | Infrastructure: Local Qdrant (`http://localhost:6333`) |
| :--- | :--- | :--- |

## Overview

In Life Sciences and Computer System Validation (CSV / GAMP 5), controlled documents—such as **Standard Operating Procedures (SOPs)**, **User Requirements Specifications (URS)**, **Validation Protocols (IQ/OQ/PQ)**, and **System Risk Assessments (SRA)**—often contain multi-clause, highly technical sentences.

- **The Problem with Single-Vector Early Interaction:** Standard vector search pools all token embeddings into a single fixed-size vector (e.g., 384 dimensions). This compresses information prematurely and loses granular token-level requirements (such as exact numerical thresholds, multi-factor criteria, or specific regulatory sections).
- **The Power of Multivector Late Interaction:** Models like **ColBERT** retain individual token-level vectors (`[num_tokens, 128]`) and compute relevance at query time using the **MaxSim** operator. Each query token is aligned with the most relevant token in the document, preserving fine-grained regulatory precision.
- **The Optimization Key (`hnsw_config=HnswConfigDiff(m=0)`):** Indexing hundreds of token-level vectors per document in an HNSW graph causes high RAM consumption and slow indexing. In Qdrant, we leave HNSW enabled on the dense vector for fast first-pass ANN retrieval, and disable HNSW on the multivector (`m=0`) to use it strictly for query-time rescoring/reranking.

---

## 🏗️ Architecture: Two-Stage Rescoring

```text
1. Document Ingestion:
   Document Text ──► Dense Embedding (384d, HNSW ON)
                 └──► ColBERT Multi-Vectors (128d / token, HNSW OFF: m=0)

2. Query Execution (Single Qdrant API Call):
   Query Text ──► [Stage 1: Dense ANN] ──► Top 10 Candidates (Fast Recall)
              └──► [Stage 2: ColBERT MaxSim Rescoring] ──► Top 3 Final Results (High Precision)
```

---

## 1. Setting Up the Collection with HNSW `m=0`

```python
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, LateInteractionTextEmbedding

client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "gxp_multivectors_demo"

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
            # HNSW is enabled by default for dense vectors (first-pass fast ANN)
        ),
        "colbert": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0),  # Disable HNSW on multivectors to save RAM
        ),
    },
)
```

---

## 2. Ingesting Token-Level Multivectors

```python
dense_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
colbert_model = LateInteractionTextEmbedding(model_name="colbert-ir/colbertv2.0")

# For each document:
# dense_vec = list(dense_model.embed([doc_text]))[0].tolist()
# colbert_vec = list(colbert_model.embed([doc_text]))[0].tolist()  # [num_tokens, 128]

client.upload_points(
    collection_name=COLLECTION_NAME,
    points=[
        models.PointStruct(
            id=doc_id,
            payload=doc_metadata,
            vector={
                "dense": dense_vec,
                "colbert": colbert_vec,
            },
        )
    ],
)
```

---

## 3. Querying with Rescoring in a Single Call

```python
query_text = "21 CFR Part 11 requirements for electronic signature verification and immutable audit trail generation"

q_dense = list(dense_model.embed([query_text]))[0].tolist()
q_colbert = list(colbert_model.embed([query_text]))[0].tolist()

results = client.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=models.Prefetch(
        query=q_dense,
        using="dense",
        limit=10,  # Fast candidate retrieval using HNSW
    ),
    query=q_colbert,  # Rescore candidates using token-level MaxSim
    using="colbert",
    limit=3,
    with_payload=True,
)

for rank, hit in enumerate(results.points, 1):
    print(f"#{rank} [MaxSim Score: {hit.score:.4f}] {hit.payload['doc_id']}: {hit.payload['title']}")
```

---

## 4. Resource & Performance Comparison

| Configuration | RAM Consumption | Upload Speed | Search Recall | Search Precision |
| :--- | :--- | :--- | :--- | :--- |
| **Dense Only** | Very Low | Very Fast | High | Moderate |
| **ColBERT with HNSW ON** | High (5–10x) | Slower | High | Highest |
| **Dense + ColBERT (`m=0`) Rescoring** | **Low (Optimized)** | **Fast** | **High** | **Highest (MaxSim)** |

---

## 5. Running the Tutorial

```bash
python tutorials/06_multivectors_and_late_interaction/multivectors_late_interaction_gxp.py
```
