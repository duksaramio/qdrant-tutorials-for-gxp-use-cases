# Tutorial 04: Hybrid Search for Life Science Quality & Computer System Validation (CSV)

| Time: 15–20 min | Level: Intermediate | Infrastructure: Local Qdrant (`http://localhost:6333`) |
| :--- | :--- | :--- |

## Overview

In Life Sciences Quality Assurance (QA) and Computer System Validation (CSV), search queries often combine **dense conceptual descriptions** with **exact alphanumeric codes and regulatory predicate citations**:
- Exact identifiers: `SOP-QA-042`, `21 CFR Part 11.10(e)`, `EU Annex 11.9`, `GAMP 5 Cat 4`, `CAPA-2023-019`, `Waters Empower 3 CDS`.
- Conceptual intent: *"unauthorized alteration of analytical records"*, *"intermittent sensor communication loss"*, *"periodic backup restoration drills"*.

Pure dense vector search struggles with exact regulatory section numbers and document codes. Pure keyword search fails on conceptual synonyms and paraphrasing.

This tutorial demonstrates how to build a **Hybrid Search Engine** on a **Local Qdrant instance (`http://localhost:6333`)** that combines:
1. **Dense Semantic Embeddings** (`sentence-transformers/all-MiniLM-L6-v2`) for conceptual similarity.
2. **Sparse BM25 Vectors** (`Qdrant/bm25` with server-side Inverse Document Frequency modifier) for exact keyword/code matches.
3. **Reciprocal Rank Fusion (RRF)** to fuse candidate lists into a unified relevance score.
4. **GxP Metadata Filtering** to restrict searches by document type (`SOP`, `CAPA`, `Deviation`) and effective year.

---

## 1. Prerequisites

Make sure local Qdrant is running on port 6333:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Verify the endpoint:
```bash
curl http://localhost:6333
```

---

## 2. Setting Up the Hybrid Collection

Create a collection configuring both named vectors:

```python
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "gxp_hybrid_quality_docs"

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense_vector": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
        )
    },
    sparse_vectors_config={
        "bm25_sparse_vector": models.SparseVectorParams(
            modifier=models.Modifier.IDF  # Boosts rare discriminative terms like 'CAPA-2023-019'
        )
    },
)
```

---

## 3. Ingesting Documents with Dual Embeddings

Embed each document's text using both dense and sparse models locally via `fastembed`:

```python
dense_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

# For each document:
# dense_vector = list(dense_model.embed([text]))[0].tolist()
# sparse_vector = list(sparse_model.embed([text]))[0]

client.upload_points(
    collection_name=COLLECTION_NAME,
    points=[
        models.PointStruct(
            id=doc_id,
            vector={
                "dense_vector": dense_vector,
                "bm25_sparse_vector": models.SparseVector(
                    indices=sparse_vector.indices.tolist(),
                    values=sparse_vector.values.tolist(),
                ),
            },
            payload=doc_metadata,
        )
    ],
)
```

---

## 4. Querying with Reciprocal Rank Fusion (RRF)

```python
query_text = "21 CFR Part 11 electronic records SOP-QA-042"

q_dense = list(dense_model.embed([query_text]))[0].tolist()
q_sparse = list(sparse_model.embed([query_text]))[0]

results = client.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[
        models.Prefetch(
            query=q_dense,
            using="dense_vector",
            limit=5,
        ),
        models.Prefetch(
            query=models.SparseVector(
                indices=q_sparse.indices.tolist(),
                values=q_sparse.values.tolist(),
            ),
            using="bm25_sparse_vector",
            limit=5,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=3,
    with_payload=True,
)
```

---

## 5. Running the Tutorial

```bash
python tutorials/04_hybrid_search/hybrid_search_gxp.py
```
