# Tutorial 08: Multi-Representation Search Across Titles, Scopes, and Body Chunks for GxP & CSV

| Time: 30–45 min | Level: Intermediate / Advanced | Infrastructure: Local Qdrant (`http://localhost:6333`) |
| :--- | :--- | :--- |

## Overview

In Life Sciences and Computer System Validation (CSV / GAMP 5), a controlled document is rarely well-represented by a single embedding:
- **The Document Title** carries the formal system name, SOP code, and regulatory identity (`SOP-QA-042: Electronic Records, Signatures, and Audit Trail Review`).
- **The Executive Scope / Abstract** carries high-level regulatory frameworks (`21 CFR Part 11`, `EU Annex 11`, `GAMP 5 Category 4`).
- **The Specific Body Chunks** contain granular test scripts, acceptance criteria, or failure mode mitigations.
- **The Lexical Sparse Title** carries exact acronyms (`Empower 3 CDS`, `RTO/RPO`, `Modbus TCP/IP`).

If all representations are merged into a single dense vector, the title gets diluted, specific test conditions are averaged out, and chunk-level grounding disappears.

This tutorial builds a **Multi-Representation Search Pipeline** on **Local Qdrant (`http://localhost:6333`)**:
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
    Dense Query              Dense Query                 Dense Query              Sparse Query
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

## 1. Multi-Representation Collection Schema

```python
from qdrant_client import QdrantClient, models

client = QdrantClient("http://localhost:6333")

COLLECTION_NAME = "gxp_multi_representation_docs"

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense_chunk": models.VectorParams(size=384, distance=models.Distance.COSINE),
        "dense_title": models.VectorParams(size=384, distance=models.Distance.COSINE),
        "dense_scope": models.VectorParams(size=384, distance=models.Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse_title": models.SparseVectorParams(modifier=models.Modifier.IDF)
    },
)

# Index fields for grouping and filtering
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="document_id", field_schema=models.PayloadSchemaType.KEYWORD)
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="system", field_schema=models.PayloadSchemaType.KEYWORD)
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="doc_type", field_schema=models.PayloadSchemaType.KEYWORD)
```

---

## 2. Ingesting Multi-Representation Chunks

Each chunk is stored as a point in Qdrant. The title, executive scope, and sparse title embeddings are reused across all chunks belonging to the same document:

```python
# For each chunk in document:
points.append(
    models.PointStruct(
        id=point_id,
        vector={
            "dense_chunk": chunk_dense_vector,
            "dense_title": doc_title_dense_vector,
            "dense_scope": doc_scope_dense_vector,
            "sparse_title": doc_title_sparse_vector,
        },
        payload={
            "document_id": doc["doc_id"],
            "document_title": doc["title"],
            "doc_type": doc["doc_type"],
            "system": doc["system"],
            "section": chunk["section"],
            "chunk_text": chunk["text"],
        },
    )
)
```

---

## 3. Querying and Grouping with the Query API

```python
response = client.query_points_groups(
    collection_name=COLLECTION_NAME,
    prefetch=[
        models.Prefetch(query=q_dense, using="dense_chunk", limit=20),
        models.Prefetch(query=q_dense, using="dense_title", limit=20),
        models.Prefetch(query=q_dense, using="dense_scope", limit=20),
        models.Prefetch(query=q_sparse, using="sparse_title", limit=20),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    group_by="document_id",
    group_size=2,  # Return top-2 matching chunks per document
    limit=3,       # Return top-3 document groups
    with_payload=True,
)
```

---

## 4. Running the Tutorial

```bash
python tutorials/08_multi_representation_search/multi_representation_search_gxp.py
```
