# Tutorial 07: Multivector Document Retrieval (PDF / VLLM Style with Mean Pooling) for Life Science Quality & CSV

| Time: 25–35 min | Level: Advanced | Infrastructure: Local Qdrant (`http://localhost:6333`) |
| :--- | :--- | :--- |

## Overview

In Life Sciences and Computer System Validation (CSV), critical GxP evidence is stored across complex, multi-page PDFs:
- **Validation Protocols & Reports (IQ/OQ/PQ/VSR):** Contain multi-column test execution tables, acceptance formulas, and electronic sign-off blocks.
- **Deviation Investigation Reports:** Contain Ishikawa fishbone root cause diagrams, Modbus/SCADA packet diagnostic logs, and CQA risk tables.
- **System Risk Assessments (GAMP 5 FMEA):** Contain Failure Mode, Severity, Occurrence, Detectability (SOD) scoring matrices.

Traditional text extraction loses visual structure (tables, layout grids, headers, signature manifests). Modern Multimodal / Vision and Late-Interaction models (such as ColPali, ColQwen, and ColBERT) represent each document page as rich patch or token multivectors (~100 to 1,000 vectors per page).

### The Scaling Challenge
Building an HNSW graph on uncompressed multivectors for thousands of validation pages creates a combinatorial explosion:
$$\text{Vectors per page} \times \text{Vectors per page} \times \text{ef\_construct} = 1000 \times 1000 \times 100 = 100\text{M comparisons per page}$$

### The Solution: Two-Stage Mean-Pooled Retrieval
1. **Ingestion:**
   - Compress/pool the multivectors into condensed structural vectors (`mean_pooled`) with **HNSW enabled** for fast candidate search.
   - Store full-resolution multivectors (`original`) with **HNSW disabled (`m=0`)**.
2. **Querying:**
   - **Stage 1 (Fast Recall):** Prefetch candidate pages using the `mean_pooled` HNSW index.
   - **Stage 2 (Fine-Grained Precision):** Rerank candidate pages using full-resolution `MaxSim` on `original` vectors in a single Qdrant API call.

---

## 🏗️ Two-Stage Multivector Architecture

```text
               ┌────────────────────────────────────────────────────────┐
               │                      User Query                        │
               │  "chromatographic peak integration algorithm formula"  │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
         Full Query Multivector                          Pooled Query Vector
                  │                                               │
                  │                              ┌────────────────┴────────────────┐
                  │                              ▼                                 ▼
                  │                     [Stage 1: Prefetch]               [Stage 1: Prefetch]
                  │                  Mean-Pooled Column Vector         Mean-Pooled Row Vector
                  │                  (Fast Candidate Search)           (Fast Candidate Search)
                  │                              │                                 │
                  │                              └────────────────┬────────────────┘
                  │                                               ▼
                  │                                 Candidate Page Pool (Top K)
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    Stage 2: Full-Resolution MaxSim    │
                      │         Late-Interaction Rerank       │
                      │       ['original' vector with m=0]    │
                      └───────────────────┬───────────────────┘
                                          │
                                          ▼
                              High-Precision Page Hits
                           (Exact Table, Form & Diagram)
```

---

## 1. Setting Up Dual Multivector Collection

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "gxp_pdf_pages_demo"

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        # 1. Full resolution multivector: HNSW OFF (m=0) to eliminate graph construction overhead
        "original": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0),
        ),
        # 2. Mean-pooled multivector: HNSW ON for fast first-stage ANN prefetch
        "mean_pooled": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
        ),
    },
)
```

---

## 2. Mean Pooling Multivectors

```python
import numpy as np

def mean_pool_multivector(vectors: np.ndarray, num_pooled_chunks: int = 4) -> np.ndarray:
    """Compresses token/patch vectors into condensed chunk vectors via mean pooling."""
    tokens_per_chunk = int(np.ceil(len(vectors) / num_pooled_chunks))
    pooled = []
    for i in range(num_pooled_chunks):
        chunk = vectors[i * tokens_per_chunk : (i + 1) * tokens_per_chunk]
        if len(chunk) > 0:
            pooled.append(chunk.mean(axis=0))
    return np.array(pooled)
```

---

## 3. Two-Stage Query Execution

```python
results = client.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[
        models.Prefetch(
            query=q_pooled.tolist(),
            using="mean_pooled",
            limit=8,  # Fast candidate retrieval via HNSW
        )
    ],
    query=q_full.tolist(),  # Full resolution ColBERT MaxSim reranking
    using="original",
    limit=3,
    with_payload=True,
)
```

---

## 4. Running the Tutorial

```bash
python tutorials/07_multivector_document_retrieval/multivector_document_retrieval_gxp.py
```
