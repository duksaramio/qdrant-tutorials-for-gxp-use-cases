# Tutorial 07: Multivector Document Retrieval (ColPali/ColBERT Style)

| Time: 25–35 min | Level: Intermediate / Advanced | Infrastructure: Local Qdrant (`http://localhost:6333`) + Langfuse (`http://localhost:3000`) |
| :--- | :--- | :--- |

## Overview

In Life Sciences and Computer System Validation (CSV), validation deliverables and quality investigations are complex multi-page PDF documents containing tables, test scripts, risk matrices, and diagrams.

This tutorial demonstrates Qdrant's optimized 2-stage multivector architecture:
1. **Ingestion:** Store mean-pooled multivectors (with HNSW enabled) alongside high-resolution token multivectors (`m=0`).
2. **Retrieval:** Prefetch candidate pages using mean-pooled vectors, then rerank with token-level MaxSim late interaction.
3. **Observability:** Monitor end-to-end multi-page PDF candidate selection and MaxSim rerank scores in **Langfuse** at `http://localhost:3000`.

---

## 🔍 Observability with Langfuse

- **`@observe(as_type="embedding")`**: Traces ColBERT late-interaction token embeddings generated for document pages and queries.
- **`@observe(as_type="span")`**: Traces the PDF page mean pooling and point upload phases.
- **`@observe(as_type="retriever")`**: Traces two-stage candidate prefetching and MaxSim reranked page ordering.

---

## 💻 Running the Tutorial

```bash
python tutorials/07_multivector_document_retrieval/multivector_document_retrieval_gxp.py
```
