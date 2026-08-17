# Tutorial 01: Semantic Search 101 for Life Science Quality and CSV

| Time: 15–20 min | Level: Beginner | Infrastructure: Local Qdrant (`http://localhost:6333`) + Local Ollama (`qwen3-embedding:8b`) |
| :--- | :--- | :--- |

## Overview

In Life Sciences and Computer System Validation (CSV), Quality and Validation engineers manage thousands of controlled documents:
- **Standard Operating Procedures (SOPs)**
- **Validation Protocols & Reports (IQ / OQ / PQ / VSR)**
- **Deviation Investigations (DEV)**
- **Corrective and Preventive Actions (CAPAs)**
- **Change Controls (CC / CR)**
- **System Risk Assessments (SRA / FMEA)**

Traditional keyword search fails when queries use colloquial phrasing or synonyms (e.g., searching for *"unauthorized modification of electronic batch records"* fails to retrieve documents titled *"21 CFR Part 11 Audit Trail Review and E-Signature Controls"*).

This tutorial demonstrates how to build a high-performance **Semantic Vector Search Engine** tailored for GxP documents using **Qdrant** and local **Ollama (`qwen3-embedding:8b`)** embeddings (4096-dimensional vectors).

---

## 🎯 What You Will Learn

1. **Local Setup:** Connect to local Qdrant server (`http://localhost:6333`) and local Ollama (`http://localhost:11434`).
2. **Collection Setup:** Create a collection configured with 4096-dimensional cosine distance vectors.
3. **Payload Engineering:** Upload GxP documents with structured metadata (`doc_type`, `system`, `effective_year`, `gamp_category`, `regulatory_predicates`).
4. **Embedding Generation:** Embed documents and queries using local `qwen3-embedding:8b` via Ollama.
5. **Semantic Retrieval:** Execute natural language queries capturing complex regulatory concepts.
6. **Payload Filtering:** Filter search results using structured metadata constraints.

---

## ⚙️ Prerequisites & Installation

Make sure your local Qdrant server and Ollama are running:
```bash
# 1. Start Qdrant Docker container
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 2. Verify Ollama has qwen3-embedding:8b
ollama list
```

Install Python dependencies:
```bash
pip install qdrant-client ollama python-dotenv tabulate
```

---

## 💻 Code Example

```python
from qdrant_client import QdrantClient, models
import ollama

client = QdrantClient("http://localhost:6333")
ollama_client = ollama.Client(host="http://localhost:11434")

# 1. Create collection with 4096 dimensions
client.create_collection(
    collection_name="gxp_quality_docs",
    vectors_config=models.VectorParams(size=4096, distance=models.Distance.COSINE),
)

# 2. Embed and upload document
text = "SOP-QA-042: Establishes 21 CFR Part 11 and EU Annex 11 audit trail review procedures."
embedding = ollama_client.embed(model="qwen3-embedding:8b", input=[text]).embeddings[0]

client.upload_points(
    collection_name="gxp_quality_docs",
    points=[
        models.PointStruct(
            id=1,
            vector=embedding,
            payload={
                "doc_id": "SOP-QA-042",
                "doc_type": "SOP",
                "system": "Enterprise QMS",
                "effective_year": 2024,
            },
        )
    ],
)

# 3. Query with semantic vector + metadata filter
q_vec = ollama_client.embed(model="qwen3-embedding:8b", input=["audit trail review requirements"]).embeddings[0]

results = client.query_points(
    collection_name="gxp_quality_docs",
    query=q_vec,
    query_filter=models.Filter(
        must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value="SOP"))]
    ),
    limit=3,
)
```

---

## 🚀 Running the Tutorial

```bash
python tutorials/01_semantic_search_101/semantic_search_101_gxp.py
```
