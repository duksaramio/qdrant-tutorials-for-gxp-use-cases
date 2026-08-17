# Semantic Search 101 for Life Science Quality & Computer System Validation (CSV)

| Time: 5 - 15 min | Level: Beginner | Domain: Life Sciences / CSV / GxP Quality |
| :--- | :--- | :--- |

## Overview

In Life Sciences and regulated GxP environments, Quality Assurance (QA) and Computer System Validation (CSV) teams navigate thousands of controlled documents:
- **Standard Operating Procedures (SOPs)**
- **User Requirements Specifications (URS)**
- **Validation Protocols (IQ/OQ/PQ)**
- **Deviation Investigations (DEV)**
- **Corrective and Preventive Actions (CAPA)**
- **Change Controls (CC)**

Traditional keyword search fails when terminology varies (e.g., searching for *"unauthorized electronic record changes"* will miss an SOP titled *"21 CFR Part 11 Audit Trail Review and E-Signature Security Controls"*).

In this 5-minute tutorial, you will build a semantic search engine to index GxP quality and CSV records, perform semantic queries across validation artifacts, and narrow down search results using metadata filters.

---

## 1. Create a Qdrant Cluster

If using Qdrant Cloud:
1. Register for a [Qdrant Cloud account](https://cloud.qdrant.io/).
2. Under **Create a Free Cluster**, select your preferred cloud provider and region.
3. Copy the **Cluster Endpoint** and **API Key**.

Alternatively, this tutorial supports **zero-setup local execution** using Qdrant's in-memory engine and FastEmbed.

---

## 2. Set up the Client Connection

Install the client:
```bash
pip install qdrant-client fastembed python-dotenv
```

Connect to Qdrant:
```python
from qdrant_client import QdrantClient, models

# For Qdrant Cloud:
client = QdrantClient(
    url="https://xyz-example.cloud.qdrant.io",
    api_key="your-api-key",
    cloud_inference=True
)

# Or for local in-memory execution:
# client = QdrantClient(":memory:")
```

---

## 3. Create a Collection

```python
COLLECTION_NAME = "gxp_quality_docs"

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=384,
        distance=models.Distance.COSINE,
    ),
)
```

---

## 4. Upload Data to the Cluster

The dataset contains realistic GxP documents from `data/gxp_quality_docs.json`:

```python
import json
from pathlib import Path

with open("../../data/gxp_quality_docs.json", "r") as f:
    documents = json.load(f)

EMBEDDING_MODEL = "sentence-transformers/all-minilm-l6-v2"

# When using Qdrant Cloud with Cloud Inference:
client.upload_points(
    collection_name=COLLECTION_NAME,
    points=[
        models.PointStruct(
            id=idx,
            vector=models.Document(
                text=doc["description"],
                model=EMBEDDING_MODEL
            ),
            payload=doc
        )
        for idx, doc in enumerate(documents)
    ],
)
```

---

## 5. Query the Engine

### Semantic Query (Without Exact Keywords)
Search for concepts rather than exact keywords:

```python
hits = client.query_points(
    collection_name=COLLECTION_NAME,
    query=models.Document(
        text="unauthorized modification of electronic batch records and missing audit trails",
        model=EMBEDDING_MODEL
    ),
    limit=3,
).points

for hit in hits:
    print(f"[{hit.payload['doc_id']}] {hit.payload['title']} (Score: {hit.score:.4f})")
```

### Narrow Down with Metadata Filters
Create payload indexes and filter by document type and effective year:

```python
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="doc_type",
    field_schema=models.PayloadSchemaType.KEYWORD,
)
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="effective_year",
    field_schema=models.PayloadSchemaType.INTEGER,
)

hits = client.query_points(
    collection_name=COLLECTION_NAME,
    query=models.Document(
        text="system hardware communication failure and instrument data interruption",
        model=EMBEDDING_MODEL
    ),
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="doc_type",
                match=models.MatchAny(any=["CAPA", "Deviation"])
            ),
            models.FieldCondition(
                key="effective_year",
                range=models.Range(gte=2023)
            )
        ]
    ),
    limit=2,
).points
```

---

## Running the Sample Code

```bash
# Run the complete Python script
python tutorials/01_semantic_search_101/semantic_search_101_gxp.py
```
