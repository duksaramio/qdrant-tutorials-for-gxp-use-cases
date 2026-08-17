"""
Semantic Search 101 for Life Science Quality and Computer System Validation (CSV)

This script demonstrates how to:
1. Initialize a Qdrant client (Qdrant Cloud or Local In-Memory)
2. Create a collection for GxP / CSV documents
3. Upload documents with payloads and semantic vector embeddings
4. Run semantic queries to retrieve relevant validation protocols, SOPs, and CAPAs
5. Apply payload filters (e.g. document type, effective year) to narrow search results
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

# Load environment variables if available
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-minilm-l6-v2")
COLLECTION_NAME = "gxp_quality_docs"

# ---------------------------------------------------------------------------
# 1. Connect to Qdrant
# ---------------------------------------------------------------------------
print("=" * 70)
print("Step 1 & 2: Connecting to Qdrant...")

if QDRANT_URL and QDRANT_API_KEY:
    print(f"Connecting to Qdrant Cloud: {QDRANT_URL}")
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        cloud_inference=True,
    )
    use_cloud_inference = True
    embedder = None
else:
    print("No QDRANT_URL / QDRANT_API_KEY found in environment.")
    print("Using local in-memory Qdrant with FastEmbed (sentence-transformers/all-MiniLM-L6-v2)...")
    client = QdrantClient(":memory:")
    use_cloud_inference = False
    embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# 2. Create Collection
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"Step 3: Creating collection '{COLLECTION_NAME}'...")

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=384,  # Dimensionality for all-minilm-l6-v2
        distance=models.Distance.COSINE,
    ),
)
print(f"Collection '{COLLECTION_NAME}' created successfully.")


# ---------------------------------------------------------------------------
# 3. Load Sample GxP Documents
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 4: Loading and uploading GxP / CSV documents...")

data_path = Path(__file__).resolve().parent.parent.parent / "data" / "gxp_quality_docs.json"
with open(data_path, "r", encoding="utf-8") as f:
    documents = json.load(f)

print(f"Loaded {len(documents)} GxP documents from {data_path.name}")

if use_cloud_inference:
    # Qdrant Cloud handles model inference natively
    points = [
        models.PointStruct(
            id=idx,
            vector=models.Document(
                text=doc["description"],
                model=EMBEDDING_MODEL,
            ),
            payload=doc,
        )
        for idx, doc in enumerate(documents)
    ]
else:
    # Local mode: compute vectors with fastembed
    texts = [doc["description"] for doc in documents]
    vectors = list(embedder.embed(texts))
    points = [
        models.PointStruct(
            id=idx,
            vector=vectors[idx].tolist(),
            payload=doc,
        )
        for idx, doc in enumerate(documents)
    ]

client.upload_points(collection_name=COLLECTION_NAME, points=points)
print(f"Successfully indexed {len(documents)} documents into '{COLLECTION_NAME}'.")


# ---------------------------------------------------------------------------
# 4. Create Payload Indexes for Filtering
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Creating payload indexes for structured metadata filtering...")

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

client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="system",
    field_schema=models.PayloadSchemaType.KEYWORD,
)
print("Payload indexes created on 'doc_type', 'effective_year', and 'system'.")


# ---------------------------------------------------------------------------
# 5. Helper Function to Query
# ---------------------------------------------------------------------------
def run_search(query_text: str, query_filter: models.Filter = None, limit: int = 3):
    if use_cloud_inference:
        query_input = models.Document(text=query_text, model=EMBEDDING_MODEL)
    else:
        query_input = list(embedder.embed([query_text]))[0].tolist()

    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_input,
        query_filter=query_filter,
        limit=limit,
    ).points


def print_results(header: str, hits):
    print("\n" + "-" * 70)
    print(f"QUERY: {header}")
    print("-" * 70)
    for rank, hit in enumerate(hits, 1):
        payload = hit.payload
        doc_id = payload.get("doc_id", "N/A")
        title = payload.get("title", "N/A")
        doc_type = payload.get("doc_type", "N/A")
        system = payload.get("system", "N/A")
        year = payload.get("effective_year", "N/A")
        score = hit.score if hasattr(hit, "score") else 0.0
        print(f"#{rank} [Score: {score:.4f}] {doc_id}: {title}")
        print(f"    Type: {doc_type} | System: {system} | Effective Year: {year}")
        print(f"    Abstract: {payload.get('description', '')[:120]}...\n")


# ---------------------------------------------------------------------------
# 6. Execute Semantic Search Queries
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 5: Executing Semantic Search Queries...")

# Query 1: Data integrity / audit trail tampering (colloquial phrasing)
query_1 = "unauthorized modification of electronic batch records and missing audit trails"
hits_1 = run_search(query_1, limit=3)
print_results(query_1, hits_1)


# Query 2: Backup and Disaster Recovery
query_2 = "loss of laboratory database data and disaster recovery verification"
hits_2 = run_search(query_2, limit=3)
print_results(query_2, hits_2)


# ---------------------------------------------------------------------------
# 7. Narrow Down Results with Metadata Filters
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Filtering Query Results: Only 'CAPA' or 'Deviation' from 2023 onwards...")

query_3 = "system hardware communication failure and instrument data interruption"
gxp_filter = models.Filter(
    must=[
        models.FieldCondition(
            key="doc_type",
            match=models.MatchAny(any=["CAPA", "Deviation"]),
        ),
        models.FieldCondition(
            key="effective_year",
            range=models.Range(gte=2023),
        ),
    ]
)

hits_filtered = run_search(query_3, query_filter=gxp_filter, limit=3)
print_results(f"{query_3}\n  [FILTER: doc_type in ['CAPA', 'Deviation'] & year >= 2023]", hits_filtered)

print("=" * 70)
print("Tutorial 101 Execution Complete!")
