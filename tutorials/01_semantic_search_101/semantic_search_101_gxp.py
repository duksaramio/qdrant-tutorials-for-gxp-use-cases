"""
Semantic Search 101 for Life Science Quality and Computer System Validation (CSV)

This script demonstrates how to:
1. Connect to local Qdrant server (http://localhost:6333) and local Ollama (qwen3-embedding:8b)
2. Instrument observability with Langfuse (http://localhost:3000) for vector embeddings and search spans
3. Create a collection for GxP / CSV documents with 4096-dimensional dense vectors
4. Upload documents with payloads and semantic vector embeddings from Ollama
5. Run semantic queries to retrieve relevant validation protocols, SOPs, and CAPAs
6. Apply payload filters (e.g. document type, effective year) to narrow search results
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
import ollama
from langfuse import get_client, observe

# Load environment variables
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
VECTOR_SIZE = 4096
COLLECTION_NAME = "gxp_quality_docs_1"

# ---------------------------------------------------------------------------
# 1. Connect to Qdrant, Ollama & Langfuse
# ---------------------------------------------------------------------------
print("=" * 70)
print("Step 1: Connecting to Qdrant, Ollama & Langfuse...")
print(f"  - Qdrant Endpoint:   {QDRANT_URL}")
print(f"  - Ollama Host:       {OLLAMA_HOST}")
print(f"  - Embedding Model:   {EMBEDDING_MODEL} ({VECTOR_SIZE} dims)")
print(f"  - Langfuse Host:     {LANGFUSE_HOST}")

client = QdrantClient(url=QDRANT_URL)
ollama_client = ollama.Client(host=OLLAMA_HOST)
langfuse = get_client()


@observe(as_type="embedding", name="ollama-qwen3-embedding")
def get_embeddings(texts: list) -> list:
    """Generates dense embeddings via local Ollama qwen3-embedding:8b with Langfuse tracking."""
    response = ollama_client.embed(model=EMBEDDING_MODEL, input=texts)
    langfuse.update_current_generation(
        model=EMBEDDING_MODEL,
        metadata={"text_count": len(texts), "dimensions": VECTOR_SIZE},
    )
    return response.embeddings


# ---------------------------------------------------------------------------
# 2. Create Collection
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"Step 2: Creating collection '{COLLECTION_NAME}'...")

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=VECTOR_SIZE,
        distance=models.Distance.COSINE,
    ),
)
print(f"Collection '{COLLECTION_NAME}' created successfully.")


# ---------------------------------------------------------------------------
# 3. Load and Ingest Sample GxP Documents
# ---------------------------------------------------------------------------
@observe(as_type="span", name="qdrant-document-ingestion")
def ingest_documents():
    data_path = Path(__file__).resolve().parent.parent.parent / "data" / "gxp_quality_docs.json"
    with open(data_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"Loaded {len(documents)} GxP documents from {data_path.name}")
    texts = [f"{doc['title']}. {doc['description']}" for doc in documents]
    embeddings = get_embeddings(texts)

    points = [
        models.PointStruct(
            id=idx + 1,
            vector=embeddings[idx],
            payload=doc,
        )
        for idx, doc in enumerate(documents)
    ]

    client.upload_points(collection_name=COLLECTION_NAME, points=points)
    langfuse.update_current_span(
        output={"indexed_count": len(documents), "collection": COLLECTION_NAME},
        metadata={"source_file": data_path.name},
    )
    return len(documents)


print("\n" + "=" * 70)
print("Step 3: Loading and embedding GxP / CSV documents via Ollama (Langfuse tracked)...")
indexed_count = ingest_documents()
print(f"Successfully indexed {indexed_count} documents into '{COLLECTION_NAME}'.")


# ---------------------------------------------------------------------------
# 4. Create Payload Indexes for Filtering
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 4: Creating payload indexes for structured metadata filtering...")

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
# 5. Helper Function to Query with Langfuse Retriever Tracing
# ---------------------------------------------------------------------------
@observe(as_type="retriever", name="qdrant-semantic-search")
def run_search(query_text: str, query_filter: models.Filter = None, limit: int = 3):
    query_vector = get_embeddings([query_text])[0]

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
    ).points

    results_summary = [
        {
            "id": hit.payload.get("doc_id"),
            "title": hit.payload.get("title"),
            "score": hit.score,
            "system": hit.payload.get("system"),
        }
        for hit in hits
    ]

    langfuse.update_current_span(
        input={"query": query_text, "filter_active": query_filter is not None},
        output=results_summary,
        metadata={"limit": limit, "collection": COLLECTION_NAME},
    )
    return hits


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
# 6. Execute Traced Semantic Search Scenarios
# ---------------------------------------------------------------------------
@observe(name="tutorial-01-semantic-search-pipeline")
def execute_tutorial_scenarios():
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

    # Query 3: Filtering Query Results
    print("\n" + "=" * 70)
    print("Step 6: Filtering Query Results: Only 'CAPA' or 'Deviation' from 2023 onwards...")

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

    # Retrieve trace URL before exiting active observe context
    return langfuse.get_trace_url()


trace_url = execute_tutorial_scenarios()

# Flush observability traces to Langfuse
langfuse.flush()

print("=" * 70)
print("Tutorial 01 Execution Complete!")
if trace_url:
    print(f"Langfuse Trace URL: {trace_url}")
print("=" * 70)
