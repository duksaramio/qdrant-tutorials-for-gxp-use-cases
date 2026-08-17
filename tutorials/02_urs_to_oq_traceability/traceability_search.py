"""
Tutorial 02: Automated URS-to-OQ Requirements Traceability Matrix (RTM) Search

In Computer System Validation (CSV / GAMP 5), building and maintaining the Requirements
Traceability Matrix (RTM) is a labor-intensive manual task.

This script demonstrates how Qdrant semantic search and local Ollama (qwen3-embedding:8b)
automatically match User Requirements (URS) to corresponding Operational Qualification (OQ)
verification test scripts, fully traced with Langfuse observability.

Target Environment: Local Qdrant (http://localhost:6333) + Local Ollama (http://localhost:11434) + Langfuse (http://localhost:3000)
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
import ollama
from langfuse import get_client, observe

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
VECTOR_SIZE = 4096
COLLECTION_NAME = "oq_test_scripts"

# ---------------------------------------------------------------------------
# 1. Initialize Clients
# ---------------------------------------------------------------------------
print("=" * 75)
print("Tutorial 02: Automated URS-to-OQ Traceability with Local Ollama, Qdrant & Langfuse")
print(f"  - Qdrant:   {QDRANT_URL}")
print(f"  - Ollama:   {OLLAMA_HOST} ({EMBEDDING_MODEL}, {VECTOR_SIZE} dims)")
print(f"  - Langfuse: {LANGFUSE_HOST}")
print("=" * 75)

client = QdrantClient(url=QDRANT_URL)
ollama_client = ollama.Client(host=OLLAMA_HOST)
langfuse = get_client()


@observe(as_type="embedding", name="ollama-qwen3-embedding")
def get_embeddings(texts: list) -> list:
    response = ollama_client.embed(model=EMBEDDING_MODEL, input=texts)
    langfuse.update_current_generation(
        model=EMBEDDING_MODEL,
        metadata={"text_count": len(texts), "dimensions": VECTOR_SIZE},
    )
    return response.embeddings


if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=VECTOR_SIZE,
        distance=models.Distance.COSINE,
    ),
)


# ---------------------------------------------------------------------------
# 2. Load and Ingest URS and OQ Test Scripts
# ---------------------------------------------------------------------------
@observe(as_type="span", name="ingest-oq-test-scripts")
def ingest_oq_scripts():
    data_path = Path(__file__).resolve().parent.parent.parent / "data" / "urs_oq_traceability_data.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    oq_tests = data["oq_test_scripts"]
    print(f"Indexing {len(oq_tests)} OQ Test Verification Scripts...")

    oq_descriptions = [
        f"Title: {t['test_title']}. Module: {t['module']}. Verification Steps: {t['steps']}. Expected: {t['expected_result']}"
        for t in oq_tests
    ]

    vectors = get_embeddings(oq_descriptions)
    points = [
        models.PointStruct(
            id=idx + 1,
            vector=vectors[idx],
            payload=oq_tests[idx],
        )
        for idx in range(len(oq_tests))
    ]

    client.upload_points(collection_name=COLLECTION_NAME, points=points)
    langfuse.update_current_span(
        output={"indexed_tests": len(oq_tests), "collection": COLLECTION_NAME},
        metadata={"source": data_path.name},
    )
    return data["user_requirements"], len(oq_tests)


urs_list, count = ingest_oq_scripts()
print(f"OQ Test Scripts ({count}) indexed successfully.\n")


# ---------------------------------------------------------------------------
# 3. Perform Automated Requirement-to-Test Mapping with Observability
# ---------------------------------------------------------------------------
@observe(as_type="retriever", name="rtm-requirement-matching")
def trace_requirement_match(urs: dict, limit: int = 2):
    statement = urs["statement"]
    query_vec = get_embeddings([statement])[0]
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=limit,
    ).points

    matched_results = [
        {
            "test_id": hit.payload.get("test_id"),
            "test_title": hit.payload.get("test_title"),
            "score": hit.score,
            "module": hit.payload.get("module"),
        }
        for hit in hits
    ]

    langfuse.update_current_span(
        input={"req_id": urs["req_id"], "statement": statement, "criticality": urs["gxp_criticality"]},
        output=matched_results,
        metadata={"limit": limit, "collection": COLLECTION_NAME},
    )
    return hits


@observe(name="tutorial-02-rtm-generation-pipeline")
def run_traceability_matrix():
    print("=" * 75)
    print("AUTOMATED TRACEABILITY MATRIX (RTM) GENERATION (Langfuse Monitored)")
    print("=" * 75)

    for urs in urs_list:
        req_id = urs["req_id"]
        statement = urs["statement"]
        category = urs["category"]
        criticality = urs["gxp_criticality"]

        print(f"\n[URS: {req_id}] ({category} | Criticality: {criticality})")
        print(f"Requirement: \"{statement}\"")
        print("-" * 75)

        hits = trace_requirement_match(urs, limit=2)

        for rank, hit in enumerate(hits, 1):
            test = hit.payload
            test_id = test["test_id"]
            test_title = test["test_title"]
            module = test["module"]
            score = hit.score if hasattr(hit, "score") else 0.0

            match_verdict = "CONFIRMED TRACE" if score > 0.55 else "POTENTIAL GAP"
            print(f"  --> Match #{rank} [{match_verdict}] [Score: {score:.4f}]")
            print(f"      Test Script: {test_id} - {test_title} (Module: {module})")
            print(f"      Test Steps: {test['steps'][:100]}...")

    return langfuse.get_trace_url()


trace_url = run_traceability_matrix()
langfuse.flush()

print("\n" + "=" * 75)
print("Automated Traceability Mapping Complete!")
if trace_url:
    print(f"Langfuse Trace URL: {trace_url}")
print("=" * 75)
