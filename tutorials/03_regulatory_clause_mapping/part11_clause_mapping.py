"""
Tutorial 03: 21 CFR Part 11 & EU Annex 11 Regulatory Clause Mapping

In CSV vendor assessments and regulatory audit preparations, life science organizations
must verify whether technical software controls satisfy specific predicate rule clauses
(e.g., 21 CFR 11.10(e) time-stamped audit trails, 11.50 signature manifestations).

This script demonstrates how to:
1. Index regulatory predicate rules and guidance clauses in local Qdrant.
2. Embed vendor technical specifications and architectural controls using Ollama (qwen3-embedding:8b).
3. Automatically map vendor technical features to exact regulatory citations, fully monitored by Langfuse.

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
COLLECTION_NAME = "regulatory_clauses"

# ---------------------------------------------------------------------------
# 1. Initialize Clients
# ---------------------------------------------------------------------------
print("=" * 75)
print("Tutorial 03: Regulatory Clause Mapping with Local Ollama, Qdrant & Langfuse")
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
# 2. Load and Index Regulatory Clauses
# ---------------------------------------------------------------------------
@observe(as_type="span", name="ingest-regulatory-clauses")
def ingest_clauses():
    data_path = Path(__file__).resolve().parent.parent.parent / "data" / "part11_compliance_clauses.json"
    with open(data_path, "r", encoding="utf-8") as f:
        clauses = json.load(f)

    print(f"Indexing {len(clauses)} Regulatory Predicate Clauses...")

    clause_texts = [
        f"{c['clause_id']} - {c['title']} ({c['regulation']}): {c['section_text']}"
        for c in clauses
    ]

    vectors = get_embeddings(clause_texts)
    points = [
        models.PointStruct(
            id=idx + 1,
            vector=vectors[idx],
            payload=clauses[idx],
        )
        for idx in range(len(clauses))
    ]

    client.upload_points(collection_name=COLLECTION_NAME, points=points)
    langfuse.update_current_span(
        output={"indexed_clauses": len(clauses), "collection": COLLECTION_NAME},
        metadata={"source": data_path.name},
    )
    return len(clauses)


clause_count = ingest_clauses()
print(f"Regulatory Clauses ({clause_count}) indexed successfully.\n")


# ---------------------------------------------------------------------------
# 3. Assess Vendor Technical Controls Against Predicate Rules with Tracing
# ---------------------------------------------------------------------------
@observe(as_type="retriever", name="vendor-feature-clause-mapping")
def map_feature_to_clauses(feat: dict, limit: int = 2):
    f_desc = feat["description"]
    query_vec = get_embeddings([f_desc])[0]

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=limit,
    ).points

    matched_clauses = [
        {
            "clause_id": hit.payload.get("clause_id"),
            "title": hit.payload.get("title"),
            "regulation": hit.payload.get("regulation"),
            "score": hit.score,
        }
        for hit in hits
    ]

    langfuse.update_current_span(
        input={"feature_id": feat["feature_id"], "description": f_desc},
        output=matched_clauses,
        metadata={"limit": limit, "collection": COLLECTION_NAME},
    )
    return hits


@observe(name="tutorial-03-clause-mapping-pipeline")
def run_compliance_assessment():
    vendor_technical_features = [
        {
            "feature_id": "VEND-SEC-01",
            "description": "Our software creates an append-only, cryptographic hash-chained log recording the user ID, UTC timestamp, previous value, and new value for every record modification.",
        },
        {
            "feature_id": "VEND-SIG-02",
            "description": "Upon document sign-off, the generated PDF contains the signer's full name, role, signing timestamp, and user-selected reason code embedded directly into the document footer.",
        },
        {
            "feature_id": "VEND-BKP-03",
            "description": "Automated daily point-in-time database snapshots are replicated across multiple geographic availability zones with automated restore testing every Sunday.",
        },
    ]

    print("=" * 75)
    print("ASSESSING VENDOR TECHNICAL CONTROLS AGAINST REGULATORY PREDICATES")
    print("=" * 75)

    for feat in vendor_technical_features:
        f_id = feat["feature_id"]
        f_desc = feat["description"]

        print(f"\n[Vendor Specification: {f_id}]")
        print(f"Statement: \"{f_desc}\"")
        print("-" * 75)

        hits = map_feature_to_clauses(feat, limit=2)

        for rank, hit in enumerate(hits, 1):
            clause = hit.payload
            cid = clause["clause_id"]
            title = clause["title"]
            reg = clause["regulation"]
            score = hit.score if hasattr(hit, "score") else 0.0

            print(f"  --> Top Predicate #{rank} [Score: {score:.4f}]")
            print(f"      Regulation: {reg} | Clause: {cid} ({title})")
            print(f"      Regulatory Text: {clause['section_text'][:110]}...")

    return langfuse.get_trace_url()


trace_url = run_compliance_assessment()
langfuse.flush()

print("\n" + "=" * 75)
print("Compliance Mapping Analysis Complete!")
if trace_url:
    print(f"Langfuse Trace URL: {trace_url}")
print("=" * 75)
