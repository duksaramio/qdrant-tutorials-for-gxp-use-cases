"""
Tutorial 02: Automated URS-to-OQ Requirements Traceability Matrix (RTM) Search

In Computer System Validation (CSV / GAMP 5), building and maintaining the Requirements
Traceability Matrix (RTM) is a labor-intensive manual task.

This script demonstrates how Qdrant semantic search matches User Requirements (URS)
to corresponding Operational Qualification (OQ) test scripts automatically.
"""

import json
from pathlib import Path
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

# ---------------------------------------------------------------------------
# 1. Initialize In-Memory Qdrant Client & FastEmbed
# ---------------------------------------------------------------------------
print("=" * 75)
print("Tutorial 02: Automated URS-to-OQ Requirements Traceability Search")
print("=" * 75)

client = QdrantClient(":memory:")
embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = "oq_test_scripts"

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=384,
        distance=models.Distance.COSINE,
    ),
)

# ---------------------------------------------------------------------------
# 2. Load URS and OQ Test Scripts
# ---------------------------------------------------------------------------
data_path = Path(__file__).resolve().parent.parent.parent / "data" / "urs_oq_traceability_data.json"
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

urs_list = data["user_requirements"]
oq_tests = data["oq_test_scripts"]

print(f"Indexing {len(oq_tests)} OQ Test Verification Scripts...")

oq_descriptions = [
    f"Title: {t['test_title']}. Module: {t['module']}. Verification Steps: {t['steps']}. Expected: {t['expected_result']}"
    for t in oq_tests
]

vectors = list(embedder.embed(oq_descriptions))
points = [
    models.PointStruct(
        id=idx,
        vector=vectors[idx].tolist(),
        payload=oq_tests[idx],
    )
    for idx in range(len(oq_tests))
]

client.upload_points(collection_name=COLLECTION_NAME, points=points)
print("OQ Test Scripts indexed successfully.\n")

# ---------------------------------------------------------------------------
# 3. Perform Automated Requirement-to-Test Mapping
# ---------------------------------------------------------------------------
print("=" * 75)
print("AUTOMATED TRACEABILITY MATRIX (RTM) GENERATION")
print("=" * 75)

for urs in urs_list:
    req_id = urs["req_id"]
    statement = urs["statement"]
    category = urs["category"]
    criticality = urs["gxp_criticality"]

    print(f"\n[URS: {req_id}] ({category} | Criticality: {criticality})")
    print(f"Requirement: \"{statement}\"")
    print("-" * 75)

    query_vec = list(embedder.embed([statement]))[0].tolist()
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=2,
    ).points

    for rank, hit in enumerate(hits, 1):
        test = hit.payload
        test_id = test["test_id"]
        test_title = test["test_title"]
        module = test["module"]
        score = hit.score if hasattr(hit, "score") else 0.0

        match_verdict = "CONFIRMED TRACE" if score > 0.45 else "POTENTIAL GAP"
        print(f"  --> Match #{rank} [{match_verdict}] [Score: {score:.4f}]")
        print(f"      Test Script: {test_id} - {test_title} (Module: {module})")
        print(f"      Test Steps: {test['steps'][:100]}...")

print("\n" + "=" * 75)
print("Automated Traceability Mapping Complete!")
print("=" * 75)
