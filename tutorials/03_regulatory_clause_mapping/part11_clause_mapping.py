"""
Tutorial 03: 21 CFR Part 11 & EU Annex 11 Regulatory Clause Mapping

In CSV vendor assessments and regulatory audit preparations, life science organizations
must verify whether technical software controls satisfy specific predicate rule clauses
(e.g., 21 CFR 11.10(e) time-stamped audit trails, 11.50 signature manifestations).

This script demonstrates how to:
1. Index regulatory predicate rules and guidance clauses in Qdrant.
2. Query vendor technical specifications and architectural controls.
3. Automatically map vendor technical features to exact regulatory citations.
"""

import json
from pathlib import Path
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

# ---------------------------------------------------------------------------
# 1. Initialize In-Memory Qdrant Client & FastEmbed
# ---------------------------------------------------------------------------
print("=" * 75)
print("Tutorial 03: 21 CFR Part 11 & EU Annex 11 Regulatory Clause Mapping")
print("=" * 75)

client = QdrantClient(":memory:")
embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = "regulatory_clauses"

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=384,
        distance=models.Distance.COSINE,
    ),
)

# ---------------------------------------------------------------------------
# 2. Load and Index Regulatory Clauses
# ---------------------------------------------------------------------------
data_path = Path(__file__).resolve().parent.parent.parent / "data" / "part11_compliance_clauses.json"
with open(data_path, "r", encoding="utf-8") as f:
    clauses = json.load(f)

print(f"Indexing {len(clauses)} Regulatory Predicate Clauses...")

clause_texts = [
    f"{c['clause_id']} - {c['title']} ({c['regulation']}): {c['section_text']}"
    for c in clauses
]

vectors = list(embedder.embed(clause_texts))
points = [
    models.PointStruct(
        id=idx,
        vector=vectors[idx].tolist(),
        payload=clauses[idx],
    )
    for idx in range(len(clauses))
]

client.upload_points(collection_name=COLLECTION_NAME, points=points)
print("Regulatory Clauses indexed successfully.\n")


# ---------------------------------------------------------------------------
# 3. Assess Vendor Technical Controls Against Predicate Rules
# ---------------------------------------------------------------------------
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

    query_vec = list(embedder.embed([f_desc]))[0].tolist()
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=2,
    ).points

    for rank, hit in enumerate(hits, 1):
        clause = hit.payload
        cid = clause["clause_id"]
        title = clause["title"]
        reg = clause["regulation"]
        score = hit.score if hasattr(hit, "score") else 0.0

        print(f"  --> Top Predicate #{rank} [Score: {score:.4f}]")
        print(f"      Regulation: {reg} | Clause: {cid} ({title})")
        print(f"      Regulatory Text: {clause['section_text'][:110]}...")

print("\n" + "=" * 75)
print("Compliance Mapping Analysis Complete!")
print("=" * 75)
