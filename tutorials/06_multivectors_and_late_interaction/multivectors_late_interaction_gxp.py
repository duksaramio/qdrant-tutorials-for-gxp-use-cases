"""
Tutorial 06: Multivectors and Late Interaction for Life Science Quality & CSV

In Life Science Quality and Computer System Validation (CSV / GAMP 5), controlled
documents (SOPs, URS, OQ protocols, System Risk Assessments) contain multiple dense
technical clauses. Single-vector pooling loses token-level specifics.

This script demonstrates how to:
1. Configure a multi-vector collection in Qdrant with HNSW disabled for multivectors (m=0)
   to save RAM and optimize ingestion throughput.
2. Ingest GxP quality documents with both dense (384d) and ColBERT multivectors (128d per token).
3. Execute single-call fast retrieval (dense ANN) + high-precision MaxSim late interaction reranking.
4. Compare single-vector pooled retrieval vs. token-level late interaction scoring.

Target Environment: Local Qdrant server at http://localhost:6333
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, LateInteractionTextEmbedding

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "gxp_multivectors_demo"

# ---------------------------------------------------------------------------
# 1. Connect to Local Qdrant & Initialize FastEmbed Models
# ---------------------------------------------------------------------------
print("=" * 80)
print(f"Step 1: Connecting to Qdrant at {QDRANT_URL}...")
client = QdrantClient(url=QDRANT_URL)

print("Initializing embedding models via FastEmbed:")
print("  - Dense (Single Vector):     sentence-transformers/all-MiniLM-L6-v2 (384 dims)")
print("  - Late Interaction (Multi): colbert-ir/colbertv2.0 (128 dims/token, MaxSim)")

dense_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
colbert_model = LateInteractionTextEmbedding(model_name="colbert-ir/colbertv2.0")

# ---------------------------------------------------------------------------
# 2. Create Optimized Multi-Vector Collection (HNSW m=0 for ColBERT)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"Step 2: Configuring multi-vector collection '{COLLECTION_NAME}'...")

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
            # HNSW is active by default for dense vectors (first-pass fast ANN retrieval)
        ),
        "colbert": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0),  # Disables HNSW graph on token vectors to save RAM
        ),
    },
)
print(f"Collection '{COLLECTION_NAME}' created with optimized HNSW m=0 configuration.")

# ---------------------------------------------------------------------------
# 3. Ingest GxP Quality & CSV Documents
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Step 3: Generating dense & token-level multivector embeddings...")

data_path = Path(__file__).resolve().parent.parent.parent / "data" / "gxp_quality_docs.json"
with open(data_path, "r", encoding="utf-8") as f:
    documents = json.load(f)

texts = [f"{doc['title']}. {doc['description']}" for doc in documents]

dense_embeddings = list(dense_model.embed(texts))
colbert_embeddings = list(colbert_model.embed(texts))

points = []
for idx, doc in enumerate(documents):
    c_vec = colbert_embeddings[idx]  # shape: [num_tokens, 128]
    point = models.PointStruct(
        id=idx + 1,
        payload=doc,
        vector={
            "dense": dense_embeddings[idx].tolist(),
            "colbert": c_vec.tolist(),
        },
    )
    points.append(point)

client.upload_points(collection_name=COLLECTION_NAME, points=points)
print(f"Uploaded {len(points)} documents with token-level multivector representations.")

# ---------------------------------------------------------------------------
# 4. Create Payload Indexes for GxP Filtering
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 5. Query: Single-Vector Early Interaction vs. Multivector Late Interaction
# ---------------------------------------------------------------------------
def compare_single_vs_multivector(query_text: str):
    print("\n" + "=" * 80)
    print(f"USER QUERY: \"{query_text}\"")
    print("=" * 80)

    q_dense = list(dense_model.embed([query_text]))[0].tolist()
    q_colbert = list(colbert_model.embed([query_text]))[0].tolist()

    # 1. Single Vector Dense Retrieval (Early Interaction / Pooled)
    dense_hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_dense,
        using="dense",
        limit=3,
    ).points

    # 2. Rescoring: Dense Prefetch (Fast ANN) + ColBERT MaxSim Late Interaction
    late_interaction_hits = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=models.Prefetch(
            query=q_dense,
            using="dense",
            limit=10,  # Retrieve top 10 candidates quickly via HNSW dense
        ),
        query=q_colbert,  # Rescore candidates using token-level MaxSim
        using="colbert",
        limit=3,
        with_payload=True,
    ).points

    print("\n[Method A: Single-Vector Dense Search (Early Interaction Pooling)]")
    for r, h in enumerate(dense_hits, 1):
        print(f"  #{r} [Cosine Score: {h.score:.4f}] {h.payload['doc_id']}: {h.payload['title']}")

    print("\n[Method B: Multi-Vector Rescoring (Dense Prefetch + ColBERT MaxSim Late Interaction)]")
    for r, h in enumerate(late_interaction_hits, 1):
        print(f"  #{r} [MaxSim Score: {h.score:.4f}] {h.payload['doc_id']}: {h.payload['title']}")
        print(f"      Type: {h.payload['doc_type']} | System: {h.payload['system']}")
        print(f"      Text: {h.payload['description'][:110]}...")


# Query 1: Multi-faceted requirement (21 CFR Part 11 audit trails & electronic signatures)
compare_single_vs_multivector(
    "21 CFR Part 11 requirements for electronic signature verification and immutable audit trail generation"
)

# Query 2: Specific operational failure & recovery
compare_single_vs_multivector(
    "remediation of automated database snapshot failures and quarterly disaster recovery restore drills"
)

# ---------------------------------------------------------------------------
# 6. Multi-Vector Late Interaction with GxP Regulatory Filtering
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("FILTERED MULTIVECTOR RESCORING (CAPA / Deviations from 2023+)")
print("=" * 80)

query_filtered = "communication dropout between sensor probe and SCADA server"
q_d = list(dense_model.embed([query_filtered]))[0].tolist()
q_c = list(colbert_model.embed([query_filtered]))[0].tolist()

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

filtered_results = client.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=models.Prefetch(
        query=q_d,
        using="dense",
        filter=gxp_filter,
        limit=5,
    ),
    query=q_c,
    using="colbert",
    limit=2,
    with_payload=True,
).points

print(f"Query: \"{query_filtered}\"")
for rank, hit in enumerate(filtered_results, 1):
    payload = hit.payload
    print(f"  #{rank} [Filtered MaxSim Score: {hit.score:.4f}] {payload['doc_id']}: {payload['title']}")
    print(f"      Type: {payload['doc_type']} | Year: {payload['effective_year']} | System: {payload['system']}")

print("\n" + "=" * 80)
print("Tutorial 06 Execution Complete!")
print("=" * 80)
