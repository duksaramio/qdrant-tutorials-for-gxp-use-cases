"""
Tutorial 06: Multivectors and Late Interaction for Life Science Quality & CSV

In Life Science Quality and Computer System Validation (CSV / GAMP 5), controlled
documents (SOPs, URS, OQ protocols, System Risk Assessments) contain multiple dense
technical clauses. Single-vector pooling loses token-level specifics.

This script demonstrates how to:
1. Configure a multi-vector collection in Qdrant with HNSW disabled for multivectors (m=0)
   to save RAM and optimize ingestion throughput.
2. Ingest GxP quality documents with both dense (Ollama qwen3-embedding:8b, 4096d) and ColBERT multivectors (128d per token).
3. Execute single-call fast retrieval (dense ANN) + high-precision MaxSim late interaction reranking.
4. Compare single-vector pooled retrieval vs. token-level late interaction scoring.
5. Track full late-interaction latency and score profiles with Langfuse (http://localhost:3000).

Target Environment: Local Qdrant server at http://localhost:6333
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import LateInteractionTextEmbedding
import ollama
from langfuse import get_client, observe

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
VECTOR_SIZE = 4096
COLLECTION_NAME = "gxp_multivectors_demo"

# ---------------------------------------------------------------------------
# 1. Connect to Local Qdrant & Initialize Models
# ---------------------------------------------------------------------------
print("=" * 80)
print(f"Step 1: Connecting to Qdrant at {QDRANT_URL}...")
client = QdrantClient(url=QDRANT_URL)
ollama_client = ollama.Client(host=OLLAMA_HOST)
langfuse = get_client()

print("Initializing embedding models:")
print(f"  - Dense (Single Vector):     Ollama {EMBEDDING_MODEL} ({VECTOR_SIZE} dims)")
print("  - Late Interaction (Multi): FastEmbed colbert-ir/colbertv2.0 (128 dims/token, MaxSim)")
print(f"  - Langfuse Observability:    {LANGFUSE_HOST}")

colbert_model = LateInteractionTextEmbedding(model_name="colbert-ir/colbertv2.0")


@observe(as_type="embedding", name="ollama-dense-embedding")
def get_dense_embeddings(texts: list) -> list:
    response = ollama_client.embed(model=EMBEDDING_MODEL, input=texts)
    langfuse.update_current_generation(
        model=EMBEDDING_MODEL,
        metadata={"text_count": len(texts), "dimensions": VECTOR_SIZE},
    )
    return response.embeddings


@observe(as_type="embedding", name="colbert-multivector-embedding")
def get_colbert_embeddings(texts: list) -> list:
    embeddings = list(colbert_model.embed(texts))
    langfuse.update_current_generation(
        model="colbert-ir/colbertv2.0",
        metadata={"text_count": len(texts), "type": "colbert_late_interaction"},
    )
    return embeddings


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
            size=VECTOR_SIZE,
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
@observe(as_type="span", name="ingest-multivector-documents")
def ingest_documents():
    data_path = Path(__file__).resolve().parent.parent.parent / "data" / "gxp_quality_docs.json"
    with open(data_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    texts = [f"{doc['title']}. {doc['description']}" for doc in documents]

    dense_embeddings = get_dense_embeddings(texts)
    colbert_embeddings = get_colbert_embeddings(texts)

    points = []
    for idx, doc in enumerate(documents):
        c_vec = colbert_embeddings[idx]  # shape: [num_tokens, 128]
        point = models.PointStruct(
            id=idx + 1,
            payload=doc,
            vector={
                "dense": dense_embeddings[idx],
                "colbert": c_vec.tolist(),
            },
        )
        points.append(point)

    client.upload_points(collection_name=COLLECTION_NAME, points=points)
    langfuse.update_current_span(
        output={"indexed_count": len(points), "collection": COLLECTION_NAME},
        metadata={"source": data_path.name},
    )
    return len(points)


print("\n" + "=" * 80)
print("Step 3: Generating dense & token-level multivector embeddings...")
count = ingest_documents()
print(f"Uploaded {count} documents with token-level multivector representations.")

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
@observe(as_type="retriever", name="colbert-late-interaction-comparison")
def compare_single_vs_multivector(query_text: str):
    print("\n" + "=" * 80)
    print(f"USER QUERY: \"{query_text}\"")
    print("=" * 80)

    q_dense = get_dense_embeddings([query_text])[0]
    q_colbert = get_colbert_embeddings([query_text])[0].tolist()

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

    print("\n[Method A: Single-Vector Dense Search (Ollama Early Interaction Pooling)]")
    for r, h in enumerate(dense_hits, 1):
        print(f"  #{r} [Cosine Score: {h.score:.4f}] {h.payload['doc_id']}: {h.payload['title']}")

    print("\n[Method B: Multi-Vector Rescoring (Dense Prefetch + ColBERT MaxSim Late Interaction)]")
    for r, h in enumerate(late_interaction_hits, 1):
        print(f"  #{r} [MaxSim Score: {h.score:.4f}] {h.payload['doc_id']}: {h.payload['title']}")
        print(f"      Type: {h.payload['doc_type']} | System: {h.payload['system']}")
        print(f"      Text: {h.payload['description'][:110]}...")

    langfuse.update_current_span(
        input={"query": query_text},
        output={
            "single_vector_hits": [{"id": h.payload["doc_id"], "score": h.score} for h in dense_hits],
            "maxsim_rerank_hits": [{"id": h.payload["doc_id"], "score": h.score} for h in late_interaction_hits],
        },
    )
    return late_interaction_hits


# ---------------------------------------------------------------------------
# 6. Multi-Vector Late Interaction with GxP Regulatory Filtering
# ---------------------------------------------------------------------------
@observe(as_type="retriever", name="filtered-multivector-rescoring")
def run_filtered_multivector(query_filtered: str):
    print("\n" + "=" * 80)
    print("FILTERED MULTIVECTOR RESCORING (CAPA / Deviations from 2023+)")
    print("=" * 80)

    q_d = get_dense_embeddings([query_filtered])[0]
    q_c = get_colbert_embeddings([query_filtered])[0].tolist()

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

    langfuse.update_current_span(
        input={"query": query_filtered, "filter": "CAPA/Deviation >= 2023"},
        output=[{"id": h.payload["doc_id"], "score": h.score} for h in filtered_results],
    )
    return filtered_results


@observe(name="tutorial-06-multivectors-pipeline")
def execute_multivector_scenarios():
    # Query 1: Multi-faceted requirement (21 CFR Part 11 audit trails & electronic signatures)
    compare_single_vs_multivector(
        "21 CFR Part 11 requirements for electronic signature verification and immutable audit trail generation"
    )

    # Query 2: Specific operational failure & recovery
    compare_single_vs_multivector(
        "remediation of automated database snapshot failures and quarterly disaster recovery restore drills"
    )

    # Query 3: Filtered query
    run_filtered_multivector("communication dropout between sensor probe and SCADA server")

    return langfuse.get_trace_url()


trace_url = execute_multivector_scenarios()
langfuse.flush()

print("\n" + "=" * 80)
print("Tutorial 06 Execution Complete!")
if trace_url:
    print(f"Langfuse Trace URL: {trace_url}")
print("=" * 80)
