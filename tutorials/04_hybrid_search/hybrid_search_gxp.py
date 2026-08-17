"""
Tutorial 04: Hybrid Search for Life Science Quality & Computer System Validation (CSV)

Combines:
1. Dense Semantic Embeddings via Local Ollama (qwen3-embedding:8b, 4096-dim)
2. Sparse Keyword Embeddings via FastEmbed (Qdrant/bm25 with Server-Side IDF Modifier)
3. Reciprocal Rank Fusion (RRF) to merge and rank results
4. Regulatory & Metadata Filtering on GxP attributes

Target Environment: Local Qdrant server at http://localhost:6333
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
import ollama

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
VECTOR_SIZE = 4096
COLLECTION_NAME = "gxp_hybrid_quality_docs"

# ---------------------------------------------------------------------------
# 1. Connect to Local Qdrant Server & Initialize Models
# ---------------------------------------------------------------------------
print("=" * 80)
print(f"Step 1: Connecting to Qdrant at {QDRANT_URL}...")
client = QdrantClient(url=QDRANT_URL)
ollama_client = ollama.Client(host=OLLAMA_HOST)

print("Initializing models locally:")
print(f"  - Dense:  Ollama {EMBEDDING_MODEL} ({VECTOR_SIZE} dims)")
print("  - Sparse: FastEmbed Qdrant/bm25 (lexical frequency + server-side IDF)")

sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")


def get_dense_embeddings(texts: list) -> list:
    return ollama_client.embed(model=EMBEDDING_MODEL, input=texts).embeddings


# ---------------------------------------------------------------------------
# 2. Create Collection with Named Dense & Sparse Vectors
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"Step 2: Configuring hybrid collection '{COLLECTION_NAME}'...")

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense_vector": models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        )
    },
    sparse_vectors_config={
        "bm25_sparse_vector": models.SparseVectorParams(
            modifier=models.Modifier.IDF  # Enable inverse document frequency calculation
        )
    },
)
print(f"Collection '{COLLECTION_NAME}' created with dense_vector ({VECTOR_SIZE}d) & bm25_sparse_vector.")

# ---------------------------------------------------------------------------
# 3. Ingest GxP Quality & CSV Documents
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Step 3: Ingesting GxP & CSV documents...")

data_path = Path(__file__).resolve().parent.parent.parent / "data" / "gxp_quality_docs.json"
with open(data_path, "r", encoding="utf-8") as f:
    documents = json.load(f)

texts = [f"{doc['title']}. {doc['description']}" for doc in documents]

print("Generating dense (Ollama) and BM25 sparse embeddings...")
dense_embeddings = get_dense_embeddings(texts)
sparse_embeddings = list(sparse_model.embed(texts))

points = []
for idx, doc in enumerate(documents):
    s_vec = sparse_embeddings[idx]
    point = models.PointStruct(
        id=idx + 1,
        payload=doc,
        vector={
            "dense_vector": dense_embeddings[idx],
            "bm25_sparse_vector": models.SparseVector(
                indices=s_vec.indices.tolist(),
                values=s_vec.values.tolist(),
            ),
        },
    )
    points.append(point)

client.upload_points(collection_name=COLLECTION_NAME, points=points)
print(f"Uploaded {len(points)} documents into '{COLLECTION_NAME}'.")

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
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="system",
    field_schema=models.PayloadSchemaType.KEYWORD,
)

# ---------------------------------------------------------------------------
# 5. Hybrid Search vs Dense-Only vs Sparse-Only Comparison
# ---------------------------------------------------------------------------
def run_comparison(query_text: str):
    print("\n" + "=" * 80)
    print(f"QUERY: \"{query_text}\"")
    print("=" * 80)

    q_dense = get_dense_embeddings([query_text])[0]
    q_sparse = list(sparse_model.embed([query_text]))[0]
    sparse_vector_obj = models.SparseVector(
        indices=q_sparse.indices.tolist(),
        values=q_sparse.values.tolist(),
    )

    # 1. Dense Only
    dense_hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_dense,
        using="dense_vector",
        limit=2,
    ).points

    # 2. Sparse (BM25) Only
    sparse_hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=sparse_vector_obj,
        using="bm25_sparse_vector",
        limit=2,
    ).points

    # 3. Hybrid (Dense + BM25 via Reciprocal Rank Fusion)
    hybrid_hits = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=q_dense,
                using="dense_vector",
                limit=5,
            ),
            models.Prefetch(
                query=sparse_vector_obj,
                using="bm25_sparse_vector",
                limit=5,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=2,
    ).points

    print("\n[A] DENSE ONLY (Ollama qwen3-embedding:8b Semantic Match):")
    for r, h in enumerate(dense_hits, 1):
        print(f"  #{r} [Score: {h.score:.4f}] {h.payload['doc_id']}: {h.payload['title']}")

    print("\n[B] BM25 SPARSE ONLY (Exact Keyword / Token Match):")
    for r, h in enumerate(sparse_hits, 1):
        print(f"  #{r} [Score: {h.score:.4f}] {h.payload['doc_id']}: {h.payload['title']}")

    print("\n[C] HYBRID (RRF FUSED):")
    for r, h in enumerate(hybrid_hits, 1):
        print(f"  #{r} [RRF Score: {h.score:.4f}] {h.payload['doc_id']}: {h.payload['title']}")
        print(f"      Type: {h.payload['doc_type']} | System: {h.payload['system']}")


# Test Query 1: Alphanumeric code & regulatory citation heavy query
run_comparison("21 CFR Part 11 electronic records SOP-QA-042")

# Test Query 2: Conceptual & system failure issue
run_comparison("unauthorized tampering with digital batch records and missing audit trail history")


# ---------------------------------------------------------------------------
# 6. Hybrid Search with Regulatory Metadata Filtering
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("HYBRID SEARCH WITH GXP METADATA FILTERING")
print("=" * 80)

query_filtered = "database snapshot backup failures and recovery drills"
print(f"Query: \"{query_filtered}\"")
print("Filter: doc_type in ['CAPA', 'Deviation'] AND effective_year >= 2023")

q_d = get_dense_embeddings([query_filtered])[0]
q_s = list(sparse_model.embed([query_filtered]))[0]

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
    prefetch=[
        models.Prefetch(
            query=q_d,
            using="dense_vector",
            filter=gxp_filter,
            limit=5,
        ),
        models.Prefetch(
            query=models.SparseVector(
                indices=q_s.indices.tolist(),
                values=q_s.values.tolist(),
            ),
            using="bm25_sparse_vector",
            filter=gxp_filter,
            limit=5,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=2,
    with_payload=True,
)

print("\n[Filtered Hybrid Results]:")
for rank, hit in enumerate(filtered_results.points, 1):
    payload = hit.payload
    print(f"  #{rank} [RRF Score: {hit.score:.4f}] {payload['doc_id']}: {payload['title']}")
    print(f"      Type: {payload['doc_type']} | Year: {payload['effective_year']} | System: {payload['system']}")
    print(f"      Description: {payload['description'][:110]}...")

print("\n" + "=" * 80)
print("Tutorial 04 Execution Complete!")
print("=" * 80)
