"""
Tutorial 05: Hybrid Search with Late-Interaction (ColBERT) Reranking for Life Science Quality & CSV

Architecture:
1. Ingestion:
   - Dense Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-dim, semantic context)
   - Sparse Embeddings: Qdrant/bm25 with IDF modifier (keyword/acronym/section matching)
   - Late Interaction Multivectors: colbert-ir/colbertv2.0 (128-dim per token, MaxSim reranking)
2. Retrieval:
   - Prefetch candidate pool using Dense + BM25 Sparse search (high recall)
   - Rerank prefetched candidates using ColBERT late interaction multi-vector (high precision)
3. Target Environment: Local Qdrant server at http://localhost:6333
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "gxp_hybrid_reranking_docs"

# ---------------------------------------------------------------------------
# 1. Connect to Local Qdrant Server & Initialize FastEmbed Models
# ---------------------------------------------------------------------------
print("=" * 80)
print(f"Step 1: Connecting to Qdrant at {QDRANT_URL}...")
client = QdrantClient(url=QDRANT_URL)

print("Initializing 3 embedding models locally via FastEmbed:")
print("  [1] Dense:           sentence-transformers/all-MiniLM-L6-v2 (384 dims)")
print("  [2] Sparse:          Qdrant/bm25 (IDF modifier enabled)")
print("  [3] Late Interaction: colbert-ir/colbertv2.0 (128 dims/token, MaxSim)")

dense_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
colbert_model = LateInteractionTextEmbedding(model_name="colbert-ir/colbertv2.0")

# ---------------------------------------------------------------------------
# 2. Create Collection with Dense, Sparse, and Late-Interaction Vectors
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
        ),
        "multi": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0),  # Disable HNSW: used solely for candidate reranking
        ),
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
    },
)
print(f"Collection '{COLLECTION_NAME}' created successfully.")

# ---------------------------------------------------------------------------
# 3. Ingest GxP Quality & Validation Documents
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Step 3: Generating 3-way vector embeddings and uploading documents...")

data_path = Path(__file__).resolve().parent.parent.parent / "data" / "gxp_quality_docs.json"
with open(data_path, "r", encoding="utf-8") as f:
    documents = json.load(f)

texts = [f"{doc['title']}. {doc['description']}" for doc in documents]

dense_embeddings = list(dense_model.embed(texts))
sparse_embeddings = list(sparse_model.embed(texts))
colbert_embeddings = list(colbert_model.embed(texts))

points = []
for idx, doc in enumerate(documents):
    s_vec = sparse_embeddings[idx]
    point = models.PointStruct(
        id=idx + 1,
        payload=doc,
        vector={
            "dense": dense_embeddings[idx].tolist(),
            "sparse": models.SparseVector(
                indices=s_vec.indices.tolist(),
                values=s_vec.values.tolist(),
            ),
            "multi": colbert_embeddings[idx].tolist(),
        },
    )
    points.append(point)

client.upload_points(collection_name=COLLECTION_NAME, points=points)
print(f"Successfully uploaded {len(points)} documents with Dense, BM25, and ColBERT vectors.")

# ---------------------------------------------------------------------------
# 4. Pipeline Execution: Dense vs. BM25 vs. Hybrid RRF vs. Late Interaction Reranking
# ---------------------------------------------------------------------------
def run_retrieval_and_reranking(query_text: str):
    print("\n" + "=" * 80)
    print(f"USER QUERY: \"{query_text}\"")
    print("=" * 80)

    # Generate query vectors for all 3 models
    q_dense = list(dense_model.embed([query_text]))[0].tolist()
    q_sparse = list(sparse_model.embed([query_text]))[0]
    q_colbert = list(colbert_model.embed([query_text]))[0].tolist()

    sparse_obj = models.SparseVector(
        indices=q_sparse.indices.tolist(),
        values=q_sparse.values.tolist(),
    )

    # 1. Dense Only Retrieval
    dense_res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_dense,
        using="dense",
        limit=3,
    ).points

    # 2. Sparse (BM25) Only Retrieval
    sparse_res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=sparse_obj,
        using="sparse",
        limit=3,
    ).points

    # 3. Hybrid Search (Dense + Sparse with Reciprocal Rank Fusion)
    prefetch_stages = [
        models.Prefetch(query=q_dense, using="dense", limit=10),
        models.Prefetch(query=sparse_obj, using="sparse", limit=10),
    ]

    rrf_res = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch_stages,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=3,
        with_payload=True,
    ).points

    # 4. Hybrid Search + Late Interaction (ColBERT) Reranking
    rerank_res = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch_stages,
        query=q_colbert,
        using="multi",
        limit=3,
        with_payload=True,
    ).points

    print("\n[Stage 1: Dense Retrieval (Semantic Meaning)]")
    for r, h in enumerate(dense_res, 1):
        print(f"  #{r} [Score: {h.score:.4f}] {h.payload['doc_id']}: {h.payload['title']}")

    print("\n[Stage 2: BM25 Sparse Retrieval (Keyword & Token Matching)]")
    for r, h in enumerate(sparse_res, 1):
        print(f"  #{r} [Score: {h.score:.4f}] {h.payload['doc_id']}: {h.payload['title']}")

    print("\n[Stage 3: Hybrid Search with RRF Fusion]")
    for r, h in enumerate(rrf_res, 1):
        print(f"  #{r} [RRF Score: {h.score:.4f}] {h.payload['doc_id']}: {h.payload['title']}")

    print("\n[Stage 4: Hybrid Search + ColBERT Late-Interaction Reranking (MaxSim Precision)]")
    for r, h in enumerate(rerank_res, 1):
        print(f"  #{r} [ColBERT Score: {h.score:.4f}] {h.payload['doc_id']}: {h.payload['title']}")
        print(f"      System: {h.payload['system']} | Type: {h.payload['doc_type']}")
        print(f"      Summary: {h.payload['description'][:110]}...")


# ---------------------------------------------------------------------------
# 5. Test Queries
# ---------------------------------------------------------------------------
# Query 1: Data integrity and periodic audit log inspection
run_retrieval_and_reranking("regulatory compliance for immutable time-stamped audit trail review")

# Query 2: System resilience & backup validation
run_retrieval_and_reranking("periodic database backup snapshot failures and disaster recovery restoration drill")

print("\n" + "=" * 80)
print("Tutorial 05 Execution Complete!")
print("=" * 80)
