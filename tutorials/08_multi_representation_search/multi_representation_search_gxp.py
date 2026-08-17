"""
Tutorial 08: Multi-Representation Search Across Titles, Scopes, and Body Chunks for GxP & CSV

A controlled GxP document (SOP, Validation Protocol, Deviation, CAPA, System Risk Assessment)
is rarely well-represented by a single vector:
- The Document Title carries the formal system name, SOP code, and regulatory identity.
- The Executive Scope carries the broad regulatory framework (21 CFR Part 11, GAMP 5).
- The Body Chunks contain the exact execution steps, test scripts, formulas, and mitigations.
- The Lexical Sparse Title carries exact acronyms (e.g. 'SOP-QA-042', 'Empower 3 CDS', 'RTO/RPO').

This tutorial demonstrates Qdrant's Multi-Representation Architecture:
1. Ingestion: Each document chunk is a point, storing named vectors for:
   - 'dense_chunk': Chunk-level semantic embedding
   - 'dense_title': Document-level title semantic embedding
   - 'dense_scope': Document-level executive scope embedding
   - 'sparse_title': Sparse BM25 title vector (with server-side IDF)
2. Retrieval:
   - The Query API runs parallel prefetches across all four representations.
   - Fuses ranked lists with Reciprocal Rank Fusion (RRF).
   - Groups results by 'document_id' using query_points_groups, returning top matching chunks per document.

Target Environment: Local Qdrant server at http://localhost:6333
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "gxp_multi_representation_docs"

# ---------------------------------------------------------------------------
# 1. Connect to Local Qdrant & Initialize FastEmbed Models
# ---------------------------------------------------------------------------
print("=" * 80)
print(f"Step 1: Connecting to Qdrant at {QDRANT_URL}...")
client = QdrantClient(url=QDRANT_URL)

print("Initializing FastEmbed models:")
print("  - Dense:  sentence-transformers/all-MiniLM-L6-v2 (384 dims)")
print("  - Sparse: Qdrant/bm25 (server-side IDF enabled)")

dense_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

# ---------------------------------------------------------------------------
# 2. Configure Multi-Representation Collection Schema
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"Step 2: Configuring multi-representation collection '{COLLECTION_NAME}'...")

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense_chunk": models.VectorParams(size=384, distance=models.Distance.COSINE),
        "dense_title": models.VectorParams(size=384, distance=models.Distance.COSINE),
        "dense_scope": models.VectorParams(size=384, distance=models.Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse_title": models.SparseVectorParams(modifier=models.Modifier.IDF)
    },
)

# Index fields for grouping and filtering
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="document_id",
    field_schema=models.PayloadSchemaType.KEYWORD,
)
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="system",
    field_schema=models.PayloadSchemaType.KEYWORD,
)
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="doc_type",
    field_schema=models.PayloadSchemaType.KEYWORD,
)
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="tags",
    field_schema=models.PayloadSchemaType.KEYWORD,
)
print(f"Collection '{COLLECTION_NAME}' schema created with 4 named vectors & payload indexes.")

# ---------------------------------------------------------------------------
# 3. Ingest Multi-Representation Document Chunks
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Step 3: Embedding and indexing GxP document chunks...")

data_path = Path(__file__).resolve().parent.parent.parent / "data" / "gxp_chunked_documents.json"
with open(data_path, "r", encoding="utf-8") as f:
    documents = json.load(f)

points = []
point_id = 1

for doc in documents:
    # Compute document-level vectors (reused across all chunks of this document)
    dense_title_vec = list(dense_model.embed([doc["title"]]))[0].tolist()
    dense_scope_vec = list(dense_model.embed([doc["scope"]]))[0].tolist()
    sparse_title_raw = list(sparse_model.embed([doc["title"]]))[0]
    sparse_title_vec = models.SparseVector(
        indices=sparse_title_raw.indices.tolist(),
        values=sparse_title_raw.values.tolist(),
    )

    # Compute chunk-level vectors
    chunk_texts = [f"{c['section']}: {c['text']}" for c in doc["chunks"]]
    chunk_dense_vecs = list(dense_model.embed(chunk_texts))

    for idx, chunk in enumerate(doc["chunks"]):
        points.append(
            models.PointStruct(
                id=point_id,
                vector={
                    "dense_chunk": chunk_dense_vecs[idx].tolist(),
                    "dense_title": dense_title_vec,
                    "dense_scope": dense_scope_vec,
                    "sparse_title": sparse_title_vec,
                },
                payload={
                    "document_id": doc["doc_id"],
                    "document_title": doc["title"],
                    "doc_type": doc["doc_type"],
                    "system": doc["system"],
                    "gamp_category": doc["gamp_category"],
                    "tags": doc["tags"],
                    "chunk_id": chunk["chunk_id"],
                    "section": chunk["section"],
                    "chunk_text": chunk["text"],
                },
            )
        )
        point_id += 1

client.upload_points(collection_name=COLLECTION_NAME, points=points)
print(f"Indexed {len(points)} chunk points across {len(documents)} GxP documents.")

# ---------------------------------------------------------------------------
# 4. Multi-Representation Grouped Retrieval Function
# ---------------------------------------------------------------------------
def retrieve_gxp_groups(
    query_text: str,
    limit: int = 3,
    group_size: int = 2,
    doc_type_filter: str = None,
    system_filter: str = None,
):
    print("\n" + "=" * 80)
    print(f"MULTI-REPRESENTATION QUERY: \"{query_text}\"")
    if doc_type_filter or system_filter:
        print(f"FILTER: doc_type={doc_type_filter} | system={system_filter}")
    print("=" * 80)

    # Generate query vectors
    q_dense = list(dense_model.embed([query_text]))[0].tolist()
    q_sparse_raw = list(sparse_model.embed([query_text]))[0]
    q_sparse = models.SparseVector(
        indices=q_sparse_raw.indices.tolist(),
        values=q_sparse_raw.values.tolist(),
    )

    # Build filter conditions
    must_conditions = []
    if doc_type_filter:
        must_conditions.append(models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type_filter)))
    if system_filter:
        must_conditions.append(models.FieldCondition(key="system", match=models.MatchValue(value=system_filter)))
    query_filter = models.Filter(must=must_conditions) if must_conditions else None

    # Execute 4-way prefetch fused with RRF, grouped by parent document_id
    response = client.query_points_groups(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=q_dense, using="dense_chunk", filter=query_filter, limit=20),
            models.Prefetch(query=q_dense, using="dense_title", filter=query_filter, limit=20),
            models.Prefetch(query=q_dense, using="dense_scope", filter=query_filter, limit=20),
            models.Prefetch(query=q_sparse, using="sparse_title", filter=query_filter, limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=query_filter,
        group_by="document_id",
        group_size=group_size,
        limit=limit,
        with_payload=True,
    )

    for rank, group in enumerate(response.groups, 1):
        doc_id = group.id
        print(f"\nDocument Group #{rank}: [{doc_id}] (Total Matching Chunks: {len(group.hits)})")
        for c_rank, hit in enumerate(group.hits, 1):
            p = hit.payload
            print(f"  --> Chunk Hit #{c_rank} [RRF Score: {hit.score:.4f}] {p['chunk_id']}: {p['section']}")
            print(f"      System: {p['system']} | Doc Type: {p['doc_type']}")
            print(f"      Text Excerpt: {p['chunk_text']}")


# ---------------------------------------------------------------------------
# 5. Execute Test Scenarios
# ---------------------------------------------------------------------------

# Scenario 1: Specific test execution challenge query
retrieve_gxp_groups(
    "ApexTrack peak detection baseline resolution and retention time repeatability"
)

# Scenario 2: Regulatory audit trail governance query
retrieve_gxp_groups(
    "Quality Assurance independent audit trail review before batch release"
)

# Scenario 3: Infrastructure failure investigation query
retrieve_gxp_groups(
    "Modbus TCP/IP packet trace buffer overrun in manufacturing VLAN"
)

# Scenario 4: Filtered multi-representation query (Risk Assessments only)
retrieve_gxp_groups(
    "compromised digital signing certificates and Hardware Security Module private key protection",
    doc_type_filter="Risk Assessment",
)

print("\n" + "=" * 80)
print("Tutorial 08 Execution Complete!")
print("=" * 80)
