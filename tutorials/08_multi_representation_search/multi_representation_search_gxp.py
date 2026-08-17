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
   - 'dense_chunk': Chunk-level semantic embedding (Ollama qwen3-embedding:8b, 4096d)
   - 'dense_title': Document-level title semantic embedding (Ollama qwen3-embedding:8b, 4096d)
   - 'dense_scope': Document-level executive scope embedding (Ollama qwen3-embedding:8b, 4096d)
   - 'sparse_title': Sparse BM25 title vector (with server-side IDF)
2. Retrieval:
   - The Query API runs parallel prefetches across all four representations.
   - Fuses ranked lists with Reciprocal Rank Fusion (RRF).
   - Groups results by 'document_id' using query_points_groups, returning top matching chunks per document.
3. Observability:
   - Langfuse (http://localhost:3000) grouped trace monitoring.

Target Environment: Local Qdrant server at http://localhost:6333
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
import ollama
from langfuse import get_client, observe

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
VECTOR_SIZE = 4096
COLLECTION_NAME = "gxp_multi_representation_docs"

# ---------------------------------------------------------------------------
# 1. Connect to Local Qdrant & Initialize Models
# ---------------------------------------------------------------------------
print("=" * 80)
print(f"Step 1: Connecting to Qdrant at {QDRANT_URL}...")
client = QdrantClient(url=QDRANT_URL)
ollama_client = ollama.Client(host=OLLAMA_HOST)
langfuse = get_client()

print("Initializing models:")
print(f"  - Dense:      Ollama {EMBEDDING_MODEL} ({VECTOR_SIZE} dims)")
print("  - Sparse:     FastEmbed Qdrant/bm25 (server-side IDF enabled)")
print(f"  - Langfuse:   {LANGFUSE_HOST}")

sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")


@observe(as_type="embedding", name="ollama-qwen3-dense-embedding")
def get_dense_embeddings(texts: list) -> list:
    response = ollama_client.embed(model=EMBEDDING_MODEL, input=texts)
    langfuse.update_current_generation(
        model=EMBEDDING_MODEL,
        metadata={"text_count": len(texts), "dimensions": VECTOR_SIZE},
    )
    return response.embeddings


@observe(as_type="embedding", name="fastembed-bm25-sparse-embedding")
def get_sparse_embeddings(texts: list) -> list:
    embeddings = list(sparse_model.embed(texts))
    langfuse.update_current_generation(
        model="Qdrant/bm25",
        metadata={"text_count": len(texts), "type": "sparse_lexical"},
    )
    return embeddings


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
        "dense_chunk": models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        "dense_title": models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        "dense_scope": models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
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
@observe(as_type="span", name="ingest-multi-representation-chunks")
def ingest_chunked_docs():
    data_path = Path(__file__).resolve().parent.parent.parent / "data" / "gxp_chunked_documents.json"
    with open(data_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    points = []
    point_id = 1

    for doc in documents:
        # Compute document-level vectors (reused across all chunks of this document)
        dense_title_vec = get_dense_embeddings([doc["title"]])[0]
        dense_scope_vec = get_dense_embeddings([doc["scope"]])[0]
        sparse_title_raw = get_sparse_embeddings([doc["title"]])[0]
        sparse_title_vec = models.SparseVector(
            indices=sparse_title_raw.indices.tolist(),
            values=sparse_title_raw.values.tolist(),
        )

        # Compute chunk-level vectors
        chunk_texts = [f"{c['section']}: {c['text']}" for c in doc["chunks"]]
        chunk_dense_vecs = get_dense_embeddings(chunk_texts)

        for idx, chunk in enumerate(doc["chunks"]):
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense_chunk": chunk_dense_vecs[idx],
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
    langfuse.update_current_span(
        output={"indexed_chunks": len(points), "parent_docs": len(documents)},
        metadata={"source": data_path.name},
    )
    return len(points), len(documents)


print("\n" + "=" * 80)
print("Step 3: Embedding and indexing GxP document chunks via Ollama...")
chunk_count, doc_count = ingest_chunked_docs()
print(f"Indexed {chunk_count} chunk points across {doc_count} GxP documents.")

# ---------------------------------------------------------------------------
# 4. Multi-Representation Grouped Retrieval Function
# ---------------------------------------------------------------------------
@observe(as_type="retriever", name="multi-rep-grouped-retrieval")
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
    q_dense = get_dense_embeddings([query_text])[0]
    q_sparse_raw = get_sparse_embeddings([query_text])[0]
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

    groups_summary = []
    for rank, group in enumerate(response.groups, 1):
        doc_id = group.id
        print(f"\nDocument Group #{rank}: [{doc_id}] (Total Matching Chunks: {len(group.hits)})")
        group_hits_info = []
        for c_rank, hit in enumerate(group.hits, 1):
            p = hit.payload
            print(f"  --> Chunk Hit #{c_rank} [RRF Score: {hit.score:.4f}] {p['chunk_id']}: {p['section']}")
            print(f"      System: {p['system']} | Doc Type: {p['doc_type']}")
            print(f"      Text Excerpt: {p['chunk_text']}")
            group_hits_info.append({"chunk_id": p["chunk_id"], "score": hit.score})
        groups_summary.append({"doc_id": doc_id, "hits": group_hits_info})

    langfuse.update_current_span(
        input={"query": query_text, "filter_doc_type": doc_type_filter, "filter_system": system_filter},
        output=groups_summary,
    )
    return response.groups


# ---------------------------------------------------------------------------
# 5. Execute Test Scenarios
# ---------------------------------------------------------------------------
@observe(name="tutorial-08-multi-representation-pipeline")
def execute_multi_representation_scenarios():
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

    return langfuse.get_trace_url()


trace_url = execute_multi_representation_scenarios()
langfuse.flush()

print("\n" + "=" * 80)
print("Tutorial 08 Execution Complete!")
if trace_url:
    print(f"Langfuse Trace URL: {trace_url}")
print("=" * 80)
