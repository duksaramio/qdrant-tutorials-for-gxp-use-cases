"""
Tutorial 07: Multivector Document Retrieval (ColPali/ColBERT Style) for Life Science Quality & CSV

In Life Sciences and Computer System Validation (CSV), validation deliverables and quality
investigations are complex multi-page PDF documents containing tables, test scripts,
risk matrices, and diagrams.

Multimodal/Vision and Late-Interaction models produce heavy multivector representations
(~100 to 1,000 vectors per document page). Building full HNSW graphs on uncompressed
multivectors causes prohibitive RAM consumption and slow index build times.

This tutorial demonstrates Qdrant's optimized 2-stage architecture:
1. Ingestion:
   - 'mean_pooled': Multivectors mean-pooled into condensed structural vectors (HNSW ON)
   - 'original': Full-resolution token/patch multivectors (HNSW OFF: m=0)
2. Retrieval:
   - Stage 1 (Fast Recall): Prefetch top candidate pages using HNSW-indexed 'mean_pooled' vectors
   - Stage 2 (Fine-Grained Precision): Rerank candidates with 'original' multivector MaxSim

Target Environment: Local Qdrant server at http://localhost:6333
"""

import json
import os
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import LateInteractionTextEmbedding

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "gxp_pdf_pages_demo"

# ---------------------------------------------------------------------------
# 1. Connect to Local Qdrant Server & Initialize ColBERT Model
# ---------------------------------------------------------------------------
print("=" * 80)
print(f"Step 1: Connecting to Qdrant at {QDRANT_URL}...")
client = QdrantClient(url=QDRANT_URL)

print("Initializing ColBERT Late-Interaction model (colbert-ir/colbertv2.0)...")
colbert_model = LateInteractionTextEmbedding(model_name="colbert-ir/colbertv2.0")

# ---------------------------------------------------------------------------
# 2. Configure 2-Stage Multi-Vector Collection (Mean-Pooled + Original)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"Step 2: Configuring multi-vector collection '{COLLECTION_NAME}'...")

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        # 1. Original high-resolution multivector: HNSW OFF (m=0) to eliminate graph overhead
        "original": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0),  # HNSW disabled: strictly used for reranking
        ),
        # 2. Mean-pooled condensed multivector: HNSW ON for fast first-stage candidate retrieval
        "mean_pooled": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
        ),
    },
)
print(f"Collection '{COLLECTION_NAME}' created with original (m=0) and mean_pooled vectors.")


# ---------------------------------------------------------------------------
# 3. Helper: Mean Pooling for Multivectors
# ---------------------------------------------------------------------------
def mean_pool_multivector(vectors: np.ndarray, num_pooled_chunks: int = 4) -> np.ndarray:
    """
    Compresses a sequence of token/patch vectors into condensed structural chunk vectors
    via mean pooling, reducing vector count while preserving semantic coverage.
    """
    tokens_per_chunk = int(np.ceil(len(vectors) / num_pooled_chunks))
    pooled = []
    for i in range(num_pooled_chunks):
        chunk = vectors[i * tokens_per_chunk : (i + 1) * tokens_per_chunk]
        if len(chunk) > 0:
            pooled.append(chunk.mean(axis=0))
    return np.array(pooled)


# ---------------------------------------------------------------------------
# 4. Ingest Multi-Page GxP Documents
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Step 3: Embedding and mean-pooling GxP PDF page records...")

data_path = Path(__file__).resolve().parent.parent.parent / "data" / "gxp_pdf_pages.json"
with open(data_path, "r", encoding="utf-8") as f:
    pages = json.load(f)

page_texts = [
    f"Document: {p['doc_title']} (Page {p['page_number']}). Section: {p['section_title']}. Content: {p['content']}"
    for p in pages
]

raw_colbert_embeddings = list(colbert_model.embed(page_texts))

points = []
for idx, page in enumerate(pages):
    full_multivector = raw_colbert_embeddings[idx]  # shape: [num_tokens, 128]
    pooled_multivector = mean_pool_multivector(full_multivector, num_pooled_chunks=4)

    point = models.PointStruct(
        id=idx + 1,
        payload=page,
        vector={
            "original": full_multivector.tolist(),
            "mean_pooled": pooled_multivector.tolist(),
        },
    )
    points.append(point)

client.upload_points(collection_name=COLLECTION_NAME, points=points)
print(f"Indexed {len(points)} GxP document pages with dual multivector configurations.")


# ---------------------------------------------------------------------------
# 5. Execute Optimized Two-Stage Multivector Retrieval
# ---------------------------------------------------------------------------
def run_twostage_document_search(query_text: str, gxp_filter: models.Filter = None):
    print("\n" + "=" * 80)
    print(f"AUDIT / CSV QUERY: \"{query_text}\"")
    if gxp_filter:
        print("APPLIED FILTER: Active GxP Metadata Constraints")
    print("=" * 80)

    # 1. Embed query with ColBERT
    q_full = list(colbert_model.embed([query_text]))[0]  # shape: [q_tokens, 128]
    q_pooled = mean_pool_multivector(q_full, num_pooled_chunks=2)

    # 2. Stage 1 (Fast Candidate Prefetch via Mean-Pooled HNSW)
    #    Stage 2 (Fine-Grained MaxSim Reranking via Original Multivector)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=q_pooled.tolist(),
                using="mean_pooled",
                filter=gxp_filter,
                limit=8,  # Prefetch top-8 candidate pages
            )
        ],
        query=q_full.tolist(),  # Full resolution MaxSim reranker
        using="original",
        query_filter=gxp_filter,
        limit=3,
        with_payload=True,
    ).points

    print("\n[Two-Stage Retrieval Results (Mean-Pooled Prefetch -> Full ColBERT MaxSim)]:")
    for rank, hit in enumerate(results, 1):
        p = hit.payload
        print(f"  #{rank} [MaxSim Score: {hit.score:.4f}] {p['page_id']} ({p['doc_title']}, Page {p['page_number']})")
        print(f"      Section: {p['section_title']}")
        print(f"      System: {p['system']} | GAMP: {p['gamp_category']} | Type: {p['doc_type']}")
        print(f"      Layout Elements: {', '.join(p['layout_elements'])}")
        print(f"      Content: {p['content'][:110]}...\n")


# ---------------------------------------------------------------------------
# 6. Test Scenario Queries
# ---------------------------------------------------------------------------

# Scenario 1: Analytical validation - Peak integration & baseline resolution formula
run_twostage_document_search(
    "chromatographic peak integration algorithm verification acceptance criteria and retention time repeatability"
)

# Scenario 2: Technical root cause investigation - Modbus buffer overrun diagnostics
run_twostage_document_search(
    "root cause analysis of Modbus communication packet buffer overrun on bioreactor DO transmitter"
)

# Scenario 3: FMEA Risk Mitigation - Electronic signature key compromise
run_twostage_document_search(
    "mitigation controls for electronic signature private key compromise in cloud EDMS"
)

# Scenario 4: Filtered search - only Validation Protocols
prot_filter = models.Filter(
    must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value="Validation Protocol"))]
)
run_twostage_document_search(
    "raw data deletion resistance and immutable audit log verification steps",
    gxp_filter=prot_filter,
)

print("=" * 80)
print("Tutorial 07 Execution Complete!")
print("=" * 80)
