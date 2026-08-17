"""
Tutorial 09: Branch-Aware Semantic Search Over Versioned GxP & CSV Document Lifecycles

In Life Science Quality (QMS / EDMS) and Computer System Validation (CSV / GAMP 5),
controlled documents evolve across branches and lifecycles:
- 'main-effective': Officially approved, legally active GxP procedures and validated baselines.
- 'draft-cc-2024': Proposed draft revisions undergoing Change Control review and approval.
- 'site-eu-annex11': Regional site overlays with EU GMP Annex 11 / Qualified Person requirements.

An un-scoped vector search leaks across branches, returning superseded versions, unauthorized drafts,
or foreign branch modifications.

This script demonstrates Qdrant's Branch-Aware Search architecture:
1. Index document versions as immutable points with deterministic UUIDv5 IDs and Ollama embeddings.
2. Track supersede and retirement events using an 'overwritten_in' nested payload schema.
3. Construct branch visibility filters that traverse ancestry lineage with fork-point cutoffs.
4. Execute point lookups and semantic vector searches scoped strictly to any branch's live view.

Target Environment: Local Qdrant server at http://localhost:6333
"""

import os
import uuid
from typing import List, Tuple, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
import ollama

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
VECTOR_SIZE = 4096
COLLECTION_NAME = "gxp_branch_aware_docs"

# ---------------------------------------------------------------------------
# 1. Connect to Local Qdrant & Initialize Ollama Client
# ---------------------------------------------------------------------------
print("=" * 80)
print(f"Step 1: Connecting to Qdrant at {QDRANT_URL}...")
client = QdrantClient(url=QDRANT_URL)
ollama_client = ollama.Client(host=OLLAMA_HOST)

print(f"Initializing Ollama model '{EMBEDDING_MODEL}' ({VECTOR_SIZE} dims)...")

# Deterministic namespace for reproducible point UUIDs
NS = uuid.UUID("00000000-0000-0000-0000-000000000042")


def point_id(branch: str, seq: int, path: str) -> str:
    """Generates a deterministic UUID based on branch name, commit seq, and document path."""
    return str(uuid.uuid5(NS, f"{branch}|{seq}|{path}"))


def get_embeddings(texts: list) -> list:
    return ollama_client.embed(model=EMBEDDING_MODEL, input=texts).embeddings


# ---------------------------------------------------------------------------
# 2. Create Collection & Payload Indexes
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"Step 2: Configuring branch-aware collection '{COLLECTION_NAME}'...")

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=VECTOR_SIZE,
        distance=models.Distance.COSINE,
    ),
)

# Payload indexes required for branch lineage and nested supersede filtering
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="path", field_schema=models.PayloadSchemaType.KEYWORD)
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="branch", field_schema=models.PayloadSchemaType.KEYWORD)
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="seq", field_schema=models.PayloadSchemaType.INTEGER)
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="overwritten_in[].by", field_schema=models.PayloadSchemaType.KEYWORD)
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="overwritten_in[].seq", field_schema=models.PayloadSchemaType.INTEGER)

print(f"Collection '{COLLECTION_NAME}' created with lineage payload indexes.")


# ---------------------------------------------------------------------------
# 3. Lineage Visibility Filter Engine
# ---------------------------------------------------------------------------
def branch_filter(branch: str, ancestry: List[Tuple[str, int]]) -> models.Filter:
    """
    Constructs a Qdrant filter matching strictly the live view of a branch:
    - Includes points created in this branch (all sequences)
    - Includes points created in ancestor branches up to their fork sequence
    - Excludes points superseded/deleted by this branch
    - Excludes points superseded/deleted by ancestor branches before the fork point
    """
    # 1. Candidate selector for current branch
    should = [
        models.FieldCondition(
            key="branch",
            match=models.MatchValue(value=branch),
        )
    ]

    # 2. Exclude files this branch overwrote or retired
    must_not = [
        models.NestedCondition(
            nested=models.Nested(
                key="overwritten_in",
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="by",
                            match=models.MatchValue(value=branch),
                        )
                    ]
                ),
            )
        )
    ]

    # 3. Add one selector and one exclusion per ancestor, bound to its fork seq
    for parent, cut in ancestry:
        # Include ancestor files created up to fork seq
        branch_selector = models.Filter(
            must=[
                models.FieldCondition(key="branch", match=models.MatchValue(value=parent)),
                models.FieldCondition(key="seq", range=models.Range(lte=cut)),
            ]
        )
        should.append(branch_selector)

        # Exclude ancestor files overwritten before or at fork seq
        overwrite_exclusion = models.NestedCondition(
            nested=models.Nested(
                key="overwritten_in",
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="by", match=models.MatchValue(value=parent)),
                        models.FieldCondition(key="seq", range=models.Range(lte=cut)),
                    ]
                ),
            )
        )
        must_not.append(overwrite_exclusion)

    return models.Filter(should=should, must_not=must_not)


def visibility_filter(branch: str, ancestry: List[Tuple[str, int]], path: Optional[str] = None) -> models.Filter:
    """Combines branch lineage visibility with an optional file path constraint."""
    must = [branch_filter(branch, ancestry)]
    if path:
        must.append(models.FieldCondition(key="path", match=models.MatchValue(value=path)))
    return models.Filter(must=must)


# ---------------------------------------------------------------------------
# 4. Storage & Retrieval Operations
# ---------------------------------------------------------------------------
def lookup(file_name: str, branch: str, ancestry: List[Tuple[str, int]]):
    """Retrieves the exact active version of a file visible in a given branch."""
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=visibility_filter(branch, ancestry, path=file_name),
        limit=1,
        with_payload=True,
    )
    return points[0] if points else None


def supersede(point, by: str, seq: int):
    """Marks an existing point as overwritten/superseded by a branch at commit seq."""
    marks = point.payload.get("overwritten_in", []) + [{"by": by, "seq": seq}]
    client.set_payload(
        collection_name=COLLECTION_NAME,
        payload={"overwritten_in": marks},
        points=[point.id],
    )


def update(file_name: str, branch: str, seq: int, content: str, ancestry: List[Tuple[str, int]], doc_type: str = "SOP"):
    """Writes a new document version, marking any previously visible version on this branch as superseded."""
    prev = lookup(file_name, branch, ancestry)
    if prev:
        supersede(prev, by=branch, seq=seq)

    vec = get_embeddings([content])[0]
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=point_id(branch, seq, file_name),
                vector=vec,
                payload={
                    "path": file_name,
                    "content": content,
                    "branch": branch,
                    "seq": seq,
                    "doc_type": doc_type,
                    "overwritten_in": [],
                },
            )
        ],
    )


def delete(file_name: str, branch: str, seq: int, ancestry: List[Tuple[str, int]]):
    """Retires a document on a branch without deleting historical points."""
    prev = lookup(file_name, branch, ancestry)
    if prev:
        supersede(prev, by=branch, seq=seq)


def search(query: str, branch: str, ancestry: List[Tuple[str, int]], limit: int = 3):
    """Executes a semantic vector search scoped strictly to a branch's live view."""
    q_vec = get_embeddings([query])[0]
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec,
        query_filter=visibility_filter(branch, ancestry),
        limit=limit,
        with_payload=True,
    ).points


# ---------------------------------------------------------------------------
# 5. Replay GxP Document Version History Across Branches
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Step 3: Simulating GxP Document Commits & Branching via Ollama...")

# Lineage tracking
root_ancestry = []  # 'main-effective' is the root baseline
A_ancestry = [("main-effective", 2)]  # 'draft-cc-2024' forks from main at seq 2
B_ancestry = [("main-effective", 3)]  # 'site-eu-overlay' forks from main at seq 3

# Root Branch: Initial Baseline Release (seq 0)
print("--> [main-effective @ seq 0]: Initial Release of Global Quality SOPs")
update(
    "SOP-QA-042.md",
    "main-effective",
    seq=0,
    content="SOP-QA-042 v1.0: Audit trail reviews for computerized systems must be conducted on a monthly basis by QA.",
    ancestry=root_ancestry,
)
update(
    "VAL-OQ-108.md",
    "main-effective",
    seq=0,
    content="VAL-OQ-108 v1.0: Waters Empower 3 CDS operational qualification protocol. Peak integration repeatability requires RSD < 1.0%.",
    ancestry=root_ancestry,
    doc_type="Validation Protocol",
)
update(
    "POL-SEC-01.md",
    "main-effective",
    seq=0,
    content="POL-SEC-01 v1.0: Password security policy requiring password expiration every 90 days across all laboratory PCs.",
    ancestry=root_ancestry,
)

# Root Branch: Change Control Update (seq 1)
print("--> [main-effective @ seq 1]: Change Control CC-089 approved: Audit trail frequency updated")
update(
    "SOP-QA-042.md",
    "main-effective",
    seq=1,
    content="SOP-QA-042 v2.0: Audit trail reviews must be conducted prior to commercial batch release by an independent QA reviewer.",
    ancestry=root_ancestry,
)

# Root Branch: Policy Retirement (seq 2)
print("--> [main-effective @ seq 2]: POL-SEC-01 retired; replaced by Global MFA standard")
delete("POL-SEC-01.md", "main-effective", seq=2, ancestry=root_ancestry)

# Branch A: Draft Change Control for LIMS (forked at main seq 2)
print("--> [draft-cc-2024 @ seq 0]: Forked from main @ seq 2. Adding CAPA-2023-019 database recovery plan")
update(
    "CAPA-2023-019.md",
    "draft-cc-2024",
    seq=0,
    content="CAPA-2023-019 Draft: Automated database backup remediation in LIMS with RTO < 4h and quarterly DR drills.",
    ancestry=A_ancestry,
    doc_type="CAPA",
)

print("--> [draft-cc-2024 @ seq 1]: Proposing AI-assisted automated audit trail review in SOP-QA-042")
update(
    "SOP-QA-042.md",
    "draft-cc-2024",
    seq=1,
    content="SOP-QA-042 Draft-CC: Proposes AI-assisted automated anomaly detection for audit trail review prior to batch release.",
    ancestry=A_ancestry,
)

# Root Branch moves forward: Validation Protocol Update (seq 3)
print("--> [main-effective @ seq 3]: Upgrading Empower CDS protocol criteria to RSD < 0.5%")
update(
    "VAL-OQ-108.md",
    "main-effective",
    seq=3,
    content="VAL-OQ-108 v2.0: Waters Empower 3 CDS OQ protocol. Peak integration repeatability acceptance tightened to RSD < 0.50%.",
    ancestry=root_ancestry,
    doc_type="Validation Protocol",
)

# Branch B: Regional EU Site Overlay (forked at main seq 3)
print("--> [site-eu-overlay @ seq 0]: Forked from main @ seq 3. Overriding SOP-QA-042 for EU Annex 11 QP sign-off")
update(
    "SOP-QA-042.md",
    "site-eu-overlay",
    seq=0,
    content="SOP-QA-042 EU-Site: Audit trail review must be verified by an EU Qualified Person (QP) in compliance with EU Annex 11.9.",
    ancestry=B_ancestry,
)


# ---------------------------------------------------------------------------
# 6. Verify Exact Document Resolution Across Branches
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Step 4: Testing Exact File Lookups Across Different Branches...")
print("=" * 80)

branches = [
    ("main-effective", root_ancestry, "Official Global Validated Baseline"),
    ("draft-cc-2024", A_ancestry, "Under Change Control Review"),
    ("site-eu-overlay", B_ancestry, "EU Regional Manufacturing Site"),
]

print("\n[File Resolution: 'SOP-QA-042.md']")
for b_name, b_anc, desc in branches:
    pt = lookup("SOP-QA-042.md", b_name, b_anc)
    print(f"  Branch '{b_name}' ({desc}):")
    print(f"    --> Version content: \"{pt.payload['content']}\"")

print("\n[File Resolution: 'VAL-OQ-108.md' (Demonstrating Fork Cutoff Isolation)]")
for b_name, b_anc, desc in branches:
    pt = lookup("VAL-OQ-108.md", b_name, b_anc)
    print(f"  Branch '{b_name}':")
    print(f"    --> Version content: \"{pt.payload['content']}\"")

print("\n[File Resolution: 'POL-SEC-01.md' (Deleted on main @ seq 2)]")
for b_name, b_anc, desc in branches:
    pt = lookup("POL-SEC-01.md", b_name, b_anc)
    status = pt.payload["content"] if pt else "NONE (Successfully excluded / retired)"
    print(f"  Branch '{b_name}': {status}")


# ---------------------------------------------------------------------------
# 7. Semantic Vector Search Scoped to Specific Branches
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Step 5: Executing Semantic Vector Search Scoped to Each Branch...")
print("=" * 80)

query = "What is the requirement and frequency for audit trail review?"
print(f"QUERY: \"{query}\"")

for b_name, b_anc, desc in branches:
    print(f"\n--- Search Results on Branch: '{b_name}' ({desc}) ---")
    hits = search(query, b_name, b_anc, limit=2)
    for rank, h in enumerate(hits, 1):
        print(f"  #{rank} [Score: {h.score:.4f}] {h.payload['path']} (Branch: {h.payload['branch']}, Seq: {h.payload['seq']})")
        print(f"      Content: \"{h.payload['content']}\"")

print("\n" + "=" * 80)
print("Tutorial 09 Execution Complete!")
print("=" * 80)
