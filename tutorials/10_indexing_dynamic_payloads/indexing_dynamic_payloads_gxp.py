"""
Tutorial 10: Indexing Payloads of Random Shape (Dynamic Attributes) for GxP & CSV

In Life Science Quality (QMS) and Computer System Validation (CSV / GAMP 5), computerized
systems across analytical QC labs, manufacturing suites, and cloud platforms generate
thousands of open-ended, system-specific telemetry and qualification attributes:
- Analytical CDS: flow_rate_ml_min, column_temp_c, detector_type, rsd_retention_time
- Bioreactor SCADA/MES: dissolved_oxygen_pct, agitation_rpm, vessel_pressure_psi
- Cloud EDMS: ectd_module, hsm_fips_level, ind_number, signing_tier
- IT Infrastructure: rto_hours, rpo_minutes, encryption_standard, backup_storage_tier

The One-Index-Per-Key Trap:
Creating a separate payload index for every distinct incoming attribute creates thousands of
indexes, causing massive RAM consumption and slow index build times.

The Solution:
Reshape open-ended dynamic attributes into fixed, typed Entity-Attribute-Value (EAV) arrays
at ingest time:
- 'attrs': Strings/Keywords (attrs[].key, attrs[].value)
- 'attrs_num': Numerical floats for range queries (attrs_num[].key, attrs_num[].value)
- 'attrs_bool': Boolean flags (attrs_bool[].key, attrs_bool[].value)
- 'attrs_flat': 'key=value' keyword terms for ultra-fast categorical matching

Target Environment: Local Qdrant server at http://localhost:6333
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
import ollama

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
VECTOR_SIZE = 4096
COLLECTION_NAME = "gxp_dynamic_payloads"

# ---------------------------------------------------------------------------
# 1. Connect to Local Qdrant & Initialize Ollama Client
# ---------------------------------------------------------------------------
print("=" * 80)
print(f"Step 1: Connecting to Qdrant at {QDRANT_URL}...")
client = QdrantClient(url=QDRANT_URL)
ollama_client = ollama.Client(host=OLLAMA_HOST)

print(f"Initializing Ollama model '{EMBEDDING_MODEL}' ({VECTOR_SIZE} dims)...")


def get_embeddings(texts: list) -> list:
    return ollama_client.embed(model=EMBEDDING_MODEL, input=texts).embeddings


# ---------------------------------------------------------------------------
# 2. Configure Collection & Fixed Payload Indexes
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"Step 2: Configuring collection '{COLLECTION_NAME}' with fixed EAV indexes...")

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=VECTOR_SIZE,
        distance=models.Distance.COSINE,
    ),
)

# Fixed indexes that NEVER grow, regardless of how many distinct attributes arrive:
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="doc_id", field_schema=models.PayloadSchemaType.KEYWORD)
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="system_category", field_schema=models.PayloadSchemaType.KEYWORD)

# 1. String EAV array (exact keyword matches on dynamic keys)
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="attrs[].key", field_schema=models.PayloadSchemaType.KEYWORD)
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="attrs[].value", field_schema=models.PayloadSchemaType.KEYWORD)

# 2. Numeric EAV array (supports Range filtering on any numeric attribute)
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="attrs_num[].key", field_schema=models.PayloadSchemaType.KEYWORD)
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="attrs_num[].value", field_schema=models.PayloadSchemaType.FLOAT)

# 3. Boolean EAV array (supports boolean flag filtering)
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="attrs_bool[].key", field_schema=models.PayloadSchemaType.KEYWORD)
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="attrs_bool[].value", field_schema=models.PayloadSchemaType.BOOL)

# 4. Concatenated 'key=value' array for 30-40% faster exact-match categorical lookups
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="attrs_flat", field_schema=models.PayloadSchemaType.KEYWORD)

print(f"Collection '{COLLECTION_NAME}' created with 8 fixed indexes covering infinite dynamic attributes.")


# ---------------------------------------------------------------------------
# 3. Ingestion Helper: Reshaping Raw Dynamic Dictionaries into Typed Arrays
# ---------------------------------------------------------------------------
def reshape_gxp_attributes(raw_attrs: Dict[str, Any]) -> Dict[str, List]:
    """
    Splits open-ended raw dictionary attributes into typed EAV arrays and flat key=value terms.
    """
    strings = []
    numbers = []
    bools = []
    flats = []

    for key, value in raw_attrs.items():
        if isinstance(value, bool):
            bools.append({"key": key, "value": value})
            flats.append(f"{key}={value}")
        elif isinstance(value, (int, float)):
            numbers.append({"key": key, "value": float(value)})
            flats.append(f"{key}={value}")
        elif isinstance(value, str):
            strings.append({"key": key, "value": value})
            flats.append(f"{key}={value}")

    return {
        "attrs": strings,
        "attrs_num": numbers,
        "attrs_bool": bools,
        "attrs_flat": flats,
    }


# ---------------------------------------------------------------------------
# 4. Ingest Heterogeneous GxP Systems & Instrument Records
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Step 3: Ingesting GxP records with heterogeneous, system-specific attributes via Ollama...")

raw_records = [
    {
        "doc_id": "RUN-HPLC-8841",
        "system_category": "Chromatography CDS",
        "title": "Waters Empower 3 HPLC Assay Release Run for Lot #LOT-9921",
        "summary": "Commercial release assay for active pharmaceutical ingredient (API). Retention time repeatability verified within acceptance limits.",
        "custom_attributes": {
            "instrument_id": "HPLC-QC-04",
            "column_temp_c": 35.0,
            "flow_rate_ml_min": 1.25,
            "rsd_retention_time_pct": 0.32,
            "detector_type": "UV-Vis Photodiode Array",
            "is_gxp_compliant": True,
            "requires_qa_signoff": True,
        },
    },
    {
        "doc_id": "RUN-HPLC-8842",
        "system_category": "Chromatography CDS",
        "title": "Waters Empower 3 Stability Run for Lot #LOT-9922 (High Temperature Challenge)",
        "summary": "Accelerated 40C/75RH stability testing run. Elevated column temperature used to test impurity separation.",
        "custom_attributes": {
            "instrument_id": "HPLC-QC-04",
            "column_temp_c": 45.0,
            "flow_rate_ml_min": 0.85,
            "rsd_retention_time_pct": 0.48,
            "detector_type": "UV-Vis Photodiode Array",
            "is_gxp_compliant": True,
            "requires_qa_signoff": True,
        },
    },
    {
        "doc_id": "BIO-BATCH-2024-09",
        "system_category": "Manufacturing MES / SCADA",
        "title": "Bioreactor Commercial Fermentation Batch Run #B-2024-09",
        "summary": "Continuous telemetry logging for 2000L production bioreactor. Monitored dissolved oxygen and vessel agitation.",
        "custom_attributes": {
            "bioreactor_id": "BIO-REACT-02",
            "vessel_volume_liters": 2000.0,
            "dissolved_oxygen_pct": 34.2,
            "agitation_rpm": 250.0,
            "vessel_pressure_psi": 1.45,
            "feed_rate_l_hr": 12.5,
            "is_gxp_compliant": True,
            "in_deviation_investigation": True,
        },
    },
    {
        "doc_id": "EDMS-SRA-2024-01",
        "system_category": "Cloud Document Management",
        "title": "Cloud-Hosted Documentum EDMS GAMP 5 System Risk Assessment",
        "summary": "FMEA system risk assessment evaluating multi-tenant SaaS controls, PKI electronic signatures, and data residency.",
        "custom_attributes": {
            "cloud_provider": "AWS-GovCloud",
            "hsm_fips_level": 3.0,
            "ectd_module": "Module 3 (Quality)",
            "soc2_type_ii_certified": True,
            "rpo_minutes": 15.0,
            "rto_hours": 3.5,
            "is_gxp_compliant": True,
        },
    },
    {
        "doc_id": "LIMS-CAPA-2023-19",
        "system_category": "Laboratory LIMS",
        "title": "LIMS Database Automated Snapshot and DR Remediation Plan",
        "summary": "CAPA addressing Oracle RMAN automated snapshot backup verification and scheduled restoration drills.",
        "custom_attributes": {
            "database_engine": "Oracle-Enterprise-19c",
            "backup_storage_tier": "AWS-S3-Glacier-Vault",
            "rpo_minutes": 10.0,
            "rto_hours": 2.0,
            "quarterly_drill_frequency": 4.0,
            "is_gxp_compliant": True,
            "in_deviation_investigation": False,
        },
    },
]

texts = [f"{item['title']}. {item['summary']}" for item in raw_records]
embeddings = get_embeddings(texts)

points = []
for idx, item in enumerate(raw_records):
    reshaped_attrs = reshape_gxp_attributes(item["custom_attributes"])

    payload = {
        "doc_id": item["doc_id"],
        "system_category": item["system_category"],
        "title": item["title"],
        "summary": item["summary"],
        **reshaped_attrs,
    }

    points.append(models.PointStruct(id=idx + 1, vector=embeddings[idx], payload=payload))

client.upload_points(collection_name=COLLECTION_NAME, points=points)
print(f"Indexed {len(points)} points containing {len(points)} reshaped dynamic payload structures.")


# ---------------------------------------------------------------------------
# 5. Query Patterns on Dynamic Payloads
# ---------------------------------------------------------------------------

# Query Pattern 1: Fast Exact Match using 'attrs_flat'
print("\n" + "=" * 80)
print("QUERY PATTERN 1: Fast Exact Categorical Match via 'attrs_flat'")
print("Filter: 'detector_type=UV-Vis Photodiode Array' AND 'instrument_id=HPLC-QC-04'")
print("=" * 80)

hits_flat = client.query_points(
    collection_name=COLLECTION_NAME,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(key="attrs_flat", match=models.MatchValue(value="detector_type=UV-Vis Photodiode Array")),
            models.FieldCondition(key="attrs_flat", match=models.MatchValue(value="instrument_id=HPLC-QC-04")),
        ]
    ),
    limit=5,
    with_payload=True,
).points

for r, h in enumerate(hits_flat, 1):
    print(f"  #{r} [{h.payload['doc_id']}] {h.payload['title']}")
    print(f"      Flat Terms: {h.payload['attrs_flat'][:4]}...")


# Query Pattern 2: Numerical Range Query using Nested 'attrs_num'
# (Finds HPLC runs where flow_rate_ml_min >= 1.0 AND column_temp_c <= 40.0)
print("\n" + "=" * 80)
print("QUERY PATTERN 2: Multi-Attribute Range Query via Nested 'attrs_num'")
print("Filter: flow_rate_ml_min >= 1.0 AND column_temp_c <= 40.0")
print("=" * 80)

hits_numeric = client.query_points(
    collection_name=COLLECTION_NAME,
    query_filter=models.Filter(
        must=[
            # Nested condition 1: flow_rate_ml_min >= 1.0
            models.NestedCondition(
                nested=models.Nested(
                    key="attrs_num",
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(key="key", match=models.MatchValue(value="flow_rate_ml_min")),
                            models.FieldCondition(key="value", range=models.Range(gte=1.0)),
                        ]
                    ),
                )
            ),
            # Nested condition 2: column_temp_c <= 40.0
            models.NestedCondition(
                nested=models.Nested(
                    key="attrs_num",
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(key="key", match=models.MatchValue(value="column_temp_c")),
                            models.FieldCondition(key="value", range=models.Range(lte=40.0)),
                        ]
                    ),
                )
            ),
        ]
    ),
    limit=5,
    with_payload=True,
).points

for r, h in enumerate(hits_numeric, 1):
    p = h.payload
    print(f"  #{r} [{p['doc_id']}] {p['title']}")
    print(f"      Numeric Attrs: {p['attrs_num']}")


# Query Pattern 3: Hybrid Semantic Search + Disaster Recovery RTO/RPO Range Filter
print("\n" + "=" * 80)
print("QUERY PATTERN 3: Semantic Search + Dynamic DR Numeric Bounds (RTO <= 4h & RPO <= 15m)")
query_text = "database disaster recovery and automated backup snapshot verification"
print(f"Semantic Query: \"{query_text}\"")
print("=" * 80)

q_vec = get_embeddings([query_text])[0]

hits_semantic_filtered = client.query_points(
    collection_name=COLLECTION_NAME,
    query=q_vec,
    query_filter=models.Filter(
        must=[
            models.NestedCondition(
                nested=models.Nested(
                    key="attrs_num",
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(key="key", match=models.MatchValue(value="rto_hours")),
                            models.FieldCondition(key="value", range=models.Range(lte=4.0)),
                        ]
                    ),
                )
            ),
            models.NestedCondition(
                nested=models.Nested(
                    key="attrs_num",
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(key="key", match=models.MatchValue(value="rpo_minutes")),
                            models.FieldCondition(key="value", range=models.Range(lte=15.0)),
                        ]
                    ),
                )
            ),
            models.NestedCondition(
                nested=models.Nested(
                    key="attrs_bool",
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(key="key", match=models.MatchValue(value="is_gxp_compliant")),
                            models.FieldCondition(key="value", match=models.MatchValue(value=True)),
                        ]
                    ),
                )
            ),
        ]
    ),
    limit=3,
    with_payload=True,
).points

for r, h in enumerate(hits_semantic_filtered, 1):
    p = h.payload
    print(f"  #{r} [Score: {h.score:.4f}] [{p['doc_id']}] {p['title']}")
    print(f"      Category: {p['system_category']}")
    print(f"      Summary: {p['summary']}")
    print(f"      Numeric Specs: {p['attrs_num']}")

print("\n" + "=" * 80)
print("Tutorial 10 Execution Complete!")
print("=" * 80)
