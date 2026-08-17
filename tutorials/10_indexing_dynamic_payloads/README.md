# Tutorial 10: Indexing Payloads of Random Shape (Dynamic Attributes) for GxP & CSV

| Time: 25–35 min | Level: Intermediate | Infrastructure: Local Qdrant (`http://localhost:6333`) |
| :--- | :--- | :--- |

## Overview

In Life Science Quality (QMS) and Computer System Validation (CSV / GAMP 5), computerized systems across analytical QC labs, manufacturing suites, clinical trials, and cloud infrastructure produce **thousands of open-ended, system-specific telemetry and qualification attributes**:
- **Analytical CDS (Empower / ChemStation):** `flow_rate_ml_min`, `column_temp_c`, `detector_type`, `rsd_retention_time_pct`
- **Bioreactor MES / SCADA (DeltaV):** `dissolved_oxygen_pct`, `agitation_rpm`, `vessel_pressure_psi`, `feed_rate_l_hr`
- **Cloud EDMS (Documentum / Veeva):** `ectd_module`, `hsm_fips_level`, `ind_number`, `soc2_type_ii_certified`
- **IT Disaster Recovery:** `rto_hours`, `rpo_minutes`, `database_engine`, `backup_storage_tier`

---

## ⚠️ The One-Index-Per-Key Trap

The naive approach is to create a new Qdrant payload index every time a new attribute key arrives:

```python
# Anti-pattern: Creating one payload index per distinct incoming attribute key
for key in incoming_telemetry_keys:
    client.create_payload_index(collection_name="gxp_data", field_name=key, field_schema=models.PayloadSchemaType.KEYWORD)
```

As the key space expands to hundreds or thousands of unique instrument attributes, RAM usage and index build times explode:

| Points | Distinct Attribute Keys (= Indexes) | Index Build Time | Extra RAM Added |
| :--- | :--- | :--- | :--- |
| 10,000 | 300 | ~19.7 s | +377 MiB |
| 10,000 | 1,000 | ~63.4 s | +1,190 MiB |
| 10,000 | 3,000 | ~220.0 s | +3,203 MiB |

---

## 💡 The Solution: Entity-Attribute-Value (EAV) Reshaping

Instead of storing arbitrary key names at the top-level payload, reshape dynamic attributes into **fixed, typed key-value arrays**:

```text
{ "flow_rate_ml_min": 1.25, "detector": "UV-Vis", "is_gxp": True }
                        ↓  Reshape before Ingestion
{
  "attrs":      [ {"key": "detector", "value": "UV-Vis"} ],
  "attrs_num":  [ {"key": "flow_rate_ml_min", "value": 1.25} ],
  "attrs_bool": [ {"key": "is_gxp", "value": True} ],
  "attrs_flat": [ "detector=UV-Vis", "flow_rate_ml_min=1.25", "is_gxp=True" ]
}
```

This reduces the total index count to a **fixed set of 8 payload indexes** that never grows, regardless of how many new instruments, telemetry channels, or systems are added.

---

## 1. Fixed Collection Schema Setup

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "gxp_dynamic_payloads"

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)

# 1. String EAV array (Keywords)
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="attrs[].key", field_schema=models.PayloadSchemaType.KEYWORD)
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="attrs[].value", field_schema=models.PayloadSchemaType.KEYWORD)

# 2. Numeric EAV array (Supports Range filters)
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="attrs_num[].key", field_schema=models.PayloadSchemaType.KEYWORD)
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="attrs_num[].value", field_schema=models.PayloadSchemaType.FLOAT)

# 3. Boolean EAV array (Supports Bool filters)
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="attrs_bool[].key", field_schema=models.PayloadSchemaType.KEYWORD)
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="attrs_bool[].value", field_schema=models.PayloadSchemaType.BOOL)

# 4. Concatenated 'key=value' array (30-40% faster exact match lookups)
client.create_payload_index(collection_name=COLLECTION_NAME, field_name="attrs_flat", field_schema=models.PayloadSchemaType.KEYWORD)
```

---

## 2. Ingest Reshaping Helper

```python
def reshape_gxp_attributes(raw_attrs: dict) -> dict:
    strings, numbers, bools, flats = [], [], [], []
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
    return {"attrs": strings, "attrs_num": numbers, "attrs_bool": bools, "attrs_flat": flats}
```

---

## 3. Querying Dynamic Attributes

### Pattern 1: High-Speed Exact Categorical Filter (`attrs_flat`)
```python
hits = client.query_points(
    collection_name=COLLECTION_NAME,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(key="attrs_flat", match=models.MatchValue(value="detector_type=UV-Vis Photodiode Array")),
            models.FieldCondition(key="attrs_flat", match=models.MatchValue(value="instrument_id=HPLC-QC-04")),
        ]
    ),
    limit=5,
)
```

### Pattern 2: Multi-Attribute Range Filter via Nested Objects
Using `models.NestedCondition` avoids the "same-element trap" by evaluating `key` and `value` on the identical object inside `attrs_num`:
```python
hits = client.query_points(
    collection_name=COLLECTION_NAME,
    query_filter=models.Filter(
        must=[
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
)
```

---

## 4. Running the Tutorial

```bash
python tutorials/10_indexing_dynamic_payloads/indexing_dynamic_payloads_gxp.py
```
