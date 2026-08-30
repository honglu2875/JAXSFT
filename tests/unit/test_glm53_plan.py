import json
from pathlib import Path

import pytest

from scripts.plan_glm53_lora import (
    validate_execution_schema_evidence,
    validate_kernel_evidence,
    validate_loader_evidence,
)


def test_glm53_kernel_evidence_is_validated_instead_of_boolean_asserted(tmp_path):
    root = Path(__file__).resolve().parents[2]
    evidence_path = root / "docs" / "results" / "glm53_fp8_v4_probe.json"
    validated = validate_kernel_evidence(evidence_path)
    assert validated["source_revision"] == "2bfda04c006a1612b82d0d023397c581c092d727"
    assert len(validated["sha256"]) == 64

    payload = json.loads(evidence_path.read_text())
    payload["tpu"]["tiled_full_weight_hlo_shape_mentions"]["bfloat16"] = 1
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="full bfloat16 weight"):
        validate_kernel_evidence(tampered)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        validate_kernel_evidence(duplicate)


def test_glm53_loader_evidence_rehashes_header_audit_and_enforces_bounds(tmp_path):
    root = Path(__file__).resolve().parents[2]
    results = root / "docs" / "results"
    evidence_path = results / "glm53_direct_sharded_loader_v4.json"
    validated = validate_loader_evidence(evidence_path)
    assert validated["source_revision"] == "fb08fcc48ea32b91b5ac33ac2628f5bb828ce7d5"
    assert validated["placed_base_per_device_bytes"] == 20_234_287_352
    assert validated["staging_per_host_bytes"] == 79_298_560

    header_name = "glm53_checkpoint_header_audit.json"
    (tmp_path / header_name).write_bytes((results / header_name).read_bytes())
    payload = json.loads(evidence_path.read_text())
    payload["sample_loader"]["largest_http_range_bytes"] += 1
    tampered = tmp_path / evidence_path.name
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="HTTP range bound"):
        validate_loader_evidence(tampered)

    header = json.loads((tmp_path / header_name).read_text())
    header["placement_plan"]["unsupported_tensor_count"] = 1
    (tmp_path / header_name).write_text(json.dumps(header))
    with pytest.raises(ValueError, match="header audit SHA-256 mismatch"):
        validate_loader_evidence(tampered)


def test_glm53_execution_schema_evidence_is_complete_and_bounded(tmp_path):
    root = Path(__file__).resolve().parents[2]
    evidence_path = root / "docs" / "results" / "glm53_execution_schema_audit.json"
    validated = validate_execution_schema_evidence(evidence_path)
    assert validated["source_revision"] == "5653518a9d7c1c165bf01049b06b240be04650d0"
    assert validated["staging_per_host_bytes"] == 150_994_944

    payload = json.loads(evidence_path.read_text())
    payload["coverage"]["logical_tensor_count"] -= 1
    tampered = tmp_path / "schema.json"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="logical tensor count"):
        validate_execution_schema_evidence(tampered)
