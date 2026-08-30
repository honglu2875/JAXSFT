import json
from pathlib import Path

import pytest

from scripts.plan_glm53_lora import validate_kernel_evidence


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
