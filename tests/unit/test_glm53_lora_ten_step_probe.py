import hashlib
import json
from pathlib import Path

import pytest

from scripts.probe_glm53_lora_ten_step import (
    EXPECTED_GLOBAL_RESUME_PAYLOAD_SHA256,
    EXPECTED_STEP_THREE,
    FINAL_STEP,
    RESUME_SOURCE_REVISION,
    RESUME_STEP,
    _checkpoint_identity,
    _require_step_three_sentinel,
    _validate_rank_artifact_files,
    _validate_three_step_evidence,
)


ROOT = Path(__file__).resolve().parents[2]


def test_three_step_resume_evidence_is_exact_and_fail_closed(tmp_path):
    evidence_path = ROOT / "docs/results/glm53_lora_three_step_v4.json"
    evidence, digest = _validate_three_step_evidence(evidence_path)
    assert len(digest) == 64
    assert evidence["source_revision"] == RESUME_SOURCE_REVISION
    assert evidence["trajectory"]["steps"][1]["step"] == RESUME_STEP

    tampered_value = json.loads(evidence_path.read_text())
    tampered_value["gate"]["fifty_step_probe_authorized"] = True
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(tampered_value))
    with pytest.raises(ValueError, match="SHA-256 identity drifted"):
        _validate_three_step_evidence(tampered)


def test_rank_artifact_files_are_hashed_before_restore(tmp_path):
    manifest_payload = b"manifest"
    npz_payload = b"npz-payload"
    (tmp_path / "rank-002.json").write_bytes(manifest_payload)
    (tmp_path / "rank-002.npz").write_bytes(npz_payload)
    identity = {
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "npz_sha256": hashlib.sha256(npz_payload).hexdigest(),
        "local_payload_sha256": "0" * 64,
    }
    actual = _validate_rank_artifact_files(
        tmp_path,
        process_index=2,
        artifact_identity=identity,
    )
    assert actual["manifest_bytes"] == len(manifest_payload)
    assert actual["npz_bytes"] == len(npz_payload)

    (tmp_path / "rank-002.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="file identity drifted"):
        _validate_rank_artifact_files(
            tmp_path,
            process_index=2,
            artifact_identity=identity,
        )


def test_step_three_is_an_exact_cross_run_determinism_sentinel():
    record = {"step": 3, **EXPECTED_STEP_THREE}
    _require_step_three_sentinel(record)
    record["loss"] += 1e-6
    with pytest.raises(ValueError, match="deterministic sentinel"):
        _require_step_three_sentinel(record)


def test_checkpoint_identity_distinguishes_resume_and_step_ten():
    resume = _checkpoint_identity(
        source_revision=RESUME_SOURCE_REVISION,
        step=RESUME_STEP,
    )
    assert "resumed_from" not in resume

    output_revision = "1" * 40
    output = _checkpoint_identity(source_revision=output_revision, step=FINAL_STEP)
    assert output["source_revision"] == output_revision
    assert output["resumed_from"] == {
        "global_payload_sha256": EXPECTED_GLOBAL_RESUME_PAYLOAD_SHA256,
        "source_revision": RESUME_SOURCE_REVISION,
        "step": RESUME_STEP,
    }
