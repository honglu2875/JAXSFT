import json
from pathlib import Path

import numpy as np
import pytest

from scripts.probe_glm53_lora_three_step import TRAINING_STATISTIC_NAMES, _float32_sha256
from scripts.summarize_glm53_lora_ten_step import (
    EXPECTED_GLOBAL_CHECKPOINT_SHA256,
    EXPECTED_SOURCE_REVISION,
    _checkpoint_contract,
    _validate_step_record,
)


ROOT = Path(__file__).resolve().parents[2]


def _step_fixture() -> tuple[dict, dict]:
    statistics = np.zeros(len(TRAINING_STATISTIC_NAMES), dtype=np.float32)
    statistics[0] = 1
    statistics[2] = 1
    statistics[8] = 1
    statistics[11] = 1
    statistics[16] = 1
    statistics[-1] = 10
    expected = {
        "step": 10,
        "loss": 1.25,
        "loss_float32_sha256": _float32_sha256(np.asarray([1.25])),
        "gradient_norm_before_clipping": 2.0,
        "gradient_norm_float32_sha256": _float32_sha256(np.asarray([2.0])),
        "training_statistics_float32_sha256": _float32_sha256(statistics),
    }
    return {
        **expected,
        "training_statistics": statistics.tolist(),
        "execute_seconds": 1.0,
        "diagnostics_seconds": 0.1,
    }, expected


def test_step_record_rehashes_statistics_and_rejects_drift():
    record, expected = _step_fixture()
    validated = _validate_step_record(record, expected)
    assert validated["step"] == 10

    record["training_statistics"][2] = 2
    with pytest.raises(ValueError, match="statistics or timing drifted"):
        _validate_step_record(record, expected)


def test_step_ten_checkpoint_contract_pins_resume_lineage():
    contract = _checkpoint_contract()
    assert contract.checkpoint_step == 10
    assert contract.global_payload_sha256 == EXPECTED_GLOBAL_CHECKPOINT_SHA256
    assert contract.identity["source_revision"] == EXPECTED_SOURCE_REVISION
    assert contract.identity["resumed_from"]["step"] == 2
    assert set(contract.artifacts_by_process_index) == {0, 1, 2, 3}


def test_recorded_acceptance_keeps_real_data_claims_closed():
    evidence = json.loads((ROOT / "docs/results/glm53_lora_ten_step_v4.json").read_text())
    assert evidence["gate"]["g6d_total_ten_step_resume_stability"] == "passed"
    assert evidence["gate"]["fifty_step_fixed_token_resume_probe_authorized"] is True
    assert evidence["gate"]["instruction_sequence_execution_authorized"] is False
    assert evidence["memory"]["step_three_to_step_ten_peak_slope_bytes"] == 0
    assert evidence["resume"]["step_three_cross_run_exact"] is True
