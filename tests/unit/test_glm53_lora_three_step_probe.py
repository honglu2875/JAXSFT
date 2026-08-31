import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jaxsft.optim import AdamWState
from scripts.probe_glm53_lora_three_step import (
    EXPECTED_CHECKPOINT_GLOBAL_ELEMENTS,
    EXPECTED_CHECKPOINT_GLOBAL_LOGICAL_BYTES,
    EXPECTED_CHECKPOINT_LEAF_COUNT,
    EXPECTED_CHECKPOINT_LOCAL_DEVICE_RESIDENT_BYTES,
    EXPECTED_CHECKPOINT_LOCAL_UNIQUE_BYTES,
    EXPECTED_CHECKPOINT_LOCAL_UNIQUE_SHARDS,
    TRAINING_STATISTIC_NAMES,
    _checkpoint_root,
    _require_checkpoint_contract,
    _training_state_statistics,
    _validate_optimizer_compile_evidence,
)


ROOT = Path(__file__).resolve().parents[2]


def test_authorizing_optimizer_compile_evidence_is_strict(tmp_path):
    evidence = ROOT / "docs/results/glm53_lora_optimizer_compile_v4.json"
    memory, digest = _validate_optimizer_compile_evidence(evidence)
    assert memory["argument_size_in_bytes"] == 20_305_797_120
    assert len(digest) == 64

    value = json.loads(evidence.read_text())
    value["gate"]["full_checkpoint_three_step_execution_authorized"] = False
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="does not authorize"):
        _validate_optimizer_compile_evidence(tampered)


def test_training_state_statistics_separate_adapter_factors_and_fp32_slots():
    adapters = {
        "layer": {
            "a": jnp.asarray([[1.0, 2.0]], jnp.bfloat16),
            "b": jnp.asarray([[0.0, 3.0]], jnp.bfloat16),
        }
    }
    first = {
        "layer": {
            "a": jnp.asarray([[0.0, 4.0]], jnp.float32),
            "b": jnp.asarray([[0.0, 0.0]], jnp.float32),
        }
    }
    second = {
        "layer": {
            "a": jnp.asarray([[0.0, 0.0]], jnp.float32),
            "b": jnp.asarray([[5.0, 0.0]], jnp.float32),
        }
    }
    statistics = np.asarray(
        _training_state_statistics(
            adapters,
            AdamWState(jnp.asarray(2, jnp.int32), first, second),
        )
    )
    assert len(statistics) == len(TRAINING_STATISTIC_NAMES) == 22
    assert statistics[0] == 1
    assert statistics[1] == 5
    assert statistics[2] == 9
    assert statistics[11] == 16
    assert statistics[16] == 25
    assert statistics[21] == 2


def test_checkpoint_contract_and_root_are_fail_closed():
    summary = {
        "root_keys": ["adapters", "optimizer"],
        "leaf_count": EXPECTED_CHECKPOINT_LEAF_COUNT,
        "global_elements_including_replicas_once": EXPECTED_CHECKPOINT_GLOBAL_ELEMENTS,
        "global_logical_bytes_including_replicas_once": (
            EXPECTED_CHECKPOINT_GLOBAL_LOGICAL_BYTES
        ),
        "local_unique_shard_count": EXPECTED_CHECKPOINT_LOCAL_UNIQUE_SHARDS,
        "local_unique_tensor_bytes": EXPECTED_CHECKPOINT_LOCAL_UNIQUE_BYTES,
        "local_device_resident_bytes": EXPECTED_CHECKPOINT_LOCAL_DEVICE_RESIDENT_BYTES,
    }
    _require_checkpoint_contract(summary)
    summary["leaf_count"] += 1
    with pytest.raises(ValueError, match="inventory"):
        _require_checkpoint_contract(summary)

    assert _checkpoint_root(Path("/tmp/jaxsft-test")) == Path("/tmp/jaxsft-test")
    with pytest.raises(ValueError, match="child of /tmp"):
        _checkpoint_root(Path("/var/tmp/jaxsft-test"))
