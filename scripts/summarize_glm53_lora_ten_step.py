#!/usr/bin/env python3
"""Strictly validate the GLM-5.3 LoRA total-10-step resume gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.probe_glm53_lora_ten_step import (
    EXPECTED_GLOBAL_RESUME_PAYLOAD_SHA256,
    EXPECTED_RESUME_STATISTICS_SHA256,
    EXPECTED_THREE_STEP_EVIDENCE_SHA256,
    FINAL_STEP,
    RESUME_SOURCE_REVISION,
    RESUME_STEP,
)
from scripts.probe_glm53_lora_three_step import TRAINING_STATISTIC_NAMES, _float32_sha256
from scripts.summarize_glm53_lora_backward import (
    EXPECTED_BASE_BYTES_PER_DEVICE,
    EXPECTED_LOADER_BYTES_PER_HOST,
)
from scripts.summarize_glm53_lora_backward_compile import (
    EXPECTED_CONFIG_SHA256,
    EXPECTED_INDEX_SHA256,
    EXPECTED_SHAPE_MENTIONS,
)
from scripts.summarize_glm53_lora_optimizer_compile import (
    EXPECTED_COMPILER_MEMORY,
    EXPECTED_EXECUTION_GATE,
)
from scripts.summarize_glm53_lora_three_step import (
    CheckpointArtifactContract,
    EXPECTED_CHECKPOINT_ARTIFACTS as EXPECTED_RESUME_ARTIFACTS,
    EXPECTED_STEP_RECORDS as EXPECTED_PRIOR_STEP_RECORDS,
    _validate_checkpoint_artifacts,
)


SOURCE_TEST = "glm53_complete_text_rank4_attention_lora_adamw_total_ten_step_v4_probe"
ACCEPTANCE_TEST = "glm53_g6d_total_ten_step_resume_stability_acceptance"
EXPECTED_SOURCE_REVISION = "d47e00b1736a6c2f0e506731cb7f8262f9438c63"
EXPECTED_HLO_SHA256 = "31e3fa377bb308209b2a5c082c65328929b4a739abd70787c0f8ea20b87ba50b"
EXPECTED_HLO_BYTES = 168_571_873
EXPECTED_GLOBAL_CHECKPOINT_SHA256 = "bec4348e8a18d803ef0455867915bd2bc49ea64e834ebebb295472c8a108e8c2"
EXPECTED_REPLICATED_CHECKPOINT_SHA256 = "8601802d0dcea09a9e26e450a6abefa5bedefc968d8ba12f231c0bd45301c169"
EXPECTED_RESULT_SHA256_BY_PROCESS_INDEX = {
    0: "4993807cd0295384915ebea7b969e85a6363e8ce248209c92084d0e6a1dd0322",
    1: "5642f30be293ac7ee1cbae87829882599c424667c885814ab5aadd3c076a714c",
    2: "7d8d57ae502f8e4499ffe3a0b8f340422ce7dbf03b1e0eb294ac4922a7516d2a",
    3: "661f41ba5cd1f29e50639b955bd35c03e2b8513abecce22e753534eff6876462",
}
EXPECTED_CHECKPOINT_ARTIFACTS = {
    0: {
        "manifest_file_bytes": 2_569_698,
        "manifest_sha256": "7b5a9ca49df80d6b29c47fe0fe1f03b81e68f0f89d91968210e82b53186a284b",
        "npz_sha256": "1052e2607e507c8588782d96b6445cb61d431c10dd6df4c836370ad6ca8d6755",
        "local_payload_sha256": "4d713a1c1c24ab95597eae6f5fb3662bd21e581afe0367486f8fa4a18da1c6a4",
        "sharded_payload_sha256": "773237c3476b0fa6d1dbe5e0da3490ea4599aa88c9a173f3aeed7e0aef6d72a8",
    },
    1: {
        "manifest_file_bytes": 2_573_268,
        "manifest_sha256": "75b4cd0d29d0e95caddeab57cde9c7ca4ce319d738987d57b66e1dab9e1a9608",
        "npz_sha256": "47dc5c14dc6526c5e7926eccc559728bdf1097fd18107a686eb7f3c5acc58b18",
        "local_payload_sha256": "1e750b314799b76839817d1811072189c7875d8545207d9cd2f3c92532f08a10",
        "sharded_payload_sha256": "8d159c3754fd2a2f10c6a63f0a80820d0653ed10008a3be49f0bc1a1bbbb836d",
    },
    2: {
        "manifest_file_bytes": 2_575_859,
        "manifest_sha256": "3595ff7fc7101584c6a8aff9add7b5916de4a288ebeca141b448e12a3b3efccc",
        "npz_sha256": "26e73f63da6bca5c8eb615b291449e90e3d063b1ebb6a0745046504a6eda2b41",
        "local_payload_sha256": "06ba0e2932221278005dd0f203feb26a5b95779ba90926109f56328a7b83ba2c",
        "sharded_payload_sha256": "91c411ec5dfd8ea465db1befe386ab031826b9f30ac9eda3c0d3c337777a733b",
    },
    3: {
        "manifest_file_bytes": 2_578_417,
        "manifest_sha256": "4eec182e035b7c72c111855f09359c95f85de4486f5b26f1168c7035ff332675",
        "npz_sha256": "8f78c370bb5f57fccbee7c7d9c8ad0a559ad7b8d8a92e47904722a6fb347171c",
        "local_payload_sha256": "b262e08b34e36f4c084eff176adc48d4d2778a32760b681ec31a0d4f6245b86b",
        "sharded_payload_sha256": "3aec9ae09e090412b40c2687eeebb14db65633f6fc2afa5ef991a5e67d05d265",
    },
}
EXPECTED_STEP_RECORDS = [
    {
        "step": 3,
        "loss": 12.266050338745117,
        "loss_float32_sha256": "275e08a1dd4a898594752ce0d3e559b36a814dd4ba209d09b7e36abcdc85e3cf",
        "gradient_norm_before_clipping": 2.7583279609680176,
        "gradient_norm_float32_sha256": "ea6a2241321d611c9500f729679b5eea799f0f3d6af3142630a38eafc4f30d36",
        "training_statistics_float32_sha256": "4a0efd4d20d4353c680e19cceee5d74f1be51710f6ef00aa2c9a0cea861e3d98",
    },
    {
        "step": 4,
        "loss": 12.228952407836914,
        "loss_float32_sha256": "fc0438a64615eb82ba57a9e83addfb31fd51a2a00386a7fb8161c01f0dc57a4b",
        "gradient_norm_before_clipping": 2.728299379348755,
        "gradient_norm_float32_sha256": "46708585f1b1f141b4c416cf66a4aaab256927e124fe99fe9be52c0d70e44de2",
        "training_statistics_float32_sha256": "473b0d1d71a02543288660328a72fb8ce8c17a7c6d680802959f79441c270481",
    },
    {
        "step": 5,
        "loss": 12.241546630859375,
        "loss_float32_sha256": "728e8cb84b20234e8e653997d6f35c03fcb36d094d29540883931689a9949a26",
        "gradient_norm_before_clipping": 2.785352945327759,
        "gradient_norm_float32_sha256": "69426cf811dee12eb6f6a270fe0b56d1d9c77db2fb740bcf71ba3df5501adc98",
        "training_statistics_float32_sha256": "99b37fccfb9c9d5496ab5f7eed020f0584b1b3fe49094ef67e144e6733e4f06c",
    },
    {
        "step": 6,
        "loss": 12.240615844726562,
        "loss_float32_sha256": "6cfd94bc2079c117f20a9acc03994fa14af01239f40d54787af8db311892c902",
        "gradient_norm_before_clipping": 2.826874256134033,
        "gradient_norm_float32_sha256": "f9bf28b8b1da1d97eea22bd6641bcc34ffa81205ffed8adb4e64a1e034d124cc",
        "training_statistics_float32_sha256": "f6904e1587bc060b75786761c339bf6c4769155c58ad784ec9fb8316a7eee6d3",
    },
    {
        "step": 7,
        "loss": 12.14979362487793,
        "loss_float32_sha256": "6637ac1b170e31da5dab681b3d28f65337d7601ed3dece83be05d993762ffafe",
        "gradient_norm_before_clipping": 2.8409149646759033,
        "gradient_norm_float32_sha256": "030f002c3bbf1cc2e220432b1c1a634b55b9348226bebc7cfbeb0de785eb064a",
        "training_statistics_float32_sha256": "e02fde90d38cec83ebed03ff10ad929067343989b5a81f2ade553835bf7e5101",
    },
    {
        "step": 8,
        "loss": 12.191780090332031,
        "loss_float32_sha256": "e5c2e2ac1ecb75c119700d36b84755550a0648c43b72773abdc35b924d5ebd78",
        "gradient_norm_before_clipping": 2.9524362087249756,
        "gradient_norm_float32_sha256": "fddd13d29afc6147ac90075d4497fe192883d02083f97df509847589ae861595",
        "training_statistics_float32_sha256": "1d40cce5003948fe713a6bcd8774a2239235ba4e19b4f66add9a149af2dd7272",
    },
    {
        "step": 9,
        "loss": 12.224834442138672,
        "loss_float32_sha256": "68205488ab3372e400b9bcf0dc67033673e66948b5b7f5e4d5c68f596dfefc33",
        "gradient_norm_before_clipping": 3.0017149448394775,
        "gradient_norm_float32_sha256": "f678a5e76a1bf63a14654ff3e84389b286e8f9d504f407435b74e2731aaf5f05",
        "training_statistics_float32_sha256": "195e9eccb53bb374e0220ae7d80d07a2b21ee91c19a6348a431f196790995ee5",
    },
    {
        "step": 10,
        "loss": 12.205331802368164,
        "loss_float32_sha256": "10ad1bf4cde39111f529782a3923c047a9b114e72d11a8f6435c4de7c117be81",
        "gradient_norm_before_clipping": 2.997069835662842,
        "gradient_norm_float32_sha256": "4a0094ab2cfdd38262efb770c91766ac186adbc480677c3f9a6942dd1507a85a",
        "training_statistics_float32_sha256": "b2fa94a3ff02c94ad62f21f8d6e9312faa54d65b101795856683cb6f77f98469",
    },
]
EXPECTED_PHASE_PEAKS = {
    "after_distributed_init": 13_824,
    "after_full_base_placement": 20_378_199_040,
    "after_optimizer_compile": 13_824,
    "after_step_two_restore": 115_997_696,
    **{f"after_step_{step}": 20_599_073_792 for step in range(3, 11)},
    "after_step_ten_checkpoint_save": 20_599_073_792,
    "after_step_ten_state_release": 20_599_073_792,
    "after_step_ten_checkpoint_restore": 20_599_073_792,
}
EXPECTED_TIMING_MAXIMA = {
    "header_seconds": 40.20018918206915,
    "compile_seconds": 619.3298299349844,
    "resume_restore_seconds": 59.83608581707813,
    "load_seconds": 1162.6232583709061,
    "checkpoint_save_seconds": 6.740858553908765,
    "checkpoint_restore_seconds": 2.782508214004338,
    "elapsed_seconds_before_shutdown": 3902.5103418470826,
    "step_execute_seconds": 248.75608139601536,
    "step_diagnostics_seconds": 0.02189321699552238,
}
EXPECTED_MINIMUM_FREE_BLOCK = 11_617_694_720
EXPECTED_MAXIMUM_PROCESS_VMHWM = 19_174_993_920


def _load(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(payload, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _checkpoint_identity() -> dict[str, Any]:
    return {
        "format_purpose": "GLM-5.3-Flash rank-4 attention-LoRA adapter-only AdamW",
        "model_repo_id": "zai-org/GLM-5.3-Flash",
        "model_revision": "04c4e9e95c5da8862dced7e5056455116f83a7e0",
        "resumed_from": {
            "global_payload_sha256": EXPECTED_GLOBAL_RESUME_PAYLOAD_SHA256,
            "source_revision": RESUME_SOURCE_REVISION,
            "step": RESUME_STEP,
        },
        "source_revision": EXPECTED_SOURCE_REVISION,
        "step": FINAL_STEP,
    }


def _checkpoint_contract() -> CheckpointArtifactContract:
    return CheckpointArtifactContract(
        checkpoint_step=FINAL_STEP,
        identity=_checkpoint_identity(),
        artifacts_by_process_index=EXPECTED_CHECKPOINT_ARTIFACTS,
        replicated_payload_sha256=EXPECTED_REPLICATED_CHECKPOINT_SHA256,
        global_payload_sha256=EXPECTED_GLOBAL_CHECKPOINT_SHA256,
    )


def _validate_step_record(record: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    exact_names = (
        "step",
        "loss",
        "loss_float32_sha256",
        "gradient_norm_before_clipping",
        "gradient_norm_float32_sha256",
        "training_statistics_float32_sha256",
    )
    if {name: record.get(name) for name in exact_names} != {name: expected[name] for name in exact_names}:
        raise ValueError("G6d trajectory scalar or SHA-256 drifted")
    statistics = np.asarray(record.get("training_statistics"), dtype=np.float32)
    step = expected["step"]
    if (
        statistics.shape != (len(TRAINING_STATISTIC_NAMES),)
        or not np.isfinite(statistics).all()
        or _float32_sha256(statistics) != expected["training_statistics_float32_sha256"]
        or statistics[0] != 1
        or statistics[-1] != step
        or statistics[2] <= 0
        or statistics[8] <= 0
        or statistics[11] <= 0
        or statistics[16] <= 0
        or not isinstance(record.get("execute_seconds"), (int, float))
        or not isinstance(record.get("diagnostics_seconds"), (int, float))
        or not math.isfinite(record["execute_seconds"])
        or not math.isfinite(record["diagnostics_seconds"])
        or record["execute_seconds"] <= 0
        or record["diagnostics_seconds"] <= 0
    ):
        raise ValueError("G6d trajectory statistics or timing drifted")
    return {
        **{name: record[name] for name in exact_names},
        "training_statistics": statistics.tolist(),
    }


def _expected_model() -> dict[str, Any]:
    return {
        "alpha": 4.0,
        "attention_lora_target_count": 191,
        "batch_size": 1,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "index_sha256": EXPECTED_INDEX_SHA256,
        "input_ids": [[1, 2]],
        "loss_token_count": 1,
        "loss_weights": [[0.0, 1.0]],
        "num_hidden_layers": 45,
        "rank": 4,
        "rematerialize_each_decoder_layer": True,
        "repo_id": "zai-org/GLM-5.3-Flash",
        "revision": "04c4e9e95c5da8862dced7e5056455116f83a7e0",
        "sequence_length": 2,
    }


def _expected_optimizer() -> dict[str, Any]:
    return {
        "adapter_dtype": "bfloat16",
        "beta1": 0.9,
        "beta2": 0.95,
        "donated_argument_numbers": [1, 2],
        "epsilon": 1e-8,
        "learning_rate": 1e-5,
        "max_grad_norm": 1.0,
        "moment_dtype": "float32",
        "name": "AdamW",
        "weight_decay": 0.1,
    }


def _expected_header_network() -> dict[str, Any]:
    return {
        "bytes_by_category": {"header": 10_684_096},
        "bytes_read_including_resolves": 10_684_158,
        "largest_request_bytes": 180_384,
        "loaded_logical_tensor_count": 0,
        "loaded_scale_tensor_count": 0,
        "loaded_target_count": 0,
        "maximum_expert_host_buffer_bytes": 0,
        "prepared_shard_count": 62,
        "request_count_including_resolves": 186,
        "requests_by_category": {"header": 124},
    }


def _expected_loader_network() -> dict[str, Any]:
    return {
        "bytes_by_category": {
            "axis0_tensor": 3_757_703_168,
            "expert_tensor": 76_101_451_776,
            "header": 10_684_096,
            "replicated_tensor": 193_428_312,
            "scale_envelope": 77_871_648,
        },
        "bytes_read_including_resolves": EXPECTED_LOADER_BYTES_PER_HOST,
        "largest_request_bytes": 79_298_560,
        "loaded_logical_tensor_count": 37_534,
        "loaded_scale_tensor_count": 36_467,
        "loaded_target_count": 1_372,
        "maximum_expert_host_buffer_bytes": 603_979_776,
        "prepared_shard_count": 62,
        "request_count_including_resolves": 38_847,
        "requests_by_category": {
            "axis0_tensor": 1_796,
            "expert_tensor": 36_288,
            "header": 124,
            "replicated_tensor": 516,
            "scale_envelope": 61,
        },
    }


def _expected_adapter_placement() -> dict[str, Any]:
    return {
        "a_partition_spec": [],
        "b_partition_spec": [None, "model"],
        "global_parameter_count": 10_289_152,
        "global_parameter_count_by_factor": {"a": 3_956_736, "b": 6_332_416},
        "parameter_bytes_by_device": {str(device): 8_705_024 for device in range(16)},
        "target_count": 191,
    }


def _expected_optimizer_placement() -> dict[str, Any]:
    return {
        "first_moment_bytes_by_device": {str(device): 17_410_048 for device in range(16)},
        "moment_dtype": "float32",
        "moment_global_elements_per_slot": 10_289_152,
        "moment_slot_count": 2,
        "optimizer_state_bytes_by_device": {str(device): 34_820_100 for device in range(16)},
        "second_moment_bytes_by_device": {str(device): 17_410_048 for device in range(16)},
        "step_bytes_by_device": {str(device): 4 for device in range(16)},
        "step_dtype": "int32",
        "step_global_elements": 1,
    }


def _expected_base_placement(process_index: int) -> dict[str, Any]:
    return {
        "all_local_devices_match_header_audit": True,
        "array_leaf_count": 1_677,
        "expected_base_bytes_per_device": EXPECTED_BASE_BYTES_PER_DEVICE,
        "global_leaf_elements_by_dtype": {
            "bfloat16": 6_303_463_936,
            "float32": 19_034_430,
            "uint8": 307_023_052_800,
        },
        "global_leaf_elements_including_scale_metadata": 313_345_551_166,
        "local_bytes_by_device": {
            str(device): EXPECTED_BASE_BYTES_PER_DEVICE for device in range(process_index * 4, process_index * 4 + 4)
        },
    }


def summarize(
    rank_paths: Sequence[Path],
    manifest_paths: Sequence[Path],
    payload_paths: Sequence[Path],
    *,
    source_revision: str,
) -> dict[str, Any]:
    if source_revision != EXPECTED_SOURCE_REVISION:
        raise ValueError("G6d source revision drifted")
    if len(rank_paths) != 4:
        raise ValueError("exactly four G6d rank results are required")
    checkpoint_evidence, manifests = _validate_checkpoint_artifacts(
        manifest_paths,
        payload_paths,
        source_revision=source_revision,
        contract=_checkpoint_contract(),
    )
    values_by_rank: dict[int, dict[str, Any]] = {}
    result_hash_by_rank: dict[str, str] = {}
    for path in rank_paths:
        value, digest = _load(path)
        process_index = value.get("runtime", {}).get("process_index")
        if (
            not isinstance(process_index, int)
            or process_index not in range(4)
            or process_index in values_by_rank
            or digest != EXPECTED_RESULT_SHA256_BY_PROCESS_INDEX.get(process_index)
        ):
            raise ValueError("G6d raw rank-result identity or process coverage drifted")
        values_by_rank[process_index] = value
        result_hash_by_rank[str(process_index)] = digest
    if set(values_by_rank) != set(range(4)):
        raise ValueError("G6d raw results do not cover four process indexes")

    expected_authorization = {
        "fifty_step_probe_authorized_before_this_run": False,
        "resume_global_payload_sha256": EXPECTED_GLOBAL_RESUME_PAYLOAD_SHA256,
        "resume_source_revision": RESUME_SOURCE_REVISION,
        "three_step_evidence_sha256": EXPECTED_THREE_STEP_EVIDENCE_SHA256,
    }
    expected_runtime_common = {
        "backend": "tpu",
        "device_kinds": ["TPU v4"],
        "distributed_initialized": True,
        "distributed_shutdown_complete": True,
        "global_device_count": 16,
        "jax_version": "0.11.0",
        "local_device_count": 4,
        "mesh_shape": {"model": 16},
        "precision": "HIGHEST",
        "process_count": 4,
    }
    expected_hostnames = {
        0: "t1v-n-a09f5679-w-2",
        1: "t1v-n-a09f5679-w-0",
        2: "t1v-n-a09f5679-w-3",
        3: "t1v-n-a09f5679-w-1",
    }
    expected_resume_statistics = EXPECTED_PRIOR_STEP_RECORDS[1]["training_statistics"]
    expected_phase_names = set(EXPECTED_PHASE_PEAKS)
    phase_peaks = {name: 0 for name in expected_phase_names}
    minimum_free_block = 2**63 - 1
    maximum_vmhwm = 0
    maxima = {name: 0.0 for name in EXPECTED_TIMING_MAXIMA}
    reference_steps: list[dict[str, Any]] | None = None

    for process_index in range(4):
        value = values_by_rank[process_index]
        if (
            set(value)
            != {
                "authorization",
                "checkpoint",
                "compile_preflight",
                "device_memory",
                "host_memory",
                "loader",
                "model",
                "optimizer",
                "resume",
                "runtime",
                "schema_version",
                "shm",
                "source_revision",
                "steps",
                "test",
                "timing",
                "training_state",
            }
            or value.get("schema_version") != 1
            or value.get("test") != SOURCE_TEST
            or value.get("source_revision") != source_revision
            or value.get("authorization") != expected_authorization
            or value.get("model") != _expected_model()
            or value.get("optimizer") != _expected_optimizer()
        ):
            raise ValueError("G6d result schema, authorization, model, or optimizer drifted")
        runtime = dict(value.get("runtime", {}))
        hostname = runtime.pop("hostname", None)
        rank = runtime.pop("process_index", None)
        if runtime != expected_runtime_common or hostname != expected_hostnames[process_index] or rank != process_index:
            raise ValueError("G6d runtime topology or clean shutdown drifted")

        compile_preflight = value.get("compile_preflight", {})
        if (
            compile_preflight.get("authorizing_g6c1_hlo_sha256")
            != "d92954cee93966bd540b5a443b8dcd8a57fb2d12585e2c71248f43ccc6d73fe5"
            or compile_preflight.get("checkpoint_payload_bytes_read") != 0
            or compile_preflight.get("compiler_memory") != EXPECTED_COMPILER_MEMORY
            or compile_preflight.get("execution_gate") != EXPECTED_EXECUTION_GATE
            or compile_preflight.get("header_network") != _expected_header_network()
            or compile_preflight.get("optimized_hlo_sha256") != EXPECTED_HLO_SHA256
            or compile_preflight.get("optimized_hlo_bytes") != EXPECTED_HLO_BYTES
            or compile_preflight.get("optimized_hlo_shape_mentions") != EXPECTED_SHAPE_MENTIONS
            or compile_preflight.get("compile_seconds") != value.get("timing", {}).get("compile_seconds")
        ):
            raise ValueError("G6d compiler, HLO, or header-only preflight drifted")

        training_state = value.get("training_state", {})
        if (
            training_state.get("adapter_placement") != _expected_adapter_placement()
            or training_state.get("optimizer_placement") != _expected_optimizer_placement()
            or training_state.get("statistic_names") != list(TRAINING_STATISTIC_NAMES)
        ):
            raise ValueError("G6d adapter or FP32 optimizer placement drifted")
        loader = dict(value.get("loader", {}))
        parameter_placement = loader.pop("parameter_placement", None)
        load_seconds = loader.pop("load_seconds", None)
        if (
            loader != _expected_loader_network()
            or parameter_placement != _expected_base_placement(process_index)
            or load_seconds != value.get("timing", {}).get("load_seconds")
        ):
            raise ValueError("G6d streaming loader or base placement drifted")

        resume = value.get("resume", {})
        resume_artifact = EXPECTED_RESUME_ARTIFACTS[process_index]
        expected_resume_restore = {
            "all_local_shards_byte_exact": True,
            "leaf_count": 1_147,
            "local_payload_sha256": resume_artifact["local_payload_sha256"],
            "local_unique_shard_count": 2_866,
            "manifest_sha256": resume_artifact["manifest_sha256"],
            "npz_sha256": resume_artifact["npz_sha256"],
        }
        if (
            resume.get("artifact_files")
            != {
                "manifest_bytes": resume_artifact["manifest_file_bytes"],
                "manifest_sha256": resume_artifact["manifest_sha256"],
                "npz_bytes": 56_172_246,
                "npz_sha256": resume_artifact["npz_sha256"],
            }
            or resume.get("checkpoint_step") != RESUME_STEP
            or resume.get("directory") != "/tmp/jaxsft-glm53-g6c1-543350c/step-00000002"
            or resume.get("restore") != expected_resume_restore
            or resume.get("statistics") != expected_resume_statistics
            or resume.get("statistics_float32_sha256") != EXPECTED_RESUME_STATISTICS_SHA256
            or resume.get("restored_before_full_base_payload_load") is not True
        ):
            raise ValueError("G6d step-two checkpoint provenance or restore drifted")

        raw_steps = value.get("steps")
        if not isinstance(raw_steps, list) or len(raw_steps) != len(EXPECTED_STEP_RECORDS):
            raise ValueError("G6d did not emit exactly steps three through ten")
        steps = [
            _validate_step_record(record, expected)
            for record, expected in zip(raw_steps, EXPECTED_STEP_RECORDS, strict=True)
        ]
        if reference_steps is None:
            reference_steps = steps
        elif steps != reference_steps:
            raise ValueError("G6d per-rank loss, gradient, or state trajectory differs")
        maxima["step_execute_seconds"] = max(
            maxima["step_execute_seconds"],
            max(float(record["execute_seconds"]) for record in raw_steps),
        )
        maxima["step_diagnostics_seconds"] = max(
            maxima["step_diagnostics_seconds"],
            max(float(record["diagnostics_seconds"]) for record in raw_steps),
        )

        manifest = manifests[process_index]
        artifact = EXPECTED_CHECKPOINT_ARTIFACTS[process_index]
        checkpoint = value.get("checkpoint", {})
        expected_checkpoint_fields = {
            "base_leaf_count": 0,
            "checkpoint_step": FINAL_STEP,
            "directory": "/tmp/jaxsft-glm53-g6d-d47e00b/step-00000010",
            "format": "jaxsft_rank_local_sharded_pytree_npz",
            "frozen_base_included": False,
            "global_elements_including_replicas_once": 30_867_457,
            "global_logical_bytes_including_replicas_once": 102_891_524,
            "identity": _checkpoint_identity(),
            "leaf_count": 1_147,
            "local_device_resident_bytes": 174_100_496,
            "local_payload_sha256": artifact["local_payload_sha256"],
            "local_unique_shard_count": 2_866,
            "local_unique_tensor_bytes": 55_398_404,
            "manifest_file": f"rank-{process_index:03d}.json",
            "manifest_file_bytes": artifact["manifest_file_bytes"],
            "manifest_sha256": artifact["manifest_sha256"],
            "npz_file_bytes": 56_172_246,
            "npz_sha256": artifact["npz_sha256"],
            "pre_save_and_restored_statistics_equal": True,
            "pre_save_statistics_float32_sha256": EXPECTED_STEP_RECORDS[-1]["training_statistics_float32_sha256"],
            "process_count": 4,
            "process_index": process_index,
            "replicated_payload_sha256": EXPECTED_REPLICATED_CHECKPOINT_SHA256,
            "restored_statistics_float32_sha256": EXPECTED_STEP_RECORDS[-1]["training_statistics_float32_sha256"],
            "root_keys": ["adapters", "optimizer"],
            "schema_version": 1,
            "sharded_payload_sha256": artifact["sharded_payload_sha256"],
        }
        checkpoint_without_restore = dict(checkpoint)
        restore = checkpoint_without_restore.pop("restore", None)
        expected_final_restore = {
            "all_local_shards_byte_exact": True,
            "leaf_count": 1_147,
            "local_payload_sha256": artifact["local_payload_sha256"],
            "local_unique_shard_count": 2_866,
            "manifest_sha256": artifact["manifest_sha256"],
            "npz_sha256": artifact["npz_sha256"],
        }
        if (
            checkpoint_without_restore != expected_checkpoint_fields
            or restore != expected_final_restore
            or manifest.get("local_payload_sha256") != artifact["local_payload_sha256"]
        ):
            raise ValueError("G6d raw step-ten checkpoint/restore evidence drifted")

        host_memory = value.get("host_memory")
        device_memory = value.get("device_memory")
        if (
            not isinstance(host_memory, dict)
            or not isinstance(device_memory, dict)
            or set(host_memory) != expected_phase_names
            or set(device_memory) != expected_phase_names
        ):
            raise ValueError("G6d memory phase coverage drifted")
        for phase in expected_phase_names:
            host_record = host_memory[phase]
            records = device_memory[phase]
            if (
                not isinstance(host_record, dict)
                or not isinstance(host_record.get("vmhwm_bytes"), int)
                or not isinstance(host_record.get("vmrss_bytes"), int)
                or not isinstance(records, list)
                or len(records) != 4
                or {record.get("device_id") for record in records}
                != set(range(process_index * 4, process_index * 4 + 4))
            ):
                raise ValueError("G6d host/device memory record shape drifted")
            maximum_vmhwm = max(maximum_vmhwm, host_record["vmhwm_bytes"])
            for record in records:
                stats = record.get("stats", {})
                if (
                    record.get("process_index") != process_index
                    or stats.get("bytes_limit") != 33_014_407_168
                    or not isinstance(stats.get("peak_bytes_in_use"), int)
                    or not isinstance(stats.get("largest_free_block_bytes"), int)
                ):
                    raise ValueError("G6d TPU memory record contract drifted")
                phase_peaks[phase] = max(phase_peaks[phase], stats["peak_bytes_in_use"])
                minimum_free_block = min(minimum_free_block, stats["largest_free_block_bytes"])

        timing = value.get("timing", {})
        expected_timing_names = {name for name in EXPECTED_TIMING_MAXIMA if not name.startswith("step_")}
        if set(timing) != expected_timing_names:
            raise ValueError("G6d timing phase coverage drifted")
        for name in expected_timing_names:
            measured = timing[name]
            if not isinstance(measured, (int, float)) or not math.isfinite(measured):
                raise ValueError("G6d timing value is not finite")
            maxima[name] = max(maxima[name], float(measured))
        shm = value.get("shm", {})
        if (
            shm.get("used_delta_during_load_bytes") != 0
            or shm.get("used_delta_total_bytes") != 0
            or shm.get("before") != shm.get("after_load")
            or shm.get("before") != shm.get("after")
        ):
            raise ValueError("G6d unexpectedly staged checkpoint payloads in RAMFS")

    if reference_steps is None:
        raise AssertionError("G6d trajectory reference was not initialized")
    if phase_peaks != EXPECTED_PHASE_PEAKS:
        raise ValueError("G6d exact TPU high-water marks drifted")
    if minimum_free_block != EXPECTED_MINIMUM_FREE_BLOCK:
        raise ValueError("G6d minimum largest-free-block measurement drifted")
    if maximum_vmhwm != EXPECTED_MAXIMUM_PROCESS_VMHWM:
        raise ValueError("G6d maximum process high-water mark drifted")
    if maxima != EXPECTED_TIMING_MAXIMA:
        raise ValueError("G6d exact timing maxima drifted")

    total_steps = [
        {
            key: record[key]
            for key in (
                "step",
                "loss",
                "loss_float32_sha256",
                "gradient_norm_before_clipping",
                "gradient_norm_float32_sha256",
                "training_statistics_float32_sha256",
            )
        }
        for record in EXPECTED_PRIOR_STEP_RECORDS[:2]
    ] + reference_steps
    losses = [float(record["loss"]) for record in total_steps]
    trajectory_loss_range = max(losses) - min(losses)
    maximum_peak = max(phase_peaks.values())
    step_peaks = [phase_peaks[f"after_step_{step}"] for step in range(3, 11)]
    return {
        "schema_version": 1,
        "test": ACCEPTANCE_TEST,
        "source_revision": source_revision,
        "topology": {
            "accelerator_type": "v4-32",
            "host_count": 4,
            "process_count": 4,
            "global_device_count": 16,
            "physical_hostnames": sorted(expected_hostnames.values()),
            "physical_hostname_order_independent_of_process_index": True,
        },
        "resume": {
            "source_step": RESUME_STEP,
            "source_revision": RESUME_SOURCE_REVISION,
            "source_global_payload_sha256": EXPECTED_GLOBAL_RESUME_PAYLOAD_SHA256,
            "source_evidence_sha256": EXPECTED_THREE_STEP_EVIDENCE_SHA256,
            "all_rank_artifact_hashes_verified_before_restore": True,
            "all_local_shards_byte_exact": True,
            "restored_before_full_base_payload_load": True,
            "step_three_cross_run_exact": True,
        },
        "trajectory": {
            "steps": total_steps,
            "all_rank_trajectories_equal": True,
            "all_losses_gradients_and_states_finite": True,
            "all_gradient_norms_exceeded_clip_threshold": True,
            "loss_monotonic_decrease": False,
            "loss_step_one_to_ten_change": losses[-1] - losses[0],
            "loss_step_one_to_ten_relative_change": (losses[-1] - losses[0]) / losses[0],
            "loss_step_three_to_ten_change": losses[-1] - losses[2],
            "loss_step_three_to_ten_relative_change": (losses[-1] - losses[2]) / losses[2],
            "loss_range": trajectory_loss_range,
            "loss_range_relative_to_step_one": trajectory_loss_range / losses[0],
            "maximum_step_execute_seconds": maxima["step_execute_seconds"],
            "maximum_step_diagnostics_seconds": maxima["step_diagnostics_seconds"],
        },
        "checkpoint": checkpoint_evidence,
        "memory": {
            "maximum_phase_peak_bytes_in_use": phase_peaks,
            "maximum_device_peak_bytes_in_use": maximum_peak,
            "headroom_after_peak_bytes_per_device": 33_014_407_168 - maximum_peak,
            "minimum_largest_free_block_bytes": minimum_free_block,
            "step_three_to_step_ten_peak_slope_bytes": step_peaks[-1] - step_peaks[0],
            "all_step_high_water_marks_equal": len(set(step_peaks)) == 1,
            "maximum_process_vmhwm_bytes": maximum_vmhwm,
        },
        "streaming": {
            "bytes_read_per_host": EXPECTED_LOADER_BYTES_PER_HOST,
            "total_bytes_read_across_hosts": EXPECTED_LOADER_BYTES_PER_HOST * 4,
            "loaded_logical_tensor_count": 37_534,
            "loaded_target_count": 1_372,
            "maximum_load_seconds": maxima["load_seconds"],
            "maximum_shm_used_delta_bytes": 0,
        },
        "execution": {
            **EXPECTED_COMPILER_MEMORY,
            **EXPECTED_EXECUTION_GATE,
            "optimized_hlo_sha256": EXPECTED_HLO_SHA256,
            "optimized_hlo_bytes": EXPECTED_HLO_BYTES,
            "all_rank_hlo_equal": True,
            "no_assignment_wide_dense_weight_in_optimized_hlo": True,
            "all_distributed_shutdowns_complete": True,
        },
        "timing": {f"maximum_{name}": value for name, value in maxima.items()},
        "rank_result_sha256_by_process_index": dict(sorted(result_hash_by_rank.items(), key=lambda item: int(item[0]))),
        "gate": {
            "g6d_total_ten_step_resume_stability": "passed",
            "resume_determinism_proven": True,
            "total_ten_step_finite_trajectory_proven": True,
            "step_ten_adapter_only_checkpoint_restore_proven": True,
            "fifty_step_fixed_token_resume_probe_authorized": True,
            "instruction_sequence_execution_authorized": False,
            "long_sequence_execution_authorized": False,
            "instruction_sft_quality_proven": False,
            "remaining_blockers": [
                "The ten-step fixed-token trajectory is finite but non-monotonic.",
                "A tokenizer/chat-template/loss-mask oracle on a real instruction sample is not yet connected.",
                "Realistic sequence lengths require a separate header-only compile and HBM gate.",
                "No instruction-dataset throughput, tuning-quality, or convergence claim is established.",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-results", type=Path, nargs=4, required=True)
    parser.add_argument("--checkpoint-manifests", type=Path, nargs=4, required=True)
    parser.add_argument("--checkpoint-payloads", type=Path, nargs=4, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.rank_results,
        args.checkpoint_manifests,
        args.checkpoint_payloads,
        source_revision=args.source_revision,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
