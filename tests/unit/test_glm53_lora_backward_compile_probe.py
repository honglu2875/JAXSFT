import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec

from jaxsft.models.glm5_3_flash import (
    attention_lora_parameter_count,
    init_params,
    tiny_config,
)
from scripts.probe_glm53_lora_backward_compile import (
    HBM_LIMIT_BYTES_PER_DEVICE,
    _abstract_attention_adapters,
    _adapter_placement,
    _execution_gate,
    _shape_mentions,
)


def test_abstract_attention_adapters_use_replicated_a_and_output_sharded_b():
    config = tiny_config(vocab_size=32)
    params = init_params(jax.random.key(0), config, dtype=jnp.float32)
    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("model",))
    adapters = _abstract_attention_adapters(params, config, mesh, rank=2)
    placement = _adapter_placement(adapters)
    assert placement["global_parameter_count"] == attention_lora_parameter_count(config, rank=2)
    assert placement["target_count"] == 9
    for pair in adapters.values():
        assert pair["a"].dtype == jnp.bfloat16
        assert pair["a"].sharding.spec == PartitionSpec()
        assert pair["b"].dtype == jnp.bfloat16
        assert pair["b"].sharding.spec == PartitionSpec(None, "model")


def test_compile_memory_gate_is_fail_closed_with_one_gib_safety_margin():
    safe = _execution_gate(
        {
            "argument_size_in_bytes": 20_000_000_000,
            "output_size_in_bytes": 20_000_000,
            "temp_size_in_bytes": 1_000_000_000,
        }
    )
    assert safe["full_checkpoint_execution_authorized"] is True

    unsafe = _execution_gate(
        {
            "argument_size_in_bytes": HBM_LIMIT_BYTES_PER_DEVICE - 1024**3,
            "output_size_in_bytes": 1,
            "temp_size_in_bytes": 0,
        }
    )
    assert unsafe["full_checkpoint_execution_authorized"] is False


def test_backward_probe_detects_flattened_and_token_topk_selected_weight_shapes():
    config = tiny_config(vocab_size=32)
    hlo = " ".join(
        (
            "bf16[4,1,32]",
            "f32[2,2,1,32]",
            "u8[4,2,16]",
            "bf16[2,2,2,16]",
        )
    )
    mentions = _shape_mentions(hlo, config)
    assert mentions["local_all_assignment_gate_dense:bf16"] == 1
    assert mentions["local_token_topk_gate_dense:f32"] == 1
    assert mentions["local_all_assignment_down_dense:u8"] == 1
    assert mentions["local_token_topk_down_dense:bf16"] == 1
