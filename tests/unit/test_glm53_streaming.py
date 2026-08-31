from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec

from jaxsft.models.glm5_3_flash import (
    BatchedBlockFP8LinearKernel,
    BlockFP8LinearKernel,
    Glm53CheckpointTensorSpec,
    SafetensorsTensorRange,
)
from jaxsft.models.glm5_3_streaming import (
    Glm53StreamingLoader,
    _array_from_payload,
    _is_axis0_sharded,
    _target_shape,
    _transform_source,
)


def _tensor(*, dtype="F8_E4M3", shape=(2048, 4096)):
    width = {"F8_E4M3": 1, "BF16": 2, "F32": 4}[dtype]
    size = int(np.prod(shape)) * width
    return SafetensorsTensorRange(
        name="weight",
        dtype=dtype,
        shape=shape,
        relative_start=0,
        relative_end=size,
        data_section_start=100,
    )


def test_streaming_transforms_preserve_checkpoint_orientation_contract():
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert np.array_equal(_transform_source(source, "identity"), source)
    assert np.array_equal(_transform_source(source, "transpose"), source.T)
    convolution = np.arange(24, dtype=np.float32).reshape(6, 1, 4)
    assert np.array_equal(_transform_source(convolution, "squeeze_conv"), convolution[:, 0])

    transpose = Glm53CheckpointTensorSpec("w", ("w",), (3, 4), "transpose")
    squeeze = Glm53CheckpointTensorSpec("c", ("c",), (6, 1, 4), "squeeze_conv")
    assert _target_shape(transpose) == (4, 3)
    assert _target_shape(squeeze) == (6, 4)


def test_streaming_payload_parser_and_axis0_policy_are_exact():
    tensor = _tensor(dtype="F32", shape=(2, 3))
    payload = np.arange(6, dtype="<f4").tobytes()
    assert np.array_equal(_array_from_payload(payload, tensor), np.arange(6).reshape(2, 3))
    with pytest.raises(ValueError, match="expected"):
        _array_from_payload(payload[:-1], tensor)

    assert _is_axis0_sharded(_tensor())
    assert not _is_axis0_sharded(_tensor(dtype="F32", shape=(16, 16)))
    with pytest.raises(ValueError, match="16-way"):
        _is_axis0_sharded(_tensor(dtype="BF16", shape=(1025, 1024)))


def _abstract_loader(tensors, *, routed_experts=2):
    loader = object.__new__(Glm53StreamingLoader)
    loader.mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("model",))
    loader.config = SimpleNamespace(n_routed_experts=routed_experts)
    loader._weight_map = {tensor.name: "shard" for tensor in tensors}
    loader._headers = {tensor.name: ("shard", tensor) for tensor in tensors}
    loader._shard_headers = {"shard": object()}
    return loader


def test_abstract_single_targets_match_loader_dtype_shape_and_sharding():
    fp8 = SafetensorsTensorRange("layer.weight", "F8_E4M3", (256, 128), 0, 32768, 100)
    scale = SafetensorsTensorRange(
        "layer.weight_scale_inv", "F32", (2, 1), 32768, 32776, 100
    )
    loader = _abstract_loader((fp8, scale))
    spec = Glm53CheckpointTensorSpec("layer.weight", ("kernel",), (256, 128), "transpose")
    loaded = loader._abstract_single_target(spec)
    assert isinstance(loaded.value, BlockFP8LinearKernel)
    assert loaded.value.shape == (128, 256)
    assert loaded.value.weight_bits.dtype == jnp.uint8
    assert loaded.value.weight_scale_inv.dtype == jnp.float32
    assert loaded.scale_source_names == ("layer.weight_scale_inv",)

    bf16 = _tensor(dtype="BF16", shape=(2048, 1024))
    loader = _abstract_loader((bf16,))
    spec = Glm53CheckpointTensorSpec("weight", ("kernel",), bf16.shape, "transpose")
    loaded = loader._abstract_single_target(spec)
    assert isinstance(loaded.value, jax.ShapeDtypeStruct)
    assert loaded.value.shape == (1024, 2048)
    assert loaded.value.dtype == jnp.bfloat16
    assert loaded.value.sharding.spec == PartitionSpec(None, "model")


def test_abstract_expert_target_matches_packed_loader_contract():
    tensors = []
    specs = []
    offset = 0
    for expert in range(2):
        name = f"experts.{expert}.weight"
        tensors.append(SafetensorsTensorRange(name, "F8_E4M3", (256, 128), offset, offset + 32768, 100))
        offset += 32768
        scale_name = f"experts.{expert}.weight_scale_inv"
        tensors.append(SafetensorsTensorRange(scale_name, "F32", (2, 1), offset, offset + 8, 100))
        offset += 8
        specs.append(
            Glm53CheckpointTensorSpec(
                name,
                ("experts",),
                (256, 128),
                "expert_transpose",
                pack_index=expert,
            )
        )
    loader = _abstract_loader(tensors)
    loaded = loader._abstract_expert_target(specs)
    assert isinstance(loaded.value, BatchedBlockFP8LinearKernel)
    assert loaded.value.shape == (2, 128, 256)
    assert loaded.value.weight_bits.sharding.spec == PartitionSpec(None, "model", None)
    assert loaded.value.weight_scale_inv.sharding.spec == PartitionSpec()
