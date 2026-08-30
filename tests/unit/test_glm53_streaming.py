import numpy as np
import pytest

from jaxsft.models.glm5_3_flash import Glm53CheckpointTensorSpec, SafetensorsTensorRange
from jaxsft.models.glm5_3_streaming import (
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
