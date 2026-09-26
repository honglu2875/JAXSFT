# LoRA primitives

`jaxsft.lora` provides model-independent LoRA over explicit dict/list/tuple
parameter paths. Frozen base parameters and trainable adapters are separate
PyTrees: differentiate and initialize AdamW only on the adapter tree.
Models must explicitly declare eligible rank-2 kernel paths in `[in, out]`
layout. This module does not add a trainer option or register a new model.

```python
import jax
import jax.numpy as jnp

from jaxsft.lora import LoRAConfig, adapter_for_path, init_lora_adapters, lora_linear
from jaxsft.optim import adamw_init

base = {"projection": jnp.ones((4, 6), jnp.float32)}
targets = (("projection",),)
config = LoRAConfig(rank=2, alpha=4)
adapters = init_lora_adapters(
    jax.random.key(0), base, targets, eligible_paths=targets,
    config=config, dtype=jnp.float32,
)
optimizer = adamw_init(adapters)

def loss(trainable, inputs):
    output = lora_linear(
        inputs, base["projection"], adapter_for_path(trainable, targets[0]),
        config=config,
    )
    return jnp.mean(output**2)

value, gradients = jax.value_and_grad(loss)(adapters, jnp.ones((2, 4)))
```

The update is `X W + (alpha / rank) X A B`. `A` uses Kaiming-uniform bounds;
`B` starts at zero, preserving the initial base output. Optional dropout
affects only the adapter input during training and requires an explicit PRNG
key. Matrix products use JAX's highest contraction precision by default.

`merge_lora_adapters` returns a new base tree with deltas computed in float32
and cast to each kernel's dtype. It represents inference with dropout disabled;
allow for rounding differences with low-precision kernels.
`flatten_lora_adapters` and `unflatten_lora_adapters` provide stable tensor
names for adapter-only serialization, with strict path, key, shape, and dtype
checks on restore. Persist the model identity, target paths, and `LoRAConfig`
alongside these tensors; the helpers do not write checkpoint files themselves.

Run the offline correctness tests with
`JAX_PLATFORMS=cpu uv run python -m pytest -q tests/unit/test_lora.py`.
