UV ?= uv
PYTHON ?= $(UV) run python
RECIPE ?= configs/recipes/qwen35_0_8b_ultrachat_smoke.yaml
PROFILE ?= configs/clusters/four-host-tpu.local.toml

.PHONY: check smoke doctor sync run status collect

check:
	$(UV) run ruff check .
	PYTHONPATH=src JAX_PLATFORMS=cpu $(PYTHON) -m pytest -q tests/unit
	$(PYTHON) -m compileall -q src scripts train_sft.py cluster.py

smoke:
	PYTHONPATH=src JAX_PLATFORMS=cpu $(PYTHON) train_sft.py --config $(RECIPE) --synthetic

doctor:
	$(PYTHON) cluster.py doctor --profile $(PROFILE)

sync:
	$(PYTHON) cluster.py sync --profile $(PROFILE)

run:
	$(PYTHON) cluster.py run --profile $(PROFILE) --recipe $(RECIPE)

status:
	$(PYTHON) cluster.py status --profile $(PROFILE)

collect:
	$(PYTHON) cluster.py collect --profile $(PROFILE)
