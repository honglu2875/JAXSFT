from dataclasses import asdict
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

from jaxsft.data.stream import InstructionBatchStream, StreamCounters


class FakeReplayStream(InstructionBatchStream):
    def __init__(self, *, process_index=0, process_count=1):
        self.process_index = process_index
        self.process_count = process_count
        self.spec = SimpleNamespace(loading_mode="materialized")
        self.snapshot = SimpleNamespace(identity_hash="fixture-tokenizer")
        self.counters = StreamCounters()
        self._iterator = self._make_iterator(epoch=0)

    def _make_iterator(self, *, epoch):
        return iter([{"value": f"epoch-{epoch}-row-{row}"} for row in range(3)])


def test_stream_state_replays_current_epoch_prefix_exactly():
    original = FakeReplayStream()
    assert original._next_row() == ({"value": "epoch-0-row-0"}, 0)
    assert original._next_row() == ({"value": "epoch-0-row-1"}, 1)
    original.counters.adapter_errors = 1
    original.counters.adapter_errors_by_reason = {"AdapterError: fixture": 1}
    original.counters.rows_emitted = 1
    state = original.state_dict()

    restored = FakeReplayStream()
    restored.load_state_dict(state)
    assert asdict(restored.counters) == state["counters"]
    assert restored._next_row() == ({"value": "epoch-0-row-2"}, 2)


def test_stream_state_restores_after_epoch_boundary_and_rejects_world_size_change():
    original = FakeReplayStream()
    for expected in range(4):
        _, row_index = original._next_row()
        assert row_index == expected
        original.counters.rows_emitted += 1
    assert original.counters.epochs == 1
    assert original.counters.rows_seen_in_epoch == 1

    restored = FakeReplayStream()
    restored.load_state_dict(original.state_dict())
    assert restored._next_row() == ({"value": "epoch-1-row-1"}, 4)

    changed_world = FakeReplayStream(process_count=2)
    with pytest.raises(ValueError, match="process_count"):
        changed_world.load_state_dict(original.state_dict())

    changed_loading_mode = FakeReplayStream()
    changed_loading_mode.spec.loading_mode = "streaming"
    with pytest.raises(ValueError, match="loading_mode"):
        changed_loading_mode.load_state_dict(original.state_dict())


@pytest.mark.parametrize(
    ("loading_mode", "expected_shuffle"),
    [("streaming", {"seed": 20, "buffer_size": 97}), ("materialized", {"seed": 20})],
)
def test_dataset_loading_mode_controls_hub_materialization(monkeypatch, loading_mode, expected_shuffle):
    calls = []

    class FakeDataset:
        def shuffle(self, **kwargs):
            calls.append(("shuffle", kwargs))
            return self

        def __iter__(self):
            return iter([{"row": 1}])

    def load_dataset(*args, **kwargs):
        calls.append(("load", args, kwargs))
        return FakeDataset()

    def split_dataset_by_node(dataset, *, rank, world_size):
        calls.append(("split", rank, world_size))
        return dataset

    datasets = ModuleType("datasets")
    datasets.load_dataset = load_dataset
    distributed = ModuleType("datasets.distributed")
    distributed.split_dataset_by_node = split_dataset_by_node
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setitem(sys.modules, "datasets.distributed", distributed)

    stream = object.__new__(InstructionBatchStream)
    stream.spec = SimpleNamespace(
        repo_id="org/data",
        config="default",
        split="train",
        revision="a" * 40,
        loading_mode=loading_mode,
        shuffle_seed=17,
        shuffle_buffer_size=97,
    )
    stream.process_index = 1
    stream.process_count = 2

    assert next(stream._make_iterator(epoch=3)) == {"row": 1}
    assert calls == [
        (
            "load",
            ("org/data",),
            {
                "name": "default",
                "split": "train",
                "revision": "a" * 40,
                "streaming": loading_mode == "streaming",
            },
        ),
        ("shuffle", expected_shuffle),
        ("split", 1, 2),
    ]
