from dataclasses import asdict

import pytest

from jaxsft.data.stream import InstructionBatchStream, StreamCounters


class FakeReplayStream(InstructionBatchStream):
    def __init__(self, *, process_index=0, process_count=1):
        self.process_index = process_index
        self.process_count = process_count
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
