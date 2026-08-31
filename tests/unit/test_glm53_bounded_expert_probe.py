from scripts.probe_glm53_bounded_expert import TOP_K, _indices, _shape_mentions


def test_bounded_expert_probe_indices_are_deterministic_and_cover_top_k():
    indices = _indices(4)
    assert indices.shape == (4, TOP_K)
    assert indices.dtype.name == "int32"
    assert set(indices[0]) == {0, 17, 63, 95, 127, 191, 255, 287}
    assert all(set(row) == set(indices[0]) for row in indices)


def test_bounded_expert_probe_hlo_detector_separates_full_assignments_from_chunk():
    hlo = " ".join(
        (
            "bf16[32,2048,4096]",
            "bf16[32,4096,2048]",
            "bf16[4,8,2048,4096]",
            "bf16[1,2048,4096]",
            "bf16[1,4096,2048]",
            "bf16[1,128,4096]",
            "bf16[1,256,2048]",
        )
    )
    mentions = _shape_mentions(hlo, token_count=4, selected_weight_batch_size=1)
    assert mentions["all_assignment_gate_dense:bf16"] == 1
    assert mentions["all_assignment_down_dense:bf16"] == 1
    assert mentions["token_topk_gate_dense:bf16"] == 1
    assert mentions["bounded_gate_dense:bf16"] == 1
    assert mentions["bounded_down_dense:bf16"] == 1
    assert mentions["local_bounded_gate_dense:bf16"] == 1
    assert mentions["local_bounded_down_dense:bf16"] == 1
