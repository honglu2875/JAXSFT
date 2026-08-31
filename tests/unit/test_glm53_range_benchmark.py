from jaxsft.models.glm5_3_flash import SafetensorsTensorRange
from scripts.benchmark_glm53_range_pool import _host_range


def test_expert_host_range_is_exact_quarter_of_source_rows():
    tensor = SafetensorsTensorRange(
        name="expert.weight",
        dtype="F8_E4M3",
        shape=(2048, 4096),
        relative_start=100,
        relative_end=100 + 2048 * 4096,
        data_section_start=1000,
    )
    assert _host_range(tensor, 2) == (
        1100 + 1024 * 4096,
        1100 + 1536 * 4096 - 1,
        512 * 4096,
    )
