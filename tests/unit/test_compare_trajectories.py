import pytest

from scripts.compare_trajectories import trajectory_stability


def _records(errors):
    return [
        {"step": step, "relative_error": error}
        for step, error in enumerate(errors, start=1)
    ]


def test_trajectory_stability_reports_flat_post_update_error_band():
    summary = trajectory_stability(
        _records([0.0, 0.0, 0.051, 0.055, 0.049, 0.053, 0.052, 0.050]),
        start_step=3,
    )
    assert abs(summary["relative_error_slope_per_step"]) < 0.001
    assert abs(summary["late_minus_early_mean_relative_error"]) < 0.01
    assert summary["early_half"]["start_step"] == 3
    assert summary["late_half"]["end_step"] == 8


def test_trajectory_stability_exposes_widening_error():
    summary = trajectory_stability(_records([0.0, 0.0, 0.01, 0.02, 0.04, 0.08]), start_step=3)
    assert summary["relative_error_slope_per_step"] > 0.02
    assert summary["late_minus_early_mean_relative_error"] > 0.04


def test_trajectory_stability_requires_a_meaningful_window():
    with pytest.raises(ValueError, match="at least four"):
        trajectory_stability(_records([0.0, 0.01, 0.02]), start_step=2)
