import tarfile
from argparse import Namespace
from pathlib import Path

import pytest

import cluster


PROFILE = Path(__file__).parents[2] / "configs" / "clusters" / "four-host-tpu.example.toml"


def test_public_cluster_profile_has_four_placeholder_hosts_and_safe_roots():
    profile = cluster.load_profile(PROFILE)
    assert profile.hosts == tuple(f"worker-{index}.example.internal" for index in range(4))
    assert str(profile.remote_workspace_root) == "/dev/shm/jaxsft-runs"
    assert profile.coordinator_host == profile.hosts[0]


@pytest.mark.parametrize("value", ["../bad", "/", "/tmp", "bad/relative", "contains/slash"])
def test_run_id_rejects_path_like_values(value):
    with pytest.raises(cluster.ClusterError):
        cluster.validate_run_id(value)


def test_source_capsule_contains_research_code_and_excludes_generated_state():
    capsule, digest, files = cluster.make_capsule()
    try:
        assert len(digest) == 64
        assert "src/jaxsft/data/adapters.py" in files
        assert "src/jaxsft/models/qwen3_5.py" in files
        assert "uv.lock" in files
        assert not any(path.startswith((".jaxsft/", "artifacts/")) for path in files)
        with tarfile.open(capsule) as archive:
            assert sorted(archive.getnames()) == files
    finally:
        capsule.unlink(missing_ok=True)


def test_launch_dry_run_carries_capsule_identity(tmp_path, monkeypatch, capsys):
    profile = cluster.load_profile(PROFILE)
    monkeypatch.setattr(cluster, "STATE_ROOT", tmp_path)
    state = cluster.RunState(
        run_id="fixture-run",
        source_sha256="a" * 64,
        remote_run_dir="/dev/shm/jaxsft-runs/fixture-run",
        hosts=profile.hosts,
    )
    cluster.save_state(profile, state)
    arguments = Namespace(
        run_id="fixture-run",
        recipe=str(Path(__file__).parents[2] / "configs" / "recipes" / "qwen35_0_8b_ultrachat_smoke.yaml"),
        dry_run=True,
    )
    assert cluster.run_remote(profile, arguments) == 0
    output = capsys.readouterr().out
    assert f"JAXSFT_SOURCE_SHA256={'a' * 64}" in output
