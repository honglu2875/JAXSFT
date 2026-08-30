import subprocess
import tarfile
from argparse import Namespace
from pathlib import Path

import pytest

import cluster


PROFILE = Path(__file__).parents[2] / "configs" / "clusters" / "four-host-tpu.example.toml"


def test_public_cluster_profile_has_four_placeholder_hosts_and_safe_roots():
    profile = cluster.load_profile(PROFILE)
    assert profile.hosts == tuple(f"worker-{index}.example.internal" for index in range(4))
    assert str(profile.remote_workspace_root) == "/var/tmp/jaxsft-runs"
    assert str(profile.remote_cache_root) == "/dev/shm/.jaxsft-cache"
    assert cluster._cache_needs_logout_protection(profile.remote_cache_root)
    assert profile.coordinator_host == profile.hosts[0]


@pytest.mark.parametrize("value", ["../bad", "/", "/tmp", "bad/relative", "contains/slash"])
def test_run_id_rejects_path_like_values(value):
    with pytest.raises(cluster.ClusterError):
        cluster.validate_run_id(value)


def test_source_capsule_contains_research_code_and_excludes_generated_state():
    repository = Path(__file__).parents[2]
    worktree = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=repository,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if worktree.returncode:
        pytest.skip("source capsule intentionally has no Git metadata")
    capsule, digest, files = cluster.make_capsule()
    second_capsule, second_digest, second_files = cluster.make_capsule()
    try:
        assert len(digest) == 64
        assert second_digest == digest
        assert second_capsule.read_bytes() == capsule.read_bytes()
        assert second_files == files
        assert "src/jaxsft/data/adapters.py" in files
        assert "src/jaxsft/models/olmo2.py" in files
        assert "src/jaxsft/models/qwen3_5.py" in files
        assert "uv.lock" in files
        assert not any(path.startswith((".jaxsft/", "artifacts/")) for path in files)
        with tarfile.open(capsule) as archive:
            assert sorted(archive.getnames()) == files
    finally:
        capsule.unlink(missing_ok=True)
        second_capsule.unlink(missing_ok=True)


def test_launch_dry_run_carries_capsule_identity(tmp_path, monkeypatch, capsys):
    profile = cluster.load_profile(PROFILE)
    monkeypatch.setattr(cluster, "STATE_ROOT", tmp_path)
    state = cluster.RunState(
        run_id="fixture-run",
        source_sha256="a" * 64,
        remote_run_dir="/var/tmp/jaxsft-runs/fixture-run",
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
    assert "bootstrap uv:" in output
    assert "/dev/shm/.jaxsft-cache/bin/uv-" in output
    assert "TPU_LOG_DIR=/var/tmp/jaxsft-runs/fixture-run/tpu-logs" in output
    assert "libtpu preflight:" in output


def test_synthetic_cluster_launch_forwards_shape_flags(tmp_path, monkeypatch, capsys):
    profile = cluster.load_profile(PROFILE)
    monkeypatch.setattr(cluster, "STATE_ROOT", tmp_path)
    state = cluster.RunState(
        run_id="synthetic-fixture",
        source_sha256="b" * 64,
        remote_run_dir="/var/tmp/jaxsft-runs/synthetic-fixture",
        hosts=profile.hosts,
    )
    cluster.save_state(profile, state)
    arguments = Namespace(
        run_id="synthetic-fixture",
        recipe=str(
            Path(__file__).parents[2] / "configs" / "recipes" / "olmo2_1b_ultrachat_loss_aware_smoke.yaml"
        ),
        dry_run=True,
        synthetic=True,
        synthetic_length=24,
        synthetic_vocab_size=96,
    )
    assert cluster.run_remote(profile, arguments) == 0
    output = capsys.readouterr().out
    assert "--synthetic --synthetic-length 24 --synthetic-vocab-size 96" in output


def test_batch_tape_cluster_launch_requires_dedicated_remote_path(tmp_path, monkeypatch, capsys):
    profile = cluster.load_profile(PROFILE)
    monkeypatch.setattr(cluster, "STATE_ROOT", tmp_path)
    state = cluster.RunState(
        run_id="tape-fixture",
        source_sha256="d" * 64,
        remote_run_dir="/var/tmp/jaxsft-runs/tape-fixture",
        hosts=profile.hosts,
    )
    cluster.save_state(profile, state)
    arguments = Namespace(
        run_id="tape-fixture",
        recipe=str(
            Path(__file__).parents[2]
            / "configs"
            / "recipes"
            / "olmo2_1b_ultrachat_trajectory_20.yaml"
        ),
        dry_run=True,
        synthetic=False,
        batch_tape="/dev/shm/.jaxsft-cache/batch-tapes/fixture",
        force_fp32=True,
    )
    assert cluster.run_remote(profile, arguments) == 0
    output = capsys.readouterr().out
    assert "test -f /dev/shm/.jaxsft-cache/batch-tapes/fixture/manifest.json" in output
    assert "--batch-tape /dev/shm/.jaxsft-cache/batch-tapes/fixture" in output
    assert "JAXSFT_FORCE_FP32=1" in output

    arguments.batch_tape = "/tmp/unscoped-tape"
    with pytest.raises(cluster.ClusterError, match="dedicated remote"):
        cluster.run_remote(profile, arguments)


def test_uv_bootstrap_uses_probe_before_uploading_controller_bytes():
    remote = cluster.PurePosixPath("/dev/shm/.jaxsft-cache/bin/uv-deadbeef")
    probe = cluster._remote_uv_probe_command(remote, "a" * 64)
    upload = cluster._remote_uv_upload_command("worker-0.example.internal", remote, "a" * 64)
    assert "exit 42" in probe
    assert "cat >" not in probe
    assert "cat >" in upload
    assert "uv-deadbeef.tmp-worker-0.example.internal" in upload


def test_ram_cache_upload_can_be_unsealed_and_sealed_in_one_session():
    cache = cluster.PurePosixPath("/dev/shm/.jaxsft-cache")
    remote = cache / "bin" / "uv-deadbeef"
    upload = cluster._remote_uv_upload_command("worker-0.example.internal", remote, "a" * 64, cache)
    assert "sudo -n chown -R" in upload
    assert "trap" in upload
    assert "root:" in upload
    assert "chmod -R g+rwX" in upload


def test_persistent_cache_upload_does_not_require_sudo():
    cache = cluster.PurePosixPath("/var/tmp/jaxsft-cache")
    remote = cache / "bin" / "uv-deadbeef"
    upload = cluster._remote_uv_upload_command("worker-0.example.internal", remote, "a" * 64)
    assert not cluster._cache_needs_logout_protection(cache)
    assert "sudo" not in upload


def test_multihost_resume_dry_run_uses_runtime_rank_step_directory(tmp_path, monkeypatch, capsys):
    profile = cluster.load_profile(PROFILE)
    monkeypatch.setattr(cluster, "STATE_ROOT", tmp_path)
    state = cluster.RunState(
        run_id="resume-destination",
        source_sha256="f" * 64,
        remote_run_dir="/var/tmp/jaxsft-runs/resume-destination",
        hosts=profile.hosts,
    )
    cluster.save_state(profile, state)
    arguments = Namespace(
        run_id=state.run_id,
        recipe=str(
            Path(__file__).parents[2] / "configs" / "recipes" / "olmo2_1b_ultrachat_loss_aware_smoke.yaml"
        ),
        dry_run=True,
        synthetic=True,
        synthetic_length=32,
        synthetic_vocab_size=128,
        resume_run_id="interrupted-source",
        resume_step=2,
        stop_after_step=3,
    )

    assert cluster.run_remote(profile, arguments) == 0
    output = capsys.readouterr().out
    checkpoint = "/var/tmp/jaxsft-runs/interrupted-source/artifacts/checkpoints/step-00000002"
    assert f"test -d {checkpoint}" in output
    assert f"--resume {checkpoint} --stop-after-step 3" in output

    arguments.resume_step = None
    with pytest.raises(cluster.ClusterError, match="provided together"):
        cluster.run_remote(profile, arguments)


def test_libtpu_preflight_is_conservative_and_recoverable():
    command = cluster._remote_libtpu_preflight_command()
    assert "fuser /dev/accel0" in command
    assert "fuser /tmp/libtpu_lockfile" in command
    assert "test -e /tmp/libtpu_lockfile || test -L /tmp/libtpu_lockfile" in command
    assert "test ! -L /tmp/libtpu_lockfile" in command
    assert "test ! -s /tmp/libtpu_lockfile" in command
    assert "unlink /tmp/libtpu_lockfile" in command
    assert "rm " not in command


def test_stop_dry_run_stages_term_then_exact_kill(tmp_path, monkeypatch, capsys):
    profile = cluster.load_profile(PROFILE)
    monkeypatch.setattr(cluster, "STATE_ROOT", tmp_path)
    state = cluster.RunState(
        run_id="stop-fixture",
        source_sha256="c" * 64,
        remote_run_dir="/var/tmp/jaxsft-runs/stop-fixture",
        hosts=profile.hosts,
    )
    cluster.save_state(profile, state)
    arguments = Namespace(run_id="stop-fixture", dry_run=True, grace_seconds=7)

    assert cluster.stop(profile, arguments) == 0
    output = capsys.readouterr().out
    assert "kill -TERM" in output
    assert "remaining=7" in output
    assert "kill -KILL" in output
    assert output.count("/var/tmp/jaxsft-runs/stop-fixture/source/train_sft.py") == 8
