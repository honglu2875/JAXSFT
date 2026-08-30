#!/usr/bin/env python3
"""Controller-only SSH orchestration for immutable JAXSFT research runs."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable

try:
    import tomllib
except ModuleNotFoundError as error:  # pragma: no cover - Python requirement diagnostic
    raise SystemExit("cluster.py requires Python 3.11+; run it with `uv run --no-project --python 3.12 python cluster.py`") from error


ROOT = Path(__file__).resolve().parent
STATE_ROOT = ROOT / ".jaxsft" / "cluster-state"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclasses.dataclass(frozen=True)
class SSHConfig:
    user: str
    identity_file: str
    connect_timeout_seconds: int
    connection_attempts: int
    known_hosts_file: str


@dataclasses.dataclass(frozen=True)
class ClusterProfile:
    source_path: Path
    name: str
    hosts: tuple[str, ...]
    coordinator_host: str
    coordinator_port: int
    remote_workspace_root: PurePosixPath
    remote_cache_root: PurePosixPath
    local_artifact_root: Path
    ssh: SSHConfig


@dataclasses.dataclass(frozen=True)
class RunState:
    run_id: str
    source_sha256: str
    remote_run_dir: str
    hosts: tuple[str, ...]
    recipe: str | None = None


class ClusterError(RuntimeError):
    pass


def _strict(raw: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ClusterError(f"unknown keys in {path}: {sorted(unknown)}")


def _safe_remote_root(value: str, field: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.is_absolute() or path in {PurePosixPath("/"), PurePosixPath("/home"), PurePosixPath("/tmp")}:
        raise ClusterError(f"{field} must be a dedicated absolute path below a safe root, got {value!r}")
    if ".." in path.parts or len(path.parts) < 3:
        raise ClusterError(f"{field} is too broad or contains '..': {value!r}")
    return path


def load_profile(path: str | Path) -> ClusterProfile:
    path = Path(path).expanduser().resolve()
    raw = tomllib.loads(path.read_text())
    _strict(
        raw,
        {
            "schema_version",
            "name",
            "hosts",
            "coordinator_host",
            "coordinator_port",
            "remote_workspace_root",
            "remote_cache_root",
            "local_artifact_root",
            "ssh",
        },
        "profile",
    )
    if raw.get("schema_version") != 1:
        raise ClusterError("cluster profile schema_version must be 1")
    hosts = raw.get("hosts")
    if not isinstance(hosts, list) or not hosts or any(not isinstance(host, str) or not host for host in hosts):
        raise ClusterError("profile.hosts must be a non-empty list of host names or IPs")
    if len(hosts) != len(set(hosts)):
        raise ClusterError("profile.hosts must be unique")
    ssh_raw = raw.get("ssh", {})
    if not isinstance(ssh_raw, dict):
        raise ClusterError("profile.ssh must be a table")
    _strict(
        ssh_raw,
        {"user", "identity_file", "connect_timeout_seconds", "connection_attempts", "known_hosts_file"},
        "profile.ssh",
    )
    ssh = SSHConfig(
        user=str(ssh_raw.get("user", "")),
        identity_file=str(ssh_raw.get("identity_file", "")),
        connect_timeout_seconds=int(ssh_raw.get("connect_timeout_seconds", 8)),
        connection_attempts=int(ssh_raw.get("connection_attempts", 2)),
        known_hosts_file=str(ssh_raw.get("known_hosts_file", "/tmp/jaxsft-known-hosts")),
    )
    coordinator = str(raw.get("coordinator_host", hosts[0]))
    if not coordinator:
        raise ClusterError("profile.coordinator_host must be non-empty")
    port = int(raw.get("coordinator_port", 12355))
    if not 1 <= port <= 65535:
        raise ClusterError("profile.coordinator_port must be between 1 and 65535")
    return ClusterProfile(
        source_path=path,
        name=str(raw["name"]),
        hosts=tuple(hosts),
        coordinator_host=coordinator,
        coordinator_port=port,
        remote_workspace_root=_safe_remote_root(str(raw["remote_workspace_root"]), "remote_workspace_root"),
        remote_cache_root=_safe_remote_root(str(raw["remote_cache_root"]), "remote_cache_root"),
        local_artifact_root=(ROOT / str(raw.get("local_artifact_root", "artifacts/cluster"))).resolve(),
        ssh=ssh,
    )


def _target(profile: ClusterProfile, host: str) -> str:
    return f"{profile.ssh.user}@{host}" if profile.ssh.user else host


def _ssh_base(profile: ClusterProfile, host: str) -> list[str]:
    options = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={profile.ssh.connect_timeout_seconds}",
        "-o",
        f"ConnectionAttempts={profile.ssh.connection_attempts}",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={profile.ssh.known_hosts_file}",
    ]
    if profile.ssh.identity_file:
        options.extend(("-i", profile.ssh.identity_file))
    options.append(_target(profile, host))
    return options


def _ssh(
    profile: ClusterProfile,
    host: str,
    command: str,
    *,
    input_bytes: bytes | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*_ssh_base(profile, host), command],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _parallel(hosts: tuple[str, ...], operation: Callable[[str], Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as executor:
        futures = {executor.submit(operation, host): host for host in hosts}
        for future in concurrent.futures.as_completed(futures):
            host = futures[future]
            try:
                results[host] = future.result()
            except Exception as error:  # preserve every host result before failing
                results[host] = error
    return results


def _check_results(results: dict[str, Any], action: str) -> None:
    failures = []
    for host, result in results.items():
        if isinstance(result, Exception):
            failures.append(f"{host}: {type(result).__name__}: {result}")
        elif isinstance(result, subprocess.CompletedProcess) and result.returncode:
            stderr = result.stderr.decode(errors="replace").strip()
            failures.append(f"{host}: exit {result.returncode}: {stderr}")
    if failures:
        raise ClusterError(f"{action} failed on {len(failures)} host(s):\n" + "\n".join(failures))


def _state_path(profile: ClusterProfile) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", profile.name)
    return STATE_ROOT / f"{safe_name}.json"


def save_state(profile: ClusterProfile, state: RunState) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    path = _state_path(profile)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(dataclasses.asdict(state), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_state(profile: ClusterProfile, run_id: str | None = None) -> RunState:
    path = _state_path(profile)
    if not path.is_file():
        raise ClusterError(f"no synced run is recorded for {profile.name}; run `cluster.py sync` first")
    raw = json.loads(path.read_text())
    state = RunState(
        run_id=str(raw["run_id"]),
        source_sha256=str(raw["source_sha256"]),
        remote_run_dir=str(raw["remote_run_dir"]),
        hosts=tuple(raw["hosts"]),
        recipe=raw.get("recipe"),
    )
    if run_id is not None and state.run_id != run_id:
        raise ClusterError(f"recorded run is {state.run_id!r}, not requested {run_id!r}")
    if state.hosts != profile.hosts:
        raise ClusterError("recorded run host list differs from the current profile")
    return state


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise ClusterError(result.stderr.decode(errors="replace"))
    files = []
    for item in result.stdout.split(b"\0"):
        if not item:
            continue
        relative = Path(os.fsdecode(item))
        unresolved = ROOT / relative
        path = unresolved.resolve()
        if unresolved.is_symlink() or ROOT not in path.parents or not path.is_file():
            raise ClusterError(f"source capsule entry escapes or is not a regular file: {relative}")
        files.append(relative)
    return sorted(files)


def make_capsule() -> tuple[Path, str, list[str]]:
    files = _tracked_files()
    temporary = tempfile.NamedTemporaryFile(prefix="jaxsft-source-", suffix=".tar", delete=False)
    temporary.close()
    capsule = Path(temporary.name)
    with tarfile.open(capsule, "w", format=tarfile.PAX_FORMAT) as archive:
        for relative in files:
            path = ROOT / relative
            info = archive.gettarinfo(str(path), arcname=relative.as_posix())
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.pax_headers = {}
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    digest = hashlib.sha256(capsule.read_bytes()).hexdigest()
    return capsule, digest, [path.as_posix() for path in files]


def default_run_id(source_sha256: str) -> str:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-jaxsft-run-{source_sha256[:8]}"


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ClusterError("run ID must contain only letters, digits, dot, underscore, or hyphen")
    return run_id


def doctor(profile: ClusterProfile, args: argparse.Namespace) -> int:
    remote_script = r'''set -eu
echo "hostname=$(hostname)"
echo "kernel=$(uname -srmo)"
echo "python=$(python3 --version 2>&1 || true)"
echo "uv=$(command -v uv || true)"
echo "jax=$(python3 -c 'import jax; print(jax.__version__)' 2>/dev/null || true)"
echo "root_disk=$(df -Pk / | tail -1)"
echo "shm=$(df -Pk /dev/shm | tail -1)"
echo "memory=$(awk '/MemTotal/ {print $2 " kB"}' /proc/meminfo)"
echo "accelerator_type=$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type 2>/dev/null || true)"
echo "worker_number=$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number 2>/dev/null || true)"'''
    if args.dry_run:
        for host in profile.hosts:
            print(shlex.join([*_ssh_base(profile, host), remote_script]))
        return 0
    results = _parallel(profile.hosts, lambda host: _ssh(profile, host, remote_script, timeout=30))
    for host in profile.hosts:
        result = results[host]
        print(f"[{host}]")
        if isinstance(result, Exception):
            print(f"ERROR {type(result).__name__}: {result}")
        else:
            print(result.stdout.decode(errors="replace").rstrip())
            if result.returncode:
                print(result.stderr.decode(errors="replace").rstrip())
    _check_results(results, "doctor")
    hostnames = []
    for result in results.values():
        fields = dict(
            line.split("=", 1) for line in result.stdout.decode(errors="replace").splitlines() if "=" in line
        )
        hostnames.append(fields.get("hostname"))
    if len(set(hostnames)) != len(hostnames):
        raise ClusterError(f"SSH targets do not resolve to unique hosts: {hostnames}")
    return 0


def sync(profile: ClusterProfile, args: argparse.Namespace) -> int:
    capsule, digest, files = make_capsule()
    try:
        run_id = validate_run_id(args.run_id or default_run_id(digest))
        remote_run = profile.remote_workspace_root / run_id
        command = (
            f"set -eu; umask 077; "
            f"test ! -e {shlex.quote(str(remote_run))}; "
            f"mkdir -p {shlex.quote(str(remote_run / 'source'))}; "
            f"tar -xf - -C {shlex.quote(str(remote_run / 'source'))}; "
            f"printf '%s\\n' {shlex.quote(digest)} > {shlex.quote(str(remote_run / 'SOURCE_SHA256'))}"
        )
        if args.dry_run:
            print(json.dumps({"run_id": run_id, "sha256": digest, "files": files}, indent=2))
            for host in profile.hosts:
                print(shlex.join([*_ssh_base(profile, host), command]))
            return 0
        payload = capsule.read_bytes()
        results = _parallel(profile.hosts, lambda host: _ssh(profile, host, command, input_bytes=payload, timeout=180))
        _check_results(results, "sync")
        state = RunState(run_id, digest, str(remote_run), profile.hosts)
        save_state(profile, state)
        print(json.dumps(dataclasses.asdict(state), indent=2))
        return 0
    finally:
        capsule.unlink(missing_ok=True)


def _recipe_relative(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    if ROOT not in resolved.parents or not resolved.is_file():
        raise ClusterError("recipe must be a file inside the JAXSFT source tree")
    return resolved.relative_to(ROOT).as_posix()


def _local_uv_binary() -> tuple[Path, str, bytes]:
    executable = shutil.which("uv")
    if executable is None:
        raise ClusterError("controller has no `uv` executable to bootstrap on workers")
    path = Path(executable).resolve()
    if not path.is_file():
        raise ClusterError(f"controller uv path is not a regular file: {path}")
    payload = path.read_bytes()
    return path, hashlib.sha256(payload).hexdigest(), payload


def _remote_uv_probe_command(remote_uv: PurePosixPath, digest: str) -> str:
    return (
        f"if test -x {shlex.quote(str(remote_uv))} && "
        f"test \"$(sha256sum {shlex.quote(str(remote_uv))} | cut -d' ' -f1)\" = {shlex.quote(digest)}; "
        f"then {shlex.quote(str(remote_uv))} --version; else exit 42; fi"
    )


def _remote_cache_unseal_command(cache: PurePosixPath) -> str:
    quoted = shlex.quote(str(cache))
    return (
        'owner="$(id -u)"; group="$(id -g)"; '
        "command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1 || "
        "{ echo 'passwordless sudo is required to protect the RAM cache' >&2; exit 48; }; "
        f"if test -d {quoted}; then sudo -n chown -R \"$owner:$group\" -- {quoted}; "
        f"else sudo -n install -d -o \"$owner\" -g \"$group\" -m 0775 {quoted}; fi"
    )


def _remote_cache_seal_command(cache: PurePosixPath) -> str:
    quoted = shlex.quote(str(cache))
    return (
        'group="$(id -g)"; '
        f"sudo -n chown -R root:\"$group\" -- {quoted} && "
        f"sudo -n chmod -R g+rwX -- {quoted}"
    )


def _cache_needs_logout_protection(cache: PurePosixPath) -> bool:
    """Return whether logind may remove this per-user RAM-cache tree."""

    return cache.is_relative_to(PurePosixPath("/dev/shm"))


def _remote_uv_upload_command(
    host: str,
    remote_uv: PurePosixPath,
    digest: str,
    cache: PurePosixPath | None = None,
) -> str:
    safe_host = re.sub(r"[^A-Za-z0-9_.-]", "_", host)
    temporary = remote_uv.with_name(remote_uv.name + f".tmp-{safe_host}")
    parent = remote_uv.parent
    unseal = "" if cache is None else _remote_cache_unseal_command(cache) + "; "
    seal = "" if cache is None else f"trap {shlex.quote(_remote_cache_seal_command(cache))} EXIT; "
    return (
        f"set -eu; {unseal}{seal}umask 077; mkdir -p {shlex.quote(str(parent))}; "
        f"test ! -e {shlex.quote(str(temporary))}; "
        f"cat > {shlex.quote(str(temporary))}; "
        f"test \"$(sha256sum {shlex.quote(str(temporary))} | cut -d' ' -f1)\" = {shlex.quote(digest)}; "
        f"chmod 700 {shlex.quote(str(temporary))}; "
        f"if test -e {shlex.quote(str(remote_uv))}; then "
        f"test \"$(sha256sum {shlex.quote(str(remote_uv))} | cut -d' ' -f1)\" = {shlex.quote(digest)}; "
        f"unlink {shlex.quote(str(temporary))}; "
        f"else mv {shlex.quote(str(temporary))} {shlex.quote(str(remote_uv))}; fi; "
        f"{shlex.quote(str(remote_uv))} --version"
    )


def _remote_libtpu_preflight_command() -> str:
    """Reject an active TPU owner and remove only an unowned empty lockfile."""

    lockfile = "/tmp/libtpu_lockfile"
    accelerator = "/dev/accel0"
    return (
        "set -eu; command -v fuser >/dev/null 2>&1 || "
        "{ echo 'fuser is required for safe libtpu lock recovery' >&2; exit 43; }; "
        f"if test -e {accelerator} && fuser {accelerator} >/dev/null 2>&1; then "
        "echo 'TPU device already has an owner' >&2; exit 44; fi; "
        f"if test -e {lockfile} || test -L {lockfile}; then "
        f"test -f {lockfile} && test ! -L {lockfile} && test ! -s {lockfile} || "
        "{ echo 'refusing unsafe libtpu lock recovery' >&2; exit 45; }; "
        f"if fuser {lockfile} >/dev/null 2>&1; then "
        "echo 'libtpu lockfile still has an owner' >&2; exit 46; fi; "
        f"unlink {lockfile}; fi"
    )


def _remote_batch_tape_path(profile: ClusterProfile, value: str | None) -> PurePosixPath | None:
    if value is None:
        return None
    path = PurePosixPath(value)
    allowed_roots = (profile.remote_cache_root, profile.remote_workspace_root)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or not any(path != root and path.is_relative_to(root) for root in allowed_roots)
    ):
        raise ClusterError("--batch-tape must be below the profile's dedicated remote cache or workspace root")
    return path


def run_remote(profile: ClusterProfile, args: argparse.Namespace) -> int:
    state = load_state(profile, args.run_id)
    recipe = _recipe_relative(args.recipe)
    remote_run = PurePosixPath(state.remote_run_dir)
    source = remote_run / "source"
    lock_digest = hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    venv = profile.remote_cache_root / "venvs" / f"uv-lock-{lock_digest[:16]}"
    artifacts = remote_run / "artifacts"
    temporary_root = remote_run / "tmp"
    tpu_log_root = remote_run / "tpu-logs"
    # Hub, dataset, uv, and compilation caches are intrinsically content keyed;
    # sharing the dedicated cache root avoids downloading the same pinned inputs
    # for every immutable run directory.
    cache = profile.remote_cache_root
    protect_cache = _cache_needs_logout_protection(cache)
    batch_tape = _remote_batch_tape_path(profile, getattr(args, "batch_tape", None))
    if getattr(args, "synthetic", False) and batch_tape is not None:
        raise ClusterError("--synthetic and --batch-tape are mutually exclusive")
    stop_after_step = getattr(args, "stop_after_step", None)
    if stop_after_step is not None and stop_after_step <= 0:
        raise ClusterError("--stop-after-step must be positive")
    resume_run_id = getattr(args, "resume_run_id", None)
    resume_step = getattr(args, "resume_step", None)
    if (resume_run_id is None) != (resume_step is None):
        raise ClusterError("--resume-run-id and --resume-step must be provided together")
    resume_checkpoint = None
    if resume_run_id is not None:
        resume_run_id = validate_run_id(resume_run_id)
        if resume_run_id == state.run_id:
            raise ClusterError("resume source run must differ from the new immutable run")
        if resume_step <= 0:
            raise ClusterError("--resume-step must be positive")
        previous_artifacts = profile.remote_workspace_root / resume_run_id / "artifacts" / "checkpoints"
        if len(profile.hosts) == 1:
            resume_checkpoint = previous_artifacts / f"step-{resume_step:08d}.pkl"
        else:
            resume_checkpoint = previous_artifacts / f"step-{resume_step:08d}"
    local_uv, uv_digest, uv_payload = _local_uv_binary()
    remote_uv = cache / "bin" / f"uv-{uv_digest[:16]}"
    env = {
        "HF_HOME": str(cache / "huggingface"),
        "HF_DATASETS_CACHE": str(cache / "datasets"),
        "JAX_COMPILATION_CACHE_DIR": str(cache / "jax-compile"),
        "UV_CACHE_DIR": str(cache / "uv"),
        "UV_PROJECT_ENVIRONMENT": str(venv),
    }
    env_text = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    input_checks = []
    if batch_tape is not None:
        input_checks.extend(
            (
                f"test -d {shlex.quote(str(batch_tape))}",
                f"test -f {shlex.quote(str(batch_tape / 'manifest.json'))}",
            )
        )
    if resume_checkpoint is not None:
        input_checks.append(
            f"test -d {shlex.quote(str(resume_checkpoint))}"
            if len(profile.hosts) > 1
            else f"test -f {shlex.quote(str(resume_checkpoint))}"
        )
    input_check_text = "" if not input_checks else " && ".join(input_checks) + "; "
    prepare_body = (
        f"mkdir -p {shlex.quote(str(artifacts))} "
        f"{shlex.quote(str(temporary_root))} {shlex.quote(str(tpu_log_root))}; "
        f"{input_check_text}"
        f"cd {shlex.quote(str(source))}; "
        f"test \"$(sha256sum uv.lock | cut -d' ' -f1)\" = {shlex.quote(lock_digest)}; "
        f"env {env_text} {shlex.quote(str(remote_uv))} sync --frozen --no-dev --no-install-project"
    )
    if not getattr(args, "synthetic", False):
        stage_env = {
            **env,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(source / "src"),
        }
        stage_env_text = " ".join(f"{key}={shlex.quote(value)}" for key, value in stage_env.items())
        stage_arguments = [
            str(venv / "bin" / "python"),
            str(source / "scripts" / "stage_recipe.py"),
            "--config",
            str(source / recipe),
        ]
        if batch_tape is not None:
            stage_arguments.append("--skip-data")
        prepare_body += (
            f"; env {stage_env_text} {shlex.join(stage_arguments)} "
            f"> {shlex.quote(str(remote_run / 'staging.json'))}"
        )
    if protect_cache:
        prepare = (
            f"set -eu; {_remote_cache_unseal_command(cache)}; "
            f"trap {shlex.quote(_remote_cache_seal_command(cache))} EXIT; {prepare_body}"
        )
    else:
        prepare = f"set -eu; mkdir -p {shlex.quote(str(cache))}; {prepare_body}"
    coordinator = f"{profile.coordinator_host}:{profile.coordinator_port}"

    def launch_command(rank: int) -> str:
        pid_file = remote_run / f"rank-{rank:03d}.pid"
        log_file = remote_run / f"rank-{rank:03d}.log"
        run_env = {
            **env,
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "JAXSFT_COORDINATOR_ADDRESS": coordinator,
            "JAXSFT_PROCESS_COUNT": str(len(profile.hosts)),
            "JAXSFT_PROCESS_ID": str(rank),
            "JAXSFT_SOURCE_SHA256": state.source_sha256,
            "JAXSFT_OUTPUT_DIR": str(artifacts),
            "JAX_PLATFORMS": "tpu",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(source / "src"),
            "PYTHONUNBUFFERED": "1",
            "TEST_TMPDIR": str(temporary_root),
            "TPU_LOG_DIR": str(tpu_log_root),
        }
        if getattr(args, "force_fp32", False):
            run_env["JAXSFT_FORCE_FP32"] = "1"
        run_env_text = " ".join(f"{key}={shlex.quote(value)}" for key, value in run_env.items())
        python = venv / "bin" / "python"
        train = source / "train_sft.py"
        config = source / recipe
        extra_arguments: list[str] = []
        if getattr(args, "synthetic", False):
            extra_arguments.extend(
                (
                    "--synthetic",
                    "--synthetic-length",
                    str(args.synthetic_length),
                    "--synthetic-vocab-size",
                    str(args.synthetic_vocab_size),
                )
            )
        if batch_tape is not None:
            extra_arguments.extend(("--batch-tape", str(batch_tape)))
        if resume_checkpoint is not None:
            extra_arguments.extend(("--resume", str(resume_checkpoint)))
        if stop_after_step is not None:
            extra_arguments.extend(("--stop-after-step", str(stop_after_step)))
        extra_text = "" if not extra_arguments else " " + shlex.join(extra_arguments)
        return (
            f"set -eu; test ! -e {shlex.quote(str(pid_file))}; "
            f"cd {shlex.quote(str(source))}; "
            f"nohup env {run_env_text} {shlex.quote(str(python))} {shlex.quote(str(train))} "
            f"--config {shlex.quote(str(config))}{extra_text} > {shlex.quote(str(log_file))} 2>&1 < /dev/null & "
            f"pid=$!; printf '%s\\n' \"$pid\" > {shlex.quote(str(pid_file))}; "
            f"kill -0 \"$pid\""
        )

    if args.dry_run:
        for rank, host in enumerate(profile.hosts):
            print(f"[{host}] bootstrap uv: {local_uv} -> {remote_uv} (sha256={uv_digest})")
            print(f"[{host}] prepare: {prepare}")
            print(f"[{host}] libtpu preflight: {_remote_libtpu_preflight_command()}")
            print(f"[{host}] launch:  {launch_command(rank)}")
        return 0
    def bootstrap_uv(host: str) -> subprocess.CompletedProcess:
        probe = _ssh(profile, host, _remote_uv_probe_command(remote_uv, uv_digest), timeout=30)
        if probe.returncode != 42:
            return probe
        return _ssh(
            profile,
            host,
            _remote_uv_upload_command(host, remote_uv, uv_digest, cache if protect_cache else None),
            input_bytes=uv_payload,
            timeout=180,
        )

    bootstrapped = _parallel(profile.hosts, bootstrap_uv)
    _check_results(bootstrapped, "uv bootstrap")
    prepared = _parallel(profile.hosts, lambda host: _ssh(profile, host, prepare, timeout=1800))
    _check_results(prepared, "dependency preparation")
    preflighted = _parallel(
        profile.hosts,
        lambda host: _ssh(profile, host, _remote_libtpu_preflight_command(), timeout=30),
    )
    _check_results(preflighted, "libtpu preflight")
    launched = _parallel(
        profile.hosts,
        lambda host: _ssh(profile, host, launch_command(profile.hosts.index(host)), timeout=30),
    )
    _check_results(launched, "launch")
    save_state(profile, dataclasses.replace(state, recipe=recipe))
    print(json.dumps({"run_id": state.run_id, "status": "launched", "hosts": profile.hosts}, indent=2))
    return 0


def status(profile: ClusterProfile, args: argparse.Namespace) -> int:
    state = load_state(profile, args.run_id)
    remote_run = PurePosixPath(state.remote_run_dir)

    def command(rank: int) -> str:
        pid_file = remote_run / f"rank-{rank:03d}.pid"
        log_file = remote_run / f"rank-{rank:03d}.log"
        return (
            f"if test -f {shlex.quote(str(pid_file))}; then "
            f"pid=$(cat {shlex.quote(str(pid_file))}); "
            f"if kill -0 \"$pid\" 2>/dev/null; then state=running; else state=exited; fi; "
            f"else pid=none; state=not-launched; fi; "
            f"printf 'state=%s pid=%s\\n' \"$state\" \"$pid\"; "
            f"test ! -f {shlex.quote(str(log_file))} || tail -n 5 {shlex.quote(str(log_file))}"
        )

    results = _parallel(
        profile.hosts,
        lambda host: _ssh(profile, host, command(profile.hosts.index(host)), timeout=30),
    )
    for host in profile.hosts:
        result = results[host]
        print(f"[{host}]")
        if isinstance(result, Exception):
            print(f"ERROR {result}")
        else:
            print(result.stdout.decode(errors="replace").rstrip())
    _check_results(results, "status")
    return 0


def stop(profile: ClusterProfile, args: argparse.Namespace) -> int:
    state = load_state(profile, args.run_id)
    remote_run = PurePosixPath(state.remote_run_dir)
    grace_seconds = int(args.grace_seconds)
    if not 0 <= grace_seconds <= 300:
        raise ClusterError("--grace-seconds must be in [0, 300]")

    def command(rank: int) -> str:
        pid_file = remote_run / f"rank-{rank:03d}.pid"
        marker = str(remote_run / "source" / "train_sft.py")
        return (
            f"set -eu; test -f {shlex.quote(str(pid_file))}; pid=$(cat {shlex.quote(str(pid_file))}); "
            f"case \"$pid\" in (*[!0-9]*|'') exit 41;; esac; "
            f"cmd=$(tr '\\000' ' ' < /proc/\"$pid\"/cmdline 2>/dev/null || true); "
            f"case \"$cmd\" in (*{shlex.quote(marker)}*) ;; (*) echo 'PID command does not match this run' >&2; exit 42;; esac; "
            f"kill -TERM \"$pid\"; remaining={grace_seconds}; "
            "while kill -0 \"$pid\" 2>/dev/null && test \"$remaining\" -gt 0; do "
            "sleep 1; remaining=$((remaining - 1)); done; "
            "if kill -0 \"$pid\" 2>/dev/null; then "
            "cmd=$(tr '\\000' ' ' < /proc/\"$pid\"/cmdline 2>/dev/null || true); "
            f"case \"$cmd\" in (*{shlex.quote(marker)}*) ;; "
            "(*) echo 'PID command changed during stop grace period' >&2; exit 47;; esac; "
            "kill -KILL \"$pid\"; printf 'forced=1\\n'; else printf 'forced=0\\n'; fi"
        )

    if args.dry_run:
        for rank, host in enumerate(profile.hosts):
            print(f"[{host}] {command(rank)}")
        return 0
    results = _parallel(
        profile.hosts,
        lambda host: _ssh(
            profile,
            host,
            command(profile.hosts.index(host)),
            timeout=max(30, grace_seconds + 15),
        ),
    )
    _check_results(results, "stop")
    forced = sum(b"forced=1" in result.stdout for result in results.values())
    print(
        f"stopped exact recorded PID on {len(profile.hosts)} hosts "
        f"({forced} required SIGKILL after {grace_seconds}s grace)"
    )
    return 0


def collect(profile: ClusterProfile, args: argparse.Namespace) -> int:
    state = load_state(profile, args.run_id)
    destination = profile.local_artifact_root / state.run_id
    if args.dry_run:
        for host in profile.hosts:
            print(f"rsync from {_target(profile, host)}:{state.remote_run_dir}/ to {destination / host}/")
        return 0
    destination.mkdir(parents=True, exist_ok=True)

    def one(host: str) -> subprocess.CompletedProcess:
        target = destination / re.sub(r"[^A-Za-z0-9_.-]", "_", host)
        target.mkdir(parents=True, exist_ok=True)
        ssh_transport = shlex.join(_ssh_base(profile, host)[:-1])
        command = [
            "rsync",
            "-az",
            "--include=rank-*.log",
            "--include=rank-*.pid",
            "--include=SOURCE_SHA256",
            "--include=staging.json",
            "--include=artifacts/***",
            "--exclude=*",
            "-e",
            ssh_transport,
            f"{_target(profile, host)}:{state.remote_run_dir}/",
            str(target) + "/",
        ]
        return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    results = _parallel(profile.hosts, one)
    _check_results(results, "collect")
    print(destination)
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    for name, function in (
        ("doctor", doctor),
        ("sync", sync),
        ("run", run_remote),
        ("status", status),
        ("stop", stop),
        ("collect", collect),
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--profile", required=True)
        command.add_argument("--run-id")
        command.add_argument("--dry-run", action="store_true")
        if name == "run":
            command.add_argument("--recipe", required=True)
            command.add_argument("--synthetic", action="store_true")
            command.add_argument("--synthetic-length", type=int, default=32)
            command.add_argument("--synthetic-vocab-size", type=int, default=128)
            command.add_argument("--batch-tape", help="absolute validated tape path already present on every host")
            command.add_argument(
                "--force-fp32",
                action="store_true",
                help="set the recorded JAXSFT_FORCE_FP32 numerical control lane",
            )
            command.add_argument("--stop-after-step", type=int, help="stop at this absolute step and checkpoint")
            command.add_argument("--resume-run-id", help="previous immutable run holding rank-local checkpoints")
            command.add_argument("--resume-step", type=int, help="completed step to restore from --resume-run-id")
        elif name == "stop":
            command.add_argument("--grace-seconds", type=int, default=15)
        command.set_defaults(function=function)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    profile = load_profile(args.profile)
    return args.function(profile, args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ClusterError as error:
        print(f"cluster error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
