# Four-host slice orchestration plan

## 1. Initial target

The first concrete inventory supplied for the project is a private four-host
TPU VM slice retained in an ignored local profile. Four SSH targets do **not**
by themselves prove accelerator
generation, chips per host, aggregate memory, JAX process order, shared storage,
internet access, or that the current machine is worker 0. `cluster doctor` must
measure those facts before a training profile is accepted.

### Controller audit, 2026-08-29

The original four targets were unreachable and were then pre-empted. A
replacement single-host TPU VM was supplied explicitly and tested, but it does
not substitute for the four-process acceptance gate.

The checked-in example is [configs/clusters/four-host-tpu.example.toml](../configs/clusters/four-host-tpu.example.toml).
Operative inventory such as hostnames, IP addresses, and credentials belongs in
an ignored `*.local.toml` profile and is not published with the source capsule.

### Measured four-host v4-32 acceptance, 2026-08-30

A later supplied slice passed the previously open gate. Read-only probes found
four unique SSH targets, worker numbers zero through three, 400 GiB host RAM on
each VM, accelerator type v4-32, and enough RAM-backed cache space. The
controller then proved four JAX processes, four local devices per process, and
16 global devices. Runtime process ranks did not follow hostname-suffix order,
which confirms why all data, artifact, and checkpoint identity comes from
`jax.process_index()`.

The acceptance sequence completed:

- a three-step synthetic topology and collective smoke;
- an interrupted step-two checkpoint, fresh four-process restore, and step
  three whose semantic model/optimizer hash exactly matched an uninterrupted
  reference;
- three real materialized-UltraChat updates of OLMo 2 1B; and
- three real materialized-UltraChat updates of Qwen3.5 0.8B.

Both real runs observed four distinct rank-local first-batch hashes, exact
agreement for every globally reduced metric, zero emitted zero-objective
samples, and clean distributed shutdown on all ranks. Pinned inputs were fully
staged before TPU ownership and the training processes ran with Hub access
disabled. Exact hostname-free metrics and hashes are in
[the v4-32 acceptance record](results/v4_32_multihost_acceptance.json).

The VM image enables `systemd-logind` `RemoveIPC`, which can recursively remove
user-owned `/dev/shm` content when a short SSH session ends. The implemented
layout therefore keeps immutable run metadata and checkpoints on persistent
storage. A dedicated RAM cache is unsealed and resealed within one SSH session,
ending root-owned and group-writable so logout cleanup does not claim it.

### Measured single-host v4-8 smoke, 2026-08-29

The replacement VM reported one JAX process, four local/global TPU devices, 400
GiB host RAM, and about 201 GiB of disposable `/dev/shm`. JAXSFT bootstrapped a
frozen run-local environment without modifying system Python, verified the
pinned Hugging Face cache, and passed:

- a five-step 25,152-parameter synthetic update across all four devices;
- a five-step, 752,393,024-parameter Qwen3.5/UltraChat SFT run with finite loss
  and gradient norms;
- a 7.52 GB full model/AdamW/data-cursor checkpoint, cold restore, and exact
  continued metric trajectory against an uninterrupted reference;
- artifact collection back to the controller.

A later OLMo 2 gate on a supplied v4-8 additionally loaded and compared the
1,484,916,736-parameter public checkpoint, completed three real UltraChat
updates, and proved a byte-identical tiny checkpoint resume. The host had no
system `uv`; this motivated the content-addressed controller bootstrap described
below.

A subsequent semantic-truncation gate materialized the pinned UltraChat split,
completed three more full OLMo 2 updates, and exited cleanly after distributed
shutdown. A follow-up synthetic run started from the empty lockfile left by the
previous clean libtpu exit and proved the conservative preflight recovery path.

The same host later completed two batch-identical 20-step OLMo 2 trajectory
gates against an independent stock Transformers/Accelerate CPU run. Production
BF16 used FP32 Adam moments plus global and explicit `HIGHEST` contraction
precision; its relative loss gap did not widen between equal post-update
halves. A `--force-fp32` controller lane set `JAXSFT_FORCE_FP32=1`, loaded all
parameters as FP32, and reduced maximum TPU/CPU relative loss error to 0.0214%.
The FP32 TPU compiler log showed FP32 contractions with highest/highest operand
precision. Exact hashes and drift statistics are in
[the trajectory result](results/olmo2_1b_trajectory_parity_20.json).

The real run's first backward compile took about 101 seconds. A measured steady
step took 0.403 seconds at 2,538 input tokens/s for length 256 and local batch
four. This is a correctness smoke, not a benchmark: it uses replicated full
parameters and the readable sequential DeltaNet kernel. The public,
hostname-free record is
[docs/results/qwen35_v4_8_smoke.json](results/qwen35_v4_8_smoke.json).
Checkpoint details are recorded separately in
[docs/results/qwen35_v4_8_resume_smoke.json](results/qwen35_v4_8_resume_smoke.json).

Operationally, TPU logs had to be directed to a writable run-local directory.
Before every launch, the controller requires `fuser`, refuses a live owner of
the accelerator or lockfile, validates that an existing libtpu lock is a
regular non-symlink zero-byte file, and only then unlinks it. This handles the
empty lockfile left by a clean libtpu exit without broad or privileged process
cleanup.

## 2. Controller modes

Support both:

- **in-slice controller:** the command runs on one configured worker, which also
  joins JAX; and
- **remote controller:** the command runs on a CPU/control machine and launches
  all workers over SSH without joining JAX.

Auto mode probes reported hostnames and must find exactly one local match or
none. Ambiguous matches fail. The artifact/checkpoint backend is independent of
controller mode.

## 3. Local profile

The implemented narrow profile fields are:

```toml
schema_version = 1
name = "local-four-host"
hosts = [
  "worker-0.example.internal",
  "worker-1.example.internal",
  "worker-2.example.internal",
  "worker-3.example.internal",
]
coordinator_host = "worker-0.example.internal"
coordinator_port = 12355

remote_workspace_root = "/var/tmp/jaxsft-runs"
remote_cache_root = "/dev/shm/.jaxsft-cache"
local_artifact_root = "artifacts/cluster"

[ssh]
user = ""
identity_file = ""
connect_timeout_seconds = 8
connection_attempts = 2
known_hosts_file = "/tmp/jaxsft-known-hosts"
```

Usernames, identity-file paths, storage credentials, Hub tokens, and cloud
secrets are local-only. Environment variable **names** may be configured;
secret values never enter recipe files, source capsules, stdout, or public
manifests.

## 4. Lifecycle

### Resolve and probe

1. Read an explicit host list and require unique entries.
2. Probe all entries concurrently through non-interactive OpenSSH; never
   create/copy/modify keys.
3. Record target-to-reported-hostname mapping and reject duplicates.
4. Probe required commands, environment, storage, and accelerator facts.
5. Run a tiny JAX program on all hosts to validate process count, local/global
   devices, and collective communication.

Transport status 255 may be retried for idempotent probes. An ordinary remote
command failure is returned immediately, not mislabeled as SSH flakiness.

### Capture source

Every run creates an immutable source capsule before remote mutation. It
contains:

- tracked files plus non-ignored untracked files from the exact working tree;
- entry program and resolved recipe;
- dependency lock and capsule manifest/hashes.

Tar ownership and timestamps are normalized, so identical source paths,
contents, and executable modes produce the same capsule bytes and source
identity across fresh clones.

Ignored caches, data, checkpoints, secrets, `.git`, and virtual environments are
excluded. Symlinks and files outside the checkout are rejected unless an
explicit safe inclusion rule materializes them into the capsule.

Unlike mirroring a live checkout into a shared fixed path, a run-specific
capsule avoids stale modules, concurrent-run interference, and the possibility
that a later edit mutates a running job.

### Synchronize

- Create a new run-specific remote directory beneath the validated workspace
  root; an existing target is a hard failure.
- Stream one tar capsule to all hosts and write its SHA-256 beside the source.
- Reuse a content-addressed, host-local dependency/model/token/data cache.
- Upload the controller's exact `uv` executable to a SHA-addressed mode-0700
  cache path when needed, then sync the frozen environment without changing
  host-wide packages. An incompatible controller/worker binary fails before
  launch rather than falling back to system package mutation.
- Key the reusable dependency environment by the exact `uv.lock` SHA-256.
- Materialize the pinned model snapshot and non-streaming dataset on every host
  before launch, record their resolved identities in `staging.json`, then set
  `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` for TPU execution.
- If the cache root is below `/dev/shm`, require passwordless `sudo` only for
  the dedicated ownership-sealing operation. Persistent cache roots use normal
  user ownership and do not require `sudo`.
- Never use `--delete` outside the exact run directory. Cache cleanup is a
  separate dry-run-first command.

### Launch

- Launch one identical command per host concurrently through OpenSSH.
- Pass a run ID, source path, resolved config path, and safe environment only.
- The worker calls `jax.distributed.initialize()` before `jax.devices()`,
  `local_device_count()`, mesh construction, or checkpoint setup.
- Runtime identity comes from `jax.process_index()` and `jax.process_count()`.
- Host suffix order is used only for inventory display.
- Each worker writes an early rank handshake and a run-specific PID file.
- TPU logs and temporary files are directed to writable paths inside the exact
  immutable run directory.
- A preflight refuses an active TPU/lock owner and safely recovers only an
  unowned empty `/tmp/libtpu_lockfile`.
- All ranks validate the same source/config hashes before compilation.
- `--force-fp32` adds only the strict `JAXSFT_FORCE_FP32=1` worker environment;
  the worker records recipe/effective dtypes, global matmul precision, and
  explicit contraction precision in its manifest, initialization event, and
  final result.

### Monitor and collect

- Preserve rank-specific stdout/stderr rather than interleaving away identity.
- Write structured heartbeat/step records atomically.
- Surface the first failing rank and retain all other rank tails.
- Periodically collect complete logs/manifests in remote-controller mode without
  copying temporary in-progress files.
- Treat successful launcher exit as necessary but insufficient: validate the
  final result, all-rank completion, checkpoint marker, and artifact hashes.

### Stop

1. Send an in-band stop or SIGTERM to PIDs recorded for this run.
2. Wait a bounded grace period and collect last logs/checkpoint status.
3. Escalate only the same recorded PIDs/process groups.
4. Report any unreachable host or surviving exact PID.

The implemented controller uses a 15-second default grace period (configurable
with `--grace-seconds`), re-reads the command line after that wait, and sends
SIGKILL only if the PID still belongs to the same immutable run.

Do not use `pkill python`, substring-only job matching, or a kill command that
can affect another run.

## 5. Doctor checks

`python cluster.py doctor` is read-only apart from an explicitly configured
known-hosts file and currently reports:

- expanded targets and reported hostnames;
- controller mode and local-host match;
- SSH latency/retry result and OpenSSH version;
- OS, kernel, architecture, Python, `uv`, `rsync`, and `pdsh` availability;
- installed JAX (when visible to system Python) and accelerator metadata;
- RAM, disk, cache free space, `/dev/shm` mount/type/size, and inode pressure;
- whether configured work/cache/artifact paths are local, shared, and writable;
- outbound Hub/object-store reachability without displaying credentials;
- clock skew sufficient to explain logs (not used for run identity);
- Multi-process collective/checkpoint behavior is covered by the explicit
  synthetic acceptance run rather than inferred from the metadata probe.

Any expected topology recorded in a cluster profile is an assertion checked
against these measurements.

## 6. Data placement

Code is small; tokenized instruction data and model weights are not. Use
content-addressed caches on every worker, keyed by immutable Hub revision and
preprocessing manifest. The options are:

- each host downloads/verifies identical source artifacts when outbound access
  is reliable;
- the controller downloads once and ships missing files;
- a shared/object store supplies source artifacts while local disks cache them.

Prepared data is deterministically sharded by global process identity. A host
must never substitute a different dataset because a cache is missing. Staging
and verification happen before measured training.

The validated profile uses `/dev/shm` only after the controller audit shows
ample capacity on the current TPU VM shape. It is disposable hot storage for
reconstructable caches, never the checkpoint backend. A root-owned dedicated
cache survives per-session `RemoveIPC` cleanup, but not VM pre-emption.

## 7. Checkpoint storage

Schema v4 implements strict restart for the current replicated data-parallel
topology. Every runtime rank removes its local pmap replica axis, computes a
container-independent SHA-256 over parameters and AdamW state, gathers that
identity across processes, and refuses the write unless all four semantic
states match. Each rank atomically writes its own payload and adjacent
completion marker under a common step directory. Payloads also contain the
rank-local deterministic data cursor, step/RNG cursor, exact source and recipe
identities, and exact runtime topology.

Resume always targets a new immutable run. The controller passes the common
step directory, and each process chooses the file named for its actual
`jax.process_index()`. Before trusted local pickle state is opened, the loader
checks the marker and file SHA-256; afterward it checks marker/payload agreement,
recomputes the semantic state hash, and validates optimizer, RNG, and data
cursors. The four-host cold-resume proof is recorded in
[ADR 0002](adr/0002-replicated-multihost-checkpoints.md).

This is intentionally not called a topology-portable sharded checkpoint. It
duplicates the replicated model/optimizer state in each rank file and requires
the same process count, local-device topology, source, and recipe on restore.
Model-axis partitioning needs a successor format with global-array metadata,
shard manifests, coordinated commit, and resharding tests. Hugging Face iterable
resume currently replays the pinned rank-local epoch prefix, which is exact but
linear in consumed rows; a captured shuffle-buffer cursor is the scaling path.

## 8. Testing without a real slice

Most orchestration tests use fake `ssh`, `pdsh`, and `rsync` executables and
temporary host roots to verify:

- host expansion/count validation;
- shell quoting and environment-name validation;
- source capsule exclusions and path containment;
- retries only for transport failures;
- failure propagation and log attribution;
- run-specific synchronization and stop targeting;
- artifact collection that skips temporary files;
- remote versus in-slice controller identity.

Forced multi-device CPU JAX tests cover rank-independent mesh/loss logic where
possible. Real-slice tests remain explicit, short, and destructive only inside
a generated run directory.

## 9. First slice acceptance test

Before loading a real model:

1. run doctor on all four targets;
2. sync a source capsule containing a toy linear model and synthetic batch;
3. initialize four JAX processes and print measured topology once per rank;
4. prove a global `psum` and a sharded parameter update;
5. save a toy optimizer/data-cursor checkpoint;
6. terminate all workers;
7. start a fresh four-host process set, restore, and perform the next update;
8. collect and verify every artifact hash from the controller.

This sequence passed on the measured v4-32 before the real OLMo 2 and Qwen3.5
runs were staged. Future topology or checkpoint-backend changes must repeat it.
