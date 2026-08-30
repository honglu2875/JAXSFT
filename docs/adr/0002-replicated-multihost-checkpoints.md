# ADR 0002: rank-local checkpoints for replicated multi-host training

- Status: accepted for the replicated data-parallel baseline
- Date: 2026-08-30

## Context

JAXSFT's first four-host path replicates the complete model and optimizer on
every TPU device while assigning a distinct deterministic data cursor to each
JAX process. A single process-0 file would either discard the other ranks'
input cursors or require an unnecessary gather. Host suffix order is also not a
safe rank identity: only `jax.process_index()` identifies the runtime process.

The initial requirement is exact restart on the same measured four-process
topology. Model-axis partitioning and topology-independent resharding are
separate future requirements.

## Decision

Checkpoint schema v4 writes one file per runtime rank beneath a common step
directory:

```text
checkpoints/step-00000002/
  rank-000.pkl
  rank-000.complete.json
  ...
  rank-003.pkl
  rank-003.complete.json
```

Before writing, every process removes its local pmap replica axis, hashes the
complete model and optimizer tree in a container-independent format, gathers
the hash in runtime-rank order, and refuses the checkpoint unless all hashes
are identical. Each rank then stores that common semantic-state hash together
with its own topology and data cursor. A temporary payload is atomically
promoted, and its adjacent completion marker is written last.

The controller resumes into a new immutable run and passes the common step
directory to every worker. Each worker selects `rank-{jax.process_index():03d}`
and checks, in order:

1. completion-marker schema, recipe, source, and exact topology;
2. payload file SHA-256 before opening the trusted local pickle;
3. payload/marker metadata agreement;
4. the recomputed semantic model/optimizer SHA-256;
5. optimizer, RNG, and rank-local data-cursor invariants.

Run metadata and checkpoints live on persistent storage. Large reproducible
model, dataset, environment, and XLA caches may live in `/dev/shm`; when they
do, the controller seals that dedicated cache root-owned between SSH sessions
to survive `systemd-logind` `RemoveIPC=yes` cleanup.

## Alternatives considered

- A process-0 checkpoint was rejected because it cannot preserve every
  rank-local data cursor and would make replicated state gathering a needless
  memory spike.
- A shared Orbax/TensorStore model-axis format remains attractive once the
  trainer has real partition specs, but adopting it before model-axis sharding
  would add machinery without proving the current baseline.
- Keeping all state only in `/dev/shm` was rejected because SSH logout cleanup
  and VM pre-emption can remove it.

## Consequences

The format gives strict, inspectable restart semantics for replicated training
and lets artifacts be collected independently from each host. It deliberately
requires the same process count, runtime-rank topology, source capsule, recipe,
and checkpoint step. The per-rank payloads duplicate model and optimizer bytes,
and pickle files are trusted local artifacts rather than an interchange format.

When model-axis sharding arrives, this ADR does not bless copying the same
format forward. A successor must define global array metadata, shard manifests,
atomic commit across hosts or object storage, and cold resharding tests.

## Evidence

On a four-host v4-32, a synthetic run stopped after step two and wrote all four
rank files with replicated-state SHA-256
`dacbbf51c7165583938aa4ac0b1a96fe1ac36f0a0f5b747d99bd36e27a4eeb59`.
A fresh four-process job restored them and completed step three. Its final state
hash, `4ae7f2ec4023763036a0f6809873779add20de9750116fbe43ab681c9d5ceab1`,
exactly matched a same-source uninterrupted run, as did every step-three
metric. See [the sanitized acceptance record](../results/v4_32_multihost_acceptance.json).
