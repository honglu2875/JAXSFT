# Architecture decision records

Decisions that constrain implementation or experiment semantics are recorded
here as they become stable enough to outlive the initial vertical slice.

Accepted decisions:

- [ADR 0002: rank-local checkpoints for replicated multi-host training](0002-replicated-multihost-checkpoints.md)

The next dedicated ADRs to extract from the implemented contracts are:

1. JAX module/parameter style (pure PyTrees versus the selected stable Flax API);
2. canonical span-to-token boundary ownership;
3. source capsule and dirty-tree policy;
4. project name and license.

Each ADR states context, decision, alternatives, consequences, evidence/spike,
and reversal conditions. Open questions belong in `PLAN.md`, not in an ADR
pretending a choice has already been made.
