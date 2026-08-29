# Architecture decision records

Decisions that constrain implementation or experiment semantics are recorded
here as they become stable enough to outlive the initial vertical slice.

The first dedicated ADRs to extract from the implemented contracts are:

1. JAX module/parameter style (pure PyTrees versus the selected stable Flax API);
2. multi-host checkpoint and artifact backend;
3. canonical span-to-token boundary ownership;
4. source capsule and dirty-tree policy;
5. project name and license.

Each ADR states context, decision, alternatives, consequences, evidence/spike,
and reversal conditions. Open questions belong in `PLAN.md`, not in an ADR
pretending a choice has already been made.
