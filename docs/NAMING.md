# Naming notes

The user selected **JAXSFT** on 2026-08-29. The repository directory and Python
package name are `jaxsft`; the project display name is `JAXSFT`.

## Candidates

| Name | What it communicates | Trade-off |
|---|---|---|
| **SpanTune** | The core research object is an annotated semantic/token span. | Less explicit about JAX in the name. |
| **TurnWeaver** | Many conversation and tool dialects are woven into one typed turn stream. | Sounds data-focused and may undersell model work. |
| **JAXSFT** | Immediately searchable and technically literal. | Generic and difficult to own as a project identity. |
| **SFTax** | Compact combination of SFT and JAX. | Pronunciation is ambiguous and resembles “S.F. tax.” |
| **TraceTune** | Tool trajectories and provenance remain traceable through loss. | “Trace” can be confused with JAX tracing/profiling. |

## Decision

Use **JAXSFT**. It is technically literal, searchable, and leaves dataset,
model, objective, and orchestration scope equally visible. The CLI and import
name are both `jaxsft`.

The lightweight web/package search performed on 2026-08-29 found no prominent
machine-learning project using JAXSFT. This is not legal or trademark
clearance. Recheck GitHub, PyPI, domains, and relevant trademark databases
immediately before publication.
