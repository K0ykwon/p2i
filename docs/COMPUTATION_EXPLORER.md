# Computation explorer

The observed Model IR stays at schema 0.1. ArchitectureIR, skill and Harness action schemas are unchanged. Version 0.4.0 adds an optional previous-observation snapshot and `/api/comparison`; saved Phase 1 IR remains loadable. No weights are stored in the snapshot.

## Recorded execution

Runtime steps include only executed runtime operations, sorted by recorded sequence. The same state drives the timeline, microscope, module highlight and graph flow. Previous/Next/Restart and the step slider pause playback. Clicking another operation selects its recorded step. Input → operator → output emphasis is illustrative dependency traversal, **not measured duration, GPU scheduling or a concurrent execution model**. Static/export/FX nodes never enter playback. Module entry calls remain available separately.

Paused graphs retain selection, pan, zoom, tensor inspection and dragging. CSS flow stops on pause and is disabled with `prefers-reduced-motion`. Manual step changes still work. Graphs with only hook evidence retain call inspection; no operation ordering is invented.

## Visual grammar

| Object or state | Treatment | Meaning |
|---|---|---|
| Module | Hierarchy/architecture component and origin badge | Actual `nn.Module`, not inferred semantics |
| Runtime operation | Solid rounded rectangle | Observed dispatched operation |
| Export/FX operation | Dashed outline | Static tracing evidence |
| Tensor | Shape-labelled dependency edge and tensor card | Recorded producer/consumer relationship |
| Selected object | Teal border | User inspection focus |
| Active execution | Amber border; unrelated nodes dim | Current recorded runtime step |
| Moving dashed edge | Input/output phase, only while playing | Direction along actual tensor dependency |
| Captured numeric value | Teal positive, amber negative | Sign and sample-relative magnitude; not importance |
| Missing captured coordinate | Hatched, disabled cell | Value unavailable, not zero |
| Before/after | Labelled observed revisions and constructor diff | Two successful traces, not a proposed execution |

Parameter counts/configuration are labelled metadata, not fabricated graph nodes. Origin comes from module provenance and may be unknown. Shape axis labels are numeric; no token/head/channel semantics are inferred.

## Local graphs and tensors

Default neighborhoods contain one operation and up to six directly connected operations. All operations remain in the selector. Raw view starts at 120 nodes and can intentionally expand to 500. Topological layers do not depend on source listing order; separate edge ports reveal fan-out/fan-in. This is a simple DAG layout, not crossing-minimal layout. Drag offsets persist for the current observation in this browser session. Retracing produces a new observation and resets offsets. Fit keeps scale at least 80%; oversized scopes require panning, not illegible shrinking.

Scalars use one value, vectors a strip, matrices a bounded heatmap. Higher ranks expose fixed-axis indices plus row/column offsets for the last two axes. Each displayed value uses the original row-major coordinate. At most 64 cells are displayed. Symbolic shapes fall back to labelled flat samples. Original shape, capture count and truncation are explicit. A prefix sample cannot supply arbitrary slices: those cells stay unavailable. CPU capture remains opt-in with the existing hard size/count limits. This is not a full activation store.

## Comparing observations

The Harness retains one previous successful observation, its constructor metadata and bounded inspection sidecar. Failed validation or tracing leaves both observed snapshots intact. Compare selects the same qualified module scope on both sides. Operation navigation aligns by module path, module-call occurrence, operator type and operator occurrence. Tensor outputs additionally match by output slot. Transient tensor/operation IDs are never used to equate values across traces.

Alignment is structural correspondence, not proof of semantic equivalence. Renamed modules appear added/removed. Repeated identical operators can be ambiguous after insertions; ordering-based alignment must be interpreted cautiously. Path changes are operation-to-operation dependency pairs, not a count of parallel tensor edges. Shape/dtype/sample changes are reported only for matched outputs. Unmatched operations and paths are explicitly added/removed. Random execution can change samples without an architecture edit; no causal claim is made. Full graph matching, synchronized zoom/drag and arbitrary activation comparison remain future work.
