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
| Mean-grid cell | Teal positive, amber negative | Signed regional mean over the actual tensor, not importance |
| Cell without finite inputs | Hatched, disabled cell | Regional mean unavailable, not zero |
| Before/after | Labelled observed revisions and constructor diff | Two successful traces, not a proposed execution |

Parameter counts/configuration are labelled metadata, not fabricated graph nodes. Origin comes from module provenance and may be unknown. Shape axis labels are numeric; no token/head/channel semantics are inferred.

## Local graphs and tensors

Default neighborhoods contain one operation and up to six directly connected operations. All operations remain in the selector. Raw view starts at 120 nodes and can intentionally expand to 500. Topological layers do not depend on source listing order; separate edge ports reveal fan-out/fan-in. This is a simple DAG layout, not crossing-minimal layout. Drag offsets persist for the current observation in this browser session. Retracing produces a new observation and resets offsets. Fit keeps scale at least 80%; oversized scopes require panning, not illegible shrinking.

Scalars use one cell, vectors a strip and matrices a bounded rectangular heatmap. For rank three and above, leading axes are explicitly averaged; the final two axes are spatially pooled. The default grid preserves every position when both final axes fit within 32: a 32×32 plane renders 32×32 cells, and 32×16 renders 32×16 cells in the same outer footprint, so each cell is smaller than in a 16×16 plane. Only axes exceeding 32 use a common downsampling factor and regional means. The card retains its available width; cells adjust their width, and short/wide grids may occupy less height. Each cell represents the mean of its contributing finite elements. Colour ranges are based on the largest absolute grid mean, with a centred zero legend; cells with no finite data are marked unavailable. No semantic axis names are inferred.

The optional CPU sidecar stores at most a 32×32 grid per tensor and whole-tensor finite-value statistics; it stores no full activation tensor in the Model IR. Normal capture is limited to the first 64 observed tensors and 262,144 elements per tensor. Unsupported devices/layouts, empty tensors and tensors above the size threshold show shape and a clear omission message. Earlier sidecars containing explicit prefixes still render as labelled legacy strips. The observed Model IR and its JSON schema remain unchanged. A 32×32 input retains all 1,024 individual finite cell values in the opt-in inspection sidecar; above the 32-axis limit only regional means are retained. This is not a guarantee that every tensor in a model has a heatmap.

## Comparing observations

The Harness retains one previous successful observation, its constructor metadata and bounded inspection sidecar. Failed validation or tracing leaves both observed snapshots intact. Compare selects the same qualified module scope on both sides. Operation navigation aligns by module path, module-call occurrence, operator type and operator occurrence. Tensor outputs additionally match by output slot. Transient tensor/operation IDs are never used to equate values across traces.

Alignment is structural correspondence, not proof of semantic equivalence. Renamed modules appear added/removed. Repeated identical operators can be ambiguous after insertions; ordering-based alignment must be interpreted cautiously. Path changes are operation-to-operation dependency pairs, not a count of parallel tensor edges. Shape/dtype/grid changes are reported only for matched outputs. Unmatched operations and paths are explicitly added/removed. Random execution can change grid means without an architecture edit; no causal claim is made. Full graph matching, synchronized zoom/drag and arbitrary activation comparison remain future work.
