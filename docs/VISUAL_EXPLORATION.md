# Focused visual exploration

This UI update uses the existing observed IR and bounded inspection sidecar. It does
not change the observed JSON schema, model analysis, Harness transactions or skills.

## Explore

Discovered child modules appear as clickable diagram nodes, with shapes, parameter
counts and origin. Arrows exist only when one child/subtree's recorded output tensor
is an input to another child/subtree at a later recorded call. Spatial placement alone
is not a dependency. Missing arrows do not prove independence: parent-level operations
may connect children indirectly. Those parent operations are listed separately and
open the computation view. No model-specific architecture layouts are used.

Click a component to descend; breadcrumbs return to its ancestors. Unexecuted modules
and containers remain inspectable. Repeated blocks are the actual registered instances.

## Computation

The default view is one selected operation plus at most six directly connected
neighbors. Input tensors, outputs, dtype/device and recorded dimensions appear in the
central explanation, with changed axis sizes emphasized. Scalar arguments appear only
when actually captured. Previous/Next and the operation selector navigate trace listing
order, which is explicitly distinct from a proven tensor dependency.

The local graph draws only IR edges, includes boundary tensor buttons and supports
click selection, pan and zoom. Fit scope retains at least 80% scale to avoid illegible
nodes; pan for overflow. Raw graph retains the original scope-wide inspection mode,
with its 120-node initial cap and intentional expansion up to 500 nodes. Neighbors
outside the loaded slice remain accessible through tensor consumers and a wider scope.

The central explanation uses reusable primitive labels (shape transformation,
convolution, matrix/projection, addition, softmax, or generic operation). Fused
operators stay fused. No semantic attention or residual computation is invented.

## Tensor values

Enable **Capture bounded whole-tensor CPU heatmaps on re-trace** in Edit and run
**Build & Re-trace**. Captured operation inputs and outputs can show a heatmap;
others retain shape metadata. An eligible rank-2 plane up to 32×32 has one colored
cell per element, so a 32×32 plane has smaller cells than a 16×16 plane in the
same footprint. Above the grid limit each cell is a regional mean, with leading
axes averaged for higher ranks. A saved prefix from an older sidecar is explicitly
labeled as such and never represented as a complete matrix. Hover/focus can reveal
a cell's numerical value without showing numbers throughout the grid. Statistics
refer to the full eligible tensor; colors use the displayed grid's symmetric scale.
Unavailable tensors have an explicit omission message. Static graphs without runtime
capture display metadata without invented values.

## Runtime

Previous/Next/Play/Pause walks actual recorded module-entry order. The active module
is highlighted and its ancestors expanded in the hierarchy, without changing the
user's chosen scope. The current call's actual primitive operations appear in the same
focused graph/explanation view. A call without its own primitive operators directs
users to recorded child calls. The optional call list remains available.

Playback is a replay of trace evidence, not a live forward pass or timing simulation.
Switching views preserves the selected scope. Reduced-motion settings remain honored;
no continuous decorative animation is added.

## Verification

`python tests/browser_acceptance.py` runs the existing explorer, Phase 2 and Phase 3
browser checks plus the new live UX checks for Transformer, CNN and MLP. No API data is
mocked in the new UX checks. The legacy large-graph fixture intentionally remains a
synthetic IR stress test. The pan test scrolls the actual graph into view before dragging.

New coverage includes known-tensor module edges, bounded neighborhoods, minimum readable
fit, raw/focused switching, operation navigation, real opt-in tensor capture/hover,
recorded call highlighting and playback controls. Screenshots are saved as `docs/ux-*.png`.
