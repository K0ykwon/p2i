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

Enable **Capture bounded CPU tensor samples on re-trace** in Edit and run **Build &
Re-trace**. The backend's existing opt-in bounds remain unchanged. The first captured
input/output in an operation can show a heatmap; other tensors retain compact shape
buttons and their samples can be inspected in the tensor panel.

A complete rank-2 tensor is labeled Complete matrix only if every element is present
and it has at most 16 columns. Otherwise cells are explicitly labeled flat-order values
or a flat prefix sample. A prefix is never represented as a complete attention map.
Hover/focus gives the recorded numerical value. Statistics are the backend's tensor
statistics; colors use the displayed sample's symmetric scale. At most 64 cells render.
Unavailable values have an explicit omitted/not-captured explanation. Static graphs
without runtime samples display metadata without invented values.

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
