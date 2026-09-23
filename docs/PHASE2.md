# Phase 2: deterministic architecture harness

p2i 0.2 adds an editable architecture layer to the existing analyzer. Observed `ModelIR` stays at schema **0.1**; old JSON files and the original trace/save/serve workflows still load. Editable `ArchitectureIR` and actions use their own schema **0.1**.

The loop is: model → observed evidence → constructor architecture → structured edit → validation → build → existing Phase 1 trace → updated evidence. No AI, natural-language commands, automatic code generation, semantic architecture recognition, source rewriting or remote code loading is involved.

## Start the editable explorer

```bash
python -m pip install -e '.[test]'
p2i demo transformer --edit
# http://127.0.0.1:8000
```

`p2i demo mlp`, `p2i demo cnn`, `p2i demo transformer` and `p2i serve model.json` retain observation-only behavior. `--edit` explicitly attaches a live model, stored examples and a Harness session. The prebuilt frontend is included in the wheel; Node is only needed to develop it.

```python
import torch
import p2i
from p2i.examples.transformer import TinyTransformer

x = torch.randn(2, 16, 64)
h = p2i.Harness(TinyTransformer().eval(), example_inputs=(x,))
original = h.retrace()
print(h.observe())
p2i.serve(h)
```

Passing `ir=existing_ir` to `Harness` reuses an existing observation. Harness construction does not automatically trace. `example_args` / `example_kwargs` work like Phase 1. Use `example_inputs=()` for a zero-argument model; omitted inputs mean constructor-only validation.

## Separate representations

| Layer | Owns | Does not claim |
| --- | --- | --- |
| ModelIR | Observed module identities, runtime calls, ATen/export/FX operations, tensor metadata | Source code or an editable program |
| ArchitectureIR | Stable architecture node IDs, constructor parameters, ordered registration edges, contracts and provenance | Arbitrary reconstruction of dynamic Python |
| Builder | Adapter construction, grafting registered children, deterministic new state and compatible state transfer | Inference of hidden dependencies |
| ValidationReport | Schema/structure/construction checks, explicit Sequential shape checks, optional forward/backward checks | Proof for every possible input |
| Latest observation | Successful re-trace of a built model, observed constructor sidecar, optional bounded runtime samples | That a pending edit already ran |

Architecture node IDs (`a0`, `a1`, …) are independent of observed IDs (`m0`, …). Shared modules have one architecture node and multiple registration edges. Edits to that node affect every alias. `h.provenance()` links architecture IDs, current build paths, original observed module IDs, latest observed module IDs and the revision of observed tensor contracts.

The observed constructor sidecar is extracted from the actual supplied/built modules using adapters and exposed by the live UI/API. It is kept separate from ModelIR to preserve its schema. Dimension-changing edits also appear directly in the new ModelIR's actual tensor shapes and parameter counts.

## Manual MLP edit

```python
import torch
import p2i
from p2i.examples.simple_mlp import SimpleMLP

h = p2i.Harness(SimpleMLP().eval(), example_inputs=(torch.randn(2, 16),))
h.retrace()

# Two explicit dimensions change together. No dependencies are guessed.
action = {
    "version": "0.1",
    "type": "batch",
    "expected_revision": 0,
    "actions": [
        {"type": "set_parameter", "target": "layers.0",
         "parameter": "out_features", "value": 24},
        {"type": "set_parameter", "target": "layers.2",
         "parameter": "in_features", "value": 24}
    ]
}
preview = h.preview(action)
assert preview.success
assert h.observe()["revision"] == 0
result = h.apply(action)
assert result.success and result.committed
report = h.validate(backward=True)
assert report.constructor_valid and report.forward_valid and report.backward_valid
built = h.build()
updated = h.retrace()
print(h.diff_json(from_revision=0, to_revision=1))
print(h.last_weights.model_dump())
```

The Transformer acceptance example changes `blocks.0.mlp.0.out_features` and `blocks.0.mlp.2.in_features` from 128 to 96 in one batch. The CNN example rejects a lone 16→24 output-channel change, then accepts a compatible convolution padding-mode change. Run all three:

```bash
python examples/harness_edit.py
```

This produces architecture, action, observed IR and validation/comparison artifacts in `artifacts/phase2/`.

## Public APIs

- `Harness(model, ir=None, example_inputs=..., example_kwargs=..., registry=..., seed=0)`
- `h.observe(level="summary")`: revision, observed revision, node counts, **observed** parameter count, recent edits and validation.
- `h.observe(level="architecture")`, `h.observe(node_id="a9")`: structured editable nodes, current parameters, restrictions and contracts. A qualified path also resolves a target.
- `h.architecture()`: a defensive copy; `.save(path)` / `ArchitectureIR.load(path)` round-trip JSON.
- `h.preview(action)`: perform the same validation as apply, without committing.
- `h.apply(action)`: transactional validation and commit. Normal invalid requests return `ActionResult`, not raw internal exceptions.
- `h.validate(backward=False)`: validate current architecture. No new trace is created.
- `h.build()`: produce a live model, recording `h.last_weights`. It checks construction; call `validate` to require forward success.
- `h.retrace(capture_values=False, **phase1_trace_options)`: require forward validation, run the existing tracer, then publish the observation only after successful runtime capture.
- `h.history()`, `h.diff(from_revision=..., to_revision=...)`, `h.undo()`, `h.redo()`.
- `h.observe_json()`, `h.validation_json()`, `h.diff_json()` produce machine-readable JSON.
- `p2i.build(architecture, registry=...)`: standalone deterministic construction for fully adapter-backed architectures, with fresh weights.
- `Harness.from_architecture(architecture, example_inputs=...)`: artifact-only operation without learned state or custom live objects.

`latest_ir`, `observed_revision`, `last_validation`, `last_weights`, `last_comparison` and `observed_configuration` provide session evidence. Treat these evidence objects as read-only. A new build is not automatically adopted as a newly trained parameter state.

## Structured actions

All actions accept `version: "0.1"` and optional integer `expected_revision`. Stale revision checks are performed inside the session lock. The JSON boundary only resolves registered adapter names; it never imports modules from a provided string.

| Type | Required fields | Constraints |
| --- | --- | --- |
| `set_parameter` | `target`, `parameter`, `value` | Supported editable constructor parameter |
| `replace_module` | `target`, `replacement: {module_type, parameters}` | Registered replacement adapter; entire old subtree replaced |
| `insert_module` | `parent`, `index`, `module: {module_type, parameters}` | Sequential or ModuleList parent |
| `remove_module` | `target` | Every registration parent must be Sequential/ModuleList; root cannot be removed |
| `wrap_module` | `target`, `module`, optional `position: "before" / "after"` | Sequential wrapper; forward validation must succeed |
| `batch` | `actions` | Atomic list of the above; up to 100 sub-actions |

```python
from p2i.actions import SetParameter
result = h.apply(SetParameter(target="layers.0", parameter="bias", value=False))
print(result.model_dump_json())
```

A successful action means schema/structure/construction validation passed, plus forward validation when examples are available. `architecture_valid` specifically describes schema/structure/construction; check `success` and `validation.forward_valid` for the complete transaction outcome. Failed forward validation rolls back. Preview success does not change the revision. No-op edits do not create revisions.

## Adapter coverage and custom classes

Built-in adapters cover Linear, Conv1d, Conv2d, ReLU, GELU, SiLU, Dropout, LayerNorm, BatchNorm1d, BatchNorm2d, Embedding, MultiheadAttention, Sequential, ModuleList, Identity, Flatten, MaxPool2d, AvgPool2d and AdaptiveAvgPool2d.

Exact module types are matched to avoid silently treating a subclass with a different forward implementation as its standard superclass. MultiheadAttention owns its internal `out_proj`; edits to that implementation detail are rejected. Edit the parent constructor instead. Replacements created by an owning adapter reveal their internal modules in the subsequent runtime trace.

Unknown classes have read-only constructors. In a live session the builder retains their trusted Python implementation through a private deep copy and grafts rebuilt registered children into it. Such nodes can be replaced or wrapped. They cannot be reconstructed from architecture JSON alone without a registered adapter. Direct source rewriting and arbitrary unknown constructor introspection are not attempted.

```python
from torch import nn
import p2i

class Scale(nn.Module):
    def __init__(self, factor=2.0):
        super().__init__()
        self.factor = factor
    def forward(self, x):
        return x * self.factor

@p2i.register_module_adapter(Scale, module_type="example.Scale")
class ScaleAdapter(p2i.ModuleAdapter):
    def extract(self, module):
        return {"factor": module.factor}
    def validate(self, parameters):
        if type(parameters.get("factor")) not in (float, int):
            raise ValueError("factor must be numeric")
    def build(self, parameters):
        self.validate(parameters)
        return Scale(**parameters)
```

For local isolation, use `p2i.architecture.registry.registry.copy()` and pass it to `Harness(..., registry=custom)`, or create a `ModuleAdapterRegistry` and call `register(module_type, adapter)`. Adapters may declare `container=True` for child grafting or `owns_children=True` for constructor-managed internals. Adapter code is explicitly registered trusted Python, never provided via action JSON.

## Validation and learned state

Schema/structure checks validate root/children, references, registration names, uniqueness, reachability and cycles. Adapter checks validate allowed constructor fields and practical type/range constraints; the constructor itself must instantiate. Explicit Sequential Linear connections through known elementwise activations are checked statically. Other shape constraints are checked by real forward execution on stored examples.

Runtime validation records category, failing module where identifiable, exception type/message, a short traceback and an obvious suggested fix. Optional backward validation sums differentiable output tensors (real part for complex outputs) into a scalar and runs backward. It is a gradient-path sanity check, not a task loss or training-quality check.

Validation runs on copies and restores PyTorch RNG state. The original model, example inputs and learned state remain the session baseline. Identical-shaped state entries are copied by stable node identity. Unchanged modules and shape-compatible replacements retain their state. Changed shapes receive deterministic initialization derived from the session seed and architecture node ID; both the old dropped entry and new initialized entry are reported. Partial tensor copying is intentionally not performed (`partially_preserved` remains empty). Shared modules and compatible tied parameter objects are preserved; conflicting tied shapes are rejected.

State dict buffers and parameter `requires_grad` flags are retained when compatible. New modules without old state use PyTorch defaults. Weights stay in memory and are not placed in architecture JSON. State from externally training a returned build is not automatically ingested; a new Harness can establish that model as a new baseline.

Undo/redo restore architecture snapshots and create **new monotonically increasing revisions**, avoiding stale-agent revision reuse. Snapshots contain only architecture metadata, not model-weight copies. Stable IDs preserve state when ModuleList indices move. Diffs contain node additions/removals, constructor changes and child registration changes. Observed evidence remains at its last successful revision until another retrace.

## Artifact CLI and external agents

```bash
p2i harness inspect architecture.json
p2i harness apply architecture.json action.json --output updated.json
p2i harness apply architecture.json action.json --output unused.json --preview
p2i harness validate updated.json --input-shape 2 16
p2i harness diff architecture.json updated.json
```

Commands emit JSON to stdout and exit 1 for a rejected action or invalid architecture. `--input-shape` adds one zero-valued float32 example; without it, shape validation only covers explicitly known Sequential constraints and construction. No pickle/checkpoint loading or Python-module importing is performed.

CLI artifacts must be fully reconstructible through the available adapters. The bundled MLP's `layers` Sequential is a convenient portable model; the custom outer demo classes require a live session or a user adapter. `artifacts/phase2/portable_mlp.architecture.json` demonstrates the portable path. Architecture-only builds have deterministic new state; they do not carry weights across CLI processes. CLI diffs take two saved files; full in-memory undo history is not persisted.

An external agent can observe JSON, propose an allowlisted action with `expected_revision`, preview it, apply it, and inspect reports. API keys, providers, prompts and agent implementations are absent.

## Explorer interaction

The original hierarchy and raw computation modes remain available. New default **Explore** cards show actual registered children, prominent shapes, origin and direct-call counts. Card position denotes containment, not inferred execution order. ModuleList containers group their real children without asserting structural equivalence. Breadcrumbs support progressive drill-down for any module class.

**Computation** focuses on the selected subtree, displays graph-slice boundary tensors and shape-labelled dependency edges, and retains pan/zoom/node limits. Selecting an operation opens a primitive-level shape explanation; matrix products, shape changes, convolutions and additions get concise operation-category treatment. A fused operation stays fused. No additions are labelled as residual connections.

**Runtime** steps through actual observed module entry records with Previous/Next/Play/Pause, IO shapes and completion state. It does not fabricate operation timings or imply independent calls ran concurrently.

**Edit** displays supported constructor fields with type-appropriate controls, current/proposed values, a structured diff and validation result. Advanced JSON supports all actions and coordinated batches. Commit is enabled only after a successful preview; changing the proposal invalidates the preview. The live architecture and observed revisions are visibly distinct until Build & Re-trace succeeds. History exposes structured diffs, undo/redo, runtime comparisons and weight-transfer reports. All views share the selected module scope.

Optional bounded CPU heatmaps can be enabled for Build & Re-trace. They are stored in a separate in-memory sidecar, not saved ModelIR. Defaults: at most 64 tensors, 262,144 elements per tensor, a 32×32 grid per tensor, and 256 operations' compact scalar arguments. Within the grid bounds each 2D element has its own cell; larger axes are pooled into regional means. The original activation tensor is not retained, though small grid cells contain individual numerical values. Hard caps prevent increasing these without limit. Large/device/layout-incompatible tensors are omitted with a reason. Inspection arithmetic is excluded from the model operation trace. Without opt-in, the original metadata-only behavior remains.

Python can use `RuntimeInspection` from `p2i.analyze.inspection` with `p2i.trace(..., inspection=capture)` and inspect `capture.as_dict()`.

Animations are brief and respect `prefers-reduced-motion`. No model name is used to select a layout or substitute a predefined architecture diagram.

## REST API

Phase 1 routes remain. New read-only routes: `GET /api/harness`, `GET /api/inspection`. A live Harness adds JSON `POST /api/harness/{preview,apply,validate,undo,redo,retrace}`. Retrace returns the updated ModelIR only on success. Mutations reject mismatched browser origins and require JSON. Default binding stays `127.0.0.1`. There are no remote ingestion or source execution endpoints.

## Tests, limitations and next phase

See [PHASE2_VALIDATION.md](PHASE2_VALIDATION.md) for exact executed results and the current environment's blocked browser/socket gates. Run the complete gates in an environment permitting local sockets:

```bash
python -m pytest -q
cd frontend && npm ci && npm run build && cd ..
python tests/browser_acceptance.py
python examples/harness_edit.py
python -m pip wheel . --no-deps -w dist
```

Browser setup (`npx playwright install chromium --only-shell`) is an explicit developer step. `tests/browser_acceptance.py` runs the unchanged Phase 1 browser script, then Phase 2 editing and exploration for Transformer, CNN and MLP against real local servers.

Remaining limits: example-specific forward validation, no general dimension-dependency inference, no arbitrary graph rewiring, no persisted learned-state/session manager, incomplete shared-storage alias analysis inherited from Phase 1, CPU-only validation, and copying costs for large/custom models. Mixed-device builds and arbitrary user side effects are not sandboxed or certified. A custom wrapper may retain non-registered references/caches that a constructor adapter should handle; successful validation only establishes behavior on the supplied examples. Internal nodes owned by adapters remain constructor-managed. HTTP edit requests run synchronously within the local server event loop; it is designed for one research session, not multi-user throughput.

Recommended Phase 3: define a versioned Skill Registry on top of adapter contracts, then an external-agent tool interface. Add bounded partial code generation only behind explicit registration and validation gates. Track provenance and reproducible acceptance evidence before introducing retrieval/promotion or Voyager-style skill discovery. These are recommendations, not implemented features.
