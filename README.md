# p2i — PyTorch to interactive inspection

A local-first research prototype: **instantiated `nn.Module` + example inputs → analysis → typed Model IR → local browser explorer**.

The Phase 1 analyzer inspects the actual supplied model. Phase 2 adds deterministic architecture editing through an optional Harness. Phase 3 adds an explicit Skill Registry, external-agent tools and bounded candidate validation. P2I does **not** autonomously recognize semantic architectures, generate code, discover skills, integrate AI, parse papers or fetch repositories. Bundled neural networks are ordinary examples, not visualization templates. Candidate source is deliberately supplied by the caller; no skill dependency is automatically installed.

## Phase 2: edit, validate, build and re-trace

```bash
p2i demo transformer --edit
```

Use `p2i.Harness` for typed/JSON actions, atomic batches, previews, forward/backward validation, deterministic rebuilding with weight-transfer reports, revisions, diffs and undo/redo. The local UI adds Explore, Runtime and Edit views while retaining the original hierarchy and computation explorer.

Read **[the Phase 2 guide](docs/PHASE2.md)** for the public API, adapters, examples and artifact CLI. **[Historical Phase 2 validation](docs/PHASE2_VALIDATION.md)** records the earlier environment limitations. **[Phase 3 validation](docs/PHASE3_VALIDATION.md)** records current regression and real-browser results. Original Phase 1 workflows remain optional and unchanged.

## Install and run

Python 3.11+; PyTorch 2.5+ (tested with 2.14.0). Create a virtual environment, activate it, then:

```bash
python -m pip install -e '.[test]'
p2i demo transformer
```

Open **http://127.0.0.1:8000**. The server blocks until Ctrl+C. The delivered source archive and wheel include the production frontend: Node is not needed to use them. Choose your platform's appropriate PyTorch wheel before installing p2i if necessary.

```python
import torch
import p2i
from examples.transformer import TinyTransformer

model = TinyTransformer().eval()
x = torch.randn(2, 16, 64)
ir = p2i.trace(model, example_inputs=(x,))
assert ir.modules and ir.tensors
ir.save("tiny_transformer.json")
p2i.serve(ir)
```

`examples` imports work from the repository root. Installed packages also expose the same models under `p2i.examples`.

## Python API

```python
ir = p2i.trace(
    model,
    example_args=(input_ids,),
    example_kwargs={"attention_mask": mask},
)
ir.save("model.json")
ir = p2i.load("model.json")
p2i.serve(ir, host="127.0.0.1", port=8000)
```

Use `example_inputs` or `example_args`, not both. Positional inputs must be a tuple; kwargs must be a dict. Neither is required for a zero-input model.

Additional trace options:

- `static=False`: skip export/FX; retain hooks and observed ATen operations.
- `dynamic_shapes=...`: forward PyTorch's dynamic-shape specification to export. Symbolic dimensions are strings in the static IR; runtime dimensions describe the actual input.
- `isolate=True` (default): deep-copy the model and example inputs before execution, preserving their shared references. Training/eval mode is respected, execution uses `torch.no_grad()`, and PyTorch RNG state is restored. Set `.eval()` yourself when desired. Model copying failures raise a clear error; `isolate=False` explicitly permits forward-pass mutations to the original objects.

Inputs normally should be leaf tensors. Custom model objects must support deepcopy for isolation. Static backends always use fresh copies and may fail independently.

The package never retains full tensor payloads in the IR. `ir.metadata.analysis` records success/partial/failure/skipped for each technique, with diagnostic messages. Runtime errors produce a partial IR, including incomplete calls, and a warning rather than invented execution results. Invalid API inputs and unreadable/invalid JSON raise errors.

## CLI

```bash
p2i --help
p2i demo mlp
p2i demo cnn --port 8001
p2i demo transformer --save tiny_transformer.json --no-serve
p2i serve tiny_transformer.json
```

`python -m p2i.cli ...` is equivalent if the script directory is not on PATH.

## Explorer

- Expand/collapse the module tree, then select a module to focus its subtree.
- Hierarchy cards show real child modules, recursive parameter counts, execution and observed shapes.
- Switch to **Computation** for runtime ATen, export, or FX graphs. Graphs are deliberately separate, not deduplicated or overlaid.
- Include/exclude descendants, pan by dragging the empty canvas, and zoom with the wheel or buttons, or use **Fit scope**.
- Select operations, graph edges, or inspector tensor buttons to inspect data flow, shape, dtype, device and provenance.
- Origin filtering retains ancestors as navigation context. A shared instance is shown with ↗ and has one module ID and multiple per-call runtime records.
- Graphs initially render at most 120 operations, with explicit expansion up to 500. Narrow the module scope for larger graphs. The banner explains omitted nodes/edges; the inspector retains full tensor consumers.
- The analysis report exposes backend failures. “Not observed” does not assert that a module can never execute. Containers such as `ModuleList` are often not called even when their children execute.

## Model IR, schema 0.1

`src/p2i/ir/model.py` is the canonical Pydantic schema, with strict fields and reference validation. `ModelIR.model_json_schema()` exposes its JSON schema.

| Structure | Meaning |
| --- | --- |
| `modules`, `module_edges` | Registered module instance graph. One ID per object, canonical parent plus shared registration edges and aliases. |
| `runtime.calls` | Individual observed module invocations, entry order, nesting and tensor IO. Repeated calls remain distinct. |
| `operations` | Observed ATen dispatches or static export/FX nodes, tagged by graph, with module mapping when known. |
| `tensors` | Metadata-only values in one graph, producer/consumer references and optional previous Python-object version. |
| `data_edges` | Tensor-labelled producer→consumer operation edges. External inputs/parameters have no producer operation. |
| `metadata.analysis` | Backend outcomes and diagnostic messages. |

IDs are deterministic traversal/observation IDs for equivalent traces, not persistent identities across architecture edits. JSON keys are sorted; no timestamps, memory addresses or activation arrays are intentionally stored. Backend error text and source paths can vary across environments.

Module parameter counts are recursive and deduplicate parameter objects within a subtree. Do not sum ancestor/descendant counts. Shared modules have one canonical parent in details; all registration parents remain in `module_edges`.

## Analysis mechanisms and fidelity

1. Enumerate actual registered modules, preserving shared identities and aliases. Inspect class source and installed package metadata for origin (`user`, `framework`, `dependency`, `unknown`). Bundled `p2i.examples` classes represent user model definitions and are labelled user code.
2. Forward pre/post hooks record real module calls and nested tensor IO (mapping, tuple, list, dataclass; mapping-based model outputs work).
3. `TorchDispatchMode` observes actual ATen operators during that same forward pass, assigning the active module/call. A fused attention operation stays fused; no semantic expansion is fabricated. The dispatcher API lives under PyTorch's internal utilities and may need version-specific maintenance.
4. Try `torch.export.export(..., strict=False)` on a fresh copy. Preserve symbolic tensor metadata and module-stack mapping. A successful export does not mean each static node was observed at runtime.
5. If export fails, try `torch.fx.symbolic_trace` plus shape propagation. Graph capture and shape propagation can succeed partially. Unknown ownership or shapes stay unknown. FX leaf-module nodes are shown exactly as captured, not presented as decomposed primitives.

Export/FX are optional. Data-dependent branches often defeat both; the observed runtime path remains usable. Graphs do not cover paths the example input did not take. Static export tensors and runtime tensors have separate namespaces; there is no claimed one-to-one equivalence.

## Development and tests

```bash
python -m pip install -e '.[test]'
cd frontend
npm ci
npm run build
cd ..
python -m pytest -q
```

The Vite production build writes to `src/p2i/server/static/`, included as Python package data. For live frontend development, run the backend at port 8000 and `npm run dev` in `frontend/`; Vite proxies `/api` to localhost.

Browser regression test (separate terminals):

```bash
p2i demo transformer --save tiny_transformer.json --no-serve
p2i serve tiny_transformer.json
# another terminal:
cd frontend
npx playwright install chromium --only-shell
npm run test:e2e
```

The browser test uses an actual FastAPI-served production build, checks hierarchy/graph navigation and inspection, and saves `docs/explorer.png`. `P2I_URL` can override the server URL. Browser installation is an explicit developer step, not library behavior.

Build a wheel after building the frontend:

```bash
python -m pip wheel . --no-deps -w dist
```

Read-only endpoints: `/api/model`, `/api/modules`, `/api/modules/{id}`, `/api/operations/{id}`, `/api/tensors/{id}`. The frontend normally fetches one complete IR. No source-code execution or file browsing API is exposed.

## Repository

```text
src/p2i/
  ir/             typed schema and invariants
  analyze/        module extraction, runtime recorder, export/FX, provenance
  architecture/   editable schema, constructor adapters and deterministic builder
  harness/        transactions, validation, revisions and JSON artifact CLI
  actions.py      typed versioned actions
  serialization/  JSON loading (ModelIR.save handles writes)
  server/         FastAPI plus packaged production frontend
  examples/       offline MLP, CNN, Transformer definitions
  cli.py          serve/demo commands
frontend/         React + TypeScript, SVG graph, Vite, browser test
examples/         convenient repository-level imports
tests/           unit and integration tests
docs/            validation notes and browser screenshot
```

## Limitations

- Research prototype tested on CPU; CUDA-specific behavior is not verified. No backward graph, profiler timings, FLOPs, storage size estimates or numerical correctness claims.
- Trace operations and browser data are held in memory; the browser node cap limits rendering, not capture size. Deep copies and separate static analysis increase memory use for large models. CPU offload, streaming and paging are not implemented.
- Runtime hooks observe `module(...)`, not calls directly to `.forward()`. Fused/native fast paths can bypass child module calls. Module ownership comes from the active hook stack or static backend metadata; unknowns remain explicit.
- Object-identity versions handle direct in-place results. Full shared-storage alias/mutation analysis across distinct tensor views is not implemented; `alias_of` does not claim complete storage aliasing. Non-tensor control dependencies and scalar values are not represented as tensor edges.
- Quantized/sparse/distributed/compiled/custom tensor subclasses and asynchronous threaded module execution are not certified. Python scalar extraction can synchronize accelerators.
- FX shape propagation can fail or lack device metadata. Both export and FX can execute Python code. Deepcopy and RNG restoration do not undo global side effects, file/network activity, or user hooks in supplied model code. Analyze only models you intentionally trust; no sandbox is claimed.
- Origins are best-effort: installed distributions are dependencies, uninstalled source is user code, uninspectable/generated source can be unknown. Local vendored libraries may need future origin overrides.
- Input container types beyond mappings/lists/tuples/dataclasses, storage alias semantics, general dynamic-control-flow visualization and dependency-only scalar nodes require future work.

## Next phase (not implemented)

Build agent-side search strategies on the existing harness: curriculum, retrieval with embeddings, semantic subgraph discovery, composition search and multi-objective agent benchmarks. P2I remains independent of the intelligence operating it. Large-model paging, full alias analysis and paper alignment remain separate work.

## Phase 3 — optional skill harness

Phase 3 adds a local SQLite Skill Registry, explicit contracts, deterministic retrieval,
versioned lifecycle, isolated candidate validation, promotion, reusable composite/custom
components, structured agent tools and evaluation/experiment history. It does not
include any AI provider or autonomous reasoning system. Existing Phase 1/2 APIs and
observed JSON 0.1 remain supported.

```bash
p2i skills list
p2i skills search dropout
p2i demo transformer --edit
python -m p2i.examples.skill_discovery
```

The live UI now includes **Skills** and **Replace with Skill** in Edit. See
[Phase 3 workflow](docs/PHASE3.md) and [Skill system and execution boundaries](docs/SKILL_SYSTEM.md)
for Python/JSON/CLI use, validation, promotion, versioning and the offline
failure→correction→promotion→reuse example. Candidate source is never automatically
trusted; in-process Python skill application requires an explicit local opt-in.

## Focused visualization update

Explore now draws actual observed tensor routes between discovered modules. Computation
opens a readable local neighborhood with a central input→operation→output explanation;
Raw graph remains available. Opt-in captured tensor samples render as explicitly labeled
heatmaps. Runtime playback highlights the recorded module and shows that call's real
operations. See [visual exploration](docs/VISUAL_EXPLORATION.md) for bounds, evidence
semantics and browser checks. Run `p2i demo transformer --edit` to try it.
