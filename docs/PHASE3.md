# Phase 3: deterministic skill experimentation

P2I provides the environment; the caller chooses the intelligence. No model provider,
API key, embedding service or reasoning agent is used. Existing observed Model IR
0.1 stays unchanged. Architecture IR 0.1 gains optional skill identity/revision and
additional tensor contract fields; old artifacts still load.

## Quick start

```bash
python -m pip install -e '.[test]'
p2i skills list
p2i skills search dropout
p2i skills inspect builtin.dropout
p2i demo transformer --edit
python -m p2i.examples.skill_discovery
```

The discovery example runs offline with an already installed environment. It deliberately
submits a shape-invalid `GatedResidual`, receives structured failure feedback, supplies
corrected source, validates, promotes, and reuses the registered component in two
separate architectures. Use `--output path/to/new-directory` to retain SQLite, JSON
reports, and both experiment histories. Use a fresh directory for each run.

## Python loop

```python
import torch
from torch import nn
import p2i

registry = p2i.SkillRegistry("skills.sqlite3")
h = p2i.Harness(nn.Sequential(nn.Linear(8, 8), nn.ReLU()),
                example_inputs=(torch.randn(2, 8),), skill_registry=registry)
h.retrace()
results = h.search_skills(query="gelu")
action = p2i.ReplaceWithSkill(target="1", skill_id=results[0].skill_id,
                             expected_revision=0)
preview = h.preview(action)
assert preview.success
assert h.apply(action).success
assert h.validate(backward=True).backward_valid
model = h.build()
ir = h.retrace()
evaluation = h.evaluate()
h.export_experiment("experiment.json")
```

`InsertSkill(parent=..., index=..., skill_id=..., parameters=...)` uses the same
transaction path. Batch actions can combine ordinary actions and skill actions.
Preview does not increment the architecture revision. Failed transactions retain
architecture, model weights and observed evidence. A committed action pins the skill
revision; later registry updates do not silently change the model. `undo`, `redo`,
`diff` and `snapshot()` work for skill edits. Snapshots copy model state and keep a
shared persistent registry, not a second copy of registry metadata.

## JSON tool protocol

```json
{"version":"0.1","tool":"search_skills","arguments":{"query":"dropout"}}
```

`p2i.tools.dispatch(request, harness=h)` returns a `ToolResult` with `success`,
`result`, and a structured `error` containing `code`, `message`, and `details`.
`tool_schemas()` / `p2i tool --schemas` exposes input JSON schemas. Supported tools:
inspect_model, inspect_node, inspect_revision, list_skills, search_skills,
inspect_skill, preview_action, apply_action, validate_model, build_model,
retrace_model, evaluate_model, register_skill, validate_skill, promote_skill,
reject_skill. The local FastAPI app also exposes `POST /api/tools`; it enforces
same-origin JSON requests. No MCP or provider SDK is required.

```bash
p2i tool request.json
p2i tool request.json --architecture model.arch.json --input-shape 2 8 --output changed.arch.json
p2i harness apply model.arch.json action.json --output changed.arch.json --input-shape 2 8
p2i skills --registry ./skills.sqlite3 list
```

Set `P2I_SKILL_REGISTRY` to share a registry between commands. JSON architecture
artifacts carry no example inputs or learned weights. CLI constructor builds start
with deterministic new weights; an in-memory Harness preserves compatible state.
Unknown Python model classes still require their original live session or an adapter.

## UI

The existing Explore / Hierarchy / Computation / Runtime / Edit views remain.
Skills adds local search, kind/status filters, contract and hyperparameter inspection,
validation issues, provenance, evaluations and version inspection. Edit offers
**Replace with Skill** only for default configurations whose known input/output
contracts fit the selected node. Explicit non-default configurations remain available
through action JSON. Preview validates and shows the diff before commit. Observed
runtime stays unchanged until Build & Re-trace succeeds. A selected bound component
shows its pinned skill revision separately from observed evidence.

## Evaluation and research records

`h.evaluate()` records parameter/module counts, forward status and operation count
only when an observation exists for the current revision. An optional trusted Python
callback receives `(built_model, example_args, example_kwargs)` and returns metrics.
No task accuracy is guessed. `compare_evaluations(a,b)` returns numeric deltas without
optimization claims. `registry.record_evaluation(skill_id, record)` explicitly associates
an evaluation with a skill; P2I does not attribute a whole-model score automatically.

`experiment_history()` and `export_experiment()` record searches, edits, skill reuse,
validation, re-trace and evaluation. Registry events separately record ingestion,
validation and promotion; `registry.export()` includes these. No activation arrays
or model weights are embedded. Callback evaluation is trusted host Python, not the
bounded candidate-code API.

## Scope and limitations

* CPU is the candidate validation baseline; CUDA validation and latency benchmarking
  are not implemented.
* Symbolic equality and constructor constraints are practical checks, not a theorem
  prover. Unknown contracts are explicitly unproven and omitted from UI replacement
  recommendations. Runtime validation remains necessary.
* Validation covers declared examples and a second `T` value when present, not all
  inputs, modes or possible Python branches.
* Explicit consecutive Sequential regions and reconstructable module subtrees can be
  extracted; arbitrary runtime graph extraction and nested portable skill dependencies
  are not supported.
* Python source is bounded to one class and torch/math imports. External package and
  P2I-skill imports in candidate source are rejected; no dependencies are installed.
* Skill application does not infer arbitrary dimensional dependencies or recognize
  attention/MLP semantics. Built-ins wrap module adapters, not architecture templates.
* See [SKILL_SYSTEM.md](SKILL_SYSTEM.md) for lifecycle and execution boundaries.

Next work belongs primarily on the caller side: bounded search policies, curriculum,
embedding retrieval, semantic subgraph discovery, composition discovery and external
agent benchmarking. None of these autonomous capabilities is implemented here.
