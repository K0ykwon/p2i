# Skill system reference

## Separate representations

Observed ModelIR describes execution. ArchitectureIR describes rebuildable modules.
SkillSpec describes an explicitly named reusable capability. A skill binding layers
metadata over a component; it never replaces or fabricates runtime observations.

SkillSpec 0.1 includes stable ID, display name, kind, lifecycle status, description,
tags, tensor contracts, typed hyperparameters, structured constraints, implementation,
origin, validation, evaluations, UTC timestamps and monotonic registry revision.
Kinds are primitive/canonical/composite/custom/discovered. Implementation is a tagged
union: `registered_module`, `composite_ir`, or `python_module`.

TensorContract is shared with ArchitectureIR: shape (integer/symbol/unknown), rank,
dtype, dtype_family, device, evidence. `evidence="declared"` distinguishes a supplied
contract from observed tensor metadata. Same symbols are unified across inputs and
outputs; integer conflicts, rank and dtype/device conflicts fail. Hyperparameter names
can appear as dimensions, e.g. `["B", "T", "dim"]`. Constraint records support equality,
divisibility and CPU device restriction, without evaluating expression strings.

## Registry

`SkillRegistry(path=None, allow_python=False)` uses SQLite transactions and append-only
records keyed by `(id, revision)`. Default path is `~/.p2i/skills.sqlite3`, overridable by
`P2I_SKILL_REGISTRY`. Each operation opens/closes its connection. Reads return parsed
copies. Metadata, bounded Python source and ArchitectureIR fragments live in JSON
records; no executable pickle or huge weight files are loaded.

Built-ins wrap the existing adapter registry: Linear, Conv1d/2d, ReLU, GELU, SiLU,
Dropout, LayerNorm, BatchNorm1d/2d, Embedding, MultiheadAttention, Sequential,
ModuleList, Softmax, Identity, Flatten, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d.
They start promoted with built-in provenance. Their trust is explicit; it is not a
claim that every possible constructor configuration has been exhaustively validated.
Containers remain subject to their PyTorch calling semantics (ModuleList has no forward).

Public methods: register, get, list, versions, update, search, validate, promote,
deprecate, reject, record_evaluation, events, export, import_file. Identical source,
contracts, parameters and metadata under the same identity deduplicate. No semantic
equivalence detection or cross-ID merging is attempted.

Export is one predictable UTF-8 JSON artifact with all historical records, embedded
source, fragments and event history. Import preserves source lineage but resets external
trust to candidate; local revalidation is required. Local revision numbers are assigned
again on import. Built-in records come from installed adapters and are not overwritten
by imported claims. Imported history is not used to authorize code execution.

## Lifecycle

External registration always starts candidate, ignoring submitted status/validation.
Successful validation produces validated; an explicitly associated evaluation can produce
evaluated; promote requires a current successful validation fingerprint. Evaluated is
optional for promotion because task-specific metrics are caller-defined. Promoted,
validated and evaluated are normal search-eligible states. Candidate, rejected and
deprecated are excluded by default. Deprecation of the latest record prevents new
application of older pinned versions. Existing built architecture snapshots remain
reproducible.

Every transition appends a revision. Implementation, contract or hyperparameter changes
create a candidate with cleared validation/evaluation and parent revision/reason.
The registered/compiled status names are reserved for interoperability; the current
pipeline uses candidate -> validated -> evaluated (optional) -> promoted directly.
A failed validation is candidate plus structured issues, never silently promoted.

Search normalizes text, intersects explicit tags, checks supplied tensor contracts and
hyperparameters, then deterministically sorts a weighted text/contract/status score
with stable ID as tie breaker. Results expose each score and reasons. Missing contract
information is reported as unproven. There are no embeddings or hidden semantic labels.

## Candidate code and isolation

```python
from p2i.skills import *
registry = SkillRegistry("skills.sqlite3")
spec = SkillSpec(
    id="user.mixer", name="Mixer", kind="custom", description="User-specified mixer",
    input_contracts=[TensorContract(shape=["B", "T", "dim"], evidence="declared")],
    output_contracts=[TensorContract(shape=["B", "T", "dim"], evidence="declared")],
    hyperparameters=[HyperparameterSpec(name="dim", type="int", default=8)],
    implementation=PythonModuleImplementation(class_name="Mixer", source=source),
)
candidate = registry.register(spec)  # AST inspection; no source import
report = registry.validate(candidate.id, timeout=30, dimensions={"T": 7})
if report.valid:
    registry.promote(candidate.id)
```

`ingest_skill(source=..., metadata=..., registry=...)` also creates corrected revisions.
`MissingCapability` carries an explicit requested contract; `candidate_from_missing`
initializes metadata using externally supplied source. P2I never writes or repairs that
source itself.

Validation creates synthetic inputs with seeded CPU randomness, then runs import,
constructor, contract, forward, finite output, backward, finite gradients and Phase 1
runtime tracing in a disposable subprocess. It checks two sequence lengths when `T`
is declared. Reports record parameters, seed, Python/PyTorch, device, test shapes,
observed counts, stage, exception and a traceback summary. Unknown symbols must be
supplied in dimensions or hyperparameters. Each input is capped at 1,000,000 elements.
Validation timeout (0–300 seconds) bounds the **whole process**, therefore import,
constructor, forward and backward are each bounded by that shared budget. On Unix,
CPU time, output file size, descriptor count and core dump limits are also applied.
There is no strong address-space/memory isolation guarantee.

AST rules require one nn.Module with explicit constructor/forward; reject top-level
entrypoints, decorators, private introspection, eval/exec/open, disallowed imports,
network/process APIs and obvious unsafe torch facilities. Only torch, torch.nn,
torch.nn.functional and math imports are accepted. Missing allowed imports produce
DEPENDENCY_MISSING feedback. No pip/conda/apt command comes from candidate metadata.

**This is engineering isolation, not an adversarial security sandbox.** A subprocess
is not an OS permission boundary. Malicious code could evade static checks and reach
host resources. Validate only source deliberately supplied and reviewed by the user;
use a separate restricted VM/container for untrusted code. Temporary cwd and an
allowlist of child environment variables reduce accidental effects, but do not prove
filesystem/network containment.

Applying a Python skill to a live autograd model necessarily executes source in the
host process. It therefore additionally requires `SkillRegistry(..., allow_python=True)`
(or explicit CLI `--allow-python`), successful validation, eligible status and the
exact validated parameter configuration. This opt-in is never read from submitted
skill JSON. Host model execution is not protected by candidate worker timeouts. There
is no promise that validation makes arbitrary input-dependent behavior safe. For a new
parameter configuration, validate it first and reapply the new validated revision.

## Composite extraction

```python
skill = p2i.skill_from_subgraph(
    h.architecture(), node_ids=["0", "1", "2"], name="ProjectionPair",
    skill_id="user.projection_pair",
    input_contracts=[TensorContract(shape=["B", 8])],
    output_contracts=[TensorContract(shape=["B", 8])],
)
registry.register(skill)
assert registry.validate(skill.id).valid
registry.promote(skill.id)
```

Multiple roots must form one consecutive Sequential region. `extract_skill(h,
node_id=..., name=..., skill_id=...)` captures one supported subtree using observed
boundaries unless explicitly overridden. Unknown custom constructors and adapter-owned
internals are rejected rather than guessed. Extraction preserves source model name,
architecture revision and source nodes. Extracted skills transfer architecture, not
learned weights.

Stable tool error codes include SCHEMA_ERROR, UNKNOWN_TOOL, HARNESS_REQUIRED,
SKILL_NOT_FOUND, SKILL_EXISTS, RESERVED_ID, STALE_SKILL_REVISION, INVALID_PARAMETER,
CONTRACT_MISMATCH, STATIC_REJECTED, SYNTAX_ERROR, DEPENDENCY_NOT_ALLOWED,
VALIDATION_REQUIRED, SKILL_NOT_REUSABLE, LOCAL_CODE_OPT_IN_REQUIRED and
VALIDATION_FAILED. Validation issues additionally identify TIMEOUT, WORKER_FAILED,
DEPENDENCY_MISSING and the underlying runtime exception class.
