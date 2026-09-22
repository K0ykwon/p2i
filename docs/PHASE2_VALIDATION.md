# Phase 2 validation status — 2026-09-22

**Implementation and the socket-free acceptance pipeline are complete. The full browser/server acceptance gate is NOT signed off in this environment.** Local socket creation and Chromium startup are denied, and the automatic approval policy rejected an elevated local test run. No passing browser result is claimed for Phase 2.

## Baseline and preservation

Before implementation, the existing full pytest suite was invoked. It stalled on FastAPI TestClient's cross-thread event-loop wakeup. Diagnostic stack traces identified the blocked request, and direct socket creation raised `PermissionError: [Errno 1] Operation not permitted`.

The released Phase 1 ZIP was extracted separately and its socket-free suite rerun: **23 passed, 3 deselected**. Its historical complete validation remains documented in `VALIDATION.md`; that is a Phase 1 historical result, not a Phase 2 browser result.

Every original Python test file and `frontend/tests/explorer.mjs` was byte-compared with the Phase 1 release archive and remains unchanged. Observed ModelIR schema remains 0.1. Existing saved JSON, CLI demos and trace/save APIs continue to work in the exercised pipelines.

## Results

| Check | Result |
| --- | --- |
| Current pytest suite excluding the 3 socket-dependent `test_models` cases | **63 passed**, 3 deselected, 2 upstream deprecation warnings |
| Phase 1 tests within that run | 23 passed |
| New Phase 2 unit/integration cases | 40 passed |
| TypeScript check and Vite production build | Passed |
| Original CLI MLP/CNN/Transformer demos | All three passed with runtime and export capture |
| Actual edit → build → forward → backward → Phase 1 re-trace | Passed for MLP, CNN and Transformer |
| Live-harness API requests through real in-process HTTPX ASGI transport | Passed: preview, apply, stale revision rejection, validate, undo/redo, retrace, observed state and bounded samples |
| Observation-only API through ASGI transport | Passed: unchanged schema, objects, 404 and editing-disabled behavior |
| Artifact CLI inspect/apply/validate/diff | Passed; machine-readable JSON and saved updated architecture |
| Wheel build and isolated install | Passed; live Harness and original APIs import from installed wheel |
| Packaged frontend assets | HTML plus exactly one current JS and CSS bundle present |
| Original full TestClient suite | Blocked by execution-environment socket restrictions; 3 cases remain pending |
| Legacy + Phase 2 browser acceptance runner | Attempted; blocked at local server bind |
| Chromium launch | Attempted independently; failed with `shutdown: Operation not permitted` |

The final pytest command was:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -k 'not test_models'
```

The tests were not deleted, weakened or marked as passing. The documented deselection only avoids the known environment deadlock; those three cases must pass in a socket-enabled environment before complete acceptance.

## Actual architecture edits

| Demo | Edit | Before parameters | After parameters | Forward / backward / re-trace |
| --- | --- | ---: | ---: | --- |
| MLP | Explicit batch: first Linear width 32→24 and next Linear input 32→24 | 676 | 508 | Passed |
| CNN | Conv2d padding mode zeros→reflect | 1,477 | 1,477 | Passed |
| Transformer | Explicit batch: block 0 MLP width 128→96 and matching next Linear input | 67,592 | 63,464 | Passed |

The incompatible CNN change `features.3.out_channels: 16→24` was correctly rejected during runtime validation. A single incompatible Linear width change was rejected before commit; the coordinated batch succeeded.

Coverage includes all required adapters, unknown read-only constructors, custom adapter registration, constructor extraction, malformed actions/targets, invalid dimensions/head counts, Sequential shape checks, failed-forward attribution, atomic rollback, replace/insert/remove/wrap, shared modules, tied parameters, ModuleList reindexing, compatible weight preservation, deterministic reinitialization reports, revisions/stale checks/diffs/undo/redo, observed configuration and provenance, preservation of original state/RNG/inputs, optional backward checks, and bounded metadata-only versus opt-in value capture.

A further artifact validation test prevents a forged editable child of an owning adapter from being accepted even though that child would be ignored by construction.

## Browser acceptance provided, not certified here

`tests/browser_acceptance.py` first runs the unchanged legacy browser test, then `frontend/tests/phase2.mjs` for Transformer, CNN and MLP against actual servers. It covers:

- default Explore overview and nested drill-down;
- computation selection and tensor metadata;
- breadcrumbs and selection retention across views;
- editable constructor parameter, preview/diff and real validation;
- commit, history and pending-versus-observed revisions;
- build/retrace and the actually rebuilt constructor value;
- no browser JavaScript errors.

Run in an environment permitting local sockets and Chromium:

```bash
python -m pytest -q
cd frontend
npm ci
npm run build
npx playwright install chromium --only-shell
cd ..
python tests/browser_acceptance.py
```

The included `docs/explorer.png` and `docs/computation.png` are the earlier Phase 1 screenshots. They are not presented as screenshots of the new Phase 2 UI. New `docs/phase2-*.png` screenshots are produced by the browser runner when it can execute successfully.

## Practical boundaries

CPU execution was validated. No GPU, distributed, compiled-model, arbitrary dynamic-program rewrite or training-quality guarantee is made. Forward/backward validation is specific to supplied examples. Portable architecture JSON does not contain weights or executable custom classes. Unknown custom roots need a live Harness or registered adapters. Full storage-alias analysis and learned-state/session persistence remain future work. All remaining acceptance gates and these limitations are documented instead of being inferred from a successful build.
