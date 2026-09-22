# Phase 3 validation record

Executed locally on 2026-09-22 with Python 3.12.14, PyTorch 2.14.0+cu130
(CPU execution), Node 24.19.0 and Chromium 134 (Playwright 1.51.1).

| Gate | Result |
| --- | --- |
| Unmodified Phase 1/2 Python baseline | 66 passed |
| Unmodified frontend baseline | TypeScript + Vite build passed |
| Final combined Python suite | 102 passed, 2 upstream deprecation warnings |
| Phase 3 unit coverage | 30 tests, including worker failures/timeouts and saved-artifact validation gate |
| Phase 3 integration coverage | 6 tests: three demo edits, discovery/reuse, API, CLI |
| Frontend production build | Passed; bundled into Python wheel |
| Legacy browser explorer acceptance | Passed |
| Phase 2 Transformer / CNN / MLP browser acceptance | All three passed |
| Phase 3 live browser acceptance | Passed |
| CLI demo MLP / CNN / Transformer | All three passed, JSON produced |
| Offline failure → correction → promotion → two-model reuse | Passed |
| Wheel build | p2i-0.3.0-py3-none-any.whl |
| Isolated wheel install smoke | Passed imports, skills, edit, trace, static UI and candidate worker |

No browser tests were blocked in this environment. Earlier Phase 2 restrictions are
historical; current tests exercised real local servers and Chromium. The browser runner
starts its server and browser together, which also works in per-command network namespaces.

Browser Phase 3 checks: Transformer progressive exploration, local skill search and
inspection, UNVALIDATED candidate state, retained selected component, compatible SiLU
replacement, preview/constructor/forward validation, diff, commit, revision history,
build/retrace and updated skill binding/observed module. The old explorer also checks
runtime/export views, shapes, tensor inspection, pan/zoom, origin filtering and node caps.

The CLI integration test starts fresh subprocesses for list/search/inspect, candidate
source ingestion, validation, promotion, architecture JSON skill application, structured
tool invocation and export. The offline controller validates a repaired custom module,
performs forward/backward and re-trace on two distinct models, records evaluations,
and exports experiment and registry JSON.

Actual browser captures:

* `phase3-skill-library.png`
* `phase3-candidate.png`
* `phase3-skill-preview.png`
* `phase3-retraced.png`
* `phase2-transformer.png`, `phase2-cnn.png`, `phase2-mlp.png`

Detailed command outputs are retained under `artifacts/phase3-*.txt`.
Warnings concern upstream FastAPI/Starlette HTTPX and AnyIO deprecations; no tests
were skipped or marked expected-failure.

These tests demonstrate the supported deterministic workflow, not correctness for
all possible custom Python programs. Candidate processes have shared wall-clock and
CPU budgets; they are not a hardened filesystem/network security sandbox. Host
application of reviewed Python skills requires explicit `allow_python=True` and has
normal host Python execution semantics. See SKILL_SYSTEM.md.
