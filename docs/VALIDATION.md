# Phase 1 validation

Environment: Python 3.12, PyTorch 2.14.0+cu130 (CPU execution), Pydantic 2.13.5, FastAPI 0.141.1, Node 24.19.0. Frontend compiled with TypeScript and Vite 6.4.3. Headless Chromium exercised the production interface via Playwright 1.51.1.

## Executed checks

- `python -m pytest -q`: **26 passed**. Two upstream Starlette/httpx/AnyIO deprecation warnings, no test failures.
- `npm run build`: TypeScript check and production Vite build passed.
- `npm run test:e2e`: passed against the real local FastAPI server, with no browser errors. Covers hierarchy expansion, repeated blocks, module selection, input/output shapes, parameter counts, user/framework origins, runtime/export graphs, operation/tensor details, pan, zoom, fit-to-scope, origin and execution filters, and explicit 120→240 node expansion on a large fixture.
- `p2i demo mlp/cnn/transformer --save ... --no-serve`: all three actual model pipelines passed; saved JSON files are included at repository root.
- `p2i serve tiny_transformer.json`: launched, served the production application, and supported the browser test.
- `python -m p2i.cli --help`: passed.
- Wheel built and imported from an isolated target directory; Transformer analysis, FastAPI model API, packaged HTML and both JS/CSS asset requests passed.

| Model | Modules | Runtime operations | Export operations | Tensor records |
| --- | ---: | ---: | ---: | ---: |
| SimpleMLP | 5 | 5 | 3 | 18 |
| TinyCNN | 9 | 9 | 8 | 32 |
| TinyTransformer | 24 | 59 | 38 | 175 |

All three demos succeeded with hierarchy, runtime hooks, dispatcher capture and export. FX was correctly skipped after successful export. Tests separately force export failure to exercise FX shape propagation and force both failures to verify runtime fallback. A real data-dependent branch also exercises tracing limitations. FX shape-propagation failure preserves a partial static graph.

Tests additionally cover deterministic IDs, JSON round-trip, rejected malformed references/edges/schema versions, symbolic dimensions, shared modules and containers, repeated calls, never-executed modules, nested tuples/dicts/dataclasses, scalar outputs, direct in-place tensor versions, model/input/RNG preservation, forward exceptions and hook cleanup, and PyTorch MultiheadAttention library-module inspection.

`explorer.png` and `computation.png` are screenshots of the running production interface, not design mockups.

## Boundaries

CPU only was validated. GPU, compiled/distributed models and all external libraries are not certified. The library preserves observed operations without semantic expansion. Storage-alias mutation analysis is incomplete; static and runtime graphs are distinct; no backward pass or profiling is captured. Large-model capture still consumes memory proportional to the recorded trace, even though browser rendering is bounded. See README for full limitations and proposed next-phase work.
