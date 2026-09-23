# Contributing

Use Python 3.11 or 3.12 and Node 22 for development. Node is not needed to run an installed distribution.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]' build
cd frontend
npm ci
npm run build
npm run test:unit
npx playwright install chromium
cd ..
python -m pytest -q
python tests/browser_acceptance.py
python -m build
```

CI uses the pinned CPU PyTorch version in `.github/workflows/ci.yml` and dependency constraints in `requirements-ci.txt`. All example models run offline. Browser tests start their own local servers and use temporary skill registries. For development use `p2i demo transformer --edit` and optionally `npm run dev` in `frontend/`.

Keep observed ModelIR backward-compatible. Never turn unobserved semantics into graph nodes or tensor axis names. Preserve original observations when presenting edits or comparisons. Add regression coverage for changed behavior, use keyboard-accessible controls, and honor reduced motion. Do not add network or telemetry requirements to the application.

Describe the user-visible problem, resulting behavior, tests actually executed and any untested environments in your PR. Rebuild bundled frontend assets when frontend code changes. Generated traces, recordings, registries and benchmark results belong in `artifacts/` and are not source files.

For distribution verification install `dist/*.whl` plus `httpx` in a separate virtual environment, then run that environment's `python -I scripts/wheel_smoke.py`. The script changes to a temporary directory and checks that it imported the installed package rather than this source checkout.
