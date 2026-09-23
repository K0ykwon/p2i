# Reproducible performance and evaluation

## System measurements

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/benchmark_explorer.py
python tests/browser_acceptance.py
python -m pytest tests/unit/test_execution_comparison.py -q
```

The benchmark traces actual PyTorch sine chains at 500, 1,000 and 3,000 operations, with and without bounded capture. It exports real IR fixtures plus trace time, JSON size, sidecar size and cumulative Linux process peak RSS to `artifacts/performance/trace.json`. It excludes static tracing and uses CPU, seed 0 and one thread. This isolates explorer scaling; it is not representative of every neural architecture. One pass after initialization is a smoke measurement. Use multiple isolated processes, randomized condition order and medians/quantiles for performance claims. Do not infer capture overhead from a single noisy pair.

The browser acceptance consumes those fixtures and records navigation-to-computation and next-step wall times in `artifacts/browser/performance.json`, with browser version and viewport. These include HTTP, Playwright and assertion overhead, not pure React render time. Graph-size checks assert bounded local rendering; no hardware-independent latency promise is asserted. CUDA smoke is collected by pytest and skipped explicitly when CUDA is absent. CPU is the CI baseline.

## Human task evaluation infrastructure

`p2i.research.TaskRecorder` is an explicit local JSONL task recorder. It does not instrument the browser, transmit telemetry, collect participant identities, or grade correctness automatically. A human researcher or external study controller must call its methods:

```python
from p2i.research import TaskRecorder
study = TaskRecorder()
study.start('find-producer-01', 'navigation', observed_revision=0)
study.navigation()  # call for each counted navigation action in the study
result = study.finish(correct=True, confidence=4, notes='Graded with preregistered rubric')
study.save('session.jsonl')
```

Fields include monotonic task duration, navigation count, observed revision, optional correctness/confidence and UTC completion time. Missing correctness remains null. Optional notes should avoid participant-identifying information. Keep any consent/participant mapping outside P2I.

Suggested balanced tasks, with targets and answers generated from the specific saved IR:

| Task | Evidence/rubric | Measures |
|---|---|---|
| Model comprehension | Identify actual modules and observed input/output shapes | Correctness, time, confidence |
| Graph navigation | Find a tensor's producer and two recorded consumers | Correctness, actions, time |
| Fault localization | Locate a failing module from an explicitly labelled failing trace | Localization accuracy, time |
| Architecture comparison | Identify changed constructor parameters between successful observations | Correctness, false-positive count |
| Edit consequence | Predict a change, then compare prediction with a validated retrace | Prediction correctness, time |
| Debugging | Propose a structured edit and verify it through the Harness | Forward validity, iterations, time |

Counterbalance model/task/interface order, include unfamiliar architectures, distinguish expertise levels and report failures. Predefine grading independently of interface condition. Pair user outcomes with system responsiveness and graph sizes. Do not treat reduced clicks or higher confidence alone as improved understanding. No user study has been conducted by this implementation.

## Investigation: bounded interventions

The existing architecture edit → validation → retrace loop already supports controlled comparisons. Direct tensor/operation intervention is deliberately **not implemented**: safely replaying aliases, in-place operations, randomness, autograd and fused kernels needs an explicit execution model first. A future design should retain immutable original observation IDs, record intervention specification/seed/input identity, label modified execution separately, and validate its outputs before publishing them. Downstream changed samples alone do not prove causal influence. This is the next research boundary, not a claimed current capability.
