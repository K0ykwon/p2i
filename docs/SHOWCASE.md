# Real Transformer showcase

Run `p2i demo transformer --edit`. The bundled offline model is ordinary PyTorch composition; there is no model-name-based frontend layout. Suggested 60–90 second browser workflow:

1. **Explore** the root, `blocks`, then `blocks.0`. Show the real hierarchy, origin badges and tensor routes.
2. Open **Runtime**. Play a few steps, pause, advance once and restart. Show the recorded operation/call IDs and amber input/output emphasis. This animation does not encode measured time.
3. Open **Computation**. Choose a reshape or addition actually present in the selector; inspect its dimensions and connected neighbors. Drag a node and return to it. Do not expand a fused SDPA node into unobserved internals.
4. In **Edit**, enable bounded whole-tensor CPU heatmaps and **Build & Re-trace**. Return to an early operation. Compare the input and output rectangular grids, their axis aggregation labels and signed colour scale. No raw activations are stored in the normal capture path.
5. At root, select `norm`. In **Edit**, change `eps` to `0.001`, **Preview & validate**, **Commit edit**, then **Build & Re-trace**.
6. Open **Compare**. Show observed r0/r1, actual constructor changes, shape/grid changes and the aligned operation microscopes. A safe epsilon edit need not change operation counts or shapes; unchanged evidence is informative.

Automated acceptance and optional real browser video:

```bash
python scripts/benchmark_explorer.py
P2I_RECORD=1 python tests/browser_acceptance.py
```

The `microscope.mjs` test records browser interactions with Playwright when explicitly enabled. It writes WebM, screenshots and measured browser timings under `artifacts/browser/`. These are actual browser captures, not a mock animation or slideshow. Acceptance runs faster than a narrated presentation; use the steps above for a paced live recording. Existing MLP/CNN drill-down tests also run, guarding against Transformer-specific assumptions.
