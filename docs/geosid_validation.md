# GeoSID framework validation

Actual local checks on 2026-11-17, based on official main
`0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1`. These results belong to this
framework, not the older RQ-only candidate or its downstream study.

## Software results

| Check | Observed result |
|---|---|
| Framework CPU math/model/pipeline/OPQ tests | 54 passed |
| Existing SID CPU regressions | 77 passed, including the five pandas-dependent integration cases run after installing that dependency |
| CUDA tests | Three passed on RTX 4060 Laptop: CF/autocast, later-residual gradients, small RQ-VAE driver/artifact roundtrip and RNG isolation |
| Distinct passed cases | 134: 131 CPU + 3 separately opted-in CUDA cases |
| Actual `python -m` CLI | Eight successful processes: enabled train/export/diagnose and disabled train for both backends |
| Persistence/export | Both enabled backends reproduced codes, raw/final index, IDs and statistics byte-for-byte after loading |
| Mapping | Explicit shuffled row IDs, duplicate embeddings, group-local suffixes; 256/256 final unique SIDs in each CLI fixture |
| Original source | No changes to `rq/`, existing quantizer implementations or shared SID tools |

The CPU suite initially reported 126 passes, eight skips and four deselections.
Three skips were CUDA opt-ins later passed on hardware; the other five were
missing-pandas checks subsequently passed. The **four deselections are genuine
upstream Windows failures**, not passes: existing-directory atomic publication
tests raise a `ctypes.CDLL(None)` error, with three associated teardown errors.
The same four failures and three teardown errors were reproduced in a separate,
unmodified official-main worktree. They remain unfixed here. Use new output
directories on Windows. This is not an all-upstream-tests-green claim.

The final CUDA process had two non-fatal pytest cache permission warnings;
all three tests completed. Peak allocated/reserved CUDA memory was
18,163,712 / 23,068,672 bytes. The existing cumulative GPU ledger charged
14.031 seconds for this task, including startup; no SFT or previous full study
was rerun. The experimental OPQ adapter is CPU-only.

Test environment: Python 3.12.5, Windows, PyTorch 2.6.0+cu124, CUDA runtime
12.4, NumPy 1.26.4, scikit-learn 1.6.1, FAISS CPU 1.12.0, pytest 8.3.5,
pandas 2.2.3. Results on other platforms/versions are not certified.

## Reproduce the focused checks

```sh
python -m pytest tests/test_geosid_core.py tests/test_geosid_rqvae.py tests/test_geosid_pipeline.py tests/test_geosid_opq.py -q
python -m pip install -r sid_generation/requirements-test.txt
python -m pytest tests/test_flash_sid_export.py tests/test_flash_sid_integration.py tests/test_flash_sid_simhash.py sid_generation/tests -q
```

The second test command includes the known Windows failures. To run only the
three CUDA cases in PowerShell, with a real GPU and an explicit time budget:

```powershell
$env:GEOSID_TEST_CUDA='1'
python -m pytest tests/test_geosid_core.py::CudaTests tests/test_geosid_rqvae.py::CudaModelTests tests/test_geosid_pipeline.py::CudaPipelineTests -q
```

Core checks include NumPy reference agreement, FP64 finite-difference gradcheck,
scale/centering derivatives, fixed/random projection behavior, dtype/device and
overflow rejection. Adapter checks cover actual residual values, later-layer
encoder gradients, absence of direct GeoReg codebook gradients, unchanged base
updates, strict original model weight loading and complete default/zero bypass.
Catalog tests reject unknown or changed OPQ items and malformed/tampered state.

## Fixed synthetic OPQ comparison

Run `python -m sid_generation.geosid.compare_opq --output outputs/geosid/paired-opq.json`.
The fixture uses seed 2024, 256 eight-dimensional heavy-tailed anisotropic rows,
six duplicate pairs and two zero vectors. All arms share one fitted original
baseline: two RQ levels with four centroids, two PQ blocks at two bits, and
8/4/8 RQ/OPQ/PQ iterations. Shaping uses 80 full-catalog Adam steps, learning
rate 0.01, geometry weight 0.1 and both keep weights 1. Projection defaults
are 256 directions/17 frequencies; diagnostics use independent fixed seed 314159.
These settings were fixed before the comparison, not selected for an improvement.

| Arm | Fixed CF mean | Final quantized MSE | Raw duplicate-excess rate | Used codes | Final unique |
|---|---:|---:|---:|---|---:|
| Original RQ-OPQ | 0.0150904413 | 0.5880779390 | 0.48046875 | 4/4 at all four positions | 256 |
| Same-prefix/rotation, PQ refit only | 0.0150904413 | 0.5880779389 | 0.48046875 | 4/4 at all four positions | 256 |
| Block shaping + same-budget PQ refit | 0.0149792861 | 0.5879092989 | 0.47656250 | 4/4 at all four positions | 256 |

Shaping's full-vector semantic MSE was `5.24165e-6`, tail-drift MSE
`5.24156e-6`, and cosine loss on targets with norm above `1e-8` was
`9.30452e-7`. The two exactly-zero targets contribute a constant `2/256` to
the all-row cosine loss; that floor is not semantic drift. FP64 diagnostics
separate numerical error from the FP32 training objective.

Both refit arms use identical FAISS initialization rules, seed and iteration
budget. Different input tables need not produce bitwise-identical initial
centroids. Prefix assignments and the caller's baseline remain unchanged.
The experiment was repeated after adding final drift diagnostics, without
changing shaping/configuration; the earlier local receipt is retained.

This one synthetic result demonstrates the implemented process and measured
trade-offs. It proves neither convergence nor recommendation improvement.
Another independent CLI fixture produced fewer raw unique codes with enabled
OPQ (166 versus 167 disabled), despite 256 final unique identities in both.
Final uniqueness is always the shared disambiguation guarantee, not algorithm
quality. No downstream HR/NDCG experiment was run for this framework.
