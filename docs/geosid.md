# Optional GeoSID-style geometry regularization

This option adds residual-level characteristic-function (CF) matching to the existing RQ-VAE. It is an independent adaptation of [GeoSID](https://aclanthology.org/2026.emnlp-main.960.pdf) to MiniOneRec, based on upstream commit `0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1`. It does not claim an official implementation reproduction or a recommendation-quality improvement. See [method_mapping.md](method_mapping.md) for the distinction between paper facts and adaptation choices.

The default is `geo_mode=none`, `geo_weight=0.0`. Existing reconstruction, quantization, commitment, KMeans and Sinkhorn behavior is retained. No trainable parameters or persistent model-state keys are added. OPQ, FLASH decoding, RL, dead-code recovery and extra balancing losses are outside this change.

## Objective and gradient path

For a residual matrix `R` of shape `B × d`, with `B >= 2`, the `cf_mean` convention is:

```text
C = R - mean_batch(R)
s = sqrt(mean_all(C²) + eps²)
X = C / s
U[:, m] = unit_normalize(N(0, I)), m = 1..256
P = X @ U
t = linspace(0, 3, 17)
phi = exp(-t² / 2)
q = trapezoidal integration coefficients, including dt = 3/16
w = q * phi / sum(q * phi)
a = mean_batch(cos(P[..., None] * t))
b = mean_batch(sin(P[..., None] * t))
G(R) = mean_projection(sum_frequency(w * ((a - phi)² + b²)))
```

The per-coordinate batch mean and single overall RMS remain differentiable. The normalized tensor is used only for the auxiliary loss. This is neither per-coordinate whitening nor per-projection standardization. There is no batch-size multiplier. Projection chunks are accumulated by their actual number of directions; the empirical CF always uses the complete batch before squaring.

For three quantizers, regularized locations are `z`, `z - stop_gradient(e1)`, and `z - stop_gradient(e1) - stop_gradient(e2)`. The final post-quantization tail is not regularized. Code vectors are selected with indices from the one original RQ forward; no second nearest-neighbor or Sinkhorn selection occurs.

MiniOneRec's original STE returns `x + (q - x).detach()`. Subtracting this output cancels the encoder derivative in later original residuals. The auxiliary detached-code residual path deliberately supplies an identity derivative to `z`, while leaving the original quantizer and reconstruction graphs intact. GeoReg-only backward adds no codebook gradient. This is a MiniOneRec-specific surrogate gradient, not a finite-difference claim about discrete selection.

```text
L_base = L_reconstruction + quant_loss_weight * L_quantization
none:     L = L_base
global:   L = L_base + geo_weight * G(z)
residual: L = L_base + geo_weight * mean_level(G(r_level))
```

GeoReg is added after the base loss. It is never multiplied by `quant_loss_weight`.

## Python API

The default forward still returns exactly `(reconstruction, rq_loss, indices)`. Existing inference calls and `get_indices` do not evaluate GeoReg, even on a model configured to train with it. The existing `Trainer` requests auxiliary results only when the regularizer is enabled.

```python
out, rq_loss, indices, aux = model(x, return_aux=True)
base, reconstruction = model.compute_loss(out, rq_loss, xs=x)
loss = base + model.geo_weight * aux["geo_loss"]
loss.backward()
```

`aux` contains the unweighted scalar `geo_loss`, differentiable `geo_level_losses`, and detached `geo_residual_rms` and `geo_scales` logs. It is returned explicitly; no graph-bearing cross-step attribute is used. When mode is `none` or weight is zero, residual reconstruction and geometric RNG sampling are skipped completely. Auxiliary loss is then zero and the per-level arrays are empty.

Frequency constants remain FP64 on CPU until each call, so casting the enclosing model to FP16/BF16 cannot round them. Training geometric arithmetic explicitly disables enclosing autocast and uses FP32 for FP16/BF16/FP32 inputs. FP64 inputs retain FP64 for numerical tests. `GeoReg` has a dedicated CPU `torch.Generator`; its state is available through `get_rng_state()` and `set_rng_state()`. It does not consume the global model, dropout, NumPy/KMeans or DataLoader streams. Training samples directions on each geometric call. Pass `geo_directions=U` to `RQVAE.forward` or `directions=U` to `GeoReg` for fixed diagnostics; fixed directions do not advance the training generator.

| Option | Default |
|---|---:|
| `geo_mode` | `none` (`none`, `global`, `residual`) |
| `geo_weight` | `0.0` |
| `geo_nproj` | `256` |
| `geo_num_frequencies` | `17` |
| `geo_tmax` | `3.0` |
| `geo_eps` | `1e-6` |
| `geo_projection_chunk` | `64` |
| `geo_seed` | `2024` |

The original CLI seed remains `2024`. Independent experimental seeds and loader generators are explicit in `configs/geosid/`. Global and residual configuration files intentionally retain zero weight: provide a calibrated positive `--geo_weight` to enable them. Projection chunking reduces temporary workspace, but does not establish a peak-memory guarantee; profile actual forward and backward execution.

## Standalone commands

Run from the repository root with the existing RQ dependencies installed: PyTorch, NumPy, scikit-learn, transformers and tqdm. The thin command delegates training to `rq.rqvae` and `Trainer`; exports and diagnostics use `rq.geosid_tools`. It does not require the research runner.

```sh
python -m sid_generation.geosid train --help
python -m sid_generation.geosid export --help

# Existing RQ architecture; replace paths with local input/output locations.
python -m sid_generation.geosid train --config configs/geosid/baseline.json --data_path catalog.npy --ckpt_dir outputs/geosid/A

# Set LAMBDA to the frozen calibrated value; use the same value for B and C.
python -m sid_generation.geosid train --config configs/geosid/global.json --geo_weight LAMBDA --data_path catalog.npy --ckpt_dir outputs/geosid/B
python -m sid_generation.geosid train --config configs/geosid/residual.json --geo_weight LAMBDA --data_path catalog.npy --ckpt_dir outputs/geosid/C

python -m sid_generation.geosid export --checkpoint CHECKPOINT.pth --embeddings catalog.npy --item-ids row_item_ids.json --output-dir outputs/geosid/export
python -m sid_generation.geosid diagnose --checkpoint CHECKPOINT.pth --embeddings catalog.npy --item-ids row_item_ids.json --output-dir outputs/geosid/diagnostics --diagnostic-seed 314159
```

`row_item_ids.json` must contain an explicit item-ID list in embedding-row order. Alternatively, `--item-meta metadata.json` validates exactly the upstream `0..N-1` convention and uses numeric row order; it does not infer row order from JSON insertion. For the official Amazon embeddings, that convention must be established from the upstream embedding generator, which sorts numeric item IDs before stacking rows. Both export commands use nearest-neighbor codes (`use_sk=False`) without collision-group Sinkhorn reassignment.

The output directory contains `raw_codes.npy`, `raw.index.json` (three tokens), `index.json` (four tokens), and `metrics.json`. The shared existing formatting helper assigns the fourth token by lexicographically sorted canonical item ID **inside each raw-code collision group**. Every singleton receives `<d_0>`. Suffixes are not global item IDs. Final SIDs are bijective for the frozen catalog; their zero collision rate is a formatting guarantee, not evidence of a better tokenizer. Adding/removing catalog members can change local suffix ranks.

Downstream training must separately register the full vocabulary in a fixed order: three positions of 256 semantic tokens plus `N` suffix tokens. Merely exporting with suffix capacity `N` does not register these tokens. Every variant must regenerate interaction/history text from original item IDs; replacing only an index file is insufficient.

`configs/geosid/smoke.json` uses a tiny CPU network, three eight-entry codebooks and two epochs for software checks only. An example synthetic invocation is:

```sh
python -c "import numpy as np; np.save('smoke.npy', np.random.default_rng(1).normal(size=(48,12)).astype(np.float32))"
python -m sid_generation.geosid train --config configs/geosid/smoke.json --data_path smoke.npy --ckpt_dir outputs/geosid/smoke --geo_weight 0.03
```

Enabled geometry requires at least two items in every batch. The training CLI rejects singleton batches/tails before model construction; choose a compatible batch size. No data is silently dropped, and this check is skipped when geometry is disabled or its weight is zero.

The smoke weight is arbitrary and only exercises the software path; it is not an experimental calibration or a recommended research setting. Official comparisons retain the common three-by-256 codebook architecture.

## Checkpoints and recovery

Old RQVAE state dictionaries load with `strict=True`, without new required keys. Legacy outer training checkpoints are accepted as non-exact continuation and default to disabled geometry when GeoReg configuration is absent. Legacy inference and exact new-training recovery are separate contracts.

New outer checkpoints save model/optimizer/scheduler state, completed zero-based epoch, update count, configuration, VQ `initted` flags, Python/NumPy/PyTorch RNG, initialized CUDA RNG, independent GeoReg RNG, DataLoader/sampler RNG states and sharing topology, loader layout, standard sampler settings, and explicit model/training behavior configuration. No research or budget-runner dependency is included. Restoring `initted` prevents KMeans from overwriting loaded codebooks. Configuration and completeness checks precede restoration. Earlier local experimental epoch checkpoints lacking this complete contract are rejected as incomplete rather than certified exact; their model weights remain strictly loadable for inference. Original upstream outer checkpoints still load as non-exact continuation.

```sh
python -m sid_generation.geosid train --config configs/geosid/residual.json --geo_weight LAMBDA --data_path catalog.npy --ckpt_dir outputs/geosid/resumed --resume CHECKPOINT.pth
```

Exact continuation is tested at completed epoch boundaries with `num_workers=0`, unchanged data/order, architecture, schedule and batch layout. Standard PyTorch RandomSampler/SequentialSampler with the standard BatchSampler are supported; custom samplers are rejected for exact resume. Dedicated and global-default loader RNG modes are both covered. Keep the same dataset content: the core checkpoint checks length/layout but does not authenticate a dataset hash. Persistent worker RNG is unsupported; other multiworker restores are explicitly not certified exact. No permutation/cursor is stored, so **arbitrary-step exact resume is not supported**. An interruption inside an epoch requires re-running from a completed epoch checkpoint and accounting for the repeated work.

## Verification and interpretation

```sh
python -m unittest tests.test_georeg tests.test_georeg_model tests.test_georeg_resume tests.test_georeg_cli -v
```

CUDA tests require explicit `GEOSID_TEST_CUDA=1` and an available CUDA device. Run `python -m unittest tests.test_georeg.CudaTests tests.test_georeg_model.CudaModelTests tests.test_georeg_resume_cuda.CudaEpochResumeTests -v` after setting that variable. Respect your own compute budget; no budget orchestrator is part of this feature. CPU tests do not require downloading a language model. The compatibility test obtains the original RQVAE source from the pinned Git commit, so that commit must be present in the local clone.

The tests cover an independent NumPy reference, uneven projection chunks, no hidden batch/projection multiplier, raw and normalized FP64 gradcheck, finite constant batches, clear invalid-input errors, autocast precision, separate RNG, original STE versus auxiliary Jacobian, later-level encoder gradients, absent GeoReg-only codebook gradients, and unchanged base gradients/updates. Disabled-mode outputs, SID, loss, gradients, update and global RNG are compared exactly against the pinned upstream model, including dropout and Sinkhorn. The declared FP32 residual comparison tolerance is `atol=2e-6`, `rtol=2e-5`; algebraically equivalent STE expressions need not be bitwise equal.

Tokenizer diagnostics report reconstruction MSE, code usage, entropy in nats, normalized entropy `H/log(K)` (defined as zero for a single-entry codebook), effective ratio `exp(H)/K`, Gini, per-prefix unique counts and duplicate excess `(N-U)/N`, collision-item fraction, same-prefix pair fraction, fixed-direction normalized CF, original residual RMS, covariance spectrum and effective rank. These are distinct metrics. Prefix sharing and balanced codes do not by themselves demonstrate better recommendations.

The local matched research study is separate from this core contribution: nine fixed-budget tokenizer runs and two A/C downstream seed pairs were completed; the four ranking metrics improved for seed 42 and declined for seed 2026. Stable recommendation benefit and convergence were not established. Software tests, synthetic smoke, finite geometry or final-SID uniqueness do not establish algorithmic benefit. Negative, unconverged and incomplete results must remain visible.
