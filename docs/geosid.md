# Optional quantizer-aware GeoSID framework

GeoSID shapes continuous representations in the spaces a quantizer actually
uses. This independent MiniOneRec adaptation offers two different optimization
paths behind a common CLI. It is **off by default** (`mode=none`, `weight=0`).
It does not establish recommendation improvements or reproduce an official
implementation. Original tokenizer entry points and all `rq/` files are unchanged.

## Architecture and support

| Component | Responsibility |
|---|---|
| `sid_generation/geosid/core.py` | Quantizer-independent CF matching, normalization, precision, dedicated projection RNG, named-block mean |
| `adapters/rqvae.py` | Subclass the existing RQVAE; preserve its base graph and expose an explicit auxiliary residual loss |
| `rqvae_backend.py` | Small opt-in AdamW driver, inference artifact, encoding and fixed-projection diagnostics |
| `adapters/rq_opq.py` | Experimental fixed-catalog RQ-prefix/OPQ-tail shaping; original FAISS model by composition |
| `__main__.py` | Explicit backend selection, input validation and publication through existing SID utilities |
| `compare_opq.py` | Fixed synthetic original/refit-only/shaping comparison; CPU only |

Supported backends are `rqvae` and **experimental, transductive** `rq-opq`.
They share geometry primitives, artifact conventions and catalog export, but
not an optimizer or differentiability assumption. See [the feasibility matrix](geosid_adaptation.md)
for the other five methods; no training backend is advertised for them.

The RQ driver is deliberately small: AdamW, constant learning rate, zero default
weight decay, unit gradient clipping and explicit epoch count. It calls the
original model's `compute_loss`; it does not copy or inherit the full upstream
Trainer with its selection/scheduler/checkpoint machinery. There is no universal
resume framework. Saved models are inference artifacts, not exact-resume claims.

## Shared geometry convention

For each `B x d` block, with `B >= 2`, subtract its coordinate-wise batch mean
and divide by `sqrt(mean(centered**2) + eps**2)`. Both statistics are
differentiable. Use 256 Gaussian directions normalized to unit columns and 17
linear frequencies from 0 through 3. Compare empirical cosine/sine means with
the standard Gaussian characteristic function. Weight squared discrepancies by
Gaussian-CF-weighted trapezoid coefficients normalized to sum to one. Average
over directions and blocks; there is no batch-size multiplier.

Normalization is a single overall RMS, not whitening. The normalized tensor
is only used for geometry. Projection chunks retain complete-batch CF moments.
Geometry explicitly disables autocast, uses FP32 for FP16/BF16/FP32 inputs and
retains FP64 for numerical checks. Constants retain CPU FP64 precision across
module casts. Unsupported devices and nonfinite arithmetic fail explicitly.
Projection sampling uses a dedicated CPU generator; diagnostics use fixed
directions from a separate declared seed.

These `cf_mean` normalization/reduction/gradient details are adaptation
conventions. The [GeoSID paper](https://aclanthology.org/2026.emnlp-main.960.pdf)
does not fully specify them, and the [official repository](https://github.com/hexinrong122/GeoSID)
contains only a code-coming-soon README at `540bd73e31e4deaef426d8773c6e1de3cc4a6a2c`
(checked 2026-11-17).

## RQ-VAE: neural residual optimization

The adapter inherits the original encoder, codebooks, decoder and loss. Its
ordinary call returns the original `(out, rq_loss, indices)` tuple. For an
explicit `return_aux=True`, a fourth dictionary supplies geometry. Disabled or
zero-weight calls skip residual construction, geometry and random sampling.
State-dict keys and strict original-weight loading remain compatible.

An enabled auxiliary call uses one original quantization pass. Starting with
`r=z`, regularize each pre-quantization residual, then subtract the selected
code vector **detached** before the next level. For three levels the locations
are `z`, `z-e1`, `z-e1-e2`; the final tail is excluded. This separate surrogate
gradient avoids the original STE's cancellation of later encoder gradients
without changing the original base graph or directly updating codebooks.

`global` regularizes only `z`; `residual` averages all quantizer inputs:

```text
loss = original_reconstruction_and_quantization_loss + geo_weight * geometry
```

Geometry is outside `quant_loss_weight`. The driver retains every item and
rejects singleton batches when geometry or BatchNorm requires at least two.

## RQ-OPQ: experimental fixed-catalog block optimization

MiniOneRec's hybrid fits an additive RQ prefix and then an OPQ/PQ tail. These
are not residual levels of a neural encoder. Let `p` be the frozen RQ prefix
reconstruction and `A` the original OPQ matrix (row-vector rotation `r @ A.T`).
Initialize an item table `Z = (x-p) @ A.T` and optimize only `Z`:

```text
y = p + Z @ A
loss = semantic_weight * MSE(y, x)
     + cosine_weight * mean(1 - cosine(y, x))
     + geo_weight * mean(GeoReg(each equal-width PQ block of Z))
```

The prefix assignments/codebooks and OPQ matrix stay fixed. FAISS is outside
autograd. After shaping, fit a fresh PQ with the declared seed/iteration budget;
export the frozen prefix plus `PQEncode(Z)` without another rotation. Measure
final quantized reconstruction using `p + PQDecode(code) @ A`, separately from
unquantized semantic drift and tail drift. The prefix can dominate full-vector
cosine, so low semantic loss alone does not guarantee an accurate residual tail.
Cosine uses `eps=1e-8`: a zero target contributes a constant loss of one. Reports
separate exact-zero and near-zero (`norm <= 1e-8`) counts, and report cosine
drift on targets above that threshold; semantic diagnostics use
FP64 accumulation, while the optimization arithmetic remains FP32.

This adapts paper section 3.4, equations (3)-(7), and Algorithm 2. Full-embedding
cosine for a mixed prefix/tail and a final PQ refit are **new conventions**;
Algorithm 2 does not specify the refit schedule. The three-arm smoke comparison
therefore includes a refit-only control with the same prefix, rotation, seed and
PQ budget. Geometry is not a direct extra loss inside FAISS clustering.

Enabled artifacts bind canonical item IDs to original FP32 row fingerprints,
optimized rows and codes. Reordered/subset known-item queries are permitted;
unknown IDs or changed embeddings are rejected. There is **no learned inductive
transform for new items**. Catalog changes require refitting, rebuilding suffixes
and downstream remapping. Memory and optimizer state scale with `N x D`; this
CPU/full-catalog implementation is not certified for large catalogs or CUDA.
Ordinary zero-weight/disabled operation delegates to the original RQOPQ without
shaping or refitting. An explicitly requested refit-only control is a separate
experimental operation, never an implicit zero-weight behavior.

## Install and run

Use the existing RQ dependencies (PyTorch, NumPy, scikit-learn). Tests require
pytest. FAISS is optional and needed
only for `rq-opq`; install the upstream pin in a dedicated environment:

```sh
python -m pip install torch==2.6.0 numpy==1.26.4 scikit-learn==1.6.1 pytest==8.3.5
python -m pip install -r sid_generation/requirements-faiss.txt
python -m sid_generation.geosid --help
```

Create synthetic local inputs (no data is included in this change):

```sh
python -c "import json,numpy as np; np.save('smoke.npy',np.random.default_rng(2024).normal(size=(256,8)).astype('float32')); open('rows.json','w').write(json.dumps(list(range(256))))"
```

Both example configs have **zero weight**. Explicitly enable geometry for a
software smoke run; these weights are illustrative, not tuned recommendations:

```sh
python -m sid_generation.geosid train --backend rqvae --config configs/geosid/rqvae-smoke.json --geo-weight 0.03 --data-path smoke.npy --item-ids rows.json --dataset-name toy --output-dir outputs/geosid/rqvae
python -m sid_generation.geosid train --backend rq-opq --config configs/geosid/rq-opq-smoke.json --geo-weight 0.1 --data-path smoke.npy --item-ids rows.json --dataset-name toy --output-dir outputs/geosid/opq
python -m sid_generation.geosid export --backend rqvae --model-path outputs/geosid/rqvae/toy.tokenizer.npz --data-path smoke.npy --item-ids rows.json --dataset-name toy --output-dir outputs/geosid/rqvae-export
python -m sid_generation.geosid diagnose --backend rq-opq --model-path outputs/geosid/opq/toy.tokenizer.npz --data-path smoke.npy --item-ids rows.json --report outputs/geosid/opq-diagnostics.json
python -m sid_generation.geosid.compare_opq --output outputs/geosid/paired-opq.json
```

Export and diagnose also work with the other backend by selecting it explicitly
and using its own artifact. Config keys are `model`, `geometry`, `training`,
`seed`, and optional matching `backend`; unknown fields are rejected.

## Catalog and artifact contract

`--item-ids` is mandatory: a unique string/integer JSON list in embedding-row
order. Metadata dictionary order and old possibly colliding SIDs are never used
to infer it. Duplicate embedding rows are valid distinct items; duplicate IDs
are invalid. Inputs must be finite real matrices representable in FP32.

Publication reuses `sid_common.build_indexes` and `sid_publish`: raw codes,
raw/final JSON indexes, item IDs, statistics, inference model and hash manifest
are produced together. Default `append` resolves collisions by stable
lexicographic item-ID rank **within each raw-SID group**, with rank zero for
singletons. Following the existing shared exporter, the suffix is added to all
items when any raw collision exists; if none exists no suffix is necessary.
Raw SID length is backend-dependent, not universally three. Final uniqueness
under append is an identity guarantee, not a geometry improvement. `allow` and
`error` retain the existing shared export semantics.

Use new output directories. Atomic replacement of existing directories depends
on the upstream publisher's platform support; on Windows it requires a new
directory. No downstream CSV/SFT conversion is performed. Regenerate downstream
histories/targets from original item IDs for any new catalog mapping.

## Validation and limits

See [validation evidence](geosid_validation.md) for actual results and commands.
The earlier RQ-only study and its 65 CPU/3 CUDA candidate tests are historical
evidence, not tests or experiments on this refactor. No complete SFT was rerun.
Both positive and negative distortion/geometry results belong in the report.
This feature does not add OPQ to the neural trainer, FLASH decoding, RL,
codebook balancing, dead-code recovery or general checkpoint recovery.
