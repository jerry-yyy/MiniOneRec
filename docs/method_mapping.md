# Mapping GeoSID to MiniOneRec

This is an independent RQ-VAE adaptation, not a claim to reproduce unpublished official GeoSID code. The implementation is based on MiniOneRec commit [`0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1`](https://github.com/AkaliKong/MiniOneRec/commit/0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1). The source audit on 2026-11-16 found the [GeoSID repository](https://github.com/hexinrong122/GeoSID) at `540bd73e31e4deaef426d8773c6e1de3cc4a6a2c`, containing README.md and stating that code was forthcoming. No unpublished implementation was copied.

| Aspect | Source fact | MiniOneRec implementation or explicit choice |
|---|---|---|
| Base system | GeoSID's RVQ experiment uses GRID; its OPQ experiment uses RPG. | Reuse MiniOneRec's existing RQVAE, quantizers and Trainer. Restrict this change to RQ-VAE. |
| Geometry target | The paper matches projected empirical CF real/imaginary components to the standard Gaussian CF. | Same target, `exp(-t²/2)`, applied to auxiliary normalized residuals. |
| Directions and grid | Appendix A.3 specifies unit Gaussian directions, default 256 projections, and 17 linear frequencies through 3.0. | Defaults match these values; configurable for software testing. |
| Frequency quadrature | The paper describes trapezoidal integration weighted by the Gaussian CF. | Normalize `q * phi` to sum to one. This is an explicit scaling convention. |
| Residual normalization | The paper states level-wise normalization without publishing the exact formula needed here. | Per-coordinate batch centering and one overall differentiable RMS with `eps=1e-6`. No whitening or per-projection z-score. |
| Reduction | Exact implementation reduction is not established by the unpublished code. | Mean over projections and levels; normalized frequency weights; no multiplier by batch size. Named `cf_mean`. |
| Placement | Residual-level matching is part of GeoSID's RVQ design. | Regularize the three quantizer inputs `z`, `z-e1`, `z-e1-e2`; exclude the final tail. Global-only is the placement ablation. |
| STE handling | The paper does not define MiniOneRec's gradient repair. | Build a separate residual path from the one original forward's indices, subtracting detached code vectors. Each residual's derivative to `z` is identity; GeoReg contributes no direct codebook gradient. |
| Base objective | MiniOneRec uses reconstruction plus weighted RQ quantization/commitment loss. | Preserve both original graphs and add `lambda * G` separately, outside `quant_loss_weight`. |
| Precision and randomness | These runtime contracts are not claimed as official implementation details. | Explicitly disable autocast within geometry; use FP32 in training, preserve FP64 tests; independent CPU RNG; fixed independent diagnostic projections shared across variants. |
| Export | Existing MiniOneRec helpers already provide stable four-token SID formatting. | Reuse the pure format helper only. Nearest-neighbor raw export; stable local collision-group suffix; singleton `d0`. No FLASH algorithm or collision-repair Sinkhorn. |
| Compatibility | Upstream VQ `initted` is not in the original model state dictionary. | No new persistent model keys; restore initialization flags and full new training state in an outer checkpoint. Legacy inference remains separate from exact epoch recovery. |
| Hyperparameter selection | This local protocol is not a claim about the paper's optimal lambda. | One seed-2024 gradient calibration and three residual candidates, with matched baseline continuation and holdout reconstruction constraint. Freeze one absolute lambda for both B and C. |

The [LeJEPA minimal SIGReg example](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/MINIMAL.md) is only a mathematical cross-check. Its example's quadrature scaling and explicit batch multiplier differ from this project's `cf_mean`; those factors are intentionally not imported.

## Scope relative to existing contributions

At the source-audit snapshot, [MiniOneRec PR #76](https://github.com/AkaliKong/MiniOneRec/pull/76) remained open and concerned balancing and dead-code recovery; neither is included here. [PR #83](https://github.com/AkaliKong/MiniOneRec/pull/83) and [PR #84](https://github.com/AkaliKong/MiniOneRec/pull/84) were merged and supplied other SID algorithms and reusable format/alignment utilities. Reusing formatting does not introduce those algorithms into this experiment. The upstream repository uses Apache License 2.0 and its README welcomes contributions. This local work does not itself publish a PR, issue, comment or maintainer message.

## What the checks establish

Mathematical tests establish the documented CF objective and normalization. Gradient tests establish the intended surrogate Jacobian and unchanged base update path. Compatibility tests compare to pinned upstream code; epoch-recovery tests establish reproducible continuation only at the declared boundary and configuration. Synthetic command tests establish train/export/diagnose integration and item alignment. None of these establishes a recommendation gain.

Raw three-token codes are used for tokenizer quality diagnostics. Final four-token zero collision is a common identity-disambiguation guarantee and must not be reported as an algorithmic win. Formal tokenizer runs are transductive over a known catalog; they are not a cold-start evaluation. Downstream item-level HR/NDCG must be measured using matched clean-base initialization, shared vocabulary and data, and fully scanned constrained candidates. Results, resource limits, convergence and unfinished work are reported separately from software correctness.

For formulas, options, standalone commands and recovery limitations, see [geosid.md](geosid.md). For the primary paper, see [He et al., GeoSID, EMNLP 2026](https://aclanthology.org/2026.emnlp-main.960.pdf).
