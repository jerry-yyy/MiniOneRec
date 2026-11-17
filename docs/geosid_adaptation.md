# Feasibility across seven SID algorithms

Audited MiniOneRec main: `0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1`
(2026-11-17). Shared output formatting does not imply shared optimization.
The paths below describe the actual implementation in this repository.

| SID algorithm | Fitting mechanism and geometry | Differentiability / reuse | This change |
|---|---|---|---|
| RQ-VAE (`rq/models/rqvae.py`) | Neural encoder/decoder, additive residual codebooks, reconstruction plus quantization/commitment | STE; detached selected-code auxiliary residuals provide later encoder gradients | Supported residual/global adapter; original model and Trainer unchanged |
| RQ-KMeans (`rq/rqkmeans_faiss.py`) | FAISS residual codebooks on fixed embeddings; optional post-fit Sinkhorn mapping | No learned encoder; first-level GeoReg has no continuous parameter to update | Feasibility only; needs a separate free-vector or centroid optimization design |
| Constrained RQ-KMeans (`rq/rqkmeans_constrained.py`) | Capacity-constrained assignment and centroid fitting; additive residuals | Discrete constrained clustering, not a PyTorch loss | Feasibility only; preserve assignment constraints and distinguish imposed balance from geometry |
| RQ-KMeans+ (`rq/rqkmeans_plus.py`) | RQVAE with `x+MLP(x)`, zero final encoder layer, loaded RQ codebooks | Existing neural residual training could reuse the adapter after matching dimensions and initialization | Promising low-cost extension, not exposed; wrapper/serialization/disabled-path tests still needed |
| FLASH SimHash (`rq/flash_sid/simhash.py`, #83) | Fixed Gaussian hyperplanes and sign-bit groups, no fitted quantizer | Training-free; no residual chain or reconstruction decoder | Feasibility only; learned shaping would change the baseline algorithm |
| PRQ-KMeans (`sid_generation/prq_kmeans.py`, #84) | Cosine/top-k centroid updates; selected-direction projection subtraction and renormalization | Projection residual is not `r-c`; no additive input decoder | Feasibility only; ambient Gaussian shaping may conflict with its orthogonal subspace |
| RQ-OPQ (`sid_generation/rq_opq.py`, #84) | FAISS additive RQ prefix, orthogonal rotation, independent PQ tail blocks | Separate free catalog variables after frozen rotation; no differentiation through FAISS | Experimental fixed-catalog tail shaping with explicit semantic constraints and matched refit control |

No train/export backend for the five feasibility-only rows is registered. They
could pass explicitly extracted matrices to the pure geometry core for research
diagnostics, but this is not an implemented end-to-end adapter.

## Paper basis and explicit adaptations

[GeoSID](https://aclanthology.org/2026.emnlp-main.960.pdf), section 3.3,
equations (1)-(2), places geometry at residual quantizer inputs. Section 3.4,
equations (3)-(7), optimizes free OPQ-rotated item vectors with MSE, cosine and
block-averaged geometry. Appendix Algorithm 2 finishes with PQ encoding but
does not specify a post-shaping refit schedule. Exact normalization, reduction
and STE handling remain incomplete in the publication.

Our core's `cf_mean`, the RQ surrogate gradient, full-catalog hybrid OPQ table,
full-vector semantic penalty and final matched PQ refit are explicitly new
adaptation choices. They are not established improvements or official code.
[The official repository](https://github.com/hexinrong122/GeoSID/tree/540bd73e31e4deaef426d8773c6e1de3cc4a6a2c)
had only a README and no released implementation at audit time.

## Organization compared with nearby contributions

- [#83](https://github.com/AkaliKong/MiniOneRec/pull/83) is merged and isolates CPU
  SimHash in `rq/flash_sid/`, with thin launchers and complete-SID integration.
  This change similarly isolates the optional algorithm and tests.
- [#84](https://github.com/AkaliKong/MiniOneRec/pull/84) is merged and isolates
  PRQ/RQ-OPQ in `sid_generation/`, with shared export and publication utilities.
  This change composes its RQOPQ and directly reuses those utilities.
- [#76](https://github.com/AkaliKong/MiniOneRec/pull/76) remained an open draft
  at audit time. It changes the RQ path for usage balancing/dead-code recovery;
  neither mechanism is included here, and that branch is not a dependency.
- The prior `feat/geosid-upstream` at `06eb7fb` modified the original RQVAE,
  RQ CLI and Trainer, including exact epoch recovery. This branch starts from
  unmodified official history, moves geometry to independent adapters and
  drops general recovery support. It preserves the old branch and its evidence.

The Apache-2.0 license and original author history remain intact. The inspected
upstream tree/community profile contained no separate contribution guide,
template, AI-code policy or CLA/DCO requirement; maintainers may request
additional review. This work was prepared with OpenAI Codex assistance and
requires final human review/submission.
