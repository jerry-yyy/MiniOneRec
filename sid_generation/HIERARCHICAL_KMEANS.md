# SEATER-style balanced hierarchical K-Means

## Overview

This optional CPU SID builder independently implements SEATER-style balanced
hierarchical K-Means and exports MiniOneRec's existing `Dataset.index.json`
format. It needs no neural-network training or GPU. Existing RQ-VAE, trainers,
SID builders and downstream data consumers are unchanged.

The contribution covers tree construction and SID adaptation only. It does not
include SEATER's recommendation model, contrastive/alignment/ranking objectives,
SFT or RL. See [validation results](HIERARCHICAL_VALIDATION.md) for measured
checks, pinned data sources and reproducibility commands.

## Algorithm

For a cluster containing N items, stop when `N <= K`. Otherwise, use
`KMeansConstrained` to form K children with:

```text
size_min = floor(N / K)
size_max = floor(N / K) + 1
```

Recursively cluster each child's **original item embeddings**. The backend
uses constrained assignments with the squared Euclidean K-Means objective;
there is no residual subtraction. RQ-KMeans instead quantizes successive
residual vectors. Defaults are `K=10` and `seed=0`.

Every split has exactly K nonempty children. A terminal cluster contains at
most K item leaves; its item edges are not another constrained K-way split.
Every item occurs exactly once. Divisible N still uses the bounds above, with
the total size forcing equal children. Duplicate and identical vectors use the
same constrained solver. Infeasible bounds, solver failures and invalid
assignments raise errors, with no ordinary K-Means or random-grouping fallback.

These bounds follow paper Algorithm 1 and the official balanced implementation.
This adapter explicitly handles `N <= K` at the root, omits redundant singleton
cluster tokens, and preserves natural variable-length paths. It does not force
the paper's equal-length identifier presentation onto arbitrary catalog sizes.

## SID Representation

Each non-root cluster containing multiple items receives a globally unique
`<node_ID>` token, including terminal groups. Each item receives a globally
unique `<leaf_RANK>` token. The SID is the root-to-item cluster-token path
followed by that item's leaf token. The root and singleton cluster tokens are
omitted. For example, an illustrative five-item, K=2 tree could export:

```json
{
  "a": ["<node_1>", "<leaf_0>"],
  "b": ["<node_1>", "<leaf_1>"],
  "c": ["<node_2>", "<leaf_2>"],
  "d": ["<node_2>", "<node_4>", "<leaf_3>"],
  "e": ["<node_2>", "<node_4>", "<leaf_4>"]
}
```

IDs are canonicalized to strings and sorted lexicographically before fitting
(`"1", "10", "2"`, for example). Child clusters are ordered by their smallest
canonical ID; cluster IDs follow preorder traversal and leaf IDs follow
canonical item rank. Item IDs are never inferred from row positions. Moving
rows together with their explicit IDs preserves the mapping in the same
numerical environment, and legal item ID `0` is retained.

Distinct node tokens preserve SEATER's separate representation for each tree
node. Distinct leaf tokens identify items and guarantee unique, prefix-free
complete SIDs without a collision suffix. Reusing local branch-position tokens
could encode paths differently, but would change these per-node semantics.
SID uniqueness alone is not evidence of recommendation quality.

The added vocabulary is `items + token-bearing cluster nodes`: 3,686 + 1,110 =
4,796 tokens in the validated catalog. For language-model hidden width d, the
logical additional rows cost `4796*d` parameters with tied input/output
embeddings, or `2*4796*d` when untied, excluding bias. At d=896 this is 4,297,216
or 8,594,432 parameters (8.196 or 16.393 MiB at 16 bits). Existing spare rows
can reduce the actual resize delta; these are arithmetic weight costs, not
measured model or training memory. Vocabulary growth is linear in catalog size.

SEATER's shared Start token is replaced by MiniOneRec's prompt. Existing
newline/EOS/padding handling replaces SEATER's End/Padding tokens. Paths are
neither padded to a fixed depth nor supplemented with invented intermediate
tokens.

## Installation

Use an isolated environment; validation used Python 3.12.5 on Windows:

```bash
python -m venv .venv-hierarchical
# Windows PowerShell:
.venv-hierarchical/Scripts/python.exe -m pip install -r sid_generation/requirements-hierarchical.txt
# POSIX: use .venv-hierarchical/bin/python instead.
```

In subsequent commands, `python` means that environment's interpreter. The
builder requires NumPy, scikit-learn, k-means-constrained and psutil; PyTorch
and FAISS are unnecessary for construction. The optional integration checks
need `requirements-hierarchical-test.txt` and a separately installed CPU
PyTorch wheel; see [the validation setup](HIERARCHICAL_VALIDATION.md#environment).

## Input Format

- `--data-path`: a nonempty, finite, real-valued `(N, D)` NumPy `.npy` array.
  Loading disables pickle. The solver works in float32.
- `--item-ids`: a JSON list with one unique string or integer ID per embedding
  row, in **embedding-row order**, such as `["42", "0", "1001"]`. Metadata JSON
  key order is not evidence of embedding alignment.
- `--dataset-name`: letters, digits, underscores or hyphens, beginning with a
  letter or digit; this becomes the artifact filename prefix.
- Optional `--dataset-dir`: `Dataset.item.json` with exactly the catalog's keys
  and available `Dataset.{train,valid,test}.inter` companions. Every referenced
  item is checked. This converter-compatible path requires canonical
  non-negative integer item IDs; the standalone builder also supports other IDs.

Catalog SHA256 fingerprints include canonical IDs, shape and original numeric
values represented as little-endian float64. Integers outside `[-2**53, 2**53]`
and floating-point precision above float64 are rejected to avoid ambiguous
fingerprints.

## Usage

Run from the repository root. Each output directory must be new; existing
directories are never overwritten. Use
`python -m sid_generation.hierarchical_cli --help` to inspect the commands.

### Fit

```bash
python -m sid_generation.hierarchical_cli fit \
  --data-path embeddings.npy --item-ids item_ids.json \
  --dataset-name MyCatalog --output-dir outputs/hierarchical \
  --branch-factor 10 --seed 0
```

Add `--dataset-dir prepared/` to validate and copy catalog metadata and
interaction companions for the existing MiniOneRec converter.

### Encode

```bash
python -m sid_generation.hierarchical_cli encode \
  --data-path embeddings.npy --item-ids item_ids.json \
  --dataset-name MyCatalog \
  --model-path outputs/hierarchical/MyCatalog.tree.json \
  --output-dir outputs/hierarchical-reencoded
```

`encode` requires the same complete catalog: dimensions, item membership and
the ID-to-vector fingerprint must match. Row permutations are allowed when IDs
and vectors move together. Unknown/missing items and changed vectors are
rejected. The saved JSON tree reconstructs the fixed mapping; no centroids,
pickle objects or unvalidated new-item insertion are involved.

### Diagnose

```bash
python -m sid_generation.hierarchical_cli diagnose \
  --model-path outputs/hierarchical/MyCatalog.tree.json \
  --manifest-path outputs/hierarchical/MyCatalog.manifest.json
```

This validates tree invariants and reports statistics. Supplying the manifest
also verifies complete artifact-hash coverage and tree/index/row-map agreement.

## Output Files

| File | Contents |
| --- | --- |
| `Dataset.index.json` | Item ID to variable-length SID token list |
| `Dataset.tree.json` | Validated topology, canonical IDs, configuration and catalog fingerprint |
| `Dataset.item_ids.json` | Explicit IDs in the original input-row order |
| `Dataset.tree_stats.json` | Depth/size distributions, token counts, collisions, runtime/RAM and fit split diagnostics |
| `Dataset.manifest.json` | Algorithm/version, dependency versions, input/mapping SHA256, configuration and output hashes |

Validated dataset companions are copied when supplied. The manifest hashes all
other output artifacts, excluding itself to avoid self-reference. Files are
staged and published together by renaming a new directory under the shared SID
writer lock. The fixed-width `build_indexes()` and existing builders are not
changed to accommodate variable-length paths.

## MiniOneRec Integration

The index is consumed directly by existing TokenExtender, dataset and prefix
constraint code. Registration must preserve each node/leaf token atomically
and the complete SID length. Allow sufficient generation length for the
catalog's longest path, newline and EOS. With metadata and `.inter` companions
copied during `fit`, run the existing converter:

```bash
python convert_dataset.py --data_dir outputs/hierarchical \
  --dataset_name MyCatalog --output_dir outputs/hierarchical-converted
```

For the public Industrial_and_Scientific catalog, run
`python -m sid_generation.prepare_hierarchical_data` to obtain pinned inputs,
validate the producer's numeric row-order contract including item 0, and
prepare an explicit row map and `.inter` companions from original-ID CSV
columns. Use `--offline` to revalidate existing downloads. Do not use converter
sampling or `--keep_longest_only` when checking all source rows.

`validate_hierarchical_integration.py` checks the actual TokenExtender,
dataset methods, evaluator callbacks and CPU prefix masks without loading
model weights. `benchmark_hierarchical.py` reproduces the measured
fit/reload/conversion workflow from prepared local data. Full commands, source
revisions, SHA256 values and validation limits are in
[HIERARCHICAL_VALIDATION.md](HIERARCHICAL_VALIDATION.md).

## Configuration

| Fit option | Default | Constraint or meaning |
| --- | --- | --- |
| `--branch-factor` | 10 | K >= 2; children per constrained split |
| `--seed` | 0 | Integer in `[0, 2**32-1]` |
| `--n-init` | 10 | Positive restart count per split |
| `--max-iter` | 300 | Positive iteration limit per restart |
| `--tol` | 0.0001 | Finite, non-negative convergence tolerance |
| `--n-jobs` | 1 | Positive backend job count |

BLAS/OpenMP threads are bounded to one during fitting. The default single
solver job differs from SEATER's `n_jobs=-2`; use the same dependency versions
and numerical environment for repeatability. Version 0.9.0 of
k-means-constrained was validated here, rather than SEATER's 0.7.2 environment.
K=10 follows the official function default and is not claimed to be optimal.

## Limitations

- This is a fixed-catalog builder. Changed membership requires a new fit;
  there is no validated incremental insertion or cross-catalog SID stability.
- Constrained assignments, repeated starts and high-dimensional inputs can be
  costly. Cross-version or cross-hardware bitwise equivalence is not promised.
- Unique leaves and balancing establish identity and capacity properties,
  not recommendation accuracy. No HR/NDCG improvement is claimed.
- Node/leaf vocabulary growth also affects optimizer state and activations.
  MiniOneRec's existing embedding-gradient mask does not establish allocations
  limited to new rows, and its decoder still computes full-vocabulary logits.
  K legal children do not imply SEATER's candidate-only inference cost.
- Tokenizer and CPU-mask checks do not establish model-level generation or
  training behavior. No model weights, `model.generate`, SFT or RL were used.
- Existing-directory replacement is intentionally unsupported by this CLI;
  choose a fresh output directory, including on Windows.

## References

- Si et al., *Generative Retrieval with Semantic Tree-Structured Identifiers
  and Contrastive Learning*, SIGIR-AP 2024.
  [DOI: 10.1145/3673791.3698408](https://doi.org/10.1145/3673791.3698408).
  [Paper Algorithm 1 and section 3.2.2](https://arxiv.org/pdf/2309.13375v2).
- [Official SEATER repository](https://github.com/Ethan00Si/SEATER_Generative_Retrieval),
  audited revision `b12cd29585720f891b1a8f41dacec5e0554fac6f`;
  [balanced tree builder](https://github.com/Ethan00Si/SEATER_Generative_Retrieval/blob/b12cd29585720f891b1a8f41dacec5e0554fac6f/utils/build_tree.py).

The audited SEATER revision had no explicit software license (`LICENSE` or
`COPYING`), and repository metadata reported no license. This is an independent
implementation based on the algorithm references; no author source was copied,
ported or vendored. MiniOneRec's existing [Apache-2.0 LICENSE](../LICENSE) is
unchanged.
