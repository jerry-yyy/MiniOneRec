# Hierarchical K-Means SID construction

## Overview

This CPU builder partitions an embedding catalog into a capacity-balanced
tree and exports MiniOneRec's existing `Dataset.index.json` format. Select
`balanced` for the original SEATER-style constrained solver, `greedy` for a
global distance-ordered capacity assignment, or `hybrid` to keep the constrained solver
on smaller nodes. The builder runs independently of model training.

The contribution covers tree construction and SID adaptation only. It does not
include SEATER's recommendation model, contrastive/alignment/ranking objectives,
SFT or RL. See [validation results](HIERARCHICAL_VALIDATION.md) for measured
checks, pinned data sources and reproducibility commands.

## Algorithm

At each node, stop if its N items fit in a terminal group (`N <= K`), including
at the root. Otherwise partition that node's **original float32 working
vectors**, with no normalization or residual subtraction. For
`q, r = divmod(N, K)`, exactly r children have q+1 items and K-r have q.

| Method | How a nonterminal node splits |
| --- | --- |
| `balanced` | SEATER Algorithm 1. Fit `KMeansConstrained` with `size_min=q` and `size_max=q+1`. Use this to reproduce the original Balanced candidate. |
| `greedy` | Fit ordinary scikit-learn K-Means to the node with k-means++ initialization and Lloyd iterations. Compute all N×K squared Euclidean distances. Visit candidates in global ascending order of (distance, item position, original centroid index). Assign an unassigned item only if that centroid still has room. The first r original centroids have capacity q+1. Stop when all items are assigned. There is one capacity pass, with no subsequent centroid refit. |
| `hybrid` | Apply greedy if this node has `N > greedy_threshold`. Use constrained otherwise. At the default threshold a 2,000-item node uses constrained. Siblings can select different solvers. |

Item positions follow canonical item-ID order within each node. Sorting the
final groups by minimum ID happens **after** capacity assignment. Identical
vectors and duplicate centroids use the same tie and capacity rules. Invalid
centers, nonfinite distances and invalid labels fail the fit.

For Greedy and Hybrid we follow the pinned Re-SEATER builder's global
candidate order and node-local test; the paper describes a nearest-available
assignment and summarizes the switch by level. The trainer sets the threshold
to 0 for speedup and 2,000 for hybrid. The reference uses FAISS above 100
items and scikit-learn for smaller nodes. Here every greedy split uses CPU
scikit-learn, the configured seed and the supplied embeddings. Comparing
identifiers byte-for-byte with the author's FAISS runs is outside this contract.

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

The validation environment used native ARM64 Python 3.12.4 on an Apple M1.
From the repository root, create and activate an isolated environment.

```bash
python3.12 -m venv .venv-hierarchical
source .venv-hierarchical/bin/activate
python -m pip install -r sid_generation/requirements-hierarchical.txt
```

In subsequent commands, `python` means that environment's interpreter. The
builder requires NumPy, scikit-learn, k-means-constrained and psutil.
The integration tests use PyTorch on CPU. The upstream regression suite also
uses FAISS. See the [test installation and commands](HIERARCHICAL_VALIDATION.md#reproducing-the-checks).

## Input Format

- `--data-path`: a nonempty, finite, real-valued `(N, D)` NumPy `.npy` array.
  Loading disables pickle. The solver works in float32.
- `--item-ids`: a required JSON array with one unique string or integer item
  ID per embedding row.
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

### Preparing item_ids.json

Entry `i` in `item_ids.json` must be the real item ID for row `i` of
`embeddings.npy`. Metadata JSON key order cannot establish that mapping.
Item ID `0` is valid and must be kept when present.

**A. Confirmed IDs 0 through N−1 in row order.** Use this command only if
you have confirmed that the embedding producer wrote item 0 first, then item
1, and so on without gaps. Run it in the directory containing `embeddings.npy`.

```bash
python -c "import json, pathlib, numpy as np; n = np.load('embeddings.npy', mmap_mode='r', allow_pickle=False).shape[0]; pathlib.Path('item_ids.json').write_text(json.dumps(list(range(n))), encoding='utf-8')"
```

**B. Noncontiguous IDs or reordered rows.** Have the producer save one item
ID per line in `embedding_rows.txt` as it writes the final embedding rows.
Apply any filtering or reordering to the IDs and vectors together. For
example, three rows written for items 42, 0 and 1001 must yield
`["42", "0", "1001"]` in that order. The following Python snippet converts
the saved row record to JSON.

```python
import json
from pathlib import Path
import numpy as np

ids = Path("embedding_rows.txt").read_text(encoding="utf-8").splitlines()
n = np.load("embeddings.npy", mmap_mode="r", allow_pickle=False).shape[0]
if len(ids) != n or len(set(ids)) != n or "" in ids:
    raise ValueError("Expected one unique, nonempty item ID per embedding row")
Path("item_ids.json").write_text(json.dumps(ids), encoding="utf-8")
```

If the row record is missing, recover the order from the producer or export
the embeddings again with their IDs. For the pinned Industrial_and_Scientific
catalog, run `python -m sid_generation.prepare_hierarchical_data` from the
repository root. It uses the verified numeric producer order and writes
`data/hierarchical-source/prepared/Industrial_and_Scientific.item_ids.json`
for use with `--item-ids`.

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

Use `--method greedy` for greedy assignment at every split, or
`--method hybrid --greedy-threshold 2000` for node-local switching. Give
each fit a separate output directory.

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

New publications use format 2, algorithm `hierarchical_kmeans`, version
`2.0.0`. A format-1 tree and its original `balanced_hierarchical_kmeans`
1.0.0 manifest are readable as Balanced. The original v1 manifest omitted
`format_version`; v2 requires it in both files. When a manifest is provided for
diagnosis, or found next to the model during encoding, its original identity,
six- or ten-parameter schema and artifact hashes are verified before migration.
Encoding writes format 2 in a fresh directory and records the source tree's
byte-level SHA256. A standalone tree can be loaded without a manifest. The
manifest records the executed fit backends and counts. An encode performs
zero splits.

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

For Industrial_and_Scientific, the preparation helper also builds `.inter`
companions from the release CSVs' original-ID columns. Use `--offline` to
revalidate existing downloads. Do not use converter
sampling or `--keep_longest_only` when checking all source rows.

`validate_hierarchical_integration.py` checks the actual TokenExtender,
dataset methods, evaluator callbacks and CPU prefix masks without loading
model weights. `benchmark_hierarchical.py` can run additional benchmarks on
prepared local data. For the tested environment, catalog checksums and
reproduction commands, see the [validation report](HIERARCHICAL_VALIDATION.md).

## Configuration

| Fit option | Default | Constraint or meaning |
| --- | --- | --- |
| `--method` | balanced | balanced, greedy or hybrid |
| `--branch-factor` | 10 | K ≥ 2, children per split for all methods |
| `--seed` | 0 | Seed in `[0, 2**32-1]` for both estimators |
| `--tol` | 0.0001 | Finite nonnegative convergence tolerance for both estimators |
| `--n-init`, `--max-iter`, `--n-jobs` | 10, 300, 1 | Constrained restarts, iterations per restart and jobs, all positive |
| `--greedy-threshold` | 2000 | Nonnegative hybrid cutoff applied separately to each nonterminal node |
| `--greedy-n-init`, `--greedy-max-iter` | 5, 20 | Ordinary K-Means restarts and iterations per restart, both positive |

BLAS/OpenMP threads are bounded to one during fitting. The default single
solver job differs from SEATER's `n_jobs=-2`; use the same dependency versions
and numerical environment for repeatability. Version 0.9.0 of
k-means-constrained was validated here, rather than SEATER's 0.7.2 environment.
K=10 follows the official function default and is not claimed to be optimal.

For one greedy node with D features, centroid fitting takes roughly
`O(restarts × iterations × NKD)` arithmetic. Building exact distances costs
`O(NKD)`. A stable sort of the flattened array gives tuple order (comparison
sort upper bound `O(NK log(NK))`), followed by at most NK candidate visits.
The matrix costs 4NK bytes, plus an integer order array (8NK bytes on a
64-bit system), sort workspace and a bounded block of coordinate differences.
The implementation avoids an N×K×D temporary. Peak process memory also
includes embeddings, solver workspace, final-group error evaluation and export.
Tree cost sums the per-node work. Statistics distinguish solver inertia from
WCSS recomputed around the final child means; WCSS never changes an assignment.

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
- Each fit or encode publishes to a fresh output directory.

## References

- Si et al., *Generative Retrieval with Semantic Tree-Structured Identifiers
  and Contrastive Learning*, SIGIR-AP 2024.
  [DOI: 10.1145/3673791.3698408](https://doi.org/10.1145/3673791.3698408).
  [Paper Algorithm 1 and section 3.2.2](https://arxiv.org/pdf/2309.13375v2).
- [Official SEATER repository](https://github.com/Ethan00Si/SEATER_Generative_Retrieval/tree/b12cd29585720f891b1a8f41dacec5e0554fac6f),
  audited revision `b12cd29585720f891b1a8f41dacec5e0554fac6f`;
  [balanced tree builder](https://github.com/Ethan00Si/SEATER_Generative_Retrieval/blob/b12cd29585720f891b1a8f41dacec5e0554fac6f/utils/build_tree.py).
- Valeau et al., *Efficient Optimization of Hierarchical Identifiers for
  Generative Recommendation*. [Re-SEATER paper, version 2](https://arxiv.org/html/2512.18434v2).
  Source semantics were checked at revision `32235415e5abc1fa97d0516199422b5d9a34471b`.
  [Greedy assignment and recursive construction](https://github.com/joshrosie/re-seater/blob/32235415e5abc1fa97d0516199422b5d9a34471b/SEATER_Generative_Retrieval/utils/build_tree_speedup.py)
  and [trainer selection](https://github.com/joshrosie/re-seater/blob/32235415e5abc1fa97d0516199422b5d9a34471b/SEATER_Generative_Retrieval/trainer.py).

The audited original SEATER revision has no explicit software license. The
audited Re-SEATER revision uses Apache-2.0. [Re-SEATER LICENSE at the cited revision](https://github.com/joshrosie/re-seater/blob/32235415e5abc1fa97d0516199422b5d9a34471b/LICENSE)

MiniOneRec independently implements SEATER's Balanced tree construction and
Re-SEATER's Greedy and Hybrid strategies without copying or directly porting
source code from either project. MiniOneRec retains its existing
[Apache-2.0 license](../LICENSE).
