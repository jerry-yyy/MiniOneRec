# Hierarchical K-Means semantic IDs

This optional CPU builder turns item embeddings into tree-structured semantic
IDs (SIDs) and writes MiniOneRec's existing `Dataset.index.json`. It follows
the balanced tree construction of [SEATER](https://doi.org/10.1145/3673791.3698408)
and adds the Greedy and Hybrid strategies from
[Re-SEATER](https://arxiv.org/html/2512.18434v2). SID construction requires
CPU-based K-Means fitting, but no neural-network training, GPU, or
language-model weights.

| Method | How each node is split | When to use it |
| --- | --- | --- |
| `balanced` (default) | Constrained K-Means, as in SEATER | Small and medium catalogs; slowest |
| `greedy` | Ordinary K-Means followed by global distance-ordered, capacity-constrained assignment | Large catalogs; fastest |
| `hybrid` | Greedy on large nodes, constrained K-Means on small ones | A middle ground |

This adds SID construction only. SEATER's recommendation model and training
objectives are not included, and no recommendation-quality improvement is
claimed.

## Installation

From the repository root (validated with Python 3.12):

```bash
python3.12 -m venv .venv-hierarchical
source .venv-hierarchical/bin/activate      # Windows: .venv-hierarchical\Scripts\activate
python -m pip install -r sid_generation/requirements-hierarchical.txt
```

The builder needs NumPy, scikit-learn, k-means-constrained and psutil. It does
not need PyTorch or FAISS.

## Quick start: Industrial_and_Scientific

The helper below downloads the released Industrial_and_Scientific embeddings
and splits from the pinned [MiniOneRec dataset](https://huggingface.co/kkknight/MiniOneRec)
revision, checks their SHA256 values, and prepares everything the builder
needs under `data/hierarchical-source/`. Add `--offline` to re-check files that
are already downloaded.

```bash
python -m sid_generation.prepare_hierarchical_data
```

Then fit the tree and convert the result into training files:

```bash
dataset=Industrial_and_Scientific
src=data/hierarchical-source

python -m sid_generation.hierarchical_cli fit \
  --data-path "$src/Amazon/index/$dataset.emb-qwen-td.npy" \
  --item-ids "$src/prepared/$dataset.item_ids.json" \
  --dataset-dir "$src/prepared" \
  --dataset-name "$dataset" \
  --output-dir outputs/hierarchical-industrial

python convert_dataset.py \
  --data_dir outputs/hierarchical-industrial \
  --dataset_name "$dataset" \
  --output_dir outputs/hierarchical-industrial-csv
```

`--dataset-dir` checks that the metadata and every interaction refer to known
items, then copies them next to the new index so the converter can use the
export directly. The output directory must not exist yet; the builder never
overwrites an earlier export.

To pick a strategy, add `--method greedy` or `--method hybrid` to the `fit`
command. Hybrid uses greedy on nodes with more than `--greedy-threshold` items
(default 2,000).

## Train with the new SIDs

`sft.sh` does not forward command-line arguments, so edit the `sft.py` call
inside it:

| `sft.py` argument | Value |
| --- | --- |
| `--category` | `Industrial_and_Scientific` |
| `--train_file` | `outputs/hierarchical-industrial-csv/train/Industrial_and_Scientific_5_2016-10-2018-11.csv` |
| `--eval_file` | `outputs/hierarchical-industrial-csv/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv` |
| `--sid_index_path` | `outputs/hierarchical-industrial/Industrial_and_Scientific.index.json` |
| `--item_meta_path` | `outputs/hierarchical-industrial/Industrial_and_Scientific.item.json` |

The other launchers take different inputs:

- **RL (`rl.sh`):** point the train, validation and info paths at
  `outputs/hierarchical-industrial-csv/`, and `--sid_index_path` and
  `--item_meta_path` at the two files in the table above.
- **Evaluation (`evaluate.sh`):** point the test and info paths at
  `outputs/hierarchical-industrial-csv/`, and use a model trained on these
  SIDs. The script reads the SIDs from the info file and takes no index or
  metadata path.

New SIDs mean a new
vocabulary: train from SFT again and use that checkpoint for RL and
evaluation. A checkpoint trained on other SIDs cannot be reused. See the
[SFT section of the main README](../README.md#4-sft) for the rest of the
training setup.

`--category` must be one of the categories `sft.py` already knows:
`Industrial_and_Scientific`, `Office_Products`, `Toys_and_Games`, `Sports` or
`Books`. Any other value raises a `KeyError`.

## Using your own embeddings

The builder needs to know which item each embedding row belongs to.
`--item-ids` takes a JSON list whose entry `i` is the item ID of row `i`. Item
ID `0` is valid. Metadata key order does not tell you the row order.

If your embedding script writes items `0` through `N-1` in order, as the
upstream [`amazon_text2emb.py`](../rq/text2emb/amazon_text2emb.py) does, the
list is simply `0..N-1`. The snippet below writes that list after checking
that the metadata IDs are exactly `0` through `N-1`. This checks the metadata
ID range, not the embedding row order. The row order must be established from
the embedding producer.

```bash
python - <<'EOF'
import json, numpy as np
d = "data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific"
n = np.load(f"{d}.emb-qwen-td.npy", mmap_mode="r").shape[0]
meta = json.load(open(f"{d}.item.json", encoding="utf-8"))
assert sorted(meta, key=int) == [str(i) for i in range(n)], "rows are not items 0..N-1"
json.dump(list(range(n)), open(f"{d}.item_ids.json", "w"))
EOF
```

Otherwise, record the item ID of each row when you write the embeddings and
save that list. Pass the folder holding `Dataset.item.json` and the
`Dataset.{train,valid,test}.inter` files as `--dataset-dir` to get a
converter-ready export.

## How the tree and SIDs are built

Starting from the full catalog, every node with more than K items is split into
K children of nearly equal size: if `N = qK + r`, then r children get q+1
items and the rest get q. Each child is clustered again from the original
embeddings, with no residuals or normalization. A node with K or fewer items
becomes a leaf group. K is `--branch-factor` (default 10).

Each non-root cluster with more than one item gets its own `<node_ID>` token,
and every item gets its own `<leaf_RANK>` token. An item's SID is the path of
node tokens from the root, followed by its leaf token. For a five-item tree
with K=2:

```json
{
  "a": ["<node_1>", "<leaf_0>"],
  "b": ["<node_1>", "<leaf_1>"],
  "c": ["<node_2>", "<leaf_2>"],
  "d": ["<node_2>", "<node_4>", "<leaf_3>"],
  "e": ["<node_2>", "<node_4>", "<leaf_4>"]
}
```

Leaf tokens make every SID unique, so no collision suffix is needed. Paths can
have different lengths; they are not padded. Item IDs are sorted as strings
before fitting, so the result does not depend on row order.

Unique tokens make the vocabulary larger than with RQ codebooks, and it grows
with the catalog. The Industrial catalog adds 4,796 tokens (3,686 leaves and
1,110 nodes), compared with 768 for three 256-entry codebooks. Every SID there
has four tokens.

## Exported files

| File | Content |
| --- | --- |
| `Dataset.index.json` | Item ID to SID tokens, used by SFT, RL and evaluation |
| `Dataset.tree.json` | The fitted tree, used to re-export or inspect it |
| `Dataset.item_ids.json` | Item ID of each input row |
| `Dataset.tree_stats.json` | Depth, cluster-size and vocabulary statistics, plus fit timing |
| `Dataset.manifest.json` | Configuration, library versions and file hashes |

The `.item.json` and `.inter` files are copied when `--dataset-dir` is given.

## Reuse or inspect a saved tree

The tree describes a fixed catalog. `encode` regenerates the same index from
the same items and embeddings, in any row order. Adding, removing or changing
items requires a new `fit`.

```bash
python -m sid_generation.hierarchical_cli encode \
  --data-path "$src/Amazon/index/$dataset.emb-qwen-td.npy" \
  --item-ids "$src/prepared/$dataset.item_ids.json" \
  --dataset-name "$dataset" \
  --model-path "outputs/hierarchical-industrial/$dataset.tree.json" \
  --output-dir outputs/hierarchical-industrial-reencoded

python -m sid_generation.hierarchical_cli diagnose \
  --model-path "outputs/hierarchical-industrial/$dataset.tree.json" \
  --manifest-path "outputs/hierarchical-industrial/$dataset.manifest.json"
```

`diagnose` checks the tree and prints its statistics. With a manifest, it also
verifies every exported file against the recorded hashes. Trees written by the
first, balanced-only version of this builder can still be read.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--method` | `balanced` | `balanced`, `greedy` or `hybrid` |
| `--branch-factor` | 10 | Children per split (K ≥ 2) |
| `--seed` | 0 | Random seed |
| `--n-init`, `--max-iter` | 10, 300 | Constrained K-Means restarts and iterations |
| `--n-jobs` | 1 | Constrained K-Means parallel jobs |
| `--greedy-threshold` | 2000 | Hybrid uses greedy on nodes with more items than this |
| `--greedy-n-init`, `--greedy-max-iter` | 5, 20 | Ordinary K-Means restarts and iterations for greedy splits |
| `--tol` | 1e-4 | Convergence tolerance for both solvers |

Fitting is limited to one BLAS thread so that results are repeatable on the
same machine and library versions. Results are not guaranteed to be identical
across platforms or library versions. K=10 matches SEATER's default and has not
been tuned.

## Limitations

- **Mixed SID lengths and beam search.** MiniOneRec's default
  `length_penalty=0.0` sums log-probabilities, which favors shorter SIDs. On
  catalogs where paths differ in length, check `sid_length_distribution` in
  `Dataset.tree_stats.json` and consider adjusting the length penalty.
- **Vocabulary cost.** Each item adds one embedding row, and the model still
  computes logits over the whole vocabulary at every step.
- **Fixed catalog.** There is no incremental insertion of new items.
- **Not bit-identical to the reference code.** With the same seed, partitions
  can differ from SEATER (which used `n_jobs=-2` and numeric ID order) and from
  Re-SEATER (which uses FAISS for large greedy nodes).
- **Scope of testing.** Tokenizer, converter and constrained-decoding checks
  ran on CPU without model weights. SFT, RL and recommendation metrics have not
  been evaluated.

## Tests and references

```bash
python -m pip install -r sid_generation/requirements-hierarchical-test.txt
python -m pytest sid_generation/tests/test_hierarchical_*.py
```

The integration tests also need a CPU build of PyTorch. See
[HIERARCHICAL_VALIDATION.md](HIERARCHICAL_VALIDATION.md) for the tested
environment, recorded results and full reproduction commands.

- Si et al., *Generative Retrieval with Semantic Tree-Structured Identifiers
  and Contrastive Learning*, SIGIR-AP 2024
  ([paper](https://arxiv.org/abs/2309.13375),
  [code](https://github.com/Ethan00Si/SEATER_Generative_Retrieval/tree/b12cd29585720f891b1a8f41dacec5e0554fac6f)).
- Valeau et al., *Efficient Optimization of Hierarchical Identifiers for
  Generative Recommendation*
  ([paper](https://arxiv.org/html/2512.18434v2),
  [code](https://github.com/joshrosie/re-seater/tree/32235415e5abc1fa97d0516199422b5d9a34471b)).

This is an independent implementation; no code from either project is
included. At the linked commits, the SEATER repository has no license and
Re-SEATER is Apache-2.0. MiniOneRec keeps its [Apache-2.0 license](../LICENSE).
