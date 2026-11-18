# Hierarchical K-Means validation

Validated on macOS ARM64.

| Test selection | Passed | Failed | Skipped | Errors |
| --- | ---: | ---: | ---: | ---: |
| Hierarchical K-Means, all three strategies | 169 | 0 | 0 | 0 |
| Upstream regression tests, clean official baseline | 81 | 0 | 0 | 0 |
| Same upstream regression tests, feature branch | 81 | 0 | 0 | 0 |

All three pytest commands exited with status 0. The upstream runs matched on
all 81 test identifiers and outcomes, including eight publication tests and
15 tests that exercised FAISS. Each upstream run also passed 117 unittest
subtests, counted within their parent tests.

## Environment

| Component | Tested version or setting |
| --- | --- |
| Hardware and OS | Apple M1, macOS 15.7.9 (24G830) |
| Interpreter | Native ARM64 CPython 3.12.4 in an isolated virtual environment |
| Numerical libraries | NumPy 2.5.5, SciPy 1.18.1, scikit-learn 1.7.2 |
| Constrained solver | k-means-constrained 0.9.0, OR-Tools 9.15.6755 |
| Test and process tools | pytest 9.2.0, psutil 7.2.2, threadpoolctl 3.8.0 |
| Upstream and integration tests | faiss-cpu 1.12.0 (NEON), pandas 2.3.3, PyTorch 2.9.1 on CPU, transformers 4.57.6, tokenizers 0.22.2 |
| Official MiniOneRec baseline | `0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1` |

## Strategy and compatibility checks

The 169 tests cover tree construction (75), export (16), CLI (18), consumer
integration (9), split strategies (34) and file formats (17). The Greedy
assignments matched an independent sort of all `(distance, row, centroid)`
candidates, including equal distances across computation blocks. Outputs from
each strategy passed the existing converter, tokenizer and CPU prefix-mask checks.

- **Fit, save and reload.** Each strategy fit the same 28-item catalog twice in fresh CLI processes with K=3 and seed=42. A third process loaded the saved tree and encoded the vectors and their IDs in reverse row order. The two fits and the re-encoding produced byte-identical trees and SID indexes for each strategy. Manifest-backed `diagnose` verified four artifact hashes per output. An independent tree traversal recovered every item's exported SID.
- **Hybrid boundary.** At the default threshold of 2,000, actual solver calls on an eight-dimensional input with K=16 showed that a 2,000-item root used constrained K-means. A 2,001-item root used Greedy, then all 16 children (125 or 126 items each) used constrained K-means. The strategy tests also exercised different solvers in siblings at the same depth.
- **Format migration.** The CLI read the original v1 Balanced tree and manifest, verified their identity and hashes, and re-encoded the catalog as v2. The new manifest recorded the source tree hash. The migrated index preserved the old ID-to-SID mapping. On the real catalog below, default Balanced reproduced the legacy Balanced node topology and index bytes on the same machine.

## Real catalog

The Industrial_and_Scientific input contained 3,686 items and 2,560-dimensional
float16 embeddings. The producer sorts by numeric item ID before stacking rows.
The check verified the embedding and metadata hashes and used the explicit row
mapping `0` through `3685`. The expected SHA256 values are in the
[data preparation helper](prepare_hierarchical_data.py). See the
[embedding producer](../rq/text2emb/amazon_text2emb.py).

All three methods used K=10, seed=0 and one solver job. Constrained K-means
used 10 restarts and up to 300 iterations. Greedy used 5 restarts and up to
20 iterations. Hybrid used the default threshold of 2,000.

| Strategy | Constrained splits | Greedy splits | Backend at parent depths 0 / 1 / 2 |
| --- | ---: | ---: | --- |
| Balanced | 111 | 0 | constrained / constrained / constrained |
| Greedy | 0 | 111 | greedy / greedy / greedy |
| Hybrid | 110 | 1 | greedy / constrained / constrained |

Every tree had 1, 10 and 100 splits at successive parent depths. Child sizes
were 368–369, 36–37 and 3–4 at depths 1, 2 and 3. All 3,686 items occurred
exactly once. Each index used 1,110 node tokens and 3,686 leaf tokens. All SIDs
had length 4, with **zero duplicate SIDs and zero prefix collisions**.
For each strategy, rebuilding with the same seed reproduced the SHA256 of both
the saved tree and the SID index. These checks establish catalog identity and
tree structure. Recommendation quality and model training are outside this scope.

## Reproducing the checks

From the repository root, activate the Python 3.12 environment in
the [installation guide](HIERARCHICAL_KMEANS.md#installation). Install the
test stack there and keep the same interpreter active for both checkouts.

```bash
python -m pip install -r sid_generation/requirements-hierarchical-test.txt \
  -r sid_generation/requirements-test.txt torch==2.9.1 \
  numpy==2.5.5 scipy==1.18.1 scikit-learn==1.7.2 ortools==9.15.6755 \
  pandas==2.3.3 transformers==4.57.6 tokenizers==0.22.2 pytest==9.2.0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -m pytest sid_generation/tests/test_hierarchical*.py -p no:cacheprovider --tb=short -ra -q
```

Create a clean worktree at the official baseline in an unused directory. Run
the identical upstream selection from each repository root. Pytest creates a
fresh temporary area for each invocation.

```bash
git worktree add --detach ../minionerec-baseline 0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1
run_upstream() {
  python -m pytest \
    sid_generation/tests/test_sid_generation.py sid_generation/tests/test_rq_regressions.py \
    sid_generation/tests/test_publication.py sid_generation/tests/test_prq_regressions.py \
    sid_generation/tests/test_minionerec_integration.py tests/test_flash_sid_simhash.py \
    tests/test_flash_sid_export.py tests/test_flash_sid_integration.py \
    -p no:cacheprovider --tb=short -ra -q
}
run_upstream
(cd ../minionerec-baseline && run_upstream)
```

To reproduce the real-catalog fits and diagnosis, run the preparation helper
to verify the pinned inputs and write the explicit row map.
Use `--offline` if the verified source cache is already populated. Run each
method with a fresh output directory.

```bash
python -m sid_generation.prepare_hierarchical_data
dataset=Industrial_and_Scientific
source_dir=data/hierarchical-source
for method in balanced greedy hybrid; do
  out="outputs/hierarchical-$method"
  python -m sid_generation.hierarchical_cli fit --method "$method" \
    --data-path "$source_dir/Amazon/index/$dataset.emb-qwen-td.npy" \
    --item-ids "$source_dir/prepared/$dataset.item_ids.json" \
    --dataset-name "$dataset" --output-dir "$out"
  python -m sid_generation.hierarchical_cli diagnose \
    --model-path "$out/$dataset.tree.json" --manifest-path "$out/$dataset.manifest.json"
done
```

Two optional scripts run further checks on these outputs. Both need the test
stack above. `validate_hierarchical_integration` loads an index with the
tokenizer and runs MiniOneRec's existing TokenExtender, dataset methods,
evaluator prefix callback and CPU logits masks; no model weights are loaded.
`--download-tokenizer` fetches only the tokenizer and config files of
Qwen2.5-0.5B-Instruct at a pinned revision. Omit it once they are downloaded.

```bash
python -m sid_generation.validate_hierarchical_integration \
  --index-path "outputs/hierarchical-balanced/$dataset.index.json" \
  --tokenizer-path outputs/hierarchical-tokenizer --download-tokenizer \
  --report-path outputs/hierarchical-balanced-integration.json
```

`benchmark_hierarchical` uses the prepared source files without network access.
It runs fit, diagnose, re-encode in reverse row order and conversion in
separate processes and records their time and memory. It checks that the
re-encoded index is byte-identical and that every converted row matches the
release CSVs, then writes `benchmark_results.json` in a new `--output-root`.
Pass `--tokenizer-path` to include the integration check above.

```bash
python -m sid_generation.benchmark_hierarchical --method hybrid \
  --output-root outputs/hierarchical-benchmark-hybrid \
  --tokenizer-path outputs/hierarchical-tokenizer
```

`--synthetic-items N` fits a seeded N×64 synthetic catalog instead, without a
tokenizer, and `--fit-only` stops after fit and diagnose. `--wall-seconds` and `--rss-mib`
stop the fit at a time or memory budget.
