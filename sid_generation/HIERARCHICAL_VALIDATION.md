# Hierarchical K-Means validation

## Environment

The November 18, 2026 Windows strategy comparison used implementation
`faab0e4cccbe01fce678296572b4cca5be05e8d8`
on the original MiniOneRec history. Its parent is the public Balanced candidate
`41d51f16acd3098ad9322dbc8d0a7a91ec906cda`, based on official
`0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1`. Historical Balanced results
are identified separately below. The three-strategy release awaits Mac validation.

Windows 11 build 22621, AMD64 Family 25 Model 97, 32 logical CPUs, Python
3.12.5. NumPy 2.5.5, SciPy 1.18.1, scikit-learn 1.7.2, k-means-constrained
0.9.0, OR-Tools 9.15.6755, pytest 9.2.0, pandas 2.3.3, transformers 4.57.6,
tokenizers 0.22.2 and PyTorch 2.9.1+cpu. All fits used one solver job and a
BLAS/OpenMP limit of one. Existing inputs and tokenizer files were used offline.

## Automated tests and upstream comparison

| Windows suite | Passed | Failed | Skipped | Teardown errors | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hierarchical tests, `faab0e4` | 169 | 0 | 0 | 0 | 16.29 |
| Clean official upstream, `0cbd94d` | 62 | 4 | 15 | 3 | 6.16 |
| Same upstream selection, `faab0e4` | 62 | 4 | 15 | 3 | 5.17 |

The 169 tests comprise 75 tree, 16 export, 18 CLI, 9 integration, 34 strategy
and 17 format tests. This includes all 115 original tests and 54 new cases.
The 13 warnings came from the existing pandas dataset code and intentionally
duplicated vectors in scikit-learn tests. The two upstream runs matched on
all 81 unique cases, including the individual failure traces, cleanup errors
and skip reasons. Both also reported 93 passed subtests.

The strategy tests include an independent ordered-tuple oracle, a cross-block
tie case, duplicate embeddings and centroids, overflow rejection, terminal
nodes, boundary sizes T−1/T/T+1, and same-depth siblings that execute different
backends. Repeated fits and joint row/ID permutations reproduce each method's
tree and index. Limiting thresholds reproduce the corresponding pure method.
The legacy fixture holds five original publication files generated at
`41d51f1` before changing the writer. Format, hash and provenance checks cover
v1 diagnosis, v2 re-export, missing method and mixed identities. A generated
catalog from each strategy also goes through the actual converter and a real
offline WordLevel tokenizer with CPU masks.

The official worktree and strategy checkout used the same absolute Python
executable and eight test files. The four publication cases exercise
`encode_without_dataset_removes_previous_owned_companions`,
`manifest_cannot_claim_unrelated_or_outside_files`,
`overwrite_removes_obsolete_owned_splits_and_preserves_other_files` and
`unsupported_exchange_keeps_old_version`. Directory exchange reaches
`ctypes.CDLL(None)`, which raises a TypeError on this Windows runtime. The
first three cases also leave an embedding memory map open during traceback
cleanup (WinError 32). Seven SID and eight RQ tests skip for missing optional
FAISS. The hierarchical CLI publishes only to new directories.

## Real catalog and integration

The pinned Industrial_and_Scientific release contains 3,686 items and finite
2,560-dimensional float16 embeddings. The producer sorts numeric IDs before
stacking. The row map explicitly covers 0 through 3685, including item 0.
Preparation checked the raw embedding, metadata and three interaction file
hashes, then verified the prepared row map and companions. Embedding SHA256
`91a1c875e4b009f3a07b5febf00c2ef62a3813a55cfe873b825e18e32df1dd0b`.
The preparation code lists each pinned SHA256.
[Dataset revision](https://huggingface.co/kkknight/MiniOneRec/tree/365a03fc6601ed36abd69cc9a0a59025a3d31cdc/Amazon)
· [Embedding producer](https://github.com/AkaliKong/MiniOneRec/blob/0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1/rq/text2emb/amazon_text2emb.py#L125-L148)
· [Input checksums](prepare_hierarchical_data.py)

Each method ran twice in fresh directories, in forward then reverse method
order. Times below are seconds. Fit covers catalog canonicalization, fitting,
diagnostics and tree validation. Export includes index/statistics generation,
hashes, artifact writing and publication. Total runs from CLI process launch
through exit and includes input checks, startup and shutdown. Peak sums the
fit process and descendants' RSS at nominal 20 ms intervals. This can miss
transients and count shared pages twice. Subsequent diagnosis, conversion and
Qwen checks are outside all four measurements.

| Method | Fit, min–max | Export, min–max | Total, min–max | Peak MiB, min–max |
| --- | ---: | ---: | ---: | ---: |
| Balanced | 28.302–28.522 | 0.169–0.208 | 29.117–29.302 | 385.223–385.477 |
| Greedy | 7.229–7.321 | 0.167–0.179 | 7.971–8.145 | 287.023–287.305 |
| Hybrid | 14.859–14.971 | 0.165–0.170 | 15.605–15.680 | 286.410–287.176 |

K=10, seed=0, tol=0.0001, T=2,000. Constrained uses 10 restarts and up to
300 iterations each, Greedy uses 5 and 20. Timing differences therefore
include both the assignment algorithm and its fitting budget.

| Method | Splits at depths 0 / 1 / 2 | Final WCSS at depth 0 | Depth 1 | Depth 2 |
| --- | --- | ---: | ---: | ---: |
| Balanced | 1 / 10 / 100 constrained | 5,928,626 | 4,038,607 | 1,857,809 |
| Greedy | 1 / 10 / 100 greedy | 6,223,405 | 4,791,161 | 2,483,200 |
| Hybrid | 1 greedy / 10 constrained / 100 constrained | 6,223,405 | 4,422,113 | 2,038,787 |

For each split, an independent traversal recovered the final child membership,
recomputed each child mean in float64, and summed squared distances from that
mean. The table aggregates those values by **parent depth**, rounded to the
nearest unit. Greedy and Hybrid share the same root partition. Hybrid's
constrained descendants reduced WCSS at both deeper levels relative to Greedy.

All three methods covered the complete catalog exactly once. Child sizes were
368–369 at depth 1, 36–37 at depth 2 and 3–4 at depth 3. Each method produced
1,110 node tokens and 3,686 item leaves (4,796 tokens), all 3,686 SIDs had
length 4, and complete-SID and prefix collisions were zero. Unique leaf tokens
give this structural guarantee. Both fits of each method produced identical
tree and index bytes. Reloading and reversing aligned input rows reproduced
the index bytes. Default Balanced also matched every v1 node and the entire
real v1 index, SHA256 `86f3e6e3b199761608d15d2b7aaa1931ab2d6c1ec5437100fc82b57c183e298a`.
Diagnosing the original real v1 manifest verified eight artifact hashes.
Encoding that v1 model also reproduced this index. The new manifest recorded
the original tree SHA256 `0b2e8ec0f9b42165d2b6d980e0ad034f2891cc9ce313f07b93d65e46c70721b3`;
the v2 tree and manifest passed a fresh diagnosis.

For **each method**, the unchanged converter preserved user, target, history
and SID mappings for all 36,259 train, 4,532 validation and 4,533 test rows
(45,324 total). The actual Qwen tokenizer registered 4,796 atomic tokens and
encoded 3,686 complete SIDs. The evaluator callback checked 22,116 legal
transitions across 8,483 prefix states. Thirteen CPU logits-mask calls covered
two representative paths and invalid-prefix handling. Five dataset classes
were exercised, with four representative tokenized preprocessing calls. The
benchmark used only the five pinned tokenizer/configuration files.
[Qwen tokenizer revision](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775)
· [Tokenization file hashes from the Balanced candidate](https://github.com/jerry-yyy/MiniOneRec/blob/41d51f16acd3098ad9322dbc8d0a7a91ec906cda/sid_generation/HIERARCHICAL_VALIDATION.md#minionerec-integration)

## Synthetic scaling

After the real catalog, each method ran once on the same 10,000 × 64 float32
standard-normal array (NumPy `default_rng(2026)`) with string row IDs. All
three finished within the predeclared limits, so the same design was run at
100,000 × 64. The budget for each fresh fit process, including export, was
180 wall seconds and 2,048 MiB sampled process-tree RSS.

| N | Method | Fit s | Export s | Total s | Peak MiB | Final WCSS by parent depth, root first |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 10,000 | Balanced | 101.264 | 0.170 | 101.960 | 177.402 | 607,172 / 563,275 / 469,430 |
| 10,000 | Greedy | 3.163 | 0.167 | 3.856 | 165.734 | 608,406 / 568,501 / 478,524 |
| 10,000 | Hybrid | 17.378 | 0.167 | 18.107 | 166.844 | 608,406 / 564,668 / 471,185 |
| 100,000 | Balanced | — | — | 180.048 (timeout) | 393.320 to cutoff | No completed tree |
| 100,000 | Greedy | 14.721 | 1.780 | 17.178 | 273.801 | 6,124,675 / 5,822,175 / 5,434,319 / 4,568,360 |
| 100,000 | Hybrid | 166.527 | 1.749 | 168.916 | 273.445 | 6,124,675 / 5,822,175 / 5,394,978 / 4,493,148 |

All three 10,000-item trees covered the catalog, with capacities
1,000 → 100 → 10, 11,110 tokens and SID length 4. Their split counts by
method/depth match the real catalog. At 100,000 items, both complete trees
had 1/10/100/1,000 splits by depth, 111,110 tokens and SID length 5. Hybrid
used Greedy at depths 0 and 1 and constrained at depths 2 and 3.
The largest shared three-way comparison is 10,000 × 64.

A separate single-run control used 512 × 16 vectors generated the same way,
K=10, T=100, and matched budgets of three starts and 20 iterations for both
estimators. Fit times were 2.221 / 2.047 / 2.074 s for Balanced / Greedy /
Hybrid. Final WCSS at depths 0 and 1 was 6,393 / 3,735 (Balanced),
6,479 / 4,106 (Greedy) and 6,479 / 3,832 (Hybrid). This control used a
different size and threshold from the real benchmark.

## Reproducing the checks

From the repository root, install the [builder and test requirements](HIERARCHICAL_KMEANS.md#installation)
and CPU PyTorch. On Windows the recorded PyTorch install was
`python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu`.
Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for local inputs.
Use an unused output path for each run. The following PowerShell selection
also works on POSIX by passing the six matching file paths directly.

```powershell
$tests = (Get-ChildItem sid_generation/tests/test_hierarchical*.py).FullName
python -m pytest @tests -p no:cacheprovider --tb=short -ra -q --basetemp outputs/hierarchical-tests-fresh --junitxml outputs/hierarchical-tests-fresh.xml
```

Run these eight files in each checkout with the **same environment's Python**.
Use a distinct temporary directory and JUnit path for each. The official
checkout can be created with `git worktree add --detach ../hkm-upstream 0cbd94d`.

```bash
python -m pytest sid_generation/tests/test_sid_generation.py \
  sid_generation/tests/test_rq_regressions.py sid_generation/tests/test_publication.py \
  sid_generation/tests/test_prq_regressions.py sid_generation/tests/test_minionerec_integration.py \
  tests/test_flash_sid_simhash.py tests/test_flash_sid_export.py tests/test_flash_sid_integration.py \
  -p no:cacheprovider --basetemp ../hkm-feature-temp --junitxml ../hkm-feature.xml --tb=short -ra -q
```

For the real benchmark, the pinned raw files must already exist in the
source cache. Use the preparation helper without
`--offline` only to populate an empty cache. Set the tokenizer path to the
verified local snapshot and run the methods serially. Repeat into fresh paths
with `--fit-only` in place of `--tokenizer-path ...` for timing replicas.

```bash
python -m sid_generation.prepare_hierarchical_data --offline
for m in balanced greedy hybrid; do
  python -m sid_generation.benchmark_hierarchical --method "$m" \
    --tokenizer-path outputs/hierarchical-validation/qwen-tokenizer \
    --output-root "outputs/hkm-real-$m" --wall-seconds 180 --rss-mib 2048
done
```

For the synthetic runs replace the tokenizer argument with
`--synthetic-items 10000` or `--synthetic-items 100000` and use a separate
output root for each size and method. The result JSON records peak, elapsed
time, any budget limit, split counts and WCSS at each depth.

## Historical Balanced results and limits

- **Windows 11, Python 3.12.5, November 17, 2026.** The earlier Balanced
  review at `6e03a86` passed 115 targeted tests. Its upstream selection and
  the clean official base each returned 62 passed, 4 failures, 15 skips and
  3 teardown errors. The recorded real build took 31.656 s including export
  and peaked at 313.707 MiB sampled process-tree RSS. That run also checked
  45,324 converter rows and the pinned Qwen tokenizer. An additional Qwen
  fixture exercised paths of lengths 1–4.
- **Apple M1, macOS 15.7.9, ARM64 Python 3.12.4.** At historical validation
  commit `bd4a46d`, the Balanced patch passed 115 targeted tests and all 81
  upstream tests in both checkouts with `faiss-cpu==1.12.0`. The 28-item
  synthetic repeat/load/reverse-encode check passed on that machine.
- A fresh Mac run of all three strategies, format migration and comparison
  against the official regression selection is pending. Cross-platform byte
  identity is outside the reproducibility contract. No recommendation metrics,
  neural weights, model generation, SFT or RL were evaluated.
