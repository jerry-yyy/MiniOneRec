# Hierarchical K-Means validation

## Environment

Primary platform: **Apple M1, macOS 15.7.9, ARM64 Python 3.12.4**, with
`faiss-cpu==1.12.0` for the upstream regression selection.
The separately executed macOS validation was reported as **MACOS_VALIDATION:
PASS** at validation commit `bd4a46d`, after applying the complete 16-file patch
to official MiniOneRec base `0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1`.
It covers 115 new tests, 81 upstream regressions in each checkout, and a
28-item synthetic catalog. It is not a full real-catalog experiment.

The real-catalog build, interaction mapping and real Qwen tokenizer results
below were executed separately on **Windows 11** on 2026-11-17. That environment
used Python 3.12.5, AMD64 Family 25 Model 97 Stepping 2 and 32 logical CPUs.
Fitting used one solver job and one BLAS/OpenMP thread.
Windows dependencies: NumPy 2.5.5, scikit-learn 1.7.2, k-means-constrained 0.9.0,
SciPy 1.18.1, OR-Tools 9.15.6755, pytest 9.2.0, pandas 2.3.3,
transformers 4.57.6, tokenizers 0.22.2 and PyTorch 2.9.1+cpu.
The generator itself does not import PyTorch. No GPU or neural-network training
was used.

In the isolated environment from the [installation guide](HIERARCHICAL_KMEANS.md#installation),
install the optional test stack separately from the builder:

```bash
python -m pip install -r sid_generation/requirements-hierarchical-test.txt
```

Install PyTorch separately for the target platform. The recorded Windows run
used this CPU-wheel command:

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
```

## Automated Tests

Run from the repository root after installing the
[optional test dependencies](HIERARCHICAL_KMEANS.md#installation):

```bash
python -m pytest \
  sid_generation/tests/test_hierarchical_tree.py \
  sid_generation/tests/test_hierarchical_export.py \
  sid_generation/tests/test_hierarchical_cli.py \
  sid_generation/tests/test_hierarchical_integration.py \
  -q --basetemp outputs/hierarchical-tests-fresh
```

Use a fresh temporary directory. These tests run offline with synthetic inputs
and a small real WordLevel tokenizer; they are not the full-catalog Qwen check.

| macOS ARM64 suite | Passed | Failed | Skipped | Errors |
| --- | ---: | ---: | ---: | ---: |
| Tree mathematics/persistence | 75 | 0 | 0 | 0 |
| Export/provenance/publication | 16 | 0 | 0 | 0 |
| CLI | 18 | 0 | 0 | 0 |
| MiniOneRec integration | 6 | 0 | 0 | 0 |
| **Total** | **115** | **0** | **0** | **0** |

The macOS run passed **115 tests, with zero failures, skips or errors**.
The earlier Windows 11 targeted-suite rerun also passed 115 tests in 22.07
seconds with Hugging Face and Transformers offline mode enabled. This targeted
result is separate from the macOS upstream regression results below.

## Algorithm Validation

Tests cover deterministic repeats, K=2/3/10, nondivisible capacity bounds,
terminal groups of at most K items, exactly-once membership, duplicate/identical
embeddings, invalid inputs, canonical ID ordering, input-row permutations,
strict rejection of changed catalogs, safe JSON persistence and corruption,
manifest hashes, token namespaces, variable-length SIDs and publication safety.

On macOS ARM64, a **28-item synthetic catalog** produced byte-identical results
across repeated fits, tree loading and reverse-row encoding. This checks the
synthetic workflow on that platform; it does not establish byte-identical
outputs between Windows and macOS.

The earlier Windows checks used an independent 30-point imbalanced fixture with
K=3, producing a **10/10/10** root split, whereas ordinary K-Means produced
**28/1/1**. All seven constrained solver
calls received the original vectors and correct capacity bounds. Independent
traversal also checked all 111 split nodes in the real catalog. This validates
recursive capacity-constrained K-Means rather than ordinary hierarchical
clustering; no residual subtraction or fallback algorithm was used.

On Windows 11, two complete default fits of the real catalog produced
byte-identical index and tree JSON. Tree save/reload and fixed-catalog encoding
after reversing embeddings and matching
IDs also reproduced the index bytes exactly. Cross-version bitwise consistency
has not been established.

## Real Catalog Results

All real-catalog measurements in this section were produced on **Windows 11**.
The macOS validation used the separate 28-item synthetic catalog described above.

The source is [kkknight/MiniOneRec](https://huggingface.co/kkknight/MiniOneRec/tree/365a03fc6601ed36abd69cc9a0a59025a3d31cdc/Amazon),
pinned to revision `365a03fc6601ed36abd69cc9a0a59025a3d31cdc`.
The preparation helper verifies these raw-byte SHA256 values before use:

| Artifact under `Amazon/` | SHA256 |
| --- | --- |
| `index/Industrial_and_Scientific.emb-qwen-td.npy` | `91a1c875e4b009f3a07b5febf00c2ef62a3813a55cfe873b825e18e32df1dd0b` |
| `index/Industrial_and_Scientific.item.json` | `937600c7a03652f55864a1d12d6e9fe8ca3e24ecc4e255a88216b25c5f949496` |
| `train/Industrial_and_Scientific_5_2016-10-2018-11.csv` | `da5f2af4f5047e18023dd4792f65a48bd7c8eab90447dd682fb48ca255d25f8c` |
| `valid/Industrial_and_Scientific_5_2016-10-2018-11.csv` | `12bbb02bced0bd754a8b4e0f9ca4086416e30d12b0b73a2c95e7b195dd67385a` |
| `test/Industrial_and_Scientific_5_2016-10-2018-11.csv` | `973dd1db4fd549f3f2b40a0f039dbae157ebb92c716197cc246401e4e82c3a46` |

The upstream [embedding producer](https://github.com/AkaliKong/MiniOneRec/blob/0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1/rq/text2emb/amazon_text2emb.py#L125-L148)
sorts numeric `(item ID, embedding)` pairs before stacking. Its file SHA256 is
`d0ac88885b8a24ad7dd7cf61c231a13b7bf419eed1346e71bb0b87ce522e2853`.
The pinned metadata has exactly IDs `0..3685`; the array has 3,686 rows, so this
producer contract establishes row `i` to ID `str(i)` for this release. Preparation
validates that schema and writes an explicit mapping, without trusting JSON key
order. Item **0 is real** and is included. Embeddings were not regenerated to
independently verify their semantic provenance.

Configuration: K=10, seed=0, n_init=10, max_iter=300, tol=0.0001, n_jobs=1.

| Measurement | Result |
| --- | ---: |
| Items / embedding dimensions | 3,686 / 2,560 |
| Input array | finite float16; 18,872,448 bytes |
| Complete CPU fit process, including startup/export | **31.656 s** |
| Fit section | 30.532 s |
| Sampled process-tree RSS peak | **313.707 MiB** |
| Cluster nodes / constrained split nodes / terminal groups | 1,111 / 111 / 1,000 |
| Internal node tokens / item leaf tokens | 1,110 / 3,686 |
| Total added tokens | **4,796** |
| Maximum / mean SID token depth | 4 / 4.0 |
| SID lengths | 4 tokens for all 3,686 items |
| Complete SID collisions / prefix collisions | **0 / 0** |

The memory figure sums launcher/interpreter RSS at nominal 20 ms intervals:
sampling can miss transients and shared pages can be counted twice. It is not a
GPU-memory or training-memory measurement. The interpreter's separate OS peak at
fit end, before export, was 313.145 MiB. Timing is one recorded measurement.

| Cluster depth (root=0) | Count | Size histogram |
| --- | ---: | --- |
| 0 | 1 | 3,686: 1 |
| 1 | 10 | 368: 4; 369: 6 |
| 2 | 100 | 36: 14; 37: 86 |
| 3 | 1,000 | 3: 314; 4: 686 |

Every split satisfied its capacity bounds. SID uniqueness follows from the
unique item leaf tokens; zero collisions do **not** establish clustering quality
or recommendation-accuracy improvements.

| Reproducible output | SHA256 |
| --- | --- |
| `Industrial_and_Scientific.index.json` | `86f3e6e3b199761608d15d2b7aaa1931ab2d6c1ec5437100fc82b57c183e298a` |
| `Industrial_and_Scientific.tree.json` | `0b2e8ec0f9b42165d2b6d980e0ad034f2891cc9ce313f07b93d65e46c70721b3` |

For future reproduction, the following builds a fresh catalog, verifies
re-encoding, runs the unchanged converter and checks all converted rows. The
preparation step downloads pinned inputs only if needed; use `--offline` when
they already exist. These commands were not rerun for documentation cleanup.

```bash
python -m sid_generation.prepare_hierarchical_data
python -m sid_generation.benchmark_hierarchical \
  --output-root outputs/hierarchical-benchmark-fresh
```

## MiniOneRec Integration

The real-catalog converter and Qwen tokenizer checks in this section ran on
**Windows 11**. They were not repeated as a complete real-catalog run on macOS.

The original `convert_dataset.py` converted **36,259 train / 4,532 valid / 4,533
test** rows. All **45,324** source user IDs, target IDs, history IDs and resulting
SIDs were checked; the item-info output covers all 3,686 items. Preparation uses
the CSV's explicit original-ID columns to reconstruct `.inter` files and ignores
old SID columns. Conversion uses neither sampling nor `--keep_longest_only`.

The actual [Qwen2.5-0.5B-Instruct tokenizer](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775)
was pinned to revision `7ae557604adf67be50417f59c2c2f167def9a775`.

| Tokenizer/config file | SHA256 |
| --- | --- |
| `config.json` | `18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45` |
| `tokenizer_config.json` | `5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583` |
| `tokenizer.json` | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |
| `merges.txt` | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` |
| `vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |

The validator executes the actual TokenExtender and evaluator prefix callback
extracted unchanged with AST, plus the actual CPU logits processor. It verified
**4,796 atomic tokens**, **3,686 complete SIDs**, and **22,116 legal transitions**
including newline/EOS across 8,483 prefix states. Thirteen CPU mask calls covered
two representative catalog paths and invalid-prefix rejection. Actual dataset
classes mapped the complete catalog; four representative tokenized `pre()` calls
retained complete SIDs, with eager preprocessing disabled.

A separate actual-Qwen fixture verified lengths 1, 2, 3 and 4, with 24 transitions,
14 prefix states, 16 CPU mask calls and six dataset `pre()` calls. This supplies
mixed-length coverage because the real catalog happens to have uniform depth.
No model weights were loaded, and `model.generate`, SFT and RL were not executed.

To validate a generated catalog, download only the pinned tokenizer/config files
on the first invocation; omit `--download-tokenizer` to use the local copy:

```bash
python -m sid_generation.validate_hierarchical_integration \
  --index-path outputs/hierarchical-benchmark-fresh/catalog/Industrial_and_Scientific.index.json \
  --tokenizer-path outputs/hierarchical-validation/qwen-tokenizer \
  --download-tokenizer \
  --report-path outputs/hierarchical-validation/qwen-integration.json
```

## Upstream Regression Comparison

The following results are scoped to **macOS ARM64**. The reported comparison
used the clean official base `0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1` and the
patched feature checkout at validation commit `bd4a46d`.

| macOS ARM64 test scope | Passed | Failed | Skipped | Errors |
| --- | ---: | ---: | ---: | ---: |
| New tests | 115 | 0 | 0 | 0 |
| Clean official baseline | 81 | 0 | 0 | 0 |
| Feature branch, same selection | 81 | 0 | 0 | 0 |

All selected directory-publication tests passed on macOS. Installing
`faiss-cpu==1.12.0` enabled the optional FAISS tests, which also passed without
skips. These all-pass upstream regression results apply to macOS; they are not
a claim that the Windows upstream regression suite passed.

To reproduce the comparison, run this exact selection from each checkout with
the same absolute Python executable. Use distinct fresh `--basetemp` and
`--junitxml` paths outside both checkouts; keep the baseline clean before and
after. The example names below are for the feature run; use `baseline` instead
of `feature` for the other checkout:

```bash
python -m pip install faiss-cpu==1.12.0
git worktree add --detach ../MiniOneRec-upstream-validation 0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1
python -m pytest \
  sid_generation/tests/test_sid_generation.py \
  sid_generation/tests/test_rq_regressions.py \
  sid_generation/tests/test_publication.py \
  sid_generation/tests/test_prq_regressions.py \
  sid_generation/tests/test_minionerec_integration.py \
  tests/test_flash_sid_simhash.py tests/test_flash_sid_export.py \
  tests/test_flash_sid_integration.py \
  --basetemp ../hierarchical-regression-feature-tmp \
  --junitxml ../hierarchical-regression-feature.xml --tb=short -ra -q
```

## Limitations

- No recommendation metrics, model generation, SFT/RL or full SEATER training
  objectives were evaluated. Tokenizer and CPU-mask checks do not establish
  model-level performance.
- Existing SID builders, Trainer and downstream consumers were not modified
  for platform validation. The hierarchical CLI publishes into new directories.
- macOS results cover the targeted suites and a 28-item synthetic workflow;
  Windows results cover the reported real-catalog and integration measurements.
  Cross-platform byte-identical catalog outputs have not been established.
  Large-catalog runtime and vocabulary/training cost require separate
  evaluation; see the [usage guide](HIERARCHICAL_KMEANS.md#limitations).
- Encoding reconstructs only the fitted catalog. New-item insertion and stable
  SIDs after catalog changes are not supported.
