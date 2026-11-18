"""Reproduce the complete CPU catalog workflow, with measured process resources.

Requires the optional test dependencies. Input preparation is a separate,
SHA256-pinned step; this runner performs no network access.
"""

import argparse
import ast
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import psutil

from .prepare_hierarchical_data import DATASET, CSV_NAME
from .sid_cli import json_write, sha256_file


def run(command, log_path):
    start = time.perf_counter()
    peak = 0
    samples = 0
    observed_processes = {}
    process_peaks = {}
    windows_peaks_available = False
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUTF8="1"))
        observed = psutil.Process(process.pid)
        while True:
            # On Windows a venv python.exe can be only a small launcher. The
            # interpreter and any solver workers must be included recursively.
            try:
                candidates = [observed, *observed.children(recursive=True)]
            except psutil.NoSuchProcess:
                candidates = []
            for candidate in candidates:
                try:
                    identity = (candidate.pid, candidate.create_time())
                    observed_processes[identity] = candidate
                except psutil.NoSuchProcess:
                    continue
            resident = 0
            for identity, candidate in observed_processes.items():
                try:
                    memory = candidate.memory_info()
                except psutil.NoSuchProcess:
                    continue
                resident += memory.rss
                os_peak = getattr(memory, "peak_wset", None)
                windows_peaks_available |= os_peak is not None
                process_peaks[identity] = max(process_peaks.get(identity, 0), memory.rss, os_peak or 0)
            peak = max(peak, resident)
            samples += 1
            if process.poll() is not None:
                break
            time.sleep(0.02)
        if process.returncode:
            raise RuntimeError(f"command failed ({process.returncode}); see {log_path}")
    return {
        "wall_seconds": time.perf_counter() - start,
        "peak_ram_bytes": peak,
        "scope": "launched process and recursively discovered descendants, including Python startup and export",
        "peak_method": "maximum summed process-tree RSS from nominal 20ms samples; lower bound on transient peaks; shared pages can be counted repeatedly",
        "observed_processes": len(observed_processes),
        "memory_samples": samples,
        "observed_process_peak_sum_bytes": sum(process_peaks.values()),
        "observed_process_peak_sum_method": (
            "sum of each observed process's OS lifetime peak working set; peaks may be nonconcurrent and shared pages counted repeatedly"
            if windows_peaks_available else
            "sum of each observed process's sampled RSS peak; peaks may be nonconcurrent and shared pages counted repeatedly"
        ),
    }


def benchmark(source, output, tokenizer=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    embeddings = source / "Amazon" / "index" / f"{DATASET}.emb-qwen-td.npy"
    prepared = source / "prepared"
    provenance = source / "data_provenance.json"
    # Revalidate pinned downloads and the row-map schema before any fit.
    run([sys.executable, "-m", "sid_generation.prepare_hierarchical_data", "--source-dir", str(source), "--offline"], output / "prepare.log")
    cli = [sys.executable, "-m", "sid_generation.hierarchical_cli"]
    common = ["--dataset-name", DATASET]
    fit = output / "catalog"
    fit_metrics = run(cli + ["fit", *common, "--data-path", str(embeddings), "--item-ids", str(prepared / f"{DATASET}.item_ids.json"),
                            "--dataset-dir", str(prepared), "--output-dir", str(fit), "--branch-factor", "10", "--seed", "0"], output / "fit.log")
    tree_path = fit / f"{DATASET}.tree.json"
    index_path = fit / f"{DATASET}.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    model_stats = json.loads((fit / f"{DATASET}.tree_stats.json").read_text(encoding="utf-8"))
    rows = np.load(embeddings, allow_pickle=False)
    ids = json.loads((prepared / f"{DATASET}.item_ids.json").read_text(encoding="utf-8"))
    np.save(output / "reordered.npy", rows[::-1], allow_pickle=False)
    json_write(output / "reordered_ids.json", ids[::-1])
    encode_metrics = run(cli + ["encode", *common, "--data-path", str(output / "reordered.npy"), "--item-ids", str(output / "reordered_ids.json"),
                               "--model-path", str(tree_path), "--output-dir", str(output / "encoded")], output / "encode.log")
    if index_path.read_bytes() != (output / "encoded" / index_path.name).read_bytes():
        raise AssertionError("reloaded/reordered index bytes differ")
    run(cli + ["diagnose", "--model-path", str(tree_path), "--manifest-path", str(fit / f"{DATASET}.manifest.json")], output / "diagnose.log")
    converted = output / "converted"
    conversion_metrics = run([sys.executable, "convert_dataset.py", "--data_dir", str(fit), "--dataset_name", DATASET,
                              "--output_dir", str(converted)], output / "convert.log")
    conversion = {}
    for split in ("train", "valid", "test"):
        with (source / "Amazon" / split / CSV_NAME).open(encoding="utf-8", newline="") as handle:
            originals = list(csv.DictReader(handle))
        with (converted / split / CSV_NAME).open(encoding="utf-8", newline="") as handle:
            final = list(csv.DictReader(handle))
        if len(final) != len(originals):
            raise AssertionError(f"{split}: conversion dropped rows")
        for original, row in zip(originals, final):
            target = str(int(original["item_id"]))
            history = [str(int(value)) for value in ast.literal_eval(original["history_item_id"])]
            assert row["user_id"] == original["user_id"]
            assert row["item_id"] == target
            assert ast.literal_eval(row["history_item_id"]) == [int(value) for value in history]
            assert row["item_sid"] == "".join(index[target])
            assert ast.literal_eval(row["history_item_sid"]) == ["".join(index[item]) for item in history]
        conversion[split] = len(final)
    with (converted / "info" / f"{DATASET}_5_2016-10-2018-11.txt").open(encoding="utf-8") as handle:
        info = [line.rstrip("\n").split("\t") for line in handle]
    assert len(info) == len(index)
    assert all(fields[0] == "".join(index[fields[-1]]) for fields in info)
    integration = {"performed": False}
    if tokenizer:
        from .validate_hierarchical_integration import validate_catalog
        integration = validate_catalog(index_path, Path(tokenizer))
    result = {
        "dataset": DATASET, "cpu_only": True,
        "host": {"platform": platform.platform(), "processor": platform.processor(), "logical_cpus": psutil.cpu_count(), "python": platform.python_version()},
        "data_provenance_sha256": sha256_file(provenance),
        "fit_process": fit_metrics, "encode_process": encode_metrics, "conversion_process": conversion_metrics,
        "tree": {key: value for key, value in model_stats.items() if key != "split_diagnostics"},
        "index_sha256": sha256_file(index_path), "tree_sha256": sha256_file(tree_path),
        "reloaded_reordered_index_byte_identical": True,
        "conversion_rows": conversion, "conversion_all_targets_and_histories_verified": True,
        "conversion_all_user_ids_verified": True,
        "integration": integration, "example": {key: index[key] for key in ("0", "1", "1001")},
        "quality_claim": "none; SID uniqueness follows from unique leaf tokens",
    }
    json_write(output / "benchmark_results.json", result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="data/hierarchical-source")
    parser.add_argument("--output-root", required=True, help="fresh benchmark directory")
    parser.add_argument("--tokenizer-path", help="local pinned Qwen tokenizer for real integration validation")
    args = parser.parse_args()
    benchmark(args.source_root, args.output_root, args.tokenizer_path)


if __name__ == "__main__":
    main()
