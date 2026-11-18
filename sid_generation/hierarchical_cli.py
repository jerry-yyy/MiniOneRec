"""Fit, re-encode and diagnose a fixed-catalog balanced hierarchical SID tree."""

import argparse
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import platform
import re
import shutil
import tempfile
import threading
import time

import numpy as np
import psutil

from .hierarchical_kmeans import HierarchicalKMeans
from .hierarchical_tree import ALGORITHM, ALGORITHM_VERSION, HierarchicalTree
from .sid_cli import catalog_inputs, json_write, sha256_file
from .sid_common import canonical_item_ids
from .sid_publish import DATASET_SUFFIXES, _writer_lock


class ResourceMonitor:
    """Sample RSS every 20 ms; use the OS lifetime working-set peak on Windows."""

    def __enter__(self):
        self.process = psutil.Process()
        self.peak = 0
        self.start = time.perf_counter()
        self.cpu_start = time.process_time()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()
        return self

    def _watch(self):
        while True:
            memory = self.process.memory_info()
            self.peak = max(self.peak, memory.rss, getattr(memory, "peak_wset", 0))
            if self.stop.wait(0.02):
                break

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        memory = self.process.memory_info()
        self.peak = max(self.peak, memory.rss, getattr(memory, "peak_wset", 0))
        self.result = {
            "wall_seconds": time.perf_counter() - self.start,
            "cpu_seconds": time.process_time() - self.cpu_start,
            "peak_ram_bytes": self.peak,
            "peak_ram_method": "OS lifetime peak working set" if hasattr(memory, "peak_wset") else "20ms sampled process RSS (lower bound)",
            "scope": "input loading, catalog validation and fit/encode; excludes interpreter startup and export",
        }


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for operation in ("fit", "encode"):
        command = commands.add_parser(operation)
        command.add_argument("--data-path", required=True, help="real-valued (N,D) .npy; pickle disabled")
        command.add_argument("--item-ids", required=True, help="JSON list: explicit item ID for each embedding row")
        command.add_argument("--dataset-name", required=True)
        command.add_argument("--output-dir", required=True, help="new directory; existing directories are never replaced")
        command.add_argument("--dataset-dir", help="validate/copy .item.json and available .inter companions")
        if operation == "fit":
            command.add_argument("--branch-factor", type=int, default=10)
            command.add_argument("--seed", type=int, default=0)
            command.add_argument("--n-init", type=int, default=10)
            command.add_argument("--max-iter", type=int, default=300)
            command.add_argument("--tol", type=float, default=1e-4)
            command.add_argument("--n-jobs", type=int, default=1)
        else:
            command.add_argument("--model-path", required=True, help="saved Dataset.tree.json; same complete catalog only")
    diagnose = commands.add_parser("diagnose", help="validate a tree and optionally all published artifact hashes")
    diagnose.add_argument("--model-path", required=True)
    diagnose.add_argument("--manifest-path", help="also verify all artifact hashes and tree/index agreement")
    return parser


def _new_output(path):
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}; choose a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output.parent.resolve() / output.name


def generate(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.dataset_name):
        raise ValueError("dataset name must contain only letters, digits, underscores or hyphens")
    output = _new_output(args.output_dir)
    with ResourceMonitor() as monitor:
        data = np.load(args.data_path, allow_pickle=False, mmap_mode="r")
        if data.ndim != 2:
            raise ValueError("embeddings must have shape (items, dimensions)")
        ids, copies = catalog_inputs(args, len(data))
        start = time.perf_counter()
        if args.operation == "fit":
            config = {key: getattr(args, key) for key in ("branch_factor", "seed", "n_init", "max_iter", "tol", "n_jobs")}
            model = HierarchicalKMeans(**config).fit(data, ids)
            tree = model.tree
            fit_seconds = time.perf_counter() - start
        else:
            tree = HierarchicalTree.load(args.model_path)
            tree.encode(data, ids)
            fit_seconds = 0.0
        index = tree.to_index()
        statistics = tree.statistics()
        if args.operation == "fit":
            statistics["split_diagnostics"] = model.split_diagnostics
    statistics["runtime"] = dict(monitor.result, fit_seconds=fit_seconds)
    dataset = args.dataset_name
    names = [dataset + suffix for suffix in (".index.json", ".tree.json", ".item_ids.json", ".tree_stats.json", ".manifest.json")]
    names.extend(source.name for source in copies)
    manifest = {
        "algorithm": ALGORITHM,
        "algorithm_version": ALGORITHM_VERSION,
        "operation": args.operation,
        "configuration": tree.config,
        "input": {"embedding_sha256": sha256_file(args.data_path), "items": len(ids), "dimension": int(data.shape[1]),
                  "embedding_filename": Path(args.data_path).name, "original_dtype": str(data.dtype)},
        "item_id_mapping": {"file": dataset + ".item_ids.json", "order": "input embedding-row order",
                            "source_sha256": sha256_file(args.item_ids), "leaf_assignment": "lexicographic canonical string ID order"},
        "catalog_sha256": tree.catalog_hash,
        "versions": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "scikit-learn", "k-means-constrained", "scipy", "ortools")}},
        "source_model_sha256": sha256_file(args.model_path) if args.operation == "encode" else None,
        "copied_dataset_inputs": [{"file": p.name, "sha256": sha256_file(p)} for p in copies],
        "tree_statistics": tree.statistics(),
        "runtime": statistics["runtime"],
        "outputs": names,
        "publication": "new directory only; staged files published by rename under shared SID writer lock",
        "scope": "fixed catalog; no incremental insertion, neural training or recommendation quality claim",
    }
    with _writer_lock(output):
        _new_output(output)
        with tempfile.TemporaryDirectory(dir=output.parent, prefix=f".{dataset}-hierarchical-") as temporary:
            stage = Path(temporary) / "catalog"
            stage.mkdir()
            json_write(stage / names[0], index)
            tree.save(stage / names[1])
            json_write(stage / names[2], ids)
            json_write(stage / names[3], statistics)
            for source in copies:
                shutil.copyfile(source, stage / source.name)
            manifest["artifact_sha256"] = {name: sha256_file(stage / name) for name in names if name != dataset + ".manifest.json"}
            json_write(stage / f"{dataset}.manifest.json", manifest)
            # Do not replace an output created by a noncooperating writer.
            _new_output(output)
            os.rename(stage, output)
    result = {"index": str(output / names[0]), "items": len(index), "statistics": statistics}
    print(json.dumps(result, indent=2))
    return result


def diagnose(args):
    tree = HierarchicalTree.load(args.model_path)
    result = tree.statistics()
    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("algorithm") != ALGORITHM or manifest.get("algorithm_version") != ALGORITHM_VERSION:
            raise ValueError("unsupported manifest algorithm/version")
        suffix = ".manifest.json"
        dataset = manifest_path.name.removesuffix(suffix)
        if not manifest_path.name.endswith(suffix) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", dataset):
            raise ValueError("invalid manifest filename")
        required = {dataset + ending for ending in (".tree.json", ".index.json", ".item_ids.json", ".tree_stats.json", suffix)}
        allowed = required | {dataset + ending for ending in DATASET_SUFFIXES}
        outputs = manifest.get("outputs")
        if (not isinstance(outputs, list) or any(not isinstance(name, str) for name in outputs)
                or len(set(outputs)) != len(outputs) or not required <= set(outputs) <= allowed):
            raise ValueError("manifest outputs must include all required artifacts without duplicates")
        hashes = manifest.get("artifact_sha256")
        if not isinstance(hashes, dict) or set(hashes) != set(outputs) - {manifest_path.name}:
            raise ValueError("manifest hashes must cover every output except the manifest itself")
        for name, expected in hashes.items():
            if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name or ":" in name:
                raise ValueError("invalid manifest artifact name")
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("invalid artifact SHA256")
            if sha256_file(manifest_path.parent / name) != expected:
                raise ValueError(f"artifact SHA256 mismatch: {name}")
        model = Path(args.model_path)
        if model.name not in hashes or sha256_file(model) != hashes[model.name]:
            raise ValueError("model does not match manifest")
        index_path = manifest_path.parent / model.name.replace(".tree.json", ".index.json")
        if json.loads(index_path.read_text(encoding="utf-8")) != tree.to_index():
            raise ValueError("index and tree disagree")
        mapping = manifest.get("item_id_mapping")
        if not isinstance(mapping, dict) or mapping.get("file") != dataset + ".item_ids.json":
            raise ValueError("manifest item ID mapping is invalid")
        ids = json.loads((manifest_path.parent / mapping["file"]).read_text(encoding="utf-8"))
        if not isinstance(ids, list) or sorted(canonical_item_ids(ids, len(tree.item_ids))) != tree.item_ids:
            raise ValueError("exported row IDs and tree catalog disagree")
        if manifest.get("catalog_sha256") != tree.catalog_hash or manifest.get("configuration") != tree.config:
            raise ValueError("manifest and tree configuration/catalog disagree")
        result["artifact_hashes_verified"] = len(hashes)
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        return diagnose(args) if args.operation == "diagnose" else generate(args)
    except (ValueError, RuntimeError, ImportError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
