"""Fit, re-encode and diagnose a fixed-catalog hierarchical K-means SID tree."""

import argparse
from collections import Counter
import hashlib
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
from .hierarchical_tree import (ALGORITHM, ALGORITHM_VERSION, CONFIG_KEYS, SPLIT_BACKENDS,
                                STRATEGY_DEFAULTS, TREE_FORMAT_VERSION, HierarchicalTree,
                                artifact_version, parse_artifact_json, read_artifact_json, validate_config)
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
            command.add_argument("--method", choices=("balanced", "greedy", "hybrid"), default="balanced",
                                 help="split strategy (default: balanced constrained K-means)")
            command.add_argument("--branch-factor", type=int, default=10, help="children per split (all methods)")
            command.add_argument("--seed", type=int, default=0, help="random seed for both solvers")
            command.add_argument("--tol", type=float, default=1e-4, help="convergence tolerance for both solvers")
            command.add_argument("--n-init", type=int, default=10, help="constrained solver restarts")
            command.add_argument("--max-iter", type=int, default=300, help="constrained solver iteration limit")
            command.add_argument("--n-jobs", type=int, default=1, help="constrained solver jobs")
            command.add_argument("--greedy-threshold", type=int, default=STRATEGY_DEFAULTS["greedy_threshold"],
                                 help="hybrid uses greedy only above this node size (default: 2000)")
            command.add_argument("--greedy-n-init", type=int, default=STRATEGY_DEFAULTS["greedy_n_init"],
                                 help="ordinary K-means restarts at greedy splits (default: 5)")
            command.add_argument("--greedy-max-iter", type=int, default=STRATEGY_DEFAULTS["greedy_max_iter"],
                                 help="ordinary K-means iteration limit at greedy splits (default: 20)")
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
            config = {key: getattr(args, key) for key in CONFIG_KEYS}
            model = HierarchicalKMeans(**config).fit(data, ids)
            tree = model.tree
            fit_seconds = time.perf_counter() - start
            source_hash = None
            backends = dict(Counter(row["backend"] for row in model.split_diagnostics))
        else:
            model_path = Path(args.model_path)
            manifest_path = None
            if model_path.name.endswith(".tree.json"):
                adjacent = model_path.with_name(model_path.name.removesuffix(".tree.json") + ".manifest.json")
                if adjacent.exists():
                    manifest_path = adjacent
            tree, source_hash, _, _ = _load_model(model_path, manifest_path)
            tree.encode(data, ids)
            fit_seconds = 0.0
            backends = {}
        export_start = time.perf_counter()
        index = tree.to_index()
        statistics = tree.statistics()
        topology_stats = dict(statistics)
        if args.operation == "fit":
            statistics["split_diagnostics"] = model.split_diagnostics
    statistics["runtime"] = dict(monitor.result, fit_seconds=fit_seconds)
    dataset = args.dataset_name
    names = [dataset + suffix for suffix in (".index.json", ".tree.json", ".item_ids.json", ".tree_stats.json", ".manifest.json")]
    names.extend(source.name for source in copies)
    manifest = {
        "format_version": TREE_FORMAT_VERSION,
        "algorithm": ALGORITHM,
        "algorithm_version": ALGORITHM_VERSION,
        "operation": args.operation,
        "configuration": tree.config,
        "fit_backends": backends,
        "backend_options": _backend_options(backends),
        "input": {"embedding_sha256": sha256_file(args.data_path), "items": len(ids), "dimension": int(data.shape[1]),
                  "embedding_filename": Path(args.data_path).name, "original_dtype": str(data.dtype)},
        "item_id_mapping": {"file": dataset + ".item_ids.json", "order": "input embedding-row order",
                            "source_sha256": sha256_file(args.item_ids), "leaf_assignment": "lexicographic canonical string ID order"},
        "catalog_sha256": tree.catalog_hash,
        "versions": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "scikit-learn", "k-means-constrained", "scipy", "ortools")}},
        "source_model_sha256": source_hash,
        "copied_dataset_inputs": [{"file": p.name, "sha256": sha256_file(p)} for p in copies],
        "tree_statistics": topology_stats,
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
    result = {"index": str(output / names[0]), "items": len(index), "statistics": statistics,
              "export_seconds": time.perf_counter() - export_start}
    print(json.dumps(result, indent=2))
    return result


def _backend_options(backends):
    options = {}
    for backend in backends:
        options[backend] = {"working_dtype": "float32", "threadpool_limit": 1, "init": "k-means++"}
        if backend == SPLIT_BACKENDS["greedy"]:
            options[backend]["algorithm"] = "lloyd"
    return options


_MANIFEST_FIELDS = {
    "algorithm", "algorithm_version", "operation", "configuration", "input", "item_id_mapping",
    "catalog_sha256", "versions", "source_model_sha256", "copied_dataset_inputs", "tree_statistics",
    "runtime", "outputs", "publication", "scope", "artifact_sha256",
}


def _verify_manifest_files(path, model_path, model_hash, payload, source_version):
    """Check the original identities, configs and file hashes before any v1 upgrade."""
    manifest = read_artifact_json(path)
    if artifact_version(manifest, manifest=True) != source_version:
        raise ValueError("manifest and tree format versions disagree")
    fields = _MANIFEST_FIELDS
    if source_version == TREE_FORMAT_VERSION:
        fields = fields | {"format_version", "fit_backends", "backend_options"}
    if set(manifest) != fields:
        raise ValueError("manifest has missing or unexpected fields")
    config = validate_config(manifest["configuration"], source_version)
    if config != validate_config(payload["config"], source_version) or manifest["catalog_sha256"] != payload["catalog_hash"]:
        raise ValueError("manifest and tree configuration/catalog disagree")
    if manifest["operation"] not in ("fit", "encode"):
        raise ValueError("invalid manifest operation")
    source_hash = manifest["source_model_sha256"]
    if ((manifest["operation"] == "fit" and source_hash is not None)
            or (manifest["operation"] == "encode" and not _is_sha256(source_hash))):
        raise ValueError("invalid source model SHA256 for manifest operation")
    suffix = ".manifest.json"
    dataset = path.name.removesuffix(suffix)
    if not path.name.endswith(suffix) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", dataset):
        raise ValueError("invalid manifest filename")
    required = {dataset + ending for ending in (".tree.json", ".index.json", ".item_ids.json", ".tree_stats.json", suffix)}
    allowed = required | {dataset + ending for ending in DATASET_SUFFIXES}
    outputs = manifest["outputs"]
    if (not isinstance(outputs, list) or any(not isinstance(name, str) for name in outputs)
            or len(set(outputs)) != len(outputs) or not required <= set(outputs) <= allowed):
        raise ValueError("manifest outputs must include all required artifacts without duplicates")
    hashes = manifest["artifact_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != set(outputs) - {path.name}:
        raise ValueError("manifest hashes must cover every output except the manifest itself")
    if model_path.name != dataset + ".tree.json" or hashes[model_path.name] != model_hash:
        raise ValueError("model does not match manifest")
    for name, expected in hashes.items():
        if not _is_sha256(expected) or sha256_file(path.parent / name) != expected:
            raise ValueError(f"artifact SHA256 mismatch: {name}")
    return manifest, dataset


def _is_sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _load_model(model_path, manifest_path=None):
    model_path = Path(model_path)
    raw = model_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    payload = parse_artifact_json(raw)
    source_version = artifact_version(payload)
    if manifest_path is None:
        return HierarchicalTree.from_dict(payload), digest, source_version, 0

    path = Path(manifest_path)
    manifest, dataset = _verify_manifest_files(path, model_path, digest, payload, source_version)
    tree = HierarchicalTree.from_dict(payload)
    index = read_artifact_json(path.parent / (dataset + ".index.json"))
    if index != tree.to_index():
        raise ValueError("index and tree disagree")
    mapping = manifest["item_id_mapping"]
    if not isinstance(mapping, dict) or mapping.get("file") != dataset + ".item_ids.json":
        raise ValueError("manifest item ID mapping is invalid")
    ids = read_artifact_json(path.parent / mapping["file"])
    if not isinstance(ids, list) or sorted(canonical_item_ids(ids, len(tree.item_ids))) != tree.item_ids:
        raise ValueError("exported row IDs and tree catalog disagree")
    inputs = manifest["input"]
    if (not isinstance(inputs, dict) or inputs.get("items") != len(ids)
            or inputs.get("dimension") != tree.dimension or not _is_sha256(inputs.get("embedding_sha256"))
            or not _is_sha256(mapping.get("source_sha256"))):
        raise ValueError("manifest input metadata and tree disagree")

    stats = tree.statistics()
    if source_version == 1:
        for key in ("method", "split_counts", "split_counts_by_depth"):
            del stats[key]  # These did not exist in v1.
    else:
        backends = {SPLIT_BACKENDS[name]: count for name, count in stats["split_counts"].items()
                    if manifest["operation"] == "fit" and count}
        used = manifest["fit_backends"]
        if (not isinstance(used, dict) or any(type(n) is not int for n in used.values())
                or used != backends or manifest["backend_options"] != _backend_options(backends)):
            raise ValueError("manifest fit backends and tree split policy disagree")
    if manifest["tree_statistics"] != stats:
        raise ValueError("manifest tree statistics and tree disagree")
    return tree, digest, source_version, len(manifest["artifact_sha256"])


def diagnose(args):
    tree, _, version_read, verified_count = _load_model(args.model_path, args.manifest_path)
    result = tree.statistics()
    result["source_format_version"] = version_read
    if args.manifest_path:
        result["artifact_hashes_verified"] = verified_count
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
