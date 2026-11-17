"""Explicit backend dispatch and shared upstream catalog publication."""

import argparse
import json
from pathlib import Path
import re
import tempfile

import numpy as np

from sid_generation.sid_cli import catalog_inputs, json_write, sha256_file
from sid_generation.sid_common import build_indexes, vectors
from sid_generation.sid_publish import generated_names, publish_catalog, validate_publication
from . import BACKENDS, backend_class


def parser():
    result = argparse.ArgumentParser(description="Optional quantizer-aware GeoSID construction")
    commands = result.add_subparsers(dest="operation", required=True)
    for operation in ("train", "export", "diagnose"):
        command = commands.add_parser(operation)
        command.add_argument("--backend", choices=BACKENDS, required=True)
        command.add_argument("--data-path", required=True, help="finite (items, dimensions) NumPy matrix")
        command.add_argument("--item-ids", required=True, help="explicit JSON list in embedding-row order")
        command.set_defaults(dataset_dir=None)
        if operation == "train":
            command.add_argument("--config", help="JSON object with model/geometry/training/seed options")
            command.add_argument("--geo-weight", type=float, help="override the default-off geometry weight")
        else:
            command.add_argument("--model-path", required=True)
        if operation == "diagnose":
            command.add_argument("--diagnostic-seed", type=int, default=314159)
            command.add_argument("--report", help="new JSON file; existing files are never overwritten")
        else:
            command.add_argument("--output-dir", required=True)
            command.add_argument("--dataset-name", required=True)
            command.add_argument("--collision-policy", choices=("append", "error", "allow"), default="append")
            command.add_argument("--overwrite", action="store_true")
    return result


def train_options(args):
    options = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    if not isinstance(options, dict) or set(options) - {"backend", "model", "geometry", "training", "seed"}:
        raise ValueError("config must contain only backend/model/geometry/training/seed")
    if options.pop("backend", args.backend) != args.backend:
        raise ValueError("config backend does not match --backend")
    if args.geo_weight is not None:
        options["geometry"] = dict(options.get("geometry", {}), weight=args.geo_weight)
    return options


def run(args):
    data = vectors(np.load(args.data_path, allow_pickle=False))
    ids, _ = catalog_inputs(args, len(data))
    implementation = backend_class(args.backend)
    if args.operation != "diagnose":
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.dataset_name):
            raise ValueError("invalid dataset name")
        output = Path(args.output_dir)
        names = generated_names(args.dataset_name)
        protected = {Path(args.data_path).resolve(), Path(args.item_ids).resolve()}
        if args.operation == "export":
            protected.add(Path(args.model_path).resolve())
        elif args.config:
            protected.add(Path(args.config).resolve())
        validate_publication(output, args.dataset_name, names, args.overwrite, protected)
    if args.operation == "train":
        model = implementation(**train_options(args)).fit(data, ids)
    else:
        model = implementation.load(args.model_path)
    codes = model.encode(data, ids)
    raw, final, statistics = build_indexes(ids, codes, model.capacities,
                                           getattr(args, "collision_policy", "append"))
    if args.operation == "diagnose":
        result = {"backend": args.backend, "geometry": model.diagnose(data, ids, args.diagnostic_seed),
                  "sid_statistics": statistics}
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.report:
            with Path(args.report).open("x", encoding="utf-8") as handle:
                handle.write(encoded)
        print(encoded)
        return result
    dataset = args.dataset_name
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "algorithm": "geosid", "backend": args.backend, "operation": args.operation,
        "configuration": model.configuration(), "input_sha256": sha256_file(args.data_path),
        "item_ids_sha256": sha256_file(args.item_ids), "items": len(data), "dimension": data.shape[1],
        "source_model_sha256": sha256_file(args.model_path) if args.operation == "export" else None,
        "outputs": names,
        "scope": "tokenizer only; no downstream quality or convergence claim",
    }
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".geosid-") as temporary:
        stage = Path(temporary)
        json_write(stage / f"{dataset}.index.json", final)
        json_write(stage / f"{dataset}.raw.index.json", raw)
        json_write(stage / f"{dataset}.item_ids.json", ids)
        json_write(stage / f"{dataset}.sid_stats.json", statistics)
        np.save(stage / f"{dataset}.codes.npy", codes, allow_pickle=False)
        model.save(stage / f"{dataset}.tokenizer.npz")
        manifest["artifact_sha256"] = {name: sha256_file(stage / name) for name in names
                                      if not name.endswith(".manifest.json")}
        json_write(stage / f"{dataset}.manifest.json", manifest)
        publish_catalog(stage, output, dataset, names, args.overwrite, protected)
    result = {"backend": args.backend, "items": len(ids),
              "raw_collision_rate": statistics["raw"]["excess_collision_rate"],
              "final_collision_rate": statistics["final"]["excess_collision_rate"],
              "index": str(output / f"{dataset}.index.json")}
    print(json.dumps(result, indent=2))
    return result


def main():
    argument_parser = parser()
    args = argument_parser.parse_args()
    try:
        run(args)
    except (ValueError, TypeError, KeyError, RuntimeError, ImportError, OSError) as exc:
        argument_parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
