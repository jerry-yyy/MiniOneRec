"""Exercise MiniOneRec's existing SID consumers without loading model weights.

TokenExtender, evaluator prefix construction, and dataset classes are compiled
unchanged from repository AST nodes. Only eager dataset preprocessing is disabled;
selected actual ``pre`` methods still run with the real tokenizer. The logits
processor runs its actual CPU tensor operations. This is not model generation.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import tempfile
from typing import List
import warnings


ROOT = Path(__file__).resolve().parents[1]
HEADER = "### Response:\n"
QWEN_REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
QWEN_FILES = ("config.json", "tokenizer_config.json", "tokenizer.json", "merges.txt", "vocab.json")


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _module(path, nodes, scope):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def token_extender(index_path):
    """Instantiate the actual sft.py TokenExtender without its training imports."""
    path = ROOT / "sft.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TokenExtender")
    scope = _module(path, [node], {"os": os, "json": json})
    index_path = Path(index_path)
    _require(index_path.name.endswith(".index.json"), "expected Dataset.index.json")
    return scope["TokenExtender"](str(index_path.parent), index_path.name[:-len(".index.json")])


def evaluator_constraints(index, tokenizer):
    """Execute the existing semantic prefix-table statements and callback verbatim."""
    path = ROOT / "evaluate.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    get_hash = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_hash")
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    selected = []
    for node in main.body:
        if isinstance(node, ast.If):
            assigned = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            if "prefixID" in assigned or "prefix_index" in assigned:
                selected.append(node)
        elif isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "hash_dict" for t in node.targets):
                selected.append(node)
        elif isinstance(node, ast.For):
            names = {n.id for n in ast.walk(node.iter) if isinstance(n, ast.Name)}
            if "prefixID" in names or "hash_dict" in names:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "prefix_allowed_tokens_fn_semantic":
            selected.append(node)
    _require(len(selected) == 6, "evaluate.py prefix construction changed; review AST extraction")
    scope = {"tokenizer": tokenizer, "base_model": "Qwen2.5",
             "info_semantic": [HEADER + "".join(path) + "\n" for path in index.values()],
             "info_titles": []}
    return _module(path, [get_hash, *selected], scope)


def _processor_class():
    # Import only this small module, not sft/evaluate and their training stacks.
    import importlib.util
    path = ROOT / "LogitProcessor.py"
    spec = importlib.util.spec_from_file_location("_hierarchical_actual_logits_processor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ConstrainedLogitsProcessor


def _dataset_classes():
    import numpy as np
    import pandas as pd
    path = ROOT / "data.py"
    names = {"Tokenizer", "BaseDataset", "CSVBaseDataset", "JSONBaseDataset", "SidSFTDataset",
             "EvalSidDataset", "SidItemFeatDataset", "FusionSeqRecDataset", "TitleHistory2SidSFTDataset"}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    _require(len(classes) == len(names), "data.py classes changed; review AST extraction")
    scope = {"Dataset": object, "List": List, "pd": pd, "np": np, "json": json,
             "random": random, "copy": copy, "os": os, "tqdm": lambda iterable: iterable}
    _module(path, classes, scope)
    scope["BaseDataset"].get_inputs = lambda self: None
    return scope


def validate_dataset_contract(index, tokenizer):
    """Verify every SID mapping and tokenize representative actual dataset rows."""
    import pandas as pd
    scope = _dataset_classes()
    full = {key: "".join(tokens) for key, tokens in index.items()}
    keys = list(index)
    sample_positions = sorted({0, len(keys) - 1,
                               min(range(len(keys)), key=lambda i: len(index[keys[i]])),
                               max(range(len(keys)), key=lambda i: len(index[keys[i]]))})
    with tempfile.TemporaryDirectory(prefix="hierarchical-integration-") as temporary:
        folder = Path(temporary)
        metadata = {key: {"title": f"Title {key}", "description": f"Description {key}"} for key in keys}
        (folder / "Demo.item.json").write_text(json.dumps(metadata), encoding="utf-8")
        (folder / "Demo.index.json").write_text(json.dumps(index), encoding="utf-8")
        rows = [{"history_item_sid": repr([full[keys[0]]]), "history_item_title": repr([f"Title {keys[0]}"]),
                 "history_item_id": repr([keys[0]]), "item_sid": full[key], "item_id": key}
                for key in keys]
        frame = pd.DataFrame(rows)
        frame.to_csv(folder / "train.csv", index=False)
        kwargs = {"item_file": str(folder / "Demo.item.json"), "index_file": str(folder / "Demo.index.json")}
        feature = scope["SidItemFeatDataset"](tokenizer=tokenizer, **kwargs)
        _require(feature.title2sid == {f"Title {key}": sid for key, sid in full.items()}, "title2sid truncated SID")
        _require(feature.sid2title == {sid: f"Title {key}" for key, sid in full.items()}, "sid2title truncated SID")
        sequence = scope["SidSFTDataset"](train_file=str(folder / "train.csv"), tokenizer=tokenizer, max_len=8192)
        evaluation = scope["EvalSidDataset"](train_file=str(folder / "train.csv"), tokenizer=tokenizer, max_len=8192)
        fusion = scope["FusionSeqRecDataset"](train_file=str(folder / "train.csv"), tokenizer=tokenizer, **kwargs)
        title = scope["TitleHistory2SidSFTDataset"](train_file=str(folder / "train.csv"), tokenizer=tokenizer, **kwargs)
        _require(title.id2sid == full, "title-history ID mapping truncated SID")
        for position, key in enumerate(keys):
            row = frame.iloc[position].copy()
            expected = full[key] + "\n"
            for dataset in (sequence, evaluation, title):
                _require(dataset.get_history(row.copy())["output"] == expected, "dataset output truncated SID")
            _require(fusion.get_history(row.copy())["target_sid"] == full[key], "Fusion SID truncated")
        pre_calls = 0
        for position in sample_positions:
            target = tokenizer.encode(full[keys[position]] + "\n", add_special_tokens=False) + [tokenizer.eos_token_id]
            for dataset in (sequence, evaluation):
                prepared = dataset.pre(position)
                labels = [token for token in prepared["labels"] if token != -100]
                _require(labels == target, "actual dataset pre() truncated or changed SID target")
                _require(prepared["input_ids"][-len(target):] == target, "SID target tokens lost")
                pre_calls += 1
        return {"mapped_items": len(keys), "dataset_classes_checked": 5,
                "actual_tokenized_pre_calls": pre_calls,
                "scope": "unchanged AST classes; eager get_inputs disabled; sampled actual pre() calls"}


def validate_index(index_path, tokenizer):
    """Use an already constructed real tokenizer (Qwen or offline test tokenizer)."""
    import torch
    index_path = Path(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    _require(isinstance(index, dict) and bool(index), "index must be a nonempty object")
    _require(all(isinstance(path, list) and path and all(isinstance(t, str) for t in path)
                 for path in index.values()), "each SID must be a nonempty token list")
    paths = [tuple(path) for path in index.values()]
    _require(len(set(paths)) == len(paths), "duplicate SID")
    path_set = set(paths)
    _require(not any(path[:n] in path_set for path in paths for n in range(1, len(path))), "SID prefix collision")
    extender = token_extender(index_path)
    tokens = extender.get_new_tokens()
    _require(tokens == sorted({token for path in paths for token in path}), "TokenExtender lost tokens")
    added = tokenizer.add_tokens(tokens)
    token_ids = {}
    for token in tokens:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        _require(len(encoded) == 1, f"SID token is not atomic: {token}")
        token_ids[token] = encoded[0]
    _require(len(set(token_ids.values())) == len(tokens), "SID token IDs alias")
    for path in paths:
        _require(tokenizer.encode("".join(path), add_special_tokens=False) == [token_ids[t] for t in path],
                 "adjacent SID tokenization changed IDs")
    scope = evaluator_constraints(index, tokenizer)
    prefix = tokenizer.encode(HEADER, add_special_tokens=False)
    _require(len(prefix) == scope["prefix_index"] == 3, "actual response header does not match Qwen offset 3")
    newline = tokenizer.encode("\n", add_special_tokens=False)
    _require(bool(newline), "tokenizer must retain newline for evaluator validation")
    callback = scope["prefix_allowed_tokens_fn_semantic"]
    expected_states = {}
    continuations = []
    transitions = 0
    for path, actual in zip(paths, scope["prefixID"]):
        continuation = [token_ids[t] for t in path] + newline + [tokenizer.eos_token_id]
        _require(actual == prefix + continuation, "wrapper/newline/EOS tokenization differs from evaluate.py")
        continuations.append(continuation)
        for position, next_token in enumerate(continuation):
            state = tuple(prefix if position == 0 else continuation[:position])
            expected_states.setdefault(state, set()).add(next_token)
            transitions += 1
    for state, expected in expected_states.items():
        _require(set(callback(0, list(state))) == expected, "evaluator accepted wrong next tokens")
    # Use an in-vocabulary base token that cannot start any SID, so this also
    # covers a realistic invalid continuation rather than an impossible ID.
    forbidden_start = set(token_ids.values()) | set(newline) | {tokenizer.eos_token_id}
    invalid_state = [next(token for token in range(len(tokenizer)) if token not in forbidden_start)]
    _require(callback(0, invalid_state) == [], "invalid prefix accepted")

    cls = _processor_class()
    sample_indices = sorted({0, len(paths) - 1,
                             min(range(len(paths)), key=lambda i: len(paths[i])),
                             max(range(len(paths)), key=lambda i: len(paths[i]))})
    mask_calls = 0
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for sample in sample_indices:
            processor = cls(callback, num_beams=2, base_model="Qwen2.5", eos_token_id=tokenizer.eos_token_id)
            continuation = continuations[sample]
            for position in range(len(continuation)):
                current = prefix + continuation[:position]
                input_ids = torch.tensor([current, current], dtype=torch.long)
                scores = torch.zeros((2, len(tokenizer)), dtype=torch.float32)
                masked = processor(input_ids, scores)
                state = prefix if position == 0 else continuation[:position]
                allowed = set(callback(0, state))
                for beam in range(2):
                    finite = set(torch.nonzero(torch.isfinite(masked[beam]), as_tuple=False).flatten().tolist())
                    _require(finite == allowed, "actual logits processor mask differs from allowed next tokens")
                mask_calls += 1
        processor = cls(callback, num_beams=1, base_model="Qwen2.5", eos_token_id=tokenizer.eos_token_id)
        processor.count = 1
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = processor(torch.tensor([prefix + invalid_state]), torch.zeros((1, len(tokenizer))))
        finite = torch.nonzero(torch.isfinite(output[0]), as_tuple=False).flatten().tolist()
        _require(finite == [tokenizer.eos_token_id] and bool(caught), "invalid prefix must warn and force EOS")
        mask_calls += 1
    finally:
        torch.set_num_threads(previous_threads)
    return {"catalog_items": len(index), "sid_lengths": sorted({len(path) for path in paths}),
            "sid_tokens_registered": len(tokens), "new_tokens_added": added,
            "atomic_tokens_checked": len(tokens), "adjacent_sid_encodings_checked": len(paths),
            "response_header_ids": prefix, "newline_ids": newline, "eos_token_id": tokenizer.eos_token_id,
            "legal_path_transitions_checked": transitions, "distinct_prefix_states_checked": len(expected_states),
            "tensor_mask_paths_sampled": len(sample_indices), "tensor_mask_calls": mask_calls,
            "tensor_device": "cpu", "invalid_prefix_forces_eos": True,
            "dataset_contract": validate_dataset_contract(index, tokenizer),
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in ("sft.py", "data.py", "evaluate.py", "LogitProcessor.py")},
            "versions": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "torch", "pandas")},
            "scope": "tokenizer + unchanged evaluator callback + CPU logits masks; no model weights or model.generate"}


def validate_catalog(index_path, tokenizer_path):
    """Validate a full catalog with local tokenizer files; never downloads weights."""
    from transformers import AutoTokenizer
    tokenizer_path = Path(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    report = validate_index(index_path, tokenizer)
    manifest = tokenizer_path / "source_manifest.json"
    if manifest.exists():
        source = json.loads(manifest.read_text(encoding="utf-8"))
        for name, details in source["files"].items():
            _require(Path(name).name == name, "tokenizer manifest contains unsafe filename")
            _require(hashlib.sha256((tokenizer_path / name).read_bytes()).hexdigest() == details["sha256"],
                     f"tokenizer source hash mismatch: {name}")
        report["tokenizer_source"] = source
    else:
        report["tokenizer_source"] = {"local_path": str(tokenizer_path), "pinned_manifest": False}
    return report


def download_tokenizer(destination):
    """Download only explicitly allowlisted files from one immutable revision."""
    import urllib.request
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    source = {"repository": QWEN_REPOSITORY, "revision": QWEN_REVISION,
              "model_weights_downloaded": False, "files": {}}
    for name in QWEN_FILES:
        url = f"https://huggingface.co/{QWEN_REPOSITORY}/resolve/{QWEN_REVISION}/{name}"
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        (destination / name).write_bytes(data)
        source["files"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "url": url}
    (destination / "source_manifest.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    return source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--download-tokenizer", action="store_true", help="Download pinned tokenizer/config only")
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args(argv)
    if args.download_tokenizer:
        download_tokenizer(args.tokenizer_path)
    report = validate_catalog(args.index_path, args.tokenizer_path)
    payload = json.dumps(report, indent=2) + "\n"
    if args.report_path:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
