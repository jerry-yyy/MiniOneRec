"""Artifact compatibility and provenance through the actual fit/encode/diagnose CLI.

hierarchical_v1.zip preserves five files from the v1 Balanced writer, including
the original line endings covered by the manifest hashes. The v1 fit CLI used
the inputs below, --dataset-name Legacy, --branch-factor 2 and default solver
parameters.
"""

import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from sid_generation.hierarchical_cli import main
from sid_generation.hierarchical_tree import HierarchicalTree


IDS = ["42", "0", "c", "7", "15", "x", "2"]
VALUES = np.array([[10, 1], [0, 0], [2, 2], [-3, 1], [11, 0], [-4, 0], [1, 3]], dtype=np.float32)


def publish(tmp, directory, operation, *options, reverse=False):
    np.save(tmp / "legacy-input.npy", VALUES[::-1] if reverse else VALUES)
    (tmp / "legacy-ids.json").write_text(json.dumps(IDS[::-1] if reverse else IDS), encoding="utf-8")
    path = tmp / directory
    main([operation, "--dataset-name", "Legacy", "--data-path", str(tmp / "legacy-input.npy"),
          "--item-ids", str(tmp / "legacy-ids.json"), "--output-dir", str(path), *options])
    return path


def read(directory, suffix):
    return json.loads((directory / ("Legacy." + suffix + ".json")).read_text(encoding="utf-8"))


def diagnose(directory):
    return main(["diagnose", "--model-path", str(directory / "Legacy.tree.json"),
                 "--manifest-path", str(directory / "Legacy.manifest.json")])


@pytest.fixture
def legacy(tmp_path):
    path = tmp_path / "v1"
    with ZipFile(Path(__file__).parent / "fixtures" / "hierarchical_v1.zip") as archive:
        archive.extractall(path)
    return path


def test_real_v1_publication_validates_and_migrates_without_changing_its_index(legacy, tmp_path):
    source = legacy / "Legacy.tree.json"
    expected_source_hash = "59ed6110f3ed061ede74c07fe5e3d057d7216b0e596129c63b6bcbb6c2cc0861"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_source_hash
    report = diagnose(legacy)
    assert report["source_format_version"] == 1 and report["artifact_hashes_verified"] == 4
    assert report["method"] == "balanced"
    assert HierarchicalTree.load(source).encode(VALUES, IDS) == read(legacy, "index")
    new = publish(tmp_path, "new", "encode", "--model-path", str(source), reverse=True)
    assert read(new, "index") == read(legacy, "index")
    assert read(new, "item_ids") == IDS[::-1]
    converted = read(new, "manifest")
    assert converted["source_model_sha256"] == expected_source_hash
    assert converted["operation"] == "encode" and converted["fit_backends"] == {}
    assert converted["format_version"] == read(new, "tree")["format_version"] == 2
    assert converted["algorithm"] == "hierarchical_kmeans" and converted["algorithm_version"] == "2.0.0"
    assert diagnose(new)["source_format_version"] == 2
    fresh = publish(tmp_path, "default", "fit", "--branch-factor", "2")
    assert read(fresh, "tree")["nodes"] == read(legacy, "tree")["nodes"]
    # Account for the fixture's CRLF endings when comparing local JSON bytes.
    expected = (legacy / "Legacy.index.json").read_bytes().replace(b"\r\n", os.linesep.encode())
    assert expected == (fresh / "Legacy.index.json").read_bytes() == (new / "Legacy.index.json").read_bytes()


@pytest.mark.parametrize("method,constrained,greedy", [
    ("balanced", 3, 0), ("greedy", 0, 3), ("hybrid", 2, 1),
])
def test_v2_cli_roundtrip_reports_only_executed_backends(tmp_path, method, constrained, greedy):
    options = ["--branch-factor", "2", "--method", method, "--greedy-threshold", "4",
               "--greedy-n-init", "2", "--greedy-max-iter", "13"]
    first = publish(tmp_path, "fit", "fit", *options)
    other = publish(tmp_path, "repeat", "fit", *options, reverse=True)
    path = first / "Legacy.tree.json"
    assert path.read_bytes() == (other / path.name).read_bytes()
    report = diagnose(first)
    assert report["split_counts"] == {"constrained": constrained, "greedy": greedy}
    assert report["source_format_version"] == 2 and report["method"] == method
    manifest = read(first, "manifest")
    expected = {backend: count for backend, count in [
        ("k_means_constrained.KMeansConstrained", constrained), ("sklearn.cluster.KMeans", greedy)] if count}
    assert manifest["fit_backends"] == expected
    assert manifest["configuration"]["greedy_n_init"] == 2
    assert manifest["configuration"]["greedy_max_iter"] == 13
    assert manifest["configuration"]["greedy_threshold"] == 4
    assert manifest["configuration"]["method"] == method
    encoded = publish(tmp_path, "encode", "encode", "--model-path", str(path), reverse=True)
    assert diagnose(encoded)["split_counts"] == report["split_counts"]
    assert read(encoded, "manifest")["fit_backends"] == {}
    assert read(encoded, "manifest")["source_model_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (encoded / "Legacy.index.json").read_bytes() == (first / "Legacy.index.json").read_bytes()


@pytest.mark.parametrize("corruption", [
    "legacy-config-with-method", "bad-version", "bool-format", "mixed-identities",
    "wrong-config", "missing-hash", "bad-tree-hash", "duplicate-key",
])
def test_legacy_checks_precede_reexport(legacy, tmp_path, corruption):
    manifest_path = legacy / "Legacy.manifest.json"
    m = read(legacy, "manifest")
    if corruption == "legacy-config-with-method":
        m["configuration"]["method"] = "balanced"
    elif corruption == "bad-version":
        m["algorithm_version"] = "1.0.1"
    elif corruption == "bool-format":
        m["format_version"] = True
    elif corruption == "mixed-identities":
        m.update(algorithm="hierarchical_kmeans", algorithm_version="2.0.0", format_version=2)
    elif corruption == "wrong-config":
        m["configuration"]["n_init"] += 1
    elif corruption == "missing-hash":
        del m["artifact_sha256"]["Legacy.tree_stats.json"]
    elif corruption == "bad-tree-hash":
        m["artifact_sha256"]["Legacy.tree.json"] = "0" * 64
    data = json.dumps(m)
    if corruption == "duplicate-key":
        data = '{"algorithm":"forged",' + data[1:]
    manifest_path.write_text(data, encoding="utf-8")
    with pytest.raises(SystemExit):
        diagnose(legacy)
    with pytest.raises(SystemExit):
        publish(tmp_path, "rejected", "encode", "--model-path", str(legacy / "Legacy.tree.json"))
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("target,field", [
    ("tree", "method"), ("manifest", "method"), ("manifest", "format_version"),
    ("manifest", "fit_backends"), ("manifest", "backend_options"),
])
def test_v2_requires_explicit_strategy_and_consistent_provenance(tmp_path, target, field):
    path = publish(tmp_path, "new", "fit", "--branch-factor", "2", "--method", "greedy")
    payload = read(path, target)
    if field == "method":
        del payload["config" if target == "tree" else "configuration"][field]
    elif field == "format_version":
        del payload[field]
    else:
        payload[field] = {}
    file = path / f"Legacy.{target}.json"
    file.write_text(json.dumps(payload), encoding="utf-8")
    if target == "tree":
        with pytest.raises(ValueError):
            HierarchicalTree.load(file)
    with pytest.raises(SystemExit):
        diagnose(path)
