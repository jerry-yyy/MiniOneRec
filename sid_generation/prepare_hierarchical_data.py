"""Fetch and validate the pinned public Industrial catalog; never load model weights.

The release contains processed CSV interactions, not .inter files. Reconstruct
the latter from explicit original-ID columns so the existing converter can run.
Existing files must match the pinned bytes; they are never silently overwritten.
"""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import urllib.request

import numpy as np


DATASET = "Industrial_and_Scientific"
REVISION = "365a03fc6601ed36abd69cc9a0a59025a3d31cdc"
REPOSITORY = "kkknight/MiniOneRec"
UPSTREAM_REVISION = "0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1"
SOURCE_PATH = "rq/text2emb/amazon_text2emb.py"
SOURCE_SHA256 = "d0ac88885b8a24ad7dd7cf61c231a13b7bf419eed1346e71bb0b87ce522e2853"
CSV_NAME = f"{DATASET}_5_2016-10-2018-11.csv"
FILES = {
    f"Amazon/index/{DATASET}.emb-qwen-td.npy":
        "91a1c875e4b009f3a07b5febf00c2ef62a3813a55cfe873b825e18e32df1dd0b",
    f"Amazon/index/{DATASET}.item.json":
        "937600c7a03652f55864a1d12d6e9fe8ca3e24ecc4e255a88216b25c5f949496",
    f"Amazon/train/{CSV_NAME}":
        "da5f2af4f5047e18023dd4792f65a48bd7c8eab90447dd682fb48ca255d25f8c",
    f"Amazon/valid/{CSV_NAME}":
        "12bbb02bced0bd754a8b4e0f9ca4086416e30d12b0b73a2c95e7b195dd67385a",
    f"Amazon/test/{CSV_NAME}":
        "973dd1db4fd549f3f2b40a0f039dbae157ebb92c716197cc246401e4e82c3a46",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(path, remote, expected, offline):
    if path.exists():
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise ValueError(f"existing source does not match pinned SHA256: {path}")
        return
    if offline:
        raise FileNotFoundError(f"offline source missing: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            request = urllib.request.Request(remote, headers={"User-Agent": "MiniOneRec-data-audit/1"})
            with urllib.request.urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, handle)
        if sha256(temporary) != expected:
            raise ValueError(f"download SHA256 mismatch: {remote}")
        # Exclusive creation avoids overwriting a concurrent download/user file.
        with path.open("xb") as output, temporary.open("rb") as source:
            shutil.copyfileobj(source, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_verified(path, data):
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"derived output differs; choose a fresh source directory: {path}")
        return
    with path.open("xb") as handle:
        handle.write(data)


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def prepare(source_dir, offline=False):
    source_dir = Path(source_dir)
    if source_dir.is_symlink():
        raise ValueError("source directory must not be a symbolic link")
    sources = []
    for name, checksum in FILES.items():
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
        path = source_dir / name
        fetch(path, url, checksum, offline)
        sources.append({"path": name, "url": url, "sha256": checksum, "bytes": path.stat().st_size})

    embeddings_path = source_dir / f"Amazon/index/{DATASET}.emb-qwen-td.npy"
    metadata_path = source_dir / f"Amazon/index/{DATASET}.item.json"
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    if embeddings.shape != (3686, 2560) or embeddings.dtype != np.dtype("float16"):
        raise ValueError(f"unexpected pinned embedding shape/dtype: {embeddings.shape}/{embeddings.dtype}")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings contain NaN/Inf")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not all(re.fullmatch(r"0|[1-9][0-9]*", key) for key in metadata):
        raise ValueError("metadata must have canonical nonnegative numeric item IDs")
    # The upstream producer casts IDs to int, sorts (ID, embedding) pairs, then
    # stacks the vectors. This explicit sort is independent of JSON key order.
    row_ids = sorted(metadata, key=int)
    if row_ids != [str(row) for row in range(len(embeddings))]:
        raise ValueError("pinned metadata IDs must match every numeric embedding row, including 0")

    output = source_dir / "prepared"
    output.mkdir(parents=True, exist_ok=True)
    write_verified(output / f"{DATASET}.item_ids.json", json_bytes(row_ids))
    write_verified(output / f"{DATASET}.item.json", metadata_path.read_bytes())
    split_stats = {}
    csv.field_size_limit(10_000_000)
    for split in ("train", "valid", "test"):
        rows = ["user_id:token\titem_id_list:token_seq\titem_id:token\n"]
        users = set()
        zero_history = 0
        zero_target = 0
        pair_digest = hashlib.sha256()
        with (source_dir / f"Amazon/{split}/{CSV_NAME}").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                user = row["user_id"]
                if not re.fullmatch(r"A[0-9]+", user):
                    raise ValueError(f"unexpected user ID: {user}")
                history = ast.literal_eval(row["history_item_id"])
                if not isinstance(history, list) or not all(type(item) is int for item in history):
                    raise ValueError("history_item_id must be a literal list of integer IDs")
                if not re.fullmatch(r"0|[1-9][0-9]*", row["item_id"]):
                    raise ValueError("target item_id must be a canonical nonnegative integer")
                target = row["item_id"]
                if any(str(item) not in metadata for item in history) or target not in metadata:
                    raise ValueError("interaction refers to an unknown catalog item")
                rows.append(f"{user[1:]}\t{' '.join(map(str, history))}\t{target}\n")
                pair_digest.update(json.dumps([user, history, int(target)], separators=(",", ":")).encode())
                pair_digest.update(b"\n")
                users.add(user)
                zero_history += history.count(0)
                zero_target += target == "0"
        write_verified(output / f"{DATASET}.{split}.inter", "".join(rows).encode("utf-8"))
        split_stats[split] = {
            "rows": len(rows) - 1, "users": len(users),
            "item_zero_history_occurrences": zero_history,
            "item_zero_targets": zero_target,
            "ordered_user_history_target_sha256": pair_digest.hexdigest(),
        }

    derived_names = [f"{DATASET}.item_ids.json", f"{DATASET}.item.json"] + [
        f"{DATASET}.{split}.inter" for split in ("train", "valid", "test")]
    report = {
        "dataset": DATASET, "repository": REPOSITORY, "revision": REVISION,
        "sources": sources,
        "embeddings": {"shape": list(embeddings.shape), "dtype": str(embeddings.dtype),
                       "bytes": embeddings_path.stat().st_size, "all_finite": True},
        "item_count": len(row_ids), "item_zero_included": True,
        "row_mapping": {
            "rule": "numeric item ID sort before stacking; all IDs 0..3685 verified",
            "item_ids_file": f"prepared/{DATASET}.item_ids.json",
            "source_url": f"https://github.com/AkaliKong/MiniOneRec/blob/{UPSTREAM_REVISION}/{SOURCE_PATH}#L38-L44",
            "sort_url": f"https://github.com/AkaliKong/MiniOneRec/blob/{UPSTREAM_REVISION}/{SOURCE_PATH}#L139-L148",
            "producer_source_sha256": SOURCE_SHA256,
            "evidence_scope": "upstream producer contract plus pinned artifact schema validation; no embedding re-encoding",
        },
        "interaction_adaptation": "lossless original-ID columns from release CSV; strip one A user prefix because convert_dataset.py restores it; preserve order and all rows",
        "splits": split_stats,
        "derived_outputs": {name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
                            for name in derived_names},
    }
    write_verified(source_dir / "data_provenance.json", json_bytes(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/hierarchical-source"))
    parser.add_argument("--offline", action="store_true", help="verify already downloaded pinned files without network")
    args = parser.parse_args(argv)
    report = prepare(args.source_dir, args.offline)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
