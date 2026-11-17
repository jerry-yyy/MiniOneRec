"""One fixed synthetic, CPU-only three-arm software/geometry check.

No tuning or recommendation evaluation. The refit-only arm separates extra PQ
training from geometric shaping. The original FAISS fit is shared by all arms.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from sid_generation.rq_opq import RQOPQ
from sid_generation.sid_common import build_indexes
from .adapters.rq_opq import RQOPQBackend


def compare():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rng = np.random.default_rng(2024)
        x = (rng.standard_t(4, size=(256, 8)) * np.geomspace(0.2, 3.0, 8)).astype(np.float32)
        x[248:254] = x[:6]
        x[254:] = 0
        ids = [f"item-{i:04d}" for i in range(len(x))]
        settings = dict(rq_levels=2, codebook_size=4, opq_subspaces=2, nbits=2,
                        rq_niter=8, opq_niter=4, pq_niter=8, seed=2024, threads=1)
        baseline = RQOPQ(**settings).fit(x)
        original_codes = baseline.encode(x).copy()
        shared = dict(steps=80, lr=0.01, semantic_weight=1.0, cosine_weight=1.0, device="cpu")
        arms = {
            "original": dict(geometry={"mode": "none", "weight": 0.0}, training=shared),
            "refit_only": dict(geometry={"mode": "none", "weight": 0.0},
                               training=dict(shared, pq_refit_only=True)),
            "block_shaping": dict(geometry={"mode": "block", "weight": 0.1}, training=shared),
        }
        result = {"seed": 2024, "items": len(x), "dimension": x.shape[1],
                  "baseline_configuration": settings, "shaping_configuration": shared,
                  "geo_weight": 0.1, "diagnostic_seed": 314159, "arms": {},
                  "claim": "single synthetic software check; no convergence or recommendation benefit claim"}
        for name, options in arms.items():
            start = time.perf_counter()
            model = RQOPQBackend(model=baseline, seed=2024, **options).fit(x, ids)
            codes = model.encode(x, ids)
            np.testing.assert_array_equal(codes[:, :2], original_codes[:, :2])
            np.testing.assert_array_equal(baseline.encode(x), original_codes)
            if name == "original":
                np.testing.assert_array_equal(codes, original_codes)
            _, _, statistics = build_indexes(ids, codes, model.capacities)
            result["arms"][name] = {
                "configuration": model.configuration(),
                "diagnostics": model.diagnose(x, ids, 314159),
                "sid_statistics": statistics, "history": model.history,
                "wall_seconds": time.perf_counter() - start,
                "rq_prefix_identical": True,
            }
        return result
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new JSON report (no model/data files)")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("choose a new report filename; previous results are preserved")
    result = compare()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({name: entry["diagnostics"] for name, entry in result["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
