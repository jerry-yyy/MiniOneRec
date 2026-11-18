"""Independent balanced hierarchical K-means SID construction.

The algorithm recursively partitions subsets of the original embeddings using
the min-cost-flow constrained assignment in k-means-constrained. It does not
subtract residuals, fit neural networks, or fall back to unconstrained grouping.
"""

import logging
import time

import numpy as np

from .hierarchical_tree import HierarchicalTree, canonical_catalog, validate_config


class HierarchicalKMeans:
    def __init__(self, branch_factor=10, seed=0, n_init=10, max_iter=300, tol=1e-4, n_jobs=1):
        self.config = validate_config({"branch_factor": branch_factor, "seed": seed, "n_init": n_init,
                                       "max_iter": max_iter, "tol": tol, "n_jobs": n_jobs})
        self.tree = None
        self.split_diagnostics = []

    def fit(self, embeddings, item_ids):
        """Build a complete deterministic catalog tree in canonical ID order.

        Equal vectors still go through the constrained solver; its finite
        capacity assignment resolves ties without dropping duplicate items.
        Child labels are canonicalized after each fit. Repeatability is scoped
        to the same dependency versions and numerical/thread environment.
        """
        from k_means_constrained import KMeansConstrained
        from threadpoolctl import threadpool_limits

        values, ids, fingerprint = canonical_catalog(embeddings, item_ids)
        branch_factor = self.config["branch_factor"]
        nodes = []
        diagnostics = []

        def build(rows, parent, depth):
            node_id = len(nodes)
            size = len(rows)
            node = {"id": node_id, "parent": parent, "depth": depth, "size": size,
                    "token": f"<node_{node_id}>" if parent is not None and size > 1 else None,
                    "children": [], "items": []}
            nodes.append(node)
            if size <= branch_factor:
                node["items"] = [ids[row] for row in rows]
                return node_id
            lower = size // branch_factor
            upper = lower + 1
            if lower < 1 or not branch_factor * lower <= size <= branch_factor * upper:
                raise ValueError(f"infeasible constrained clustering at node {node_id}: N={size}, K={branch_factor}")
            estimator = KMeansConstrained(n_clusters=branch_factor, size_min=lower, size_max=upper,
                                         random_state=self.config["seed"], n_init=self.config["n_init"],
                                         max_iter=self.config["max_iter"], tol=self.config["tol"],
                                         n_jobs=self.config["n_jobs"])
            start = time.perf_counter()
            try:
                labels = np.asarray(estimator.fit_predict(values[rows]))
            except Exception as exc:
                raise RuntimeError(f"constrained K-means failed at node {node_id} (N={size}, "
                                   f"K={branch_factor}, sizes={lower}..{upper}); no fallback was used") from exc
            elapsed = time.perf_counter() - start
            if labels.shape != (size,) or not np.issubdtype(labels.dtype, np.integer):
                raise RuntimeError("constrained solver returned invalid cluster labels")
            if np.any(labels < 0) or np.any(labels >= branch_factor):
                raise RuntimeError("constrained solver returned cluster labels outside [0, K)")
            counts = np.bincount(labels, minlength=branch_factor)
            if np.any(counts < lower) or np.any(counts > upper):
                raise RuntimeError(f"constrained solver violated capacities at node {node_id}: {counts.tolist()}")
            groups = [rows[labels == label] for label in range(branch_factor)]
            groups.sort(key=lambda group: ids[group[0]])
            record = {"node_id": node_id, "depth": depth, "items": size, "size_min": lower,
                      "size_max": upper, "child_sizes": [len(group) for group in groups],
                      "inertia": float(estimator.inertia_), "iterations": int(estimator.n_iter_),
                      "elapsed_seconds": elapsed}
            diagnostics.append(record)
            logging.info("cluster node=%d depth=%d items=%d sizes=%s seconds=%.3f", node_id, depth,
                         size, record["child_sizes"], elapsed)
            for group in groups:
                node["children"].append(build(group, node_id, depth + 1))
            return node_id

        start = time.perf_counter()
        # Bound BLAS/OpenMP oversubscription; n_jobs remains an explicit solver
        # parameter. Canonical rows and single-thread defaults stabilize ties.
        with threadpool_limits(limits=1):
            build(np.arange(len(ids)), None, 0)
        fitted = HierarchicalTree(nodes=nodes, item_ids=ids, dimension=values.shape[1],
                                  catalog_hash=fingerprint, config=self.config)
        self.tree = fitted
        self.split_diagnostics = diagnostics
        self.fit_seconds = time.perf_counter() - start
        return self

    def encode(self, embeddings, item_ids):
        if self.tree is None:
            raise ValueError("fit or load a tree before encoding")
        return self.tree.encode(embeddings, item_ids)
