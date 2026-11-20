"""Feature selection / reduction comparison on the Phase 2 graph features.

Several selection (keep a subset of the original features) and reduction
(project to new components) algorithms are tried at several sizes ``k`` on the
same time-ordered split the models use. Each (method, k) is fitted on the
train split only and scored with a fixed Random Forest:

- ``edge`` table: one row per flow record (``edge_attr`` -> ``edge_y``)
- ``node`` table: one row per host and window (``x`` -> ``node_y``)

The winner per table is the method/k with the best validation F1; among
candidates within ``tolerance`` of the best, the one with the fewest features
wins (a smaller feature set is cheaper at inference and less likely to
overfit). Test F1 is reported for every candidate but never used to choose.

Output: ``results/phase2/feature_selection.json`` (read by the dashboard).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from common.config import parse_args_with_config, run_metadata


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _tables(graphs: list, kind: str) -> tuple[np.ndarray, np.ndarray]:
    if kind == "edge":
        X = np.concatenate([g.edge_attr.numpy() for g in graphs])
        y = np.concatenate([g.edge_y.numpy() for g in graphs])
    else:
        X = np.concatenate([g.x.numpy() for g in graphs])
        y = np.concatenate([g.node_y.numpy() for g in graphs])
    return _signed_log1p(X.astype(np.float64)), y.astype(np.int64)


def _subsample(X, y, n, rng):
    if len(X) <= n:
        return X, y
    idx = rng.choice(len(X), n, replace=False)
    return X[idx], y[idx]


# Each method returns (transform_fn, kept_feature_indices or None).
def _fit_selector(name: str, k: int, X, y, seed: int):
    if name == "anova_f":
        sel = SelectKBest(f_classif, k=k).fit(X, y)
    elif name == "mutual_info":
        sel = SelectKBest(lambda a, b: mutual_info_classif(a, b, random_state=seed), k=k).fit(X, y)
    elif name == "rf_importance":
        rf = RandomForestClassifier(n_estimators=60, max_depth=10, n_jobs=-1, random_state=seed).fit(X, y)
        keep = np.sort(np.argsort(rf.feature_importances_)[::-1][:k])
        return (lambda Z: Z[:, keep]), keep
    elif name == "l1_logistic":
        # Rank by |coef| of an L1 model: the features it keeps longest are the strongest.
        lr = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=300, random_state=seed).fit(X, y)
        keep = np.sort(np.argsort(np.abs(lr.coef_[0]))[::-1][:k])
        return (lambda Z: Z[:, keep]), keep
    elif name == "rfe":
        sel = RFE(LogisticRegression(max_iter=300, random_state=seed), n_features_to_select=k, step=1).fit(X, y)
    elif name == "pca":
        m = PCA(n_components=k, random_state=seed).fit(X)
        return m.transform, None
    elif name == "truncated_svd":
        m = TruncatedSVD(n_components=k, random_state=seed).fit(X)
        return m.transform, None
    elif name == "fast_ica":
        m = FastICA(n_components=k, random_state=seed, max_iter=300, whiten="unit-variance").fit(X)
        return m.transform, None
    else:
        raise ValueError(f"Unknown method: {name}")
    keep = np.flatnonzero(sel.get_support())
    return (lambda Z: Z[:, keep]), keep


SELECTION = ["anova_f", "mutual_info", "rf_importance", "l1_logistic", "rfe"]
REDUCTION = ["pca", "truncated_svd", "fast_ica"]
LABELS = {
    "all_features": "All features (baseline)",
    "anova_f": "ANOVA F-test",
    "mutual_info": "Mutual information",
    "rf_importance": "RF importance",
    "l1_logistic": "L1 logistic",
    "rfe": "RFE (logistic)",
    "pca": "PCA",
    "truncated_svd": "Truncated SVD",
    "fast_ica": "FastICA",
}


def _score(Xtr, ytr, Xva, yva, Xte, yte, seed: int, n_estimators: int) -> tuple[float, float, float]:
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=12, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
    ).fit(Xtr, ytr)
    fit_s = time.time() - t0
    return (
        float(f1_score(yva, clf.predict(Xva), zero_division=0)),
        float(f1_score(yte, clf.predict(Xte), zero_division=0)),
        fit_s,
    )


def compare(kind: str, graphs: dict, names: list[str], ks: list[int], max_rows: int, seed: int,
            n_estimators: int, tolerance: float) -> dict:
    rng = np.random.default_rng(seed)
    Xtr, ytr = _subsample(*_tables(graphs["train"], kind), max_rows, rng)
    Xva, yva = _subsample(*_tables(graphs["val"], kind), max_rows, rng)
    Xte, yte = _subsample(*_tables(graphs["test"], kind), max_rows, rng)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
    n_feat = Xtr.shape[1]
    ks = sorted({k for k in ks if 1 <= k < n_feat})
    print(f"[{kind}] {len(Xtr)} train / {len(Xva)} val / {len(Xte)} test rows, {n_feat} features, "
          f"attack share {ytr.mean():.1%}")

    results = []
    val, test, fit_s = _score(Xtr, ytr, Xva, yva, Xte, yte, seed, n_estimators)
    results.append({"method": "all_features", "type": "baseline", "k": n_feat,
                    "val_f1": val, "test_f1": test, "fit_seconds": fit_s, "features": None})
    print(f"  {'all_features':<14} k={n_feat:<3} val_f1={val:.4f} test_f1={test:.4f}")

    for name in names:
        for k in ks:
            try:
                t0 = time.time()
                transform, keep = _fit_selector(name, k, Xtr, ytr, seed)
                select_s = time.time() - t0
                val, test, fit_s = _score(transform(Xtr), ytr, transform(Xva), yva, transform(Xte), yte,
                                          seed, n_estimators)
            except Exception as exc:  # one failing method must not abort the comparison
                print(f"  {name:<14} k={k:<3} FAILED: {type(exc).__name__}: {exc}")
                continue
            results.append({"method": name, "type": "selection" if name in SELECTION else "reduction",
                            "k": k, "val_f1": val, "test_f1": test, "fit_seconds": fit_s + select_s,
                            "features": None if keep is None else keep.tolist()})
            print(f"  {name:<14} k={k:<3} val_f1={val:.4f} test_f1={test:.4f}")

    def pick(rows):
        best_val = max(r["val_f1"] for r in rows)
        near = [r for r in rows if r["val_f1"] >= best_val - tolerance]
        return min(near, key=lambda r: (r["k"], -r["val_f1"]))

    best_val = max(r["val_f1"] for r in results)
    winner = pick(results)
    # Reduction methods create new components, which the live OpenFlow feature extractor
    # cannot compute; models use the best method that keeps original columns.
    applicable = pick([r for r in results if r["type"] != "reduction"])
    return {"n_features": n_feat, "rows": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
            "results": results, "best": {k: winner[k] for k in ("method", "type", "k", "val_f1", "test_f1", "features")},
            "applied": {k: applicable[k] for k in ("method", "type", "k", "val_f1", "test_f1", "features")},
            "best_val_f1_any": best_val}


def run(graph_path: str, output_path: str, methods: list[str], edge_ks: list[int], node_ks: list[int],
        max_rows: int = 40000, seed: int = 42, rf_estimators: int = 60, tolerance: float = 0.005,
        metadata: dict | None = None) -> dict:
    graphs = torch.load(graph_path, weights_only=False)
    meta = graphs["meta"]
    out = {"tolerance": tolerance, "labels": LABELS, "run": metadata or {}, "tables": {}}
    for kind, ks in (("edge", edge_ks), ("node", node_ks)):
        res = compare(kind, graphs, methods, ks, max_rows, seed, rf_estimators, tolerance)
        names = meta["edge_feature_names"] if kind == "edge" else meta["node_feature_names"]
        res["feature_names"] = names
        b = res["best"]
        b["feature_names"] = [names[i] for i in b["features"]] if b["features"] else None
        a = res["applied"]
        a["features"] = a["features"] or list(range(res["n_features"]))
        a["feature_names"] = [names[i] for i in a["features"]]
        out["tables"][kind] = res
        what = f"{b['k']} of {res['n_features']} features" if b["type"] != "reduction" else f"{b['k']} components"
        print(f"[{kind}] BEST: {LABELS[b['method']]} ({what}) val_f1={b['val_f1']:.4f} test_f1={b['test_f1']:.4f}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {output_path}")
    return out


def load_selection(path: str | None) -> dict | None:
    """Columns the models should use, ``{"node": [...], "edge": [...], "methods": {...}}``, or None."""
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        tables = json.load(f)["tables"]
    return {
        "node": tables["node"]["applied"]["features"],
        "edge": tables["edge"]["applied"]["features"],
        "methods": {k: {f: tables[k]["applied"][f] for f in ("method", "k", "val_f1")} for k in ("node", "edge")},
        "source": path,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Compare feature selection / reduction methods.")
    p.add_argument("--graph-path", required=True)
    p.add_argument("--output-path", default="results/phase2/feature_selection.json")
    p.add_argument("--methods", nargs="+", default=SELECTION + REDUCTION)
    p.add_argument("--edge-ks", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    p.add_argument("--node-ks", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    p.add_argument("--max-rows", type=int, default=40000)
    p.add_argument("--rf-estimators", type=int, default=60)
    p.add_argument("--tolerance", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=42)
    args, _ = parse_args_with_config(p, "feature_selection")
    run(args.graph_path, args.output_path, args.methods, args.edge_ks, args.node_ks, args.max_rows,
        args.seed, args.rf_estimators, args.tolerance, run_metadata(args))


if __name__ == "__main__":
    main()
