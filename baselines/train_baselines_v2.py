"""Phase 2 baselines on exactly the GNN's features and split.

Everything is read from the v2 graph file, so tabular models and the GNN share
the same OpenFlow-compatible features and the same time-ordered split. Results
are reported at three levels so they can be compared with each GNN output:

- flow:   per flow record (edge_attr -> edge label), the classic tabular IDS,
- window: flow predictions aggregated per window  (vs. GNN graph head),
- host:   (a) max flow score over a host's outgoing flows,
          (b) a model on per-host window features  (vs. GNN node head).
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from common.config import parse_args_with_config, run_metadata
from common.metrics import (
    best_threshold,
    binary_metrics,
    coordinated_subset_metrics,
    detection_rate_by_class,
    multiclass_metrics,
)
from preprocessing.labels import ATTACK_CLASSES, CLASS_TO_ID

BENIGN_ID = CLASS_TO_ID["Benign"]


@dataclass
class FlowTable:
    X: np.ndarray
    y: np.ndarray
    y_multi: np.ndarray
    first: np.ndarray
    reverse: np.ndarray
    graph: np.ndarray  # graph index of each record
    src: np.ndarray  # global node index of each record's source


@dataclass
class NodeTable:
    X: np.ndarray
    y: np.ndarray
    graph: np.ndarray


def flow_table(graphs) -> FlowTable:
    parts = {k: [] for k in ("X", "y", "y_multi", "first", "reverse", "graph", "src")}
    offset = 0
    for gi, g in enumerate(graphs):
        e = g.edge_index.shape[1]
        parts["X"].append(g.edge_attr.numpy())
        parts["y"].append(g.edge_y.numpy())
        parts["y_multi"].append(g.edge_multi.numpy())
        parts["first"].append(g.edge_first.numpy())
        parts["reverse"].append(g.edge_reverse.numpy())
        parts["graph"].append(np.full(e, gi))
        parts["src"].append(g.edge_index[0].numpy() + offset)
        offset += g.num_nodes
    return FlowTable(**{k: np.concatenate(v) for k, v in parts.items()})


def node_table(graphs) -> NodeTable:
    return NodeTable(
        X=np.concatenate([g.x.numpy() for g in graphs]),
        y=np.concatenate([g.node_y.numpy() for g in graphs]),
        graph=np.concatenate([np.full(g.num_nodes, gi) for gi, g in enumerate(graphs)]),
    )


def make_model(kind: str, multiclass: bool, seed: int, rf_estimators: int, xgb_estimators: int):
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=rf_estimators,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )
    return XGBClassifier(
        n_estimators=xgb_estimators,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        eval_metric="mlogloss" if multiclass else "logloss",
    )


def fit(model, X, y):
    if isinstance(model, XGBClassifier):
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model


def _encode(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """XGBoost needs contiguous class ids; returns (encoded y, class ids)."""
    classes = np.unique(y)
    return np.searchsorted(classes, y), classes


def window_scores(flows: FlowTable, flow_score: np.ndarray, n_graphs: int) -> np.ndarray:
    """Mean attack probability over each window's forward flow records."""
    fwd = ~flows.reverse
    sums = np.bincount(flows.graph[fwd], weights=flow_score[fwd], minlength=n_graphs)
    counts = np.bincount(flows.graph[fwd], minlength=n_graphs)
    return sums / np.maximum(counts, 1)


def window_classes(
    flows: FlowTable, class_proba: np.ndarray, classes: np.ndarray, is_attack: np.ndarray, n_graphs: int
) -> np.ndarray:
    """Flagged windows get the attack class with the highest summed probability."""
    fwd = ~flows.reverse
    attack_cols = [i for i, c in enumerate(classes) if c != BENIGN_ID]
    totals = np.zeros((n_graphs, len(attack_cols)))
    for j, col in enumerate(attack_cols):
        totals[:, j] = np.bincount(flows.graph[fwd], weights=class_proba[fwd, col], minlength=n_graphs)
    pred = classes[np.array(attack_cols)][totals.argmax(axis=1)]
    return np.where(is_attack, pred, BENIGN_ID)


def host_scores_from_flows(flows: FlowTable, flow_score: np.ndarray, n_nodes: int) -> np.ndarray:
    """Host score = max attack probability over the host's outgoing forward flows."""
    fwd = ~flows.reverse
    scores = np.zeros(n_nodes)
    np.maximum.at(scores, flows.src[fwd], flow_score[fwd])
    return scores


def evaluate_kind(kind: str, splits: dict, seed: int, rf_estimators: int, xgb_estimators: int) -> dict:
    tr_f, va_f, te_f = (flow_table(splits[s]) for s in ("train", "val", "test"))
    n_val, n_test = len(splits["val"]), len(splits["test"])
    val_y = np.array([int(g.y) for g in splits["val"]])
    test_y = np.array([int(g.y) for g in splits["test"]])
    test_multi = np.array([int(g.y_multi) for g in splits["test"]])

    # --- flow level (train on each record once) ---
    fit_mask = tr_f.first
    binary = fit(make_model(kind, False, seed, rf_estimators, xgb_estimators), tr_f.X[fit_mask], tr_f.y[fit_mask])
    va_score = binary.predict_proba(va_f.X)[:, 1]
    te_score = binary.predict_proba(te_f.X)[:, 1]
    flow_thr = best_threshold(va_f.y[va_f.first], va_score[va_f.first])
    flow_metrics = binary_metrics(te_f.y[te_f.first], te_score[te_f.first], flow_thr)

    y_enc, classes = _encode(tr_f.y_multi[fit_mask])
    multi = fit(make_model(kind, True, seed, rf_estimators, xgb_estimators), tr_f.X[fit_mask], y_enc)
    te_proba = multi.predict_proba(te_f.X)
    flow_multi_pred = classes[te_proba.argmax(axis=1)]
    flow_multi = multiclass_metrics(te_f.y_multi[te_f.first], flow_multi_pred[te_f.first])

    # --- window level ---
    val_w = window_scores(va_f, va_score, n_val)
    test_w = window_scores(te_f, te_score, n_test)
    win_thr = best_threshold(val_y, val_w)
    win_pred = (test_w >= win_thr).astype(int)
    window = {
        "binary": binary_metrics(test_y, test_w, win_thr),
        "multiclass": multiclass_metrics(test_multi, window_classes(te_f, te_proba, classes, win_pred == 1, n_test)),
        "detection_rate_by_class": detection_rate_by_class(test_multi, win_pred),
        "coordinated_subset": coordinated_subset_metrics(test_multi, test_w, win_thr),
    }

    # --- host level ---
    tr_n, va_n, te_n = (node_table(splits[s]) for s in ("train", "val", "test"))
    va_host = host_scores_from_flows(va_f, va_score, len(va_n.y))
    te_host = host_scores_from_flows(te_f, te_score, len(te_n.y))
    host_thr = best_threshold(va_n.y, va_host)

    host_model = fit(make_model(kind, False, seed, rf_estimators, xgb_estimators), tr_n.X, tr_n.y)
    va_hf = host_model.predict_proba(va_n.X)[:, 1]
    te_hf = host_model.predict_proba(te_n.X)[:, 1]
    hf_thr = best_threshold(va_n.y, va_hf)

    return {
        "flow": {"binary": flow_metrics, "multiclass": flow_multi},
        "window": window,
        "host": {
            "from_flow_scores": binary_metrics(te_n.y, te_host, host_thr),
            "host_feature_model": binary_metrics(te_n.y, te_hf, hf_thr),
        },
        "_scores": {"window": test_w, "host_from_flows": te_host, "host_features": te_hf},
    }


def run(
    graph_path: str,
    metrics_out: str,
    scores_out: str | None = None,
    seed: int = 42,
    rf_estimators: int = 200,
    xgb_estimators: int = 300,
    models: tuple[str, ...] = ("random_forest", "xgboost"),
    metadata: dict | None = None,
) -> dict:
    splits = torch.load(graph_path, weights_only=False)
    out = {
        "run": metadata or {},
        "graph_path": graph_path,
        "features": splits["meta"]["edge_feature_names"],
        "node_features": splits["meta"]["node_feature_names"],
        "split_sizes": {s: len(splits[s]) for s in ("train", "val", "test")},
    }
    scores = {}
    for kind in models:
        print(f"Training {kind} ...")
        result = evaluate_kind(kind, splits, seed, rf_estimators, xgb_estimators)
        for level, arr in result.pop("_scores").items():
            scores[f"{kind}_{level}"] = arr
        out[kind] = result
        w = result["window"]["binary"]
        print(
            f"  flow F1={result['flow']['binary']['f1']:.4f} | window F1={w['f1']:.4f} "
            f"FPR={w['false_positive_rate']:.4f} | window macro-F1 (multi)={result['window']['multiclass']['macro_f1']:.4f} "
            f"| host F1 (flows)={result['host']['from_flow_scores']['f1']:.4f} "
            f"(features)={result['host']['host_feature_model']['f1']:.4f}"
        )

    os.makedirs(os.path.dirname(metrics_out) or ".", exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved baseline metrics -> {metrics_out}")
    if scores_out:
        np.savez_compressed(scores_out, **scores)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 baselines on the GNN's features and split.")
    parser.add_argument("--graph-path", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--scores-out", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-estimators", type=int, default=200)
    parser.add_argument("--xgb-estimators", type=int, default=300)
    args, _ = parse_args_with_config(parser, "baselines_v2")
    run(
        graph_path=args.graph_path,
        metrics_out=args.metrics_out,
        scores_out=args.scores_out,
        seed=args.seed,
        rf_estimators=args.rf_estimators,
        xgb_estimators=args.xgb_estimators,
        metadata=run_metadata(args),
    )


if __name__ == "__main__":
    main()
