"""Evaluate a trained Phase 2 checkpoint on any v2 graph file.

Used for transfer checks, where the graphs come from a different source than
the training data: InSDN-trained model -> Mininet traffic, or the CICIDS2017 <->
InSDN cross-dataset check. The normalisation statistics and thresholds stored in
the checkpoint are reused unchanged (no re-fitting on the target data).

    python -m models.evaluate_v2 --checkpoint-path models/checkpoints/gnn_v2_gat.pt \
        --graph-path data/graphs/mininet_v2.pt --split all --out results/phase2/transfer_mininet.json
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch_geometric.loader import DataLoader

from common.config import run_metadata
from common.metrics import binary_metrics, coordinated_subset_metrics, detection_rate_by_class, multiclass_metrics
from models.export import load_checkpoint_model
from models.train_gnn_v2 import _attack_class, predict, window_multiclass_pred
from preprocessing.graph_dataset import GraphFeatureStats, apply_stats_to_graphs


def evaluate(checkpoint_path: str, graph_path: str, split: str = "test", batch_size: int = 64) -> dict:
    model, ckpt = load_checkpoint_model(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = torch.load(graph_path, weights_only=False)
    graphs = data["train"] + data["val"] + data["test"] if split == "all" else data[split]
    graphs = apply_stats_to_graphs(graphs, GraphFeatureStats.from_dict(ckpt["feature_stats"]))
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, exclude_keys=["node_ips"])

    pred = predict(model, loader, device)
    pred["attack_class"] = _attack_class(model, loader, device)
    thr = ckpt["thresholds"]
    win_pred = (pred["graph_score"] >= thr["window"]).astype(int)
    return {
        "checkpoint": checkpoint_path,
        "graph_path": graph_path,
        "split": split,
        "graphs": len(graphs),
        "thresholds": thr,
        "window": {
            "binary": binary_metrics(pred["y"], pred["graph_score"], thr["window"]),
            "multiclass": multiclass_metrics(pred["y_multi"], window_multiclass_pred(pred, thr["window"])),
            "detection_rate_by_class": detection_rate_by_class(pred["y_multi"], win_pred),
            "coordinated_subset": coordinated_subset_metrics(pred["y_multi"], pred["graph_score"], thr["window"]),
        },
        "host": binary_metrics(pred["node_y"], pred["node_score"], thr["node"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--graph-path", required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = evaluate(args.checkpoint_path, args.graph_path, args.split)
    result["run"] = run_metadata(args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    w, h = result["window"]["binary"], result["host"]
    print(f"{result['graphs']} graphs | window F1={w['f1']:.4f} FPR={w['false_positive_rate']} | "
          f"attack-type major macro-F1={result['window']['multiclass']['major_macro_f1']} | host F1={h['f1']:.4f}")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
