"""Train and evaluate the Phase 2 multi-task GNN on pre-split v2 graphs.

Loss = alpha * node_loss (attacker vs. not) + (1 - alpha) * graph_loss (attack
class). Model selection uses the mean of validation window F1, node F1 and
window multi-class macro-F1 over well-supported classes. Test metrics use ``common.metrics`` so they are
directly comparable with ``baselines/train_baselines_v2.py``.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from common.config import parse_args_with_config, run_metadata
from common.metrics import (
    best_threshold,
    binary_metrics,
    coordinated_subset_metrics,
    detection_rate_by_class,
    multiclass_metrics,
)
from models.gnn_v2 import MultiTaskGNN, attack_probability
from preprocessing.graph_dataset import apply_stats_to_graphs, fit_feature_stats
from preprocessing.labels import ATTACK_CLASSES, CLASS_TO_ID

BENIGN_ID = CLASS_TO_ID["Benign"]


def class_weights(labels: list[int], num_classes: int, max_weight: float = 10.0) -> torch.Tensor:
    """Square-root inverse-frequency weights, clipped so rare classes don't dominate."""
    counts = Counter(labels)
    total = sum(counts.values())
    w = [
        min(max_weight, float(np.sqrt(total / (len(counts) * counts[c])))) if counts.get(c) else 0.0
        for c in range(num_classes)
    ]
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict(model, loader, device) -> dict:
    model.eval()
    out = {k: [] for k in ("graph_score", "graph_pred", "y", "y_multi", "node_score", "node_y")}
    for batch in loader:
        batch = batch.to(device)
        graph_logits, node_logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        out["graph_score"].append(attack_probability(graph_logits, BENIGN_ID).cpu())
        out["graph_pred"].append(graph_logits.argmax(dim=1).cpu())
        out["y"].append(batch.y.view(-1).cpu())
        out["y_multi"].append(batch.y_multi.view(-1).cpu())
        out["node_score"].append(torch.softmax(node_logits, dim=1)[:, 1].cpu())
        out["node_y"].append(batch.node_y.cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def window_multiclass_pred(pred: dict, threshold: float) -> np.ndarray:
    """Benign unless the binary score passes the threshold; then best attack class."""
    return np.where(pred["graph_score"] >= threshold, pred["attack_class"], BENIGN_ID)


def selection_score(pred: dict) -> tuple[float, dict]:
    g = binary_metrics(pred["y"], pred["graph_score"], 0.5)
    n = binary_metrics(pred["node_y"], pred["node_score"], 0.5)
    m = multiclass_metrics(pred["y_multi"], pred["graph_pred"])
    # Classes with a handful of validation windows would make selection noisy.
    parts = {"window_f1": g["f1"], "node_f1": n["f1"], "multiclass_macro_f1": m["major_macro_f1"] or m["macro_f1"]}
    return float(np.mean(list(parts.values()))), parts


def _attack_class(model, loader, device) -> np.ndarray:
    """Most likely non-benign class per window."""
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            logits[:, BENIGN_ID] = float("-inf")
            preds.append(logits.argmax(dim=1).cpu())
    return torch.cat(preds).numpy()


def plot_curves(history: list[dict], out_path: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(epochs, [h["train_loss"] for h in history], label="train")
    a.plot(epochs, [h["val_loss"] for h in history], label="val")
    a.set_title("Loss")
    a.set_xlabel("epoch")
    a.legend()
    for key in ("window_f1", "node_f1", "multiclass_macro_f1"):
        b.plot(epochs, [h["val"][key] for h in history], label=key)
    b.set_title("Validation")
    b.set_xlabel("epoch")
    b.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(
    graph_path: str,
    checkpoint_path: str,
    metrics_path: str,
    scores_out: str | None = None,
    curves_path: str | None = None,
    conv_type: str = "gat",
    use_edge_features: bool = True,
    hidden_dim: int = 64,
    heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.2,
    alpha: float = 0.5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    epochs: int = 60,
    patience: int = 10,
    seed: int = 42,
    extra_graph_paths: list[str] | None = None,
    extra_oversample: int = 1,
    metadata: dict | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = torch.load(graph_path, weights_only=False)
    meta = data["meta"]
    # Extra sources (e.g. labelled Mininet traffic) join each split; their training
    # graphs are repeated so a small live dataset isn't drowned out.
    for extra in extra_graph_paths or []:
        more = torch.load(extra, weights_only=False)
        data["train"] = data["train"] + more["train"] * max(1, extra_oversample)
        data["val"] = data["val"] + more["val"]
        data["test"] = data["test"] + more["test"]
        print(f"+ {extra}: train {len(more['train'])} (x{extra_oversample}), val {len(more['val'])}, test {len(more['test'])}")
    stats = fit_feature_stats(data["train"])
    splits = {s: apply_stats_to_graphs(data[s], stats) for s in ("train", "val", "test")}
    loaders = {
        s: DataLoader(g, batch_size=batch_size, shuffle=(s == "train"), exclude_keys=["node_ips"])
        for s, g in splits.items()
    }

    num_classes = len(ATTACK_CLASSES)
    model_config = {
        "node_feat_dim": splits["train"][0].x.shape[1],
        "edge_feat_dim": splits["train"][0].edge_attr.shape[1],
        "num_classes": num_classes,
        "hidden_dim": hidden_dim,
        "heads": heads,
        "num_layers": num_layers,
        "dropout": dropout,
        "conv_type": conv_type,
        "use_edge_features": use_edge_features,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskGNN(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{conv_type} edge_features={use_edge_features} params={n_params:,} device={device}")

    graph_w = class_weights([int(g.y_multi) for g in splits["train"]], num_classes).to(device)
    node_w = class_weights(torch.cat([g.node_y for g in splits["train"]]).tolist(), 2).to(device)
    graph_loss_fn = nn.CrossEntropyLoss(weight=graph_w)
    node_loss_fn = nn.CrossEntropyLoss(weight=node_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    def batch_loss(batch):
        graph_logits, node_logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        return alpha * node_loss_fn(node_logits, batch.node_y) + (1 - alpha) * graph_loss_fn(
            graph_logits, batch.y_multi.view(-1)
        )

    best, best_epoch, wait, history = -1.0, 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = batch_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_loss = float(np.mean([batch_loss(b.to(device)).item() for b in loaders["val"]]))
        score, parts = selection_score(predict(model, loaders["val"], device))
        scheduler.step(score)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "val": parts})
        print(
            f"Epoch {epoch:03d} | loss={np.mean(losses):.4f} val_loss={val_loss:.4f} | "
            + " ".join(f"{k}={v:.4f}" for k, v in parts.items())
            + f" | lr={optimizer.param_groups[0]['lr']:.1e}"
        )
        if score > best + 1e-4:
            best, best_epoch, wait = score, epoch, 0
            torch.save(model.state_dict(), checkpoint_path + ".tmp")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(checkpoint_path + ".tmp", map_location=device))
    os.remove(checkpoint_path + ".tmp")

    val_pred = predict(model, loaders["val"], device)
    window_thr = best_threshold(val_pred["y"], val_pred["graph_score"])
    node_thr = best_threshold(val_pred["node_y"], val_pred["node_score"])

    test = predict(model, loaders["test"], device)
    test["attack_class"] = _attack_class(model, loaders["test"], device)
    win_pred = (test["graph_score"] >= window_thr).astype(int)
    results = {
        "window": {
            "binary": binary_metrics(test["y"], test["graph_score"], window_thr),
            "multiclass": multiclass_metrics(test["y_multi"], window_multiclass_pred(test, window_thr)),
            "multiclass_argmax": multiclass_metrics(test["y_multi"], test["graph_pred"]),
            "detection_rate_by_class": detection_rate_by_class(test["y_multi"], win_pred),
            "coordinated_subset": coordinated_subset_metrics(test["y_multi"], test["graph_score"], window_thr),
        },
        "host": binary_metrics(test["node_y"], test["node_score"], node_thr),
    }

    thresholds = {"window": window_thr, "node": node_thr}
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "feature_stats": stats.to_dict(),
            "thresholds": thresholds,
            "classes": ATTACK_CLASSES,
            "node_feature_names": meta["node_feature_names"],
            "edge_feature_names": meta["edge_feature_names"],
            "graph_meta": {k: v for k, v in meta.items() if k != "run"},
            "epoch": best_epoch,
        },
        checkpoint_path,
    )
    out = {
        "run": metadata or {},
        "model_config": model_config,
        "parameters": n_params,
        "best_epoch": best_epoch,
        "best_val_selection_score": best,
        "thresholds": thresholds,
        "class_weights": {"graph": graph_w.tolist(), "node": node_w.tolist()},
        "split_sizes": {s: len(g) for s, g in splits.items()},
        "extra_graph_paths": extra_graph_paths or [],
        "extra_oversample": extra_oversample,
        "test": results,
        "history": history,
        "checkpoint": checkpoint_path,
    }
    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if scores_out:
        np.savez_compressed(scores_out, **{k: v for k, v in test.items()})
    if curves_path:
        plot_curves(history, curves_path)
    w, h = results["window"]["binary"], results["host"]
    print(
        f"TEST window F1={w['f1']:.4f} FPR={w['false_positive_rate']:.4f} AUC={w['roc_auc']:.4f} | "
        f"multi macro-F1={results['window']['multiclass']['macro_f1']:.4f} | host F1={h['f1']:.4f}"
    )
    print(f"Saved -> {checkpoint_path}, {metrics_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Phase 2 multi-task GNN.")
    parser.add_argument("--graph-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--scores-out", default=None)
    parser.add_argument("--curves-path", default=None)
    parser.add_argument("--conv-type", choices=["gat", "gcn", "sage"], default="gat")
    parser.add_argument("--no-edge-features", dest="use_edge_features", action="store_false")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extra-graph-paths", nargs="*", default=[], help="more v2 graph files to train on")
    parser.add_argument("--extra-oversample", type=int, default=1)
    args, _ = parse_args_with_config(parser, "train_gnn_v2")
    kwargs = {k: v for k, v in vars(args).items() if k != "config"}
    run(**kwargs, metadata=run_metadata(args))


if __name__ == "__main__":
    main()
