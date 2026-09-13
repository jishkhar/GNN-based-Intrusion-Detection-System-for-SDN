from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn

from common.config import parse_args_with_config, run_metadata
from models.gat_model import GATIntrusionDetector
from preprocessing.graph_dataset import create_loaders


def _scores(model, loader, device):
    model.eval()
    ys, scores = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            prob_attack = torch.softmax(out, dim=1)[:, 1]
            ys.extend(batch.y.view(-1).cpu().tolist())
            scores.extend(prob_attack.cpu().tolist())
    return ys, scores


def _ranking_metrics(ys, scores) -> dict:
    if len(set(ys)) < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {
        "roc_auc": float(roc_auc_score(ys, scores)),
        "pr_auc": float(average_precision_score(ys, scores)),
    }


def plot_training_curves(history: list[dict], out_path: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax_loss.plot(epochs, [h["argmax"]["loss"] for h in history], label="val")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend()
    ax_f1.plot(epochs, [h["argmax"]["f1"] for h in history], label="val F1 @0.5")
    ax_f1.plot(epochs, [h["threshold_tuned"]["f1"] for h in history], label="val F1 tuned")
    ax_f1.set_title("Validation F1 (attack class)")
    ax_f1.set_xlabel("epoch")
    ax_f1.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _metrics_from_scores(ys, scores, threshold: float = 0.5):
    ps = [int(score >= threshold) for score in scores]
    report = classification_report(ys, ps, labels=[0, 1], output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(ys, ps)),
        "precision": float(precision_score(ys, ps, zero_division=0)),
        "recall": float(recall_score(ys, ps, zero_division=0)),
        "f1": float(f1_score(ys, ps, zero_division=0)),
        "class_0": {
            "precision": float(report["0"]["precision"]),
            "recall": float(report["0"]["recall"]),
            "f1": float(report["0"]["f1-score"]),
        },
        "class_1": {
            "precision": float(report["1"]["precision"]),
            "recall": float(report["1"]["recall"]),
            "f1": float(report["1"]["f1-score"]),
        },
        "prediction_counts": {str(k): int(v) for k, v in Counter(ps).items()},
        "threshold": float(threshold),
    }


def _eval_with_loss(model, loader, device, criterion):
    model.eval()
    ys, scores = [], []
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            prob_attack = torch.softmax(out, dim=1)[:, 1]
            y = batch.y.view(-1)
            losses.append(float(criterion(out, y).item()))
            ys.extend(y.cpu().tolist())
            scores.extend(prob_attack.cpu().tolist())
    metrics = _metrics_from_scores(ys, scores, threshold=0.5)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        **metrics,
    }


def _best_threshold(ys, scores) -> tuple[float, dict]:
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_metrics = _metrics_from_scores(ys, scores, threshold=best_threshold)
    for threshold in thresholds:
        metrics = _metrics_from_scores(ys, scores, threshold=float(threshold))
        if (metrics["f1"], metrics["recall"], metrics["accuracy"]) > (
            best_metrics["f1"],
            best_metrics["recall"],
            best_metrics["accuracy"],
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def run(
    graph_path: str,
    checkpoint_path: str,
    metrics_path: str,
    epochs: int = 40,
    seed: int = 42,
    batch_size: int = 128,
    hidden_dim: int = 64,
    heads: int = 4,
    dropout: float = 0.25,
    lr: float = 7e-4,
    weight_decay: float = 1e-4,
    patience: int = 10,
    test_size: float = 0.15,
    val_size: float = 0.15,
    split_mode: str = "time",
    split_gap: int = 0,
    curves_path: str | None = None,
    metadata: dict | None = None,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    split_config = {
        "test_size": test_size,
        "val_size": val_size,
        "seed": seed,
        "split_mode": split_mode,
        "split_gap": split_gap,
    }
    splits = create_loaders(graph_path=graph_path, batch_size=batch_size, **split_config)
    sample_batch = next(iter(splits.train_loader))
    node_feat_dim = sample_batch.x.shape[1]
    edge_feat_dim = sample_batch.edge_attr.shape[1] if getattr(sample_batch, "edge_attr", None) is not None else 0

    train_labels = [int(graph.y.item()) for graph in splits.train_loader.dataset]
    class_counts = Counter(train_labels)
    total_count = sum(class_counts.values())
    # Use a mild inverse-frequency weighting; threshold tuning handles the final
    # precision/recall trade-off without forcing attack-only predictions.
    class_weights = torch.tensor(
        [np.sqrt(total_count / (2.0 * max(class_counts.get(cls, 1), 1))) for cls in range(2)],
        dtype=torch.float32,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = {
        "node_feat_dim": node_feat_dim,
        "edge_feat_dim": edge_feat_dim,
        "hidden_dim": hidden_dim,
        "heads": heads,
        "dropout": dropout,
        "num_classes": 2,
    }
    model = GATIntrusionDetector(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_threshold_value = 0.5
    best_epoch = 0
    min_delta = 1e-4
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in splits.train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            y = batch.y.view(-1)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else 0.0
        val_metrics = _eval_with_loss(model, splits.val_loader, device, criterion)
        val_y, val_scores = _scores(model, splits.val_loader, device)
        val_threshold, threshold_metrics = _best_threshold(val_y, val_scores)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "argmax": val_metrics,
                "threshold_tuned": threshold_metrics,
            }
        )

        print(
            f"Epoch {epoch:03d} | loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"argmax_f1={val_metrics['f1']:.4f} | tuned_f1={threshold_metrics['f1']:.4f} | "
            f"tuned_recall={threshold_metrics['recall']:.4f} | threshold={val_threshold:.2f} | "
            f"pred_counts={threshold_metrics['prediction_counts']}"
        )

        is_better = (
            threshold_metrics["f1"] > best_val_f1 + min_delta
            or (
                abs(threshold_metrics["f1"] - best_val_f1) <= min_delta
                and val_metrics["loss"] < best_val_loss - min_delta
            )
        )
        if is_better:
            best_val_loss = val_metrics["loss"]
            best_val_f1 = threshold_metrics["f1"]
            best_threshold_value = val_threshold
            best_epoch = epoch
            wait = 0
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "feature_stats": splits.feature_stats.to_dict() if splits.feature_stats else None,
                    "decision_threshold": best_threshold_value,
                    "best_val_metrics": threshold_metrics,
                    "best_val_loss": best_val_loss,
                    "epoch": best_epoch,
                    "seed": seed,
                    "split_config": split_config,
                },
                checkpoint_path,
            )
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_y, test_scores = _scores(model, splits.test_loader, device)
    test_metrics = _metrics_from_scores(test_y, test_scores, threshold=best_threshold_value)
    test_metrics.update(_ranking_metrics(test_y, test_scores))
    argmax_test_metrics = _metrics_from_scores(test_y, test_scores, threshold=0.5)
    if curves_path:
        plot_training_curves(history, curves_path)
        print(f"Saved training curves -> {curves_path}")
    out = {
        "best_val_f1": best_val_f1,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "decision_threshold": best_threshold_value,
        "batch_size": batch_size,
        "train_class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "class_weights": class_weights.tolist(),
        "test": test_metrics,
        "argmax_test": argmax_test_metrics,
        "history": history,
        "model_config": model_config,
        "feature_stats": splits.feature_stats.to_dict() if splits.feature_stats else None,
        "checkpoint": checkpoint_path,
        "split_config": split_config,
        "split_sizes": {
            "train": len(splits.train_loader.dataset),
            "val": len(splits.val_loader.dataset),
            "test": len(splits.test_loader.dataset),
        },
        "run": metadata or {},
    }

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved GNN metrics -> {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GAT on graph snapshots (Phase 1 MVP).")
    parser.add_argument("--graph-path", default="data/graphs/cicids_graphs.pt")
    parser.add_argument("--checkpoint-path", default="models/checkpoints/best_gat.pt")
    parser.add_argument("--metrics-path", default="results/gnn_metrics.json")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--split-mode", choices=["time", "random"], default="time")
    parser.add_argument("--split-gap", type=int, default=0)
    parser.add_argument("--curves-path", default="results/training_curves.png")
    args, _ = parse_args_with_config(parser, "train_gnn")
    run(
        graph_path=args.graph_path,
        checkpoint_path=args.checkpoint_path,
        metrics_path=args.metrics_path,
        epochs=args.epochs,
        seed=args.seed,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        test_size=args.test_size,
        val_size=args.val_size,
        split_mode=args.split_mode,
        split_gap=args.split_gap,
        curves_path=args.curves_path,
        metadata=run_metadata(args),
    )


if __name__ == "__main__":
    main()
