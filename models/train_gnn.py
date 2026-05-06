from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch import nn

from models.gat_model import GATIntrusionDetector
from preprocessing.graph_dataset import create_loaders


def _eval(model, loader, device):
    model.eval()
    ys, ps = [], []
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            ys.extend(batch.y.view(-1).cpu().tolist())
            ps.extend(pred.cpu().tolist())
    return {
        "accuracy": float(accuracy_score(ys, ps)),
        "precision": float(precision_score(ys, ps, zero_division=0)),
        "recall": float(recall_score(ys, ps, zero_division=0)),
        "f1": float(f1_score(ys, ps, zero_division=0)),
    }


def _eval_with_loss(model, loader, device, criterion):
    model.eval()
    ys, ps = [], []
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            y = batch.y.view(-1)
            losses.append(float(criterion(out, y).item()))
            ys.extend(y.cpu().tolist())
            ps.extend(pred.cpu().tolist())
    report = classification_report(ys, ps, output_dict=True, zero_division=0)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
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
    }


def run(graph_path: str, checkpoint_path: str, metrics_path: str, epochs: int = 40, seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    splits = create_loaders(graph_path=graph_path, batch_size=16, test_size=0.15, val_size=0.15, seed=seed)
    sample_batch = next(iter(splits.train_loader))
    node_feat_dim = sample_batch.x.shape[1]

    train_labels = [int(graph.y.item()) for graph in splits.train_loader.dataset]
    class_counts = Counter(train_labels)
    total_count = sum(class_counts.values())
    class_weights = torch.tensor(
        [total_count / (2.0 * max(class_counts.get(cls, 1), 1)) for cls in range(2)],
        dtype=torch.float32,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GATIntrusionDetector(node_feat_dim=node_feat_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_val_loss = float("inf")
    best_val_f1 = -1.0
    patience = 8
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
            optimizer.step()
            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else 0.0
        val_metrics = _eval_with_loss(model, splits.val_loader, device, criterion)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        print(
            f"Epoch {epoch:03d} | loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | pred_counts={val_metrics['prediction_counts']}"
        )

        if val_metrics["loss"] < (best_val_loss - min_delta):
            best_val_loss = val_metrics["loss"]
            best_val_f1 = val_metrics["f1"]
            wait = 0
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_metrics = _eval(model, splits.test_loader, device)
    out = {
        "best_val_f1": best_val_f1,
        "best_val_loss": best_val_loss,
        "train_class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "class_weights": class_weights.tolist(),
        "test": test_metrics,
        "history": history,
        "checkpoint": checkpoint_path,
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
    args = parser.parse_args()
    run(
        graph_path=args.graph_path,
        checkpoint_path=args.checkpoint_path,
        metrics_path=args.metrics_path,
        epochs=args.epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
