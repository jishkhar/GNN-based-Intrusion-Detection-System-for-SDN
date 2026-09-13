from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    roc_auc_score,
)

from common.config import parse_args_with_config, run_metadata
from models.gat_model import GATIntrusionDetector
from preprocessing.graph_dataset import GraphFeatureStats, create_loaders


def run(graph_path: str, checkpoint_path: str, report_out: str, cm_out: str, metadata: dict | None = None) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
        model_config = checkpoint["model_config"]
        threshold = float(checkpoint.get("decision_threshold", 0.5))
        feature_stats = GraphFeatureStats.from_dict(checkpoint.get("feature_stats"))
        normalize = feature_stats is not None
        # Older checkpoints predate time splits; they were trained on the random split.
        split_config = checkpoint.get("split_config") or {"split_mode": "random"}
    else:
        model_state = checkpoint
        model_config = None
        threshold = 0.5
        feature_stats = None
        normalize = False
        split_config = {"split_mode": "random"}

    split_kwargs = {"test_size": 0.15, "val_size": 0.15, "seed": 42, **split_config}
    splits = create_loaders(
        graph_path=graph_path,
        batch_size=16,
        normalize=normalize,
        feature_stats=feature_stats,
        **split_kwargs,
    )
    sample_batch = next(iter(splits.train_loader))
    if model_config is None:
        model_config = {
            "node_feat_dim": sample_batch.x.shape[1],
            "edge_feat_dim": 0,
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GATIntrusionDetector(**model_config).to(device)
    model.load_state_dict(model_state)
    model.eval()

    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for batch in splits.test_loader:
            batch = batch.to(device)
            out = model(batch)
            prob_attack = torch.softmax(out, dim=1)[:, 1]
            pred = (prob_attack >= threshold).long()
            y_true.extend(batch.y.view(-1).cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            y_score.extend(prob_attack.cpu().tolist())

    report = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0)
    report["decision_threshold"] = threshold
    if len(set(y_true)) == 2:
        report["roc_auc"] = float(roc_auc_score(y_true, y_score))
        report["pr_auc"] = float(average_precision_score(y_true, y_score))
    report["split_config"] = split_kwargs
    report["run"] = metadata or {}
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title("GNN Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_out, dpi=160)
    plt.close(fig)
    print(f"Saved report -> {report_out}")
    print(f"Saved confusion matrix -> {cm_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained GNN model.")
    parser.add_argument("--graph-path", default="data/graphs/cicids_graphs.pt")
    parser.add_argument("--checkpoint-path", default="models/checkpoints/best_gat.pt")
    parser.add_argument("--report-out", default="results/gnn_classification_report.json")
    parser.add_argument("--cm-out", default="results/gnn_confusion_matrix.png")
    args, _ = parse_args_with_config(parser, "evaluate")
    run(args.graph_path, args.checkpoint_path, args.report_out, args.cm_out, metadata=run_metadata(args))


if __name__ == "__main__":
    main()
