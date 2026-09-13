"""Export a trained Phase 2 checkpoint as a self-contained TorchScript model.

The ``.pt`` file embeds ``ids_bundle.json`` (normalisation stats, class names,
decision thresholds, feature names, graph-window settings), so the inference
engine needs only this one file. A readable copy of the bundle is written next
to it. The export is verified against the eager model on test graphs.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch_geometric.loader import DataLoader

from common.config import parse_args_with_config, run_metadata
from models.gnn_v2 import MultiTaskGNN
from preprocessing.graph_dataset import GraphFeatureStats, apply_stats_to_graphs

BUNDLE_NAME = "ids_bundle.json"


def load_checkpoint_model(checkpoint_path: str) -> tuple[MultiTaskGNN, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MultiTaskGNN(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval(), ckpt


def make_bundle(ckpt: dict, metadata: dict | None = None) -> dict:
    return {
        "format_version": 1,
        "model_config": ckpt["model_config"],
        "classes": ckpt["classes"],
        "thresholds": ckpt["thresholds"],
        "feature_stats": ckpt["feature_stats"],
        "node_feature_names": ckpt["node_feature_names"],
        "edge_feature_names": ckpt["edge_feature_names"],
        "graph_meta": ckpt.get("graph_meta", {}),
        "export": metadata or {},
    }


def load_exported(path: str, device: str = "cpu") -> tuple[torch.jit.ScriptModule, dict]:
    """Load an exported model and its bundle (used by the inference engine)."""
    files = {BUNDLE_NAME: ""}
    model = torch.jit.load(path, map_location=device, _extra_files=files)
    return model.eval(), json.loads(files[BUNDLE_NAME])


@torch.no_grad()
def verify(eager: MultiTaskGNN, scripted, graph_path: str, stats: GraphFeatureStats, n: int = 100) -> float:
    graphs = torch.load(graph_path, weights_only=False)["test"][:n]
    graphs = apply_stats_to_graphs(graphs, stats)
    max_diff = 0.0
    for batch in DataLoader(graphs, batch_size=1, exclude_keys=["node_ips"]):
        a = eager(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        b = scripted(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        max_diff = max(max_diff, *(float((x - y).abs().max()) for x, y in zip(a, b)))
    return max_diff


def run(checkpoint_path: str, output_path: str, graph_path: str | None = None, metadata: dict | None = None) -> None:
    eager, ckpt = load_checkpoint_model(checkpoint_path)
    scripted = torch.jit.script(eager)
    bundle = make_bundle(ckpt, metadata)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.jit.save(scripted, output_path, _extra_files={BUNDLE_NAME: json.dumps(bundle)})
    with open(os.path.splitext(output_path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    print(f"Exported -> {output_path}")

    reloaded, _ = load_exported(output_path)
    if graph_path:
        stats = GraphFeatureStats.from_dict(bundle["feature_stats"])
        diff = verify(eager, reloaded, graph_path, stats)
        print(f"Max |eager - torchscript| over 100 test graphs: {diff:.2e}")
        if diff > 1e-4:
            raise RuntimeError(f"Exported model deviates from eager model (max diff {diff})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the Phase 2 GNN to TorchScript.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--graph-path", default=None, help="Graph file used to verify the export")
    args, _ = parse_args_with_config(parser, "export")
    run(args.checkpoint_path, args.output_path, args.graph_path, metadata=run_metadata(args))


if __name__ == "__main__":
    main()
