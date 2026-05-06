from __future__ import annotations

from dataclasses import dataclass

import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader


@dataclass
class GraphSplits:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def load_graphs(path: str):
    # These are local, project-generated PyG `Data` objects, so we intentionally
    # opt out of PyTorch 2.6's `weights_only=True` default for this trusted file.
    graphs = torch.load(path, weights_only=False)
    if not graphs:
        raise ValueError(f"No graphs loaded from {path}")
    return graphs


def split_graphs(graphs, test_size: float = 0.15, val_size: float = 0.15, seed: int = 42):
    labels = [int(g.y.item()) for g in graphs]
    train_graphs, test_graphs = train_test_split(
        graphs, test_size=test_size, random_state=seed, stratify=labels
    )

    train_labels = [int(g.y.item()) for g in train_graphs]
    val_ratio = val_size / (1.0 - test_size)
    train_graphs, val_graphs = train_test_split(
        train_graphs, test_size=val_ratio, random_state=seed, stratify=train_labels
    )
    return train_graphs, val_graphs, test_graphs


def create_loaders(
    graph_path: str,
    batch_size: int = 16,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> GraphSplits:
    graphs = load_graphs(graph_path)
    train_graphs, val_graphs, test_graphs = split_graphs(
        graphs,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
    )

    return GraphSplits(
        train_loader=DataLoader(train_graphs, batch_size=batch_size, shuffle=True),
        val_loader=DataLoader(val_graphs, batch_size=batch_size, shuffle=False),
        test_loader=DataLoader(test_graphs, batch_size=batch_size, shuffle=False),
    )
