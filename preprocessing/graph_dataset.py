from __future__ import annotations

from dataclasses import dataclass

import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader


@dataclass
class GraphFeatureStats:
    node_mean: torch.Tensor
    node_std: torch.Tensor
    edge_mean: torch.Tensor | None = None
    edge_std: torch.Tensor | None = None

    def to_dict(self) -> dict:
        return {
            "node_mean": self.node_mean.cpu().tolist(),
            "node_std": self.node_std.cpu().tolist(),
            "edge_mean": self.edge_mean.cpu().tolist() if self.edge_mean is not None else None,
            "edge_std": self.edge_std.cpu().tolist() if self.edge_std is not None else None,
        }

    @classmethod
    def from_dict(cls, values: dict | None) -> "GraphFeatureStats | None":
        if not values:
            return None
        edge_mean = values.get("edge_mean")
        edge_std = values.get("edge_std")
        return cls(
            node_mean=torch.tensor(values["node_mean"], dtype=torch.float32),
            node_std=torch.tensor(values["node_std"], dtype=torch.float32),
            edge_mean=torch.tensor(edge_mean, dtype=torch.float32) if edge_mean is not None else None,
            edge_std=torch.tensor(edge_std, dtype=torch.float32) if edge_std is not None else None,
        )


@dataclass
class GraphSplits:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    feature_stats: GraphFeatureStats | None = None


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


def _signed_log1p(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sign(tensor) * torch.log1p(torch.abs(tensor))


def fit_feature_stats(graphs) -> GraphFeatureStats:
    x = torch.cat([_signed_log1p(g.x.float()) for g in graphs], dim=0)
    node_mean = x.mean(dim=0)
    node_std = x.std(dim=0).clamp_min(1e-6)

    edge_tensors = [
        _signed_log1p(g.edge_attr.float())
        for g in graphs
        if getattr(g, "edge_attr", None) is not None and g.edge_attr.numel() > 0
    ]
    if edge_tensors:
        edge_attr = torch.cat(edge_tensors, dim=0)
        edge_mean = edge_attr.mean(dim=0)
        edge_std = edge_attr.std(dim=0).clamp_min(1e-6)
    else:
        edge_mean = None
        edge_std = None

    return GraphFeatureStats(
        node_mean=node_mean,
        node_std=node_std,
        edge_mean=edge_mean,
        edge_std=edge_std,
    )


def apply_feature_stats(graph, stats: GraphFeatureStats):
    graph = graph.clone()
    graph.x = (_signed_log1p(graph.x.float()) - stats.node_mean) / stats.node_std
    if (
        stats.edge_mean is not None
        and stats.edge_std is not None
        and getattr(graph, "edge_attr", None) is not None
        and graph.edge_attr.numel() > 0
    ):
        graph.edge_attr = (_signed_log1p(graph.edge_attr.float()) - stats.edge_mean) / stats.edge_std
    return graph


def apply_stats_to_graphs(graphs, stats: GraphFeatureStats):
    return [apply_feature_stats(g, stats) for g in graphs]


def create_loaders(
    graph_path: str,
    batch_size: int = 16,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
    normalize: bool = True,
    feature_stats: GraphFeatureStats | None = None,
) -> GraphSplits:
    graphs = load_graphs(graph_path)
    train_graphs, val_graphs, test_graphs = split_graphs(
        graphs,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
    )
    if normalize:
        feature_stats = feature_stats or fit_feature_stats(train_graphs)
        train_graphs = apply_stats_to_graphs(train_graphs, feature_stats)
        val_graphs = apply_stats_to_graphs(val_graphs, feature_stats)
        test_graphs = apply_stats_to_graphs(test_graphs, feature_stats)

    return GraphSplits(
        train_loader=DataLoader(train_graphs, batch_size=batch_size, shuffle=True),
        val_loader=DataLoader(val_graphs, batch_size=batch_size, shuffle=False),
        test_loader=DataLoader(test_graphs, batch_size=batch_size, shuffle=False),
        feature_stats=feature_stats if normalize else None,
    )
