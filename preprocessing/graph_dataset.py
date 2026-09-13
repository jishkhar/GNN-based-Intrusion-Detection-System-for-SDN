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


def _graph_group(graph) -> tuple:
    source = getattr(graph, "source_id", None)
    source = int(source.item()) if torch.is_tensor(source) else (source if source is not None else 0)
    return source, int(graph.y.view(-1)[0].item())


def _graph_time(graph, fallback: int) -> float:
    start = getattr(graph, "window_start", None)
    if start is None:
        return float(fallback)
    return float(start.item()) if torch.is_tensor(start) else float(start)


def time_split_indices(
    groups: list,
    times: list[float],
    test_size: float = 0.15,
    val_size: float = 0.15,
    gap: int = 0,
) -> tuple[list[int], list[int], list[int]]:
    """Chronological split within each group.

    Each group (e.g. one capture file and class) is sorted by time; its earliest
    windows go to train, then val, then test. ``gap`` windows are dropped at each
    boundary so overlapping sliding windows never appear on both sides.
    """
    by_group: dict = {}
    for idx, (group, t) in enumerate(zip(groups, times)):
        by_group.setdefault(group, []).append((t, idx))

    train, val, test = [], [], []
    for members in by_group.values():
        order = [idx for _, idx in sorted(members)]
        n = len(order)
        if n < 3:
            # Too small to split in time; keep it in training rather than dropping it.
            train.extend(order)
            continue
        n_test = max(1, int(round(n * test_size)))
        n_val = max(1, int(round(n * val_size)))
        n_train = n - n_val - n_test
        g = gap if n_train - 2 * gap >= 1 else 0
        train.extend(order[: n_train - g])
        val.extend(order[n_train : n_train + n_val - g])
        test.extend(order[n_train + n_val :])
    return sorted(train), sorted(val), sorted(test)


def split_graphs(
    graphs,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
    mode: str = "time",
    gap: int = 0,
):
    """Split graphs into train/val/test.

    ``mode="time"`` (default) is leakage-safe for sliding windows;
    ``mode="random"`` reproduces the Phase 1 stratified random split.
    """
    if mode == "time":
        groups = [_graph_group(g) for g in graphs]
        times = [_graph_time(g, i) for i, g in enumerate(graphs)]
        tr, va, te = time_split_indices(groups, times, test_size, val_size, gap)
        return [graphs[i] for i in tr], [graphs[i] for i in va], [graphs[i] for i in te]
    if mode != "random":
        raise ValueError(f"Unknown split mode: {mode}")

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
    split_mode: str = "time",
    split_gap: int = 0,
) -> GraphSplits:
    graphs = load_graphs(graph_path)
    train_graphs, val_graphs, test_graphs = split_graphs(
        graphs,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
        mode=split_mode,
        gap=split_gap,
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
