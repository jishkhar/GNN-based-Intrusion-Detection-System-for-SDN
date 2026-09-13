"""Phase 2 multi-task GNN: per-host attacker scores + window attack type.

Traffic graphs are directed, but a pure attacker (e.g. a spoofed DDoS source)
has no incoming edges and would never receive messages. Message passing
therefore runs on both edge directions, with a direction flag appended to the
edge features.

Edge features reach every conv type: GAT also uses them inside attention, and
for all types each node is initialised with the mean embedding of its incoming
and outgoing flows, so GAT / GCN / GraphSAGE ablations differ only in the conv.

``forward`` takes plain tensors (not a ``Data`` object) so it can be exported
with TorchScript for the inference engine.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_max_pool, global_mean_pool
from torch_geometric.utils import scatter

CONV_TYPES = ("gat", "gcn", "sage")


def _mlp(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(out_dim, out_dim), nn.ReLU())


class MultiTaskGNN(nn.Module):
    # Constants to TorchScript, so untaken branches (e.g. passing edge_attr to
    # GCN/SAGE) are never compiled.
    __constants__ = ["use_edge_features", "edge_in_conv"]

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        conv_type: str = "gat",
        use_edge_features: bool = True,
    ) -> None:
        super().__init__()
        if conv_type not in CONV_TYPES:
            raise ValueError(f"conv_type must be one of {CONV_TYPES}")
        self.conv_type = conv_type
        self.use_edge_features = use_edge_features and edge_feat_dim > 0
        self.edge_in_conv = conv_type == "gat" and self.use_edge_features
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        self.node_encoder = _mlp(node_feat_dim, hidden_dim, dropout)
        if self.use_edge_features:
            # +1 input for the direction flag (0 = original, 1 = reversed copy).
            self.edge_encoder = _mlp(edge_feat_dim + 1, hidden_dim, dropout)
            self.node_input = nn.Linear(hidden_dim * 3, hidden_dim)
        else:
            self.edge_encoder = nn.Identity()
            self.node_input = nn.Linear(hidden_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        edge_dim = hidden_dim if self.use_edge_features else None
        for _ in range(num_layers):
            if conv_type == "gat":
                conv = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, edge_dim=edge_dim)
            elif conv_type == "gcn":
                conv = GCNConv(hidden_dim, hidden_dim)
            else:
                conv = SAGEConv(hidden_dim, hidden_dim)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Node head sees its input embedding too (skip connection).
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2)
        )
        graph_dim = hidden_dim * 2 + (hidden_dim * 2 if self.use_edge_features else 0)
        self.graph_head = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns ``(graph_logits [B, C], node_logits [N, 2])``."""
        num_nodes = x.size(0)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        src, dst = edge_index[0], edge_index[1]
        mp_index = torch.cat([edge_index, torch.stack([dst, src])], dim=1)

        h = self.node_encoder(x)
        if self.use_edge_features:
            num_edges = edge_attr.size(0)
            flag = torch.cat(
                [torch.zeros(num_edges, 1, device=x.device), torch.ones(num_edges, 1, device=x.device)]
            )
            edge_emb = self.edge_encoder(torch.cat([torch.cat([edge_attr, edge_attr], dim=0), flag], dim=1))
            fwd_emb = edge_emb[:num_edges]
            out_mean = scatter(fwd_emb, src, dim=0, dim_size=num_nodes, reduce="mean")
            in_mean = scatter(fwd_emb, dst, dim=0, dim_size=num_nodes, reduce="mean")
            h = self.node_input(torch.cat([h, out_mean, in_mean], dim=1))
        else:
            edge_emb = torch.zeros(0, self.hidden_dim, device=x.device)
            fwd_emb = edge_emb
            h = self.node_input(h)
        h0 = F.relu(h)

        h = h0
        for conv, norm in zip(self.convs, self.norms):
            if self.edge_in_conv:
                out = conv(h, mp_index, edge_attr=edge_emb)
            else:
                out = conv(h, mp_index)
            h = norm(h + F.elu(out))
            h = F.dropout(h, p=self.dropout, training=self.training)

        node_logits = self.node_head(torch.cat([h, h0], dim=1))

        parts = [global_mean_pool(h, batch, num_graphs), global_max_pool(h, batch, num_graphs)]
        if self.use_edge_features:
            edge_batch = batch[src]
            parts.append(global_mean_pool(fwd_emb, edge_batch, num_graphs))
            parts.append(global_max_pool(fwd_emb, edge_batch, num_graphs))
        graph_logits = self.graph_head(torch.cat(parts, dim=1))
        return graph_logits, node_logits


def attack_probability(graph_logits: Tensor, benign_index: int = 0) -> Tensor:
    """Binary attack score for a window: 1 - P(Benign)."""
    return 1.0 - torch.softmax(graph_logits, dim=1)[:, benign_index]
