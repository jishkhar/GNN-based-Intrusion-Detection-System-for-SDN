from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool


class GATIntrusionDetector(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int = 0,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.edge_feat_dim = edge_feat_dim
        edge_dim = edge_feat_dim if edge_feat_dim > 0 else None
        self.gat1 = GATConv(node_feat_dim, hidden_dim, heads=heads, dropout=dropout, edge_dim=edge_dim)
        self.gat2 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim,
        )

        graph_dim = hidden_dim * 2
        if edge_feat_dim > 0:
            edge_hidden_dim = hidden_dim
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_feat_dim, edge_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(edge_hidden_dim, edge_hidden_dim),
                nn.ReLU(),
            )
            graph_dim += edge_hidden_dim * 2
        else:
            self.edge_encoder = None

        self.graph_head = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)
        if self.edge_feat_dim <= 0:
            edge_attr = None

        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        graph_parts = [global_mean_pool(x, batch), global_max_pool(x, batch)]

        if self.edge_encoder is not None and edge_attr is not None and edge_attr.numel() > 0:
            edge_x = self.edge_encoder(edge_attr)
            edge_batch = batch[edge_index[0]]
            graph_parts.extend([global_mean_pool(edge_x, edge_batch), global_max_pool(edge_x, edge_batch)])

        g = torch.cat(graph_parts, dim=1)
        logits = self.graph_head(g)
        return logits
