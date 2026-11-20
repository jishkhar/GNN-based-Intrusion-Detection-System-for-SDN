import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from inference.inference_engine import InferenceEngine
from preprocessing.graph_dataset import select_graph_features


def test_select_graph_features_subsets_columns_without_touching_original():
    g = Data(x=torch.arange(12.0).view(3, 4), edge_attr=torch.arange(10.0).view(2, 5),
             edge_index=torch.tensor([[0, 1], [1, 2]]))
    out = select_graph_features([g], {"node": [1, 3], "edge": [0, 4]})[0]
    assert out.x.tolist() == [[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]]
    assert out.edge_attr.tolist() == [[0.0, 4.0], [5.0, 9.0]]
    assert g.x.shape == (3, 4) and g.edge_attr.shape == (2, 5)
    assert select_graph_features([g], None)[0] is g


def test_engine_serves_model_trained_on_selected_columns(tmp_path):
    from models.export import run as export_run
    from models.gnn_v2 import MultiTaskGNN
    from preprocessing.labels import ATTACK_CLASSES

    node_cols, edge_cols = [0, 4, 9], [1, 3]
    cfg = {"node_feat_dim": 3, "edge_feat_dim": 2, "num_classes": len(ATTACK_CLASSES), "hidden_dim": 16,
           "heads": 2, "num_layers": 2, "dropout": 0.0, "conv_type": "gat", "use_edge_features": True}
    ckpt = {
        "model_state_dict": MultiTaskGNN(**cfg).state_dict(), "model_config": cfg,
        "feature_stats": {"node_mean": [0.0] * 3, "node_std": [1.0] * 3, "edge_mean": [0.0] * 2, "edge_std": [1.0] * 2},
        "thresholds": {"window": 0.5, "node": 0.5}, "classes": ATTACK_CLASSES,
        "node_feature_names": [], "edge_feature_names": [], "graph_meta": {"window_size": 50},
        "feature_selection": {"node": node_cols, "edge": edge_cols},
    }
    torch.save(ckpt, tmp_path / "c.pt")
    export_run(str(tmp_path / "c.pt"), str(tmp_path / "m.pt"))
    engine = InferenceEngine(str(tmp_path / "m.pt"), device="cpu")
    records = pd.DataFrame({
        "src_ip": ["10.0.0.1", "10.0.0.2"], "dst_ip": ["10.0.0.3", "10.0.0.3"], "src_port": [40000, 40001],
        "dst_port": [80, 80], "ip_proto": [6, 6], "packets": [5.0, 7.0], "bytes": [500.0, 700.0],
        "duration": [1.0, 2.0],
    })
    pred = engine.predict(records)
    assert pred.num_flows == 2 and set(pred.node_scores) == {"10.0.0.1", "10.0.0.2", "10.0.0.3"}
