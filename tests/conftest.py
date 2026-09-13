import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from preprocessing.labels import ATTACK_CLASSES  # noqa: E402
from preprocessing.openflow_features import EDGE_DIM, EDGE_FEATURE_NAMES, NODE_DIM, NODE_FEATURE_NAMES  # noqa: E402


@pytest.fixture(scope="session")
def tiny_model_path(tmp_path_factory):
    """An untrained, exported MultiTaskGNN with identity normalisation."""
    from models.export import run as export_run
    from models.gnn_v2 import MultiTaskGNN

    torch.manual_seed(0)
    config = {"node_feat_dim": NODE_DIM, "edge_feat_dim": EDGE_DIM, "num_classes": len(ATTACK_CLASSES),
              "hidden_dim": 16, "heads": 2, "num_layers": 2, "dropout": 0.0, "conv_type": "gat",
              "use_edge_features": True}
    model = MultiTaskGNN(**config)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_config": config,
        "feature_stats": {"node_mean": [0.0] * NODE_DIM, "node_std": [1.0] * NODE_DIM,
                          "edge_mean": [0.0] * EDGE_DIM, "edge_std": [1.0] * EDGE_DIM},
        "thresholds": {"window": 0.5, "node": 0.5},
        "classes": ATTACK_CLASSES,
        "node_feature_names": NODE_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "graph_meta": {"window_size": 50},
    }
    d = tmp_path_factory.mktemp("model")
    ckpt_path, out_path = str(d / "ckpt.pt"), str(d / "tiny_ids.pt")
    torch.save(ckpt, ckpt_path)
    export_run(ckpt_path, out_path)
    return out_path


def flow_entry(src, dst, packets=10, nbytes=1000, dpid=1, sport=40000, dport=80, proto=6, duration=1.0):
    return {"dpid": dpid, "ipv4_src": src, "ipv4_dst": dst, "ip_proto": proto, "tp_src": sport, "tp_dst": dport,
            "packet_count": packets, "byte_count": nbytes, "duration_sec": int(duration),
            "duration_nsec": int((duration % 1) * 1e9)}
