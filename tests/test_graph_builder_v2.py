import numpy as np
import pandas as pd
import torch

from preprocessing import graph_builder_v2 as gb
from preprocessing.labels import CLASS_TO_ID


def write_capture(path, n, label, src_fn, dst="10.0.0.100", start_minute=0):
    rows = []
    for i in range(n):
        rows.append(
            {
                "Src IP": src_fn(i), "Dst IP": dst, "Src Port": 40000 + i, "Dst Port": 80 if label == "Normal" else i % 1000,
                "Protocol": 6, "Timestamp": f"1/1/2020 10:{start_minute + i // 60:02d}", "Flow Duration": 1000,
                "Tot Fwd Pkts": 2, "Tot Bwd Pkts": 1, "TotLen Fwd Pkts": 0, "TotLen Bwd Pkts": 0,
                "Label": label, "Attack": "Benign" if label == "Normal" else "Probe",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def build(tmp_path, overlay_prob=0.0):
    write_capture(tmp_path / "a_benign.csv", 400, "Normal", lambda i: f"192.168.0.{i % 20}")
    write_capture(tmp_path / "b_probe.csv", 400, "Probe", lambda i: "10.66.66.66")
    return gb.run(
        input_glob=str(tmp_path / "*.csv"),
        output_path=str(tmp_path / "graphs.pt"),
        window_mode="count",
        window_size=40,
        window_stride=20,
        overlay_prob=overlay_prob,
    )


def test_splits_labels_and_node_labels(tmp_path):
    out = build(tmp_path)
    assert set(out) == {"train", "val", "test", "meta"}
    for split in ("train", "val", "test"):
        classes = {int(g.y_multi) for g in out[split]}
        assert classes == {CLASS_TO_ID["Benign"], CLASS_TO_ID["Probe"]}  # both streams in every split
    probe = next(g for g in out["test"] if int(g.y) == 1)
    ips = probe.node_ips
    attacker = ips.index("10.66.66.66")
    victim = ips.index("10.0.0.100")
    assert probe.node_y[attacker] == 1
    assert probe.node_y[victim] == 0  # victim replies (reverse records) don't make it an attacker
    assert probe.edge_reverse.any() and (~probe.edge_reverse).any()
    assert probe.x.shape[1] == len(out["meta"]["node_feature_names"])
    assert torch.load(tmp_path / "graphs.pt", weights_only=False)["meta"]["stats"]["graphs"] > 0


def test_no_flow_shared_between_splits(tmp_path):
    out = build(tmp_path)
    # Count-mode windows record their first flow index in window_start.
    size = out["meta"]["window_size"]
    for sid in (0, 1):
        spans = {
            s: [(float(g.window_start), float(g.window_start) + size) for g in out[s] if int(g.source_id) == sid]
            for s in ("train", "val", "test")
        }
        assert max(e for _, e in spans["train"]) <= min(s for s, _ in spans["val"])
        assert max(e for _, e in spans["val"]) <= min(s for s, _ in spans["test"])


def test_overlay_mixes_benign_hosts_into_attack_windows(tmp_path):
    out = build(tmp_path, overlay_prob=1.0)
    attack = [g for g in out["train"] if int(g.y) == 1]
    assert attack and all((g.node_y == 0).any() and (g.node_y == 1).any() for g in attack)
    first = np.concatenate([g.edge_first.numpy() for g in out["train"]])
    assert (~first).any()  # overlaid benign records are marked as already seen
