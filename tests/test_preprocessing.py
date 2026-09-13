import numpy as np
import pandas as pd
import pytest
import torch

from preprocessing.clean_data import clean_dataframe
from preprocessing.graph_builder import parse_timestamps
from preprocessing.graph_dataset import split_graphs, time_split_indices
from preprocessing.labels import binary_label, canonical_class
from preprocessing.openflow_features import (
    EDGE_DIM,
    EDGE_FEATURE_NAMES,
    NODE_DIM,
    NODE_FEATURE_NAMES,
    dataset_to_flow_records,
    edge_features,
    factorize_nodes,
    node_features,
    openflow_stats_to_flow_records,
)


# ----------------------------------------------------------------- labels
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("BENIGN", "Benign"),
        ("Normal", "Benign"),  # InSDN benign label (was mislabelled as attack in Phase 1)
        ("DDoS ", "DDoS"),
        ("DoS Hulk", "DoS"),
        ("PortScan", "Probe"),
        ("Web Attack \x96 Brute Force", "BruteForce"),
        ("Web Attack � XSS", "WebAttack"),
        ("BOTNET", "Botnet"),
        ("something new", "Other"),  # unknown labels are never benign
    ],
)
def test_canonical_class(raw, expected):
    assert canonical_class(raw) == expected


def test_binary_label():
    assert binary_label("Normal") == 0
    assert binary_label("BENIGN") == 0
    assert binary_label("Probe") == 1


def test_clean_dataframe_keeps_attack_type():
    df = pd.DataFrame({" Label": ["Normal", "DDoS", "DDoS"], "x": [1.0, np.inf, np.inf]})
    cleaned, stats = clean_dataframe(df, "Label")
    assert list(cleaned["Label"]) == [0, 1]  # duplicate removed
    assert list(cleaned["Attack"]) == ["Benign", "DDoS"]
    assert np.isfinite(cleaned["x"]).all()
    assert stats.class_counts == {"Benign": 1, "DDoS": 1}


def test_parse_mixed_timestamps():
    ts = parse_timestamps(pd.Series(["12/1/2020 1:14", "25/12/2019 05:20:05 PM"]))
    assert ts[0] == pd.Timestamp("2020-01-12 01:14")
    assert ts[1] == pd.Timestamp("2019-12-25 17:20:05")


# ------------------------------------------------------ openflow features
def insdn_rows():
    return pd.DataFrame(
        {
            "Src IP": ["10.0.0.1", "10.0.0.1", "10.0.0.2"],
            "Dst IP": ["10.0.0.9", "10.0.0.9", "10.0.0.9"],
            "Src Port": [40000, 40001, 50000],
            "Dst Port": [22, 80, 53],
            "Protocol": [6, 6, 17],
            "Timestamp": ["1/1/2020 10:00"] * 3,
            "Flow Duration": [2_000_000, 0, 500_000],
            "Tot Fwd Pkts": [4, 1, 2],
            "Tot Bwd Pkts": [2, 0, 2],
            "TotLen Fwd Pkts": [100, 0, 80],
            "TotLen Bwd Pkts": [50, 0, 160],
            "Label": ["Probe", "Probe", "Normal"],
        }
    )


def test_dataset_to_flow_records_splits_directions():
    rec = dataset_to_flow_records(insdn_rows())
    # 3 forward records + 2 reverse (the row without backward packets has none)
    assert len(rec) == 5
    assert rec["is_reverse"].sum() == 2
    first_rev = rec[(rec.flow_idx == 0) & rec.is_reverse].iloc[0]
    assert (first_rev.src_ip, first_rev.dst_ip, first_rev.src_port, first_rev.dst_port) == ("10.0.0.9", "10.0.0.1", 22, 40000)
    # bytes = payload + per-packet Ethernet/IP/TCP headers (OpenFlow counts whole frames)
    fwd0 = rec[(rec.flow_idx == 0) & ~rec.is_reverse].iloc[0]
    assert fwd0.bytes == 100 + 4 * 54
    assert fwd0.duration == pytest.approx(2.0)
    assert set(rec["Label"]) == {"Probe", "Normal"}


def test_server_first_rows_are_direction_normalised():
    # CICFlowMeter saw the victim's reply first: src is the service port (80).
    row = insdn_rows().iloc[[0]].assign(**{"Src IP": "10.0.0.9", "Dst IP": "10.0.0.1", "Src Port": 80, "Dst Port": 51000})
    rec = dataset_to_flow_records(row)
    client = rec[~rec.is_reverse].iloc[0]
    assert (client.src_ip, client.dst_port) == ("10.0.0.1", 80)  # the initiator is the client side
    raw = dataset_to_flow_records(row, normalize_direction=False)
    assert raw[~raw.is_reverse].iloc[0].src_ip == "10.0.0.9"
    # the records are identical, only the direction flag differs
    assert sorted(rec.src_ip) == sorted(raw.src_ip) and rec.bytes.sum() == raw.bytes.sum()


def test_feature_shapes_and_values():
    rec = dataset_to_flow_records(insdn_rows())
    src, dst, ips = factorize_nodes(rec)
    x = node_features(rec, src, dst, len(ips))
    e = edge_features(rec)
    assert x.shape == (3, NODE_DIM) and e.shape == (5, EDGE_DIM)
    assert np.isfinite(x).all() and np.isfinite(e).all()
    xf = pd.DataFrame(x, index=ips, columns=NODE_FEATURE_NAMES)
    assert xf.loc["10.0.0.1", "out_flows"] == 2
    assert xf.loc["10.0.0.1", "unique_dst_ports"] == 2
    assert xf.loc["10.0.0.1", "dst_port_entropy"] == pytest.approx(1.0)  # two ports, equally often
    assert xf.loc["10.0.0.9", "in_degree"] == 2
    ef = pd.DataFrame(e, columns=EDGE_FEATURE_NAMES)
    assert ef["proto_tcp"].iloc[0] == 1 and ef["dport_well_known"].iloc[0] == 1


def test_icmp_has_no_port_buckets():
    rec = pd.DataFrame({"src_ip": ["a"], "dst_ip": ["b"], "src_port": [0], "dst_port": [0], "ip_proto": [1],
                        "packets": [1.0], "bytes": [98.0], "duration": [0.0]})
    ef = pd.DataFrame(edge_features(rec), columns=EDGE_FEATURE_NAMES)
    assert ef.filter(like="port").sum(axis=1).iloc[0] == 0
    assert ef["proto_icmp"].iloc[0] == 1


def test_openflow_stats_conversion():
    rec = openflow_stats_to_flow_records([{"ipv4_src": "1.1.1.1", "ipv4_dst": "2.2.2.2", "ip_proto": 17, "udp_src": 5353,
                                           "udp_dst": 53, "packet_count": 3, "byte_count": 300, "duration_sec": 1,
                                           "duration_nsec": 500_000_000}])
    row = rec.iloc[0]
    assert (row.src_port, row.dst_port, row.duration) == (5353, 53, 1.5)


# ---------------------------------------------------------------- splits
def test_time_split_is_chronological_with_gap():
    groups = ["a"] * 20 + ["b"] * 10
    times = list(range(20)) + list(range(10))
    tr, va, te = time_split_indices(groups, times, test_size=0.2, val_size=0.2, gap=1)
    assert not set(tr) & set(va) and not set(va) & set(te)
    a_train, a_val, a_test = ([i for i in s if i < 20] for s in (tr, va, te))
    assert max(a_train) < min(a_val) and max(a_val) < min(a_test)
    assert min(a_val) - max(a_train) == 2  # one window dropped as gap
    assert any(i >= 20 for i in te)  # every group reaches the test split


def test_split_graphs_time_mode_uses_graph_metadata():
    graphs = []
    for i in range(30):
        data = type("G", (), {})()
        data.y = torch.tensor([i % 2])
        data.source_id = torch.tensor([0])
        data.window_start = torch.tensor([float(i)])
        graphs.append(data)
    tr, va, te = split_graphs(graphs, 0.2, 0.2, mode="time")
    assert len(tr) + len(va) + len(te) == 30
    assert max(float(g.window_start) for g in tr if int(g.y) == 0) < min(float(g.window_start) for g in te if int(g.y) == 0)
