"""Mininet data tools, tested on synthetic recordings (no Mininet needed)."""
import json
import subprocess
import sys

import pandas as pd
import torch

from conftest import ROOT
from preprocessing.labels import CLASS_TO_ID
from scripts.analyze_mitigation import analyze_run, read_packets, summarize

HOSTS = {
    "h1": {"ip": "10.0.0.1", "switch": "s1", "role": "attacker"},
    "h2": {"ip": "10.0.0.2", "switch": "s1", "role": "client"},
    "h4": {"ip": "10.0.0.4", "switch": "s2", "role": "server"},
}
RUN = {"scenario": "dos", "attack_type": "DoS", "attacker_ips": ["10.0.0.1"], "spoofed_sources": False,
       "victim_ip": "10.0.0.4", "start": 1030.0, "end": 1060.0, "benign_start": 1000.0, "benign_end": 1080.0,
       "hosts": HOSTS}


def recorded_flows() -> pd.DataFrame:
    rows = []
    for t in range(1000, 1081, 2):
        for k in range(3):  # benign client, a few short HTTP flows per poll
            rows.append({"received_at": t, "dpid": 1, "ipv4_src": "10.0.0.2", "ipv4_dst": "10.0.0.4", "ip_proto": 6,
                         "tp_src": 30000 + t * 3 + k, "tp_dst": 80, "packet_count": 6, "byte_count": 900,
                         "duration_sec": 0, "duration_nsec": 200_000_000})
        if 1030 <= t <= 1060:  # SYN flood: a new source port per packet
            for k in range(40):
                rows.append({"received_at": t, "dpid": 1, "ipv4_src": "10.0.0.1", "ipv4_dst": "10.0.0.4",
                             "ip_proto": 6, "tp_src": 1000 + t * 40 + k, "tp_dst": 80, "packet_count": 1,
                             "byte_count": 54, "duration_sec": 0, "duration_nsec": 0})
    return pd.DataFrame(rows)


def test_label_mininet_flows_builds_labelled_graphs(tmp_path):
    flows = tmp_path / "flows.csv"
    recorded_flows().to_csv(flows, index=False)
    runs = tmp_path / "runs"
    runs.mkdir()
    for i in range(4):  # four runs of the same scenario -> the 4th goes to test
        run = {**RUN, "start": RUN["start"], "end": RUN["end"]}
        (runs / f"run{i}_dos.json").write_text(json.dumps(run))
    out = tmp_path / "mininet_v2.pt"
    subprocess.run([sys.executable, "scripts/label_mininet_flows.py", "--flows", str(flows), "--runs",
                    str(runs / "*.json"), "--out", str(out)], cwd=ROOT, check=True, capture_output=True)
    data = torch.load(out, weights_only=False)
    assert data["test"] and data["train"]
    graphs = data["test"]
    attack = [g for g in graphs if int(g.y_multi) == CLASS_TO_ID["DoS"]]
    benign = [g for g in graphs if int(g.y) == 0]
    assert attack and benign
    g = attack[len(attack) // 2]
    ips = g.node_ips
    assert g.node_y[ips.index("10.0.0.1")] == 1
    assert g.node_y[ips.index("10.0.0.4")] == 0 and g.node_y[ips.index("10.0.0.2")] == 0


def test_analyze_mitigation_drop_rate():
    lines = []
    for i in range(int((1060 - 1000) * 100)):
        t = 1000 + i / 100
        if 1030 <= t < 1034:  # 100 pps attack until the rule lands at t=1034
            lines.append(f"{t:.6f} IP 10.0.0.1.4{i % 1000:03d} > 10.0.0.4.80: Flags [S], length 0")
        if i % 20 == 0:  # 5 pps benign
            lines.append(f"{t:.6f} IP 10.0.0.2.35000 > 10.0.0.4.80: Flags [P.], length 100")
    packets = read_packets(text="\n".join(lines))
    log = [{"event": "rule_added", "ts": 1034.0, "rule_id": "r-1", "action": "drop", "match": {"ipv4_src": "10.0.0.1"},
            "attack_type": "DoS", "confidence": 0.99}]
    result = analyze_run(RUN, packets, log, poll_interval=2.0)
    assert result["mitigated"] and result["time_to_mitigation_s"] == 4.0
    assert result["time_to_mitigation_polls"] == 2.0
    assert result["drop_rate"] == 1.0
    assert result["attack_pps_before"] > 90 and result["benign_pps_after"] > 4
    assert result["false_blocks"] == []
    summary = summarize([result])
    assert summary["drop_rate"]["runs_meeting_70pct"] == 1


def test_analyze_mitigation_flags_false_blocks():
    log = [{"event": "rule_added", "ts": 1040.0, "rule_id": "r-9", "action": "drop", "match": {"ipv4_src": "10.0.0.2"},
            "attack_type": "DoS", "confidence": 0.9}]
    result = analyze_run(RUN, [], log, poll_interval=2.0)
    assert result["false_blocks"] == ["10.0.0.2"] and result["mitigated"] is False


def test_analyze_benign_run_counts_any_rule_as_false():
    run = {"scenario": "benign", "attack_type": "Benign", "attacker_ips": [], "hosts": HOSTS,
           "benign_start": 1000.0, "benign_end": 1080.0}
    log = [{"event": "rule_added", "ts": 1010.0, "match": {"ipv4_src": "10.0.0.2"}}]
    result = analyze_run(run, [], log, poll_interval=2.0)
    assert result["false_rules_in_benign_run"] == 1 and result["false_blocks"] == ["10.0.0.2"]
    assert summarize([result])["benign_runs_with_rules"] == 1
