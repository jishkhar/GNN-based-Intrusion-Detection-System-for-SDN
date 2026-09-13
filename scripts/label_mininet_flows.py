#!/usr/bin/env python3
"""Turn recorded Mininet traffic into labelled v2 graphs (TODO M6.3).

Inputs:
- the flow-stats log the IDS writes when ``ids_service.record_flows_path`` is set
  (every controller upload, cumulative counters), and
- the ground-truth files from ``scripts/inject_attack.py``.

The recorded polls are replayed through the same ``FlowCollector`` the live IDS
uses, and each window is cut into the same 200-record chunks the inference
engine uses, so every graph is exactly what the model sees live. Flow
labels come from the ground truth: during an attack, traffic from an attacker
IP (or, for spoofed DDoS, from any IP outside the topology) to the victim is
attack traffic, and the victim's replies are attack traffic marked as reverse
(so the victim is not labelled as an attacker).

Two sources of label noise are handled explicitly:
- the collector is warmed up on the polls before each run, so flow entries left
  in the switches by the previous run are not mistaken for new traffic (the live
  collector has that history too), and
- traffic from attacker hosts or non-topology (spoofed) IPs outside this run's
  attack period is stale traffic from an earlier run (e.g. a controller backlog);
  chunks dominated by it are dropped rather than labelled benign.

Runs are split whole into train / val / test (no run appears in two splits),
and the output uses the v2 format, so it works with ``models.evaluate_v2``
(transfer check) and ``models.train_gnn_v2`` (fine-tuning).

    python scripts/label_mininet_flows.py --flows data/mininet/flows.csv \
        --runs "data/mininet/runs/*.json" --out data/graphs/mininet_v2.pt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller.flow_collector import FlowCollector  # noqa: E402
from preprocessing.graph_builder_v2 import build_graph  # noqa: E402
from preprocessing.labels import ATTACK_CLASSES, CLASS_TO_ID  # noqa: E402
from preprocessing.openflow_features import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES, chunk_records  # noqa: E402

BENIGN_ID = CLASS_TO_ID["Benign"]


def label_records(records: pd.DataFrame, run: dict, t: float, window_seconds: float) -> pd.DataFrame:
    rec = records.copy()
    rec["is_attack"] = False
    rec["is_reverse"] = False
    rec["attack_id"] = BENIGN_ID
    topology_ips = {h["ip"] for h in run["hosts"].values()}
    attacker_hosts = {h["ip"] for h in run["hosts"].values() if h["role"] == "attacker"}
    # Attacker hosts and spoofed addresses send nothing benign in the lab topology.
    rec["stale"] = rec["src_ip"].isin(attacker_hosts) | ~rec["src_ip"].isin(topology_ips)
    attack_type = run.get("attack_type", "Benign")
    if attack_type != "Benign" and run["start"] <= t <= run["end"] + window_seconds:
        attackers = set(run["attacker_ips"])
        victim = run["victim_ip"]
        src_bad = rec["src_ip"].isin(attackers)
        if run.get("spoofed_sources"):
            src_bad |= ~rec["src_ip"].isin(topology_ips)
        forward = src_bad & (rec["dst_ip"] == victim)
        dst_bad = rec["dst_ip"].isin(attackers) | (~rec["dst_ip"].isin(topology_ips) if run.get("spoofed_sources") else False)
        reverse = (rec["src_ip"] == victim) & dst_bad
        rec.loc[forward | reverse, "is_attack"] = True
        rec.loc[reverse, "is_reverse"] = True
        rec.loc[forward | reverse, "attack_id"] = CLASS_TO_ID[attack_type]
    rec.loc[rec["is_attack"], "stale"] = False
    return rec


def chunk_label(rec: pd.DataFrame, min_attack_frac: float) -> int:
    fwd = rec[~rec["is_reverse"]]
    if len(fwd) == 0 or fwd["is_attack"].mean() < min_attack_frac:
        return BENIGN_ID
    return int(Counter(fwd.loc[fwd["is_attack"], "attack_id"]).most_common(1)[0][0])


def run_graphs(flows: pd.DataFrame, run: dict, run_id: int, window_seconds: float, min_attack_frac: float,
               chunk_size: int = 200, poll_stride: int = 2, warmup_seconds: float = 60.0,
               max_stale_frac: float = 0.1) -> tuple[list, int]:
    t0, t1 = run.get("benign_start", run.get("start")), run.get("benign_end", run.get("end"))
    polls = flows[(flows["received_at"] >= t0 - warmup_seconds) & (flows["received_at"] <= t1)]
    collector = FlowCollector(window_seconds=window_seconds)
    graphs, dropped, k = [], 0, 0
    for t, poll in polls.groupby("received_at", sort=True):
        collector.ingest(poll.to_dict("records"), now=t)
        if t < t0:
            continue  # warm-up: learn which flow entries already existed
        k += 1
        if (k - 1) % poll_stride:
            continue  # consecutive windows overlap almost entirely
        records = collector.get_current_window(now=t)
        if len(records) < 2:
            continue
        rec = label_records(records, run, t, window_seconds)
        rec["source_id"] = run_id
        rec["flow_idx"] = rec.groupby(["src_ip", "dst_ip", "ip_proto", "src_port", "dst_port"], sort=False).ngroup()
        for chunk in chunk_records(rec, chunk_size):
            if len(chunk) < 2:
                continue
            if chunk["stale"].mean() > max_stale_frac:
                dropped += 1
                continue
            g = build_graph(chunk, chunk_label(chunk, min_attack_frac))
            g.source_id = torch.tensor([run_id])
            g.window_start = torch.tensor([t - t0], dtype=torch.float64)
            graphs.append(g)
    return graphs, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flows", default="data/mininet/flows.csv")
    parser.add_argument("--runs", default="data/mininet/runs/*.json")
    parser.add_argument("--out", default="data/graphs/mininet_v2.pt")
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--min-attack-frac", type=float, default=0.1)
    parser.add_argument("--chunk-records", type=int, default=200, help="must match the inference engine (2 x window_size)")
    parser.add_argument("--poll-stride", type=int, default=2, help="use every Nth poll's window")
    parser.add_argument("--test-every", type=int, default=4, help="every Nth run of a scenario goes to test")
    parser.add_argument("--val-every", type=int, default=5, help="every Nth run of a scenario goes to val")
    args = parser.parse_args()

    flows = pd.read_csv(args.flows)
    runs = [json.load(open(p)) for p in sorted(glob.glob(args.runs))]
    if not runs:
        raise SystemExit(f"No ground-truth files match {args.runs}")

    out = {"train": [], "val": [], "test": []}
    seen: Counter = Counter()
    for run_id, run in enumerate(runs):
        graphs, dropped = run_graphs(flows, run, run_id, args.window_seconds, args.min_attack_frac,
                                     args.chunk_records, args.poll_stride)
        k = seen[run["scenario"]]
        seen[run["scenario"]] += 1
        split = "test" if k % args.test_every == args.test_every - 1 else "val" if k % args.val_every == args.val_every - 1 else "train"
        out[split].extend(graphs)
        counts = Counter(ATTACK_CLASSES[int(g.y_multi)] for g in graphs)
        print(f"run {run_id} {run['scenario']:<10} -> {split:<5} {len(graphs)} graphs {dict(counts)}"
              f"{f' ({dropped} stale chunks dropped)' if dropped else ''}")

    all_graphs = out["train"] + out["val"] + out["test"]
    out["meta"] = {
        "source": "mininet",
        "classes": ATTACK_CLASSES,
        "node_feature_names": NODE_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "window_mode": "time",
        "window_seconds": args.window_seconds,
        "chunk_records": args.chunk_records,
        "runs": len(runs),
        "stats": {
            "graphs": len(all_graphs),
            "mean_nodes": float(np.mean([g.num_nodes for g in all_graphs])) if all_graphs else 0,
            "mean_edges": float(np.mean([g.edge_index.shape[1] for g in all_graphs])) if all_graphs else 0,
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(out, args.out)
    print({s: len(out[s]) for s in ("train", "val", "test")}, "->", args.out)


if __name__ == "__main__":
    main()
