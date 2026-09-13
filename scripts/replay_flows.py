#!/usr/bin/env python3
"""Replay held-out InSDN flows through the live IDS as OpenFlow flow stats.

Tests the whole live path (collector -> GNN -> classifier -> mitigation, and
optionally the HTTP API) without Mininet. Flows come from the last
``--holdout`` fraction of each capture file (the region used for testing), are
turned into flow-stats entries and sent as polls every ``--interval`` seconds:

    benign  --benign-polls  |  attack + benign  --attack-polls  |  benign  --benign-polls

once per attack type. Reports per attack: detected?, detection delay, predicted
type, attacker-IP precision/recall, rules created; and false alerts in benign
phases (ignoring the first window after an attack, which still contains it).

    python scripts/replay_flows.py                                   # in-process, simulated clock
    python scripts/replay_flows.py --url http://127.0.0.1:3000/api/flows --realtime
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.graph_builder_v2 import load_records  # noqa: E402
from preprocessing.labels import ATTACK_CLASSES  # noqa: E402

DEFAULT_ATTACKS = ["DDoS", "DoS", "Probe", "BruteForce"]


def holdout_records(pattern: str, holdout: float) -> pd.DataFrame:
    import glob

    parts = []
    for source_id, path in enumerate(sorted(glob.glob(pattern))):
        rec = load_records(path)
        # Like the graph split: the latest `holdout` share of each class stream
        # in each file, so every attack type has held-out flows.
        flow_class = rec.drop_duplicates("flow_idx").set_index("flow_idx")["attack_id"]
        keep = []
        for _, idx in flow_class.groupby(flow_class).groups.items():
            idx = sorted(idx)
            keep.extend(idx[int(len(idx) * (1 - holdout)) :])
        rec = rec[rec["flow_idx"].isin(set(keep))].copy()
        rec["flow_key"] = source_id * 10_000_000 + rec["flow_idx"]
        parts.append(rec)
    return pd.concat(parts, ignore_index=True)


def to_entries(flows: pd.DataFrame, dpid: int = 1) -> list[dict]:
    """Flow records -> OpenFlow flow-stats entries (cumulative counters)."""
    return [
        {
            "dpid": dpid,
            "ipv4_src": r.src_ip,
            "ipv4_dst": r.dst_ip,
            "ip_proto": int(r.ip_proto),
            "tp_src": int(r.src_port),
            "tp_dst": int(r.dst_port),
            "packet_count": int(r.packets),
            "byte_count": int(r.bytes),
            "duration_sec": int(r.duration),
            "duration_nsec": int((r.duration % 1) * 1e9),
        }
        for r in flows.itertuples(index=False)
    ]


class FlowSource:
    """Hands out consecutive flows (both directions) of one class stream."""

    def __init__(self, records: pd.DataFrame) -> None:
        self.records = records
        self.keys = records["flow_key"].drop_duplicates().to_numpy()
        self.pos = 0

    def take(self, n: int) -> pd.DataFrame:
        if len(self.keys) == 0:
            return self.records.iloc[:0]
        idx = [(self.pos + i) % len(self.keys) for i in range(n)]
        self.pos = (self.pos + n) % len(self.keys)
        return self.records[self.records["flow_key"].isin(self.keys[idx])]


class Sender:
    def __init__(self, url: str | None, api_key: str | None, config: str) -> None:
        self.url, self.api_key = url, api_key
        if url is None:
            from common.config import load_config
            from inference.service import IDSService, ServiceConfig

            cfg = ServiceConfig.from_dict(load_config(config).get("ids_service"))
            cfg.log_dir = tempfile.mkdtemp(prefix="ids_replay_logs_")
            cfg.record_flows_path = None
            self.service = IDSService(cfg)
            self.window_seconds = cfg.window_seconds

    def send(self, entries: list[dict], now: float) -> dict:
        if self.url is None:
            return self.service.process_flows(entries, now=now)
        import requests

        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        resp = requests.post(self.url, json={"flows": entries, "controller": "replay"}, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()


def run(args) -> dict:
    records = holdout_records(args.input_glob, args.holdout)
    attack_names = {c: ATTACK_CLASSES.index(c) for c in args.attacks}
    benign = FlowSource(records[records["attack_id"] == 0])
    sender = Sender(args.url, args.api_key, args.config)
    window_seconds = getattr(sender, "window_seconds", 10.0)

    now = time.time()
    polls = []

    def poll(flows: pd.DataFrame, phase: str, attack: str | None) -> None:
        nonlocal now
        response = sender.send(to_entries(flows), now)
        truth_attackers = set(flows.loc[flows["is_attack"] & ~flows["is_reverse"], "src_ip"])
        polls.append({"t": now, "phase": phase, "attack": attack, "truth_attackers": truth_attackers, **response})
        if args.realtime:
            time.sleep(args.interval)
        now += args.interval

    for _ in range(args.benign_polls):
        poll(benign.take(args.benign_flows), "benign", None)
    for name, class_id in attack_names.items():
        attack = FlowSource(records[records["attack_id"] == class_id])
        if len(attack.keys) == 0:
            print(f"skip {name}: no held-out flows")
            continue
        for _ in range(args.attack_polls):
            flows = pd.concat([benign.take(args.benign_flows), attack.take(args.attack_flows)])
            poll(flows, "attack", name)
        for _ in range(args.benign_polls):
            poll(benign.take(args.benign_flows), "benign", name)

    # IPs that only ever send benign traffic in the held-out data: blocking one is a false block.
    # Any host seen in benign traffic (either direction) that never initiates an attack flow.
    fwd = records[~records["is_reverse"]]
    ever_attacker = set(fwd.loc[fwd["is_attack"], "src_ip"])
    benign = records[~records["is_attack"]]
    benign_only = (set(benign["src_ip"]) | set(benign["dst_ip"])) - ever_attacker

    report = {"config": vars(args), "attacks": {}, "benign": {}}
    blocked_so_far: set = set()
    for name in attack_names:
        phase = [p for p in polls if p["phase"] == "attack" and p["attack"] == name]
        if not phase:
            continue
        start, end = phase[0]["t"], phase[-1]["t"]
        blocked_so_far |= {
            a["match"]["ipv4_src"]
            for p in polls if p["t"] <= end
            for a in p["actions"] if a["op"] == "add" and "ipv4_src" in a["match"]
        }
        hits = [p for p in phase if p["level"] == "ATTACK"]
        truth = set().union(*(p["truth_attackers"] for p in phase))
        actions = [a for p in phase for a in p["actions"] if a["op"] == "add"]
        new_blocked = {a["match"].get("ipv4_src") for a in actions} - {None}
        report["attacks"][name] = {
            "detected": bool(hits),
            "detection_delay_s": (hits[0]["t"] - start) if hits else None,
            "windows_flagged": f"{len(hits)}/{len(phase)}",
            "predicted_types": dict(pd.Series([p["attack_type"] for p in hits]).value_counts()) if hits else {},
            "true_attacker_ips": len(truth),
            "rules_created": len(actions),
            "rule_actions": sorted({a["action"] for a in actions}),
            "rate_limited_victims": sorted({a["match"]["ipv4_dst"] for a in actions if "ipv4_dst" in a["match"]}),
            # Attackers blocked by the end of the phase (including rules from earlier phases).
            "attacker_block_coverage": (len(truth & blocked_so_far) / len(truth)) if truth else None,
            "new_blocks": sorted(new_blocked),
            "false_blocks": sorted(new_blocked & benign_only),
        }
    all_blocked = {
        a["match"]["ipv4_src"] for p in polls for a in p["actions"] if a["op"] == "add" and "ipv4_src" in a["match"]
    }
    report["blocking"] = {
        "src_ips_blocked": len(all_blocked),
        "false_blocks": sorted(all_blocked & benign_only),
        "benign_only_ips_seen": len(benign_only),
    }
    # Benign windows: skip the ones whose window still contains attack flows.
    attack_end = {}
    for p in polls:
        if p["phase"] == "attack":
            attack_end[p["attack"]] = p["t"]
    clean = [
        p for p in polls
        if p["phase"] == "benign" and (p["attack"] is None or p["t"] - attack_end[p["attack"]] > window_seconds)
    ]
    false_alerts = [p for p in clean if p["level"] == "ATTACK"]
    report["benign"] = {
        "windows": len(clean),
        "false_attack_windows": len(false_alerts),
        "false_positive_rate": len(false_alerts) / max(1, len(clean)),
        "suspicious_windows": sum(p["level"] == "SUSPICIOUS" for p in clean),
    }
    lat = [p["latency_ms"] for p in polls]
    report["latency_ms"] = {
        "p50": float(np.percentile(lat, 50)),
        "p90": float(np.percentile(lat, 90)),
        "p99": float(np.percentile(lat, 99)),
        "max": float(np.max(lat)),
        "mean_window_flows": float(np.mean([p["window_flows"] for p in polls])),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-glob", default="data/insdn/cleaned/*_cleaned.csv")
    parser.add_argument("--config", default="configs/phase2.yaml")
    parser.add_argument("--url", default=None, help="IDS /api/flows URL; default runs the service in-process")
    parser.add_argument("--api-key", default=os.environ.get("IDS_API_KEY"))
    parser.add_argument("--realtime", action="store_true", help="sleep between polls (use with --url)")
    parser.add_argument("--holdout", type=float, default=0.15)
    parser.add_argument("--attacks", nargs="+", default=DEFAULT_ATTACKS)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--benign-polls", type=int, default=15)
    parser.add_argument("--attack-polls", type=int, default=10)
    parser.add_argument("--benign-flows", type=int, default=20, help="benign flows per poll")
    parser.add_argument("--attack-flows", type=int, default=40, help="attack flows per poll")
    parser.add_argument("--out", default="results/phase2/replay_report.json")
    args = parser.parse_args()

    report = run(args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else str(o))
    print(json.dumps({k: report[k] for k in ("attacks", "blocking", "benign", "latency_ms")}, indent=2, default=str))
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
