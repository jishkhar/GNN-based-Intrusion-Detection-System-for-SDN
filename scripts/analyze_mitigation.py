#!/usr/bin/env python3
"""Measure mitigation effectiveness of Mininet attack runs (TODO M7.2 / M10.3).

For each ground-truth file from ``scripts/inject_attack.py`` this reads the
victim-side packet capture and the IDS audit log and reports:

- time to mitigation: attack start -> first IDS rule covering the attack
  (an attacker source or the victim), also in controller polling intervals,
- attack packet rate reaching the victim before vs. after that rule and the
  resulting drop rate (target: >= 70 %),
- the effect on benign traffic to the victim (collateral damage),
- rules that blocked a benign host of the topology (false blocks).

    python scripts/analyze_mitigation.py --runs "data/mininet/runs/*.json" \
        --log logs/mitigation_log.jsonl --out results/phase2/mitigation_report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess

import numpy as np

PACKET_RE = re.compile(r"^(\d+\.\d+) IP (\d+\.\d+\.\d+\.\d+)(?:\.\d+)? > (\d+\.\d+\.\d+\.\d+)(?:\.\d+)?:")


def read_packets(pcap: str | None = None, text: str | None = None) -> list[tuple[float, str, str]]:
    """(timestamp, src, dst) of IPv4 packets, from a pcap (via tcpdump) or tcpdump -tt -nn text."""
    if text is None:
        text = subprocess.run(["tcpdump", "-nn", "-tt", "-r", pcap], capture_output=True, text=True, check=False).stdout
    out = []
    for line in text.splitlines():
        m = PACKET_RE.match(line)
        if m:
            out.append((float(m.group(1)), m.group(2), m.group(3)))
    return out


def read_log(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def analyze_run(run: dict, packets: list, log: list, poll_interval: float) -> dict:
    topology_ips = {h["ip"] for h in run["hosts"].values()}
    benign_hosts = {h["ip"] for h in run["hosts"].values() if h["role"] in ("client", "server")}
    result = {"scenario": run["scenario"], "attack_type": run.get("attack_type")}
    if run.get("attack_type") in (None, "Benign"):
        # Benign-only run: any rule at all is a false positive.
        t0, t1 = run.get("benign_start", 0), run.get("benign_end", float("inf"))
        rules = [e for e in log if e.get("event") == "rule_added" and t0 <= e["ts"] <= t1]
        result["false_rules_in_benign_run"] = len(rules)
        result["false_blocks"] = sorted({r["match"].get("ipv4_src") or r["match"].get("ipv4_dst") for r in rules})
        return result
    victim = run["victim_ip"]
    attackers = set(run.get("attacker_ips", []))
    spoofed = run.get("spoofed_sources", False)
    start, end = run["start"], run["end"]

    def is_attack_src(ip: str) -> bool:
        return ip in attackers or (spoofed and ip not in topology_ips)

    rules = [e for e in log if e.get("event") == "rule_added" and start <= e["ts"] <= end + 30]
    covering = [r for r in rules if r["match"].get("ipv4_dst") == victim or is_attack_src(r["match"].get("ipv4_src", ""))]
    false_blocks = sorted({r["match"]["ipv4_src"] for r in rules if r["match"].get("ipv4_src") in benign_hosts})
    result["rules"] = [{k: r.get(k) for k in ("rule_id", "action", "match", "attack_type", "confidence")} for r in rules]
    result["false_blocks"] = false_blocks

    to_victim = [(t, s) for t, s, d in packets if d == victim and start - 30 <= t <= end]
    attack_ts = np.array([t for t, s in to_victim if is_attack_src(s)])
    benign_ts = np.array([t for t, s in to_victim if s in topology_ips and not is_attack_src(s)])

    def rate(ts: np.ndarray, a: float, b: float) -> float | None:
        return float(((ts >= a) & (ts < b)).sum() / (b - a)) if b > a else None

    if not covering:
        result.update({"mitigated": False, "attack_pps": rate(attack_ts, start, end)})
        return result
    t_rule = min(r["ts"] for r in covering)
    settle = 1.0  # the rule reaches the switch with the controller's next poll response
    # Measure "before" from the first attack packet: tools like nmap start sending seconds late.
    first = attack_ts[(attack_ts >= start) & (attack_ts < t_rule)].min() if ((attack_ts >= start) & (attack_ts < t_rule)).any() else start
    before = rate(attack_ts, first, t_rule)
    after = rate(attack_ts, t_rule + settle, end)
    result.update(
        {
            "mitigated": True,
            "time_to_mitigation_s": t_rule - start,
            "time_to_mitigation_polls": (t_rule - start) / poll_interval,
            "attack_pps_before": before,
            "attack_pps_after": after,
            "drop_rate": (1 - after / before) if before and after is not None else None,
            "benign_pps_before": rate(benign_ts, start - 30, start),
            "benign_pps_after": rate(benign_ts, t_rule + settle, end),
        }
    )
    return result


def summarize(results: list[dict]) -> dict:
    attacks = [r for r in results if r.get("attack_type") not in (None, "Benign")]
    mitigated = [r for r in attacks if r.get("mitigated")]
    ttm = [r["time_to_mitigation_s"] for r in mitigated]
    drops = [r["drop_rate"] for r in mitigated if r.get("drop_rate") is not None]
    summary = {
        "attack_runs": len(attacks),
        "mitigated_runs": len(mitigated),
        "time_to_mitigation_s": {"mean": float(np.mean(ttm)), "p50": float(np.percentile(ttm, 50)),
                                 "p90": float(np.percentile(ttm, 90))} if ttm else None,
        "drop_rate": {"mean": float(np.mean(drops)), "min": float(np.min(drops)),
                      "runs_meeting_70pct": int(sum(d >= 0.7 for d in drops))} if drops else None,
        "false_blocks": sorted({ip for r in results for ip in r.get("false_blocks", [])}),
        "benign_runs_with_rules": sum(1 for r in results if r.get("false_rules_in_benign_run")),
        "attack_runs_with_false_blocks": sum(1 for r in attacks if r.get("false_blocks")),
        # Benign packets/s reaching the victim after mitigation vs. before the attack (collateral damage).
        "benign_traffic_kept": float(np.mean([r["benign_pps_after"] / r["benign_pps_before"] for r in mitigated
                                              if r.get("benign_pps_before") and r.get("benign_pps_after") is not None]))
        if any(r.get("benign_pps_before") for r in mitigated) else None,
        "by_scenario": {},
    }
    for name in sorted({r["scenario"] for r in attacks}):
        runs = [r for r in attacks if r["scenario"] == name]
        d = [r["drop_rate"] for r in runs if r.get("drop_rate") is not None]
        t = [r["time_to_mitigation_s"] for r in runs if r.get("mitigated")]
        kept = [(r["benign_pps_before"], r["benign_pps_after"]) for r in runs
                if r.get("benign_pps_before") and r.get("benign_pps_after") is not None]
        summary["by_scenario"][name] = {"runs": len(runs), "mitigated": len(t),
                                        "mean_drop_rate": float(np.mean(d)) if d else None,
                                        "mean_time_to_mitigation_s": float(np.mean(t)) if t else None,
                                        "benign_traffic_kept": sum(b for _, b in kept) / sum(a for a, _ in kept) if kept else None}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="data/mininet/runs/*.json")
    parser.add_argument("--log", default="logs/mitigation_log.jsonl")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--out", default="results/phase2/mitigation_report.json")
    args = parser.parse_args()

    log = read_log(args.log)
    results = []
    for path in sorted(glob.glob(args.runs)):
        run = json.load(open(path))
        packets = read_packets(run["pcap"]) if run.get("pcap") and os.path.exists(run["pcap"]) else []
        results.append({"file": os.path.basename(path), **analyze_run(run, packets, log, args.poll_interval)})
    report = {"summary": summarize(results), "runs": results}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
