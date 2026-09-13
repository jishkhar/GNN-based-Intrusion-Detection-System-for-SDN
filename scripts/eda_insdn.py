#!/usr/bin/env python3
"""Exploratory analysis of InSDN as topology graphs (TODO M1.4).

Answers the question the GNN design rests on: do attack types differ in
*graph structure* (who talks to whom, how widely), not just in per-flow
statistics? Writes plots to results/phase2/eda/ and a summary to
Docs/eda_summary.md.

    python scripts/eda_insdn.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

from preprocessing.labels import ATTACK_CLASSES  # noqa: E402
from preprocessing.openflow_features import NODE_FEATURE_NAMES  # noqa: E402

PALETTE = {"attacker": "#ef4444", "victim": "#f59e0b", "host": "#3b82f6"}


def class_table(pattern: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(pattern)):
        df = pd.read_csv(path, usecols=["Attack", "Src IP", "Dst IP"])
        for cls, g in df.groupby("Attack"):
            rows.append({"file": os.path.basename(path).replace("_cleaned.csv", ""), "class": cls, "flows": len(g),
                         "src_ips": g["Src IP"].nunique(), "dst_ips": g["Dst IP"].nunique()})
    return pd.DataFrame(rows)


def graph_stats(graphs) -> pd.DataFrame:
    rows = []
    idx = {n: i for i, n in enumerate(NODE_FEATURE_NAMES)}
    for g in graphs:
        x = g.x.numpy()
        att = g.node_y.numpy() == 1
        in_deg = x[:, idx["in_degree"]]
        rows.append(
            {
                "class": ATTACK_CLASSES[int(g.y_multi)],
                # attack window with a benign window overlaid on it (a real share of benign records)
                "overlaid": bool(int(g.y) == 1 and (g.edge_y == 0).float().mean() > 0.2),
                "nodes": g.num_nodes,
                "edges": g.edge_index.shape[1],
                "attackers": int(att.sum()),
                "max_in_degree": float(in_deg.max()),
                "attacker_out_degree": float(x[att, idx["out_degree"]].mean()) if att.any() else np.nan,
                "attacker_port_entropy": float(x[att, idx["dst_port_entropy"]].mean()) if att.any() else np.nan,
                "attacker_flows": float(x[att, idx["out_flows"]].mean()) if att.any() else np.nan,
                "density": g.edge_index.shape[1] / max(1, g.num_nodes * (g.num_nodes - 1)),
            }
        )
    return pd.DataFrame(rows)


def draw_examples(graphs, out_path: str) -> None:
    wanted = ["Benign", "DDoS", "DoS", "Probe"]
    picks = {}
    for g in graphs:
        cls = ATTACK_CLASSES[int(g.y_multi)]
        att = g.node_y.numpy() == 1
        pure = not (int(g.y) == 1 and (g.edge_y == 0).float().mean() > 0.2)  # skip benign-overlaid windows
        if cls in wanted and cls not in picks and pure and g.num_nodes >= 3:
            picks[cls] = g
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 4.2))
    for ax, (cls, g) in zip(np.atleast_1d(axes), picks.items()):
        G = nx.DiGraph()
        ei = g.edge_index.numpy()
        G.add_nodes_from(range(g.num_nodes))
        G.add_edges_from(zip(ei[0], ei[1]))
        att = g.node_y.numpy() == 1
        victims = set(ei[1][att[ei[0]]]) - set(np.where(att)[0])
        colors = [PALETTE["attacker"] if att[n] else PALETTE["victim"] if n in victims else PALETTE["host"] for n in G.nodes]
        pos = nx.spring_layout(G, seed=1, k=0.6)
        nx.draw_networkx(G, pos, ax=ax, node_color=colors, node_size=40, with_labels=False, arrows=False,
                         edge_color="#999999", width=0.4)
        ax.set_title(f"{cls}: {g.num_nodes} hosts, {ei.shape[1]} flow records")
        ax.axis("off")
    fig.suptitle("InSDN windows as graphs (red = attacker, amber = victim, blue = other host)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-glob", default="data/insdn/cleaned/*_cleaned.csv")
    parser.add_argument("--graph-path", default="data/graphs/insdn_v2.pt")
    parser.add_argument("--out-dir", default="results/phase2/eda")
    parser.add_argument("--summary", default="Docs/eda_summary.md")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    classes = class_table(args.input_glob)
    data = torch.load(args.graph_path, weights_only=False)
    graphs = data["train"]  # describe the training split only
    stats = graph_stats(graphs)
    pure = stats[~stats["overlaid"]]

    per_class = pure.groupby("class").agg(
        windows=("nodes", "size"),
        hosts=("nodes", "median"),
        flow_records=("edges", "median"),
        attackers=("attackers", "median"),
        max_in_degree=("max_in_degree", "median"),
        attacker_out_degree=("attacker_out_degree", "median"),
        attacker_port_entropy=("attacker_port_entropy", "median"),
        attacker_flows=("attacker_flows", "median"),
    ).reindex([c for c in ATTACK_CLASSES if c in set(pure["class"])])

    # Plots
    fig, ax = plt.subplots(figsize=(8, 4))
    totals = classes.groupby("class")["flows"].sum().reindex([c for c in ATTACK_CLASSES if c in set(classes["class"])])
    ax.bar(totals.index, totals.values, color="#4f6ef7")
    ax.set_yscale("log")
    ax.set_ylabel("flows (log scale)")
    ax.set_title("InSDN class distribution (flows)")
    for i, v in enumerate(totals.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/class_distribution.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title in zip(axes, ["attacker_out_degree", "attacker_port_entropy", "max_in_degree"],
                              ["Attacker fan-out (distinct destinations)", "Attacker dst-port entropy (bits)",
                               "Largest in-degree in window (victim fan-in)"]):
        order = [c for c in ["Benign", "DDoS", "DoS", "Probe", "BruteForce"] if c in set(pure["class"])]
        data_by = [pure.loc[pure["class"] == c, col].dropna().values for c in order]
        ax.boxplot(data_by, tick_labels=order, showfliers=False)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/structure_by_class.png", dpi=150)
    plt.close(fig)
    draw_examples(graphs, f"{args.out_dir}/example_graphs.png")

    meta = data["meta"]
    lines = [
        "# EDA Summary — InSDN as Topology Graphs",
        "",
        "Generated by `scripts/eda_insdn.py` from the cleaned InSDN CSVs and the Phase 2 training graphs",
        f"(`{args.graph_path}`, training split only, {len(graphs):,} windows of {int(meta['window_size'])} flows).",
        "",
        "## 1. Dataset",
        "",
        "| File | Class | Flows | Source IPs | Destination IPs |",
        "|---|---|---:|---:|---:|",
        *[f"| {r.file} | {r['class']} | {r.flows:,} | {r.src_ips:,} | {r.dst_ips:,} |" for _, r in classes.iterrows()],
        "",
        "Observations:",
        "",
        "- Benign traffic (68,423 flows) comes from a separate capture (`Normal_data`) recorded at different",
        "  times than the attacks, so raw windows are all-benign or all-attack. The graph builder therefore",
        "  overlays benign windows onto half of the attack windows (within the same split) so that attackers",
        "  must be found among normal hosts.",
        "- DDoS uses spoofed sources: ~122k DDoS flows come from ~122k distinct source IPs (one flow each).",
        "  Blocking source IPs is useless against it; the mitigation engine rate-limits the victim instead.",
        "- WebAttack (192 flows), Botnet (164) and U2R/`Other` (17) are too small for window-level evaluation:",
        "  after the time-ordered split they have 1–9 test windows. Metrics therefore report a macro-F1 over",
        "  classes with at least 20 test windows next to the all-class figure.",
        "- Timestamps are mostly minute-resolution and use mixed formats, so windows are N consecutive flows",
        "  (capture order) rather than W seconds.",
        "- **Flow direction noise.** CICFlowMeter's forward direction is whichever side it saw first, and InSDN",
        "  labels flows by capture period, so many *victim -> attacker* rows carry the attack label (e.g. 36,951",
        "  Probe rows in metasploitable-2 go from the victim 192.168.3.130 to the scanner). Those rows have the",
        "  service port as source port. The converter treats the lower-port side as the server, so only the",
        "  initiator becomes an attacker node: this corrects 27% of metasploitable-2 rows, 5% of OVS and 20%",
        "  of benign rows. Without it the node head learns to flag victims (servers) as attackers.",
        "- The real attacker in every non-spoofed InSDN attack is one host (200.175.2.130); victims are",
        "  192.168.20.131-134, 192.168.3.130 and 172.17.0.2.",
        "",
        "## 2. Graph structure per class (pure windows, medians)",
        "",
        "| Class | Windows | Hosts | Flow records | Attackers | Max in-degree | Attacker fan-out | Attacker port entropy | Flows per attacker |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {c} | {int(r.windows):,} | {r.hosts:.0f} | {r.flow_records:.0f} | {r.attackers:.0f} | {r.max_in_degree:.0f} | "
            + ("—" if np.isnan(r.attacker_out_degree) else f"{r.attacker_out_degree:.0f}") + " | "
            + ("—" if np.isnan(r.attacker_port_entropy) else f"{r.attacker_port_entropy:.2f}") + " | "
            + ("—" if np.isnan(r.attacker_flows) else f"{r.attacker_flows:.0f}") + " |"
            for c, r in per_class.iterrows()
        ],
        "",
        "Each class has a distinct *shape*, which a per-flow model cannot see directly:",
        "",
        "- **DDoS**: many one-flow attacker nodes converge on one victim (star with very high in-degree).",
        "- **Probe**: one attacker, one victim, many destination ports (high port entropy on a single edge set).",
        "- **DoS**: few attackers, one victim, many flows per attacker on few ports.",
        "- **Benign**: a few busy internal hosts talking to many external services, low port entropy per host.",
        "",
        "![Class distribution](../results/phase2/eda/class_distribution.png)",
        "![Structure by class](../results/phase2/eda/structure_by_class.png)",
        "![Example graphs](../results/phase2/eda/example_graphs.png)",
        "",
    ]
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(per_class.round(2).to_string())
    print(f"Saved -> {args.summary}, {args.out_dir}/")


if __name__ == "__main__":
    main()
