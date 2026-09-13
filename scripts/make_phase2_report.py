#!/usr/bin/env python3
"""Compile Phase 2 results into results/phase2/final_results.md (+ plots).

Reads whatever result files exist (missing ones are skipped):
- results/phase1_timesplit/*, results/gnn_metrics.json   (Phase 1, random vs time split)
- results/phase2/baselines_v2.json, gnn_v2_gat.json, ablation_*.json
- results/phase2/latency_report.json, replay_report.json
- results/phase2/mininet_*.json (transfer / fine-tune evaluations, mitigation reports)

    python scripts/make_phase2_report.py
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

R = "results/phase2"


def load(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def f(v, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def pct(v) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


def model_rows(baselines: dict | None, gnn: dict | None) -> list[tuple[str, dict]]:
    rows = []
    for key, name in (("random_forest", "Random Forest"), ("xgboost", "XGBoost")):
        if baselines and key in baselines:
            b = baselines[key]
            rows.append((name, {"window": b["window"], "host": b["host"]["host_feature_model"],
                                "host_flows": b["host"]["from_flow_scores"], "flow": b["flow"]}))
    if gnn:
        t = gnn["test"]
        rows.append(("**GAT-IDS (ours)**", {"window": t["window"], "host": t["host"]}))
    return rows


def main_table(rows) -> list[str]:
    out = [
        "| Model | Window F1 | Window FPR | Window ROC-AUC | Attack-type macro-F1 (major) | Attack-type macro-F1 (all) | Host F1 | Host precision | Host recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in rows:
        w, h = r["window"], r["host"]
        out.append(
            f"| {name} | {f(w['binary']['f1'])} | {f(w['binary']['false_positive_rate'])} | {f(w['binary']['roc_auc'])} | "
            f"{f(w['multiclass'].get('major_macro_f1'))} | {f(w['multiclass']['macro_f1'])} | {f(h['f1'])} | "
            f"{f(h['precision'])} | {f(h['recall'])} |"
        )
    return out


def per_class_table(rows) -> list[str]:
    classes = []
    for _, r in rows:
        for c in r["window"]["multiclass"]["per_class"]:
            if c not in classes:
                classes.append(c)
    out = ["| Class | Test windows | " + " | ".join(n for n, _ in rows) + " |",
           "|---|---:|" + "---:|" * len(rows)]
    for c in classes:
        support = next((r["window"]["multiclass"]["per_class"][c]["support"] for _, r in rows
                        if c in r["window"]["multiclass"]["per_class"]), 0)
        cells = [f(r["window"]["multiclass"]["per_class"].get(c, {}).get("f1")) for _, r in rows]
        out.append(f"| {c} | {support} | " + " | ".join(cells) + " |")
    return out


def roc_plot(out_path: str) -> bool:
    gnn = f"{R}/gnn_v2_gat_scores.npz"
    base = f"{R}/baseline_scores.npz"
    if not (os.path.exists(gnn) and os.path.exists(base)):
        return False
    g, b = np.load(gnn), np.load(base)
    y = g["y"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, score in (("GAT-IDS", g["graph_score"]), ("XGBoost", b["xgboost_window"]),
                        ("Random Forest", b["random_forest_window"])):
        if len(score) != len(y):
            continue
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(fpr, tpr, label=name)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("false positive rate (log)")
    ax.set_ylabel("true positive rate")
    ax.set_title("Window-level ROC (InSDN test split)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def confusion_plot(rows, out_path: str) -> bool:
    mats = [(n.strip("*"), r["window"]["multiclass"]["confusion_matrix"]) for n, r in rows]
    if not mats:
        return False
    fig, axes = plt.subplots(1, len(mats), figsize=(5.2 * len(mats), 4.8))
    for ax, (name, cm) in zip(np.atleast_1d(axes), mats):
        labels, m = cm["labels"], np.array(cm["matrix"])
        keep = [i for i in range(len(labels)) if m[i].sum() + m[:, i].sum() > 0]
        m = m[np.ix_(keep, keep)]
        norm = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(keep)), [labels[i] for i in keep], rotation=45, ha="right")
        ax.set_yticks(range(len(keep)), [labels[i] for i in keep])
        for i in range(len(keep)):
            for j in range(len(keep)):
                ax.text(j, i, m[i, j], ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_title(name)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    fig.suptitle("Window attack-type confusion matrices (InSDN test split, row-normalised colour)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def mitigation_timeline_plot(session: str, out_path: str) -> bool:
    """Attack packets/s reaching the victim around the IDS rule, one run per attack type."""
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.analyze_mitigation import read_log, read_packets

    runs = {}
    for path in sorted(__import__("glob").glob(f"{session}/runs/*.json")):
        run = load(path)
        if run and run.get("attack_type") not in (None, "Benign") and run["scenario"] not in runs and os.path.exists(run.get("pcap", "")):
            runs[run["scenario"]] = run
    if not runs:
        return False
    log = read_log(f"{session}/logs/mitigation_log.jsonl")
    topology = None
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.4), sharey=False)
    for ax, (name, run) in zip(np.atleast_1d(axes), sorted(runs.items())):
        topology = {h["ip"] for h in run["hosts"].values()}
        attackers = set(run["attacker_ips"])
        spoofed = run.get("spoofed_sources")
        pk = read_packets(run["pcap"])
        t0 = run["start"]
        att = np.array([t for t, src, dst in pk if dst == run["victim_ip"] and (src in attackers or (spoofed and src not in topology))]) - t0
        ben = np.array([t for t, src, dst in pk if dst == run["victim_ip"] and src in topology and src not in attackers]) - t0
        bins = np.arange(-10, run["end"] - t0 + 10, 1.0)
        ax.plot(bins[:-1], np.histogram(att, bins)[0], color="#ef4444", label="attack pkts/s")
        ax.plot(bins[:-1], np.histogram(ben, bins)[0], color="#3b82f6", label="benign pkts/s")
        rules = [e["ts"] - t0 for e in log if e.get("event") == "rule_added" and run["start"] <= e["ts"] <= run["end"]]
        if rules:
            ax.axvline(min(rules), color="black", ls="--", lw=1, label="IDS rule")
        ax.axvspan(0, run["end"] - t0, color="#f59e0b", alpha=0.08)
        ax.set_title(f"{run['attack_type']}")
        ax.set_xlabel("seconds from attack start")
    np.atleast_1d(axes)[0].set_ylabel("packets/s at victim")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Traffic reaching the victim before/after mitigation (final live evaluation, shaded = attack)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=f"{R}/final_results.md")
    parser.add_argument("--session", default="data/mininet/final_eval3", help="lab session for the traffic plot")
    args = parser.parse_args()

    L = ["# Phase 2 Results", "",
         "Generated by `scripts/make_phase2_report.py`. All models use the same OpenFlow-compatible features and the",
         "same time-ordered InSDN split (`data/graphs/insdn_v2.pt`). Thresholds are tuned on validation data only.", ""]

    # Targets from Docs/GNN_IDS_Architecture.md section 10
    final, lat_rep, ft_eval = load(f"{R}/mininet_mitigation_final.json"), load(f"{R}/latency_report.json"), \
        load(f"{R}/mininet_detection_after_finetune.json")
    base_rep, gnn_rep = load(f"{R}/baselines_v2.json"), load(f"{R}/gnn_v2_gat.json")
    if final and lat_rep and ft_eval and base_rep and gnn_rep:
        s = final["summary"]
        svc = next(iter(lat_rep["devices"].values()))["service"]["service_total"]
        best_base = max(base_rep[k]["window"]["multiclass"]["major_macro_f1"] for k in ("random_forest", "xgboost"))
        ours = gnn_rep["test"]["window"]["multiclass"]["major_macro_f1"]
        L += ["## 0. Results vs. architecture targets", "",
              "| Target (Architecture doc §10) | Result | Met? |", "|---|---|---|",
              f"| F1 gain over XGBoost on coordinated attacks | Attack-type macro-F1 {ours:.3f} vs {best_base:.3f} (error rate "
              f"{100*(1-best_base):.1f}% → {100*(1-ours):.1f}%); binary detection saturated for all models (≥ 0.997) | "
              "Partly: large error reduction, but the +15–25 % F1 target is not meaningful at a 0.97 baseline |",
              f"| Inference latency < 50 ms | Full service path p90 {svc['p90']:.1f} ms, p99 {svc['p99']:.1f} ms | Yes |",
              f"| ≥ 70 % attack traffic dropped | {s['drop_rate']['runs_meeting_70pct']}/{s['mitigated_runs']} live attack runs, "
              f"mean {100*s['drop_rate']['mean']:.1f} % | Yes |",
              f"| … within 2 controller round-trips (~100 ms) | Median {s['time_to_mitigation_s']['p50']:.1f} s from attack start "
              "(2 s stats polling, 10 s window, 2-window confirmation) | No: bounded by polling, not by the model |",
              f"| False positive rate < 5 % | Lab: {100*ft_eval['window']['binary']['false_positive_rate']:.1f} % of benign windows flagged; "
              f"live: {s['benign_runs_with_rules']} rules in benign-only runs, {s.get('attack_runs_with_false_blocks', 0)} attack runs "
              "with a benign host blocked | Yes |", ""]

    # Phase 1 leakage check
    p1, p1t = load("results/gnn_metrics.json"), load("results/phase1_timesplit/gnn_metrics.json")
    if p1 and p1t:
        L += ["## 1. Phase 1 model: effect of the leakage fix", "",
              "| Split | Test F1 | Precision | Recall | ROC-AUC |", "|---|---:|---:|---:|---:|",
              f"| Random (Phase 1) | {f(p1['test']['f1'])} | {f(p1['test']['precision'])} | {f(p1['test']['recall'])} | — |",
              f"| Time-ordered | {f(p1t['test']['f1'])} | {f(p1t['test']['precision'])} | {f(p1t['test']['recall'])} | {f(p1t['test'].get('roc_auc'))} |",
              "", "The Phase 1 graphs have no real topology (see `Docs/design_decisions.md` D1), so these numbers are",
              "kept only to show the leakage effect.", ""]

    baselines, gnn = load(f"{R}/baselines_v2.json"), load(f"{R}/gnn_v2_gat.json")
    rows = model_rows(baselines, gnn)
    if rows:
        sizes = (gnn or baselines)["split_sizes"]
        L += ["## 2. GNN vs. baselines (InSDN test split)", "",
              f"Windows: train {sizes['train']:,} / val {sizes['val']:,} / test {sizes['test']:,}. Baselines score flows and",
              "aggregate them per window (mean attack probability); host scores come from a model on the same per-host",
              "window features the GNN receives.", ""]
        L += main_table(rows)
        if baselines:
            L += ["", "Baseline host detection from flow scores (max over a host's outgoing flows): "
                  + ", ".join(f"{n}: F1 {f(r['host_flows']['f1'])}" for n, r in rows if 'host_flows' in r) + ".",
                  "Flow-level F1: " + ", ".join(f"{n}: {f(r['flow']['binary']['f1'])}" for n, r in rows if 'flow' in r) + "."]
        L += ["", "### Per-class window F1 (attack type)", ""] + per_class_table(rows)
        L += ["", "### Coordinated attacks (Benign + DDoS + Probe + Botnet windows)", "",
              "| Model | F1 | FPR | Detection rate DDoS | Detection rate Probe |", "|---|---:|---:|---:|---:|"]
        for name, r in rows:
            c, d = r["window"]["coordinated_subset"], r["window"]["detection_rate_by_class"]
            L.append(f"| {name} | {f(c['f1'])} | {f(c['false_positive_rate'])} | {pct(d.get('DDoS', {}).get('flagged_rate'))} | "
                     f"{pct(d.get('Probe', {}).get('flagged_rate'))} |")
        if roc_plot(f"{R}/roc_windows.png"):
            L += ["", "![Window ROC](roc_windows.png)"]
        if confusion_plot(rows, f"{R}/confusion_matrices.png"):
            L += ["", "![Confusion matrices](confusion_matrices.png)"]
        if os.path.exists(f"{R}/gnn_v2_gat_curves.png"):
            L += ["", "![GAT training curves](gnn_v2_gat_curves.png)"]
        L.append("")

    ablations = [("GAT + edge features (main)", gnn)] + [
        (name, load(path)) for name, path in (
            ("GCN + edge features", f"{R}/ablation_gcn.json"),
            ("GraphSAGE + edge features", f"{R}/ablation_sage.json"),
            ("GAT without edge features", f"{R}/ablation_gat_noedge.json"),
            ("GAT, window 50 flows", f"{R}/ablation_window_50.json"),
            ("GAT, window 200 flows", f"{R}/ablation_window_200.json"),
        )]
    ablations = [(n, a) for n, a in ablations if a]
    if len(ablations) > 1:
        L += ["## 3. Ablations", "",
              "| Variant | Params | Window F1 | Attack-type macro-F1 (major) | Host F1 | Best epoch |",
              "|---|---:|---:|---:|---:|---:|"]
        for name, a in ablations:
            t = a["test"]
            L.append(f"| {name} | {a['parameters']:,} | {f(t['window']['binary']['f1'])} | "
                     f"{f(t['window']['multiclass'].get('major_macro_f1'))} | {f(t['host']['f1'])} | {a['best_epoch']} |")
        L += ["", "Window-size variants are evaluated on their own test windows (different sizes), so compare them with care.", ""]

    lat = load(f"{R}/latency_report.json")
    if lat:
        L += ["## 4. Detection latency", "", "| Device | Records in window | p50 | p90 | p99 | GNN forward p50 |",
              "|---|---:|---:|---:|---:|---:|"]
        for dev, res in lat["devices"].items():
            for size, e in res["engine"].items():
                L.append(f"| {res['name']} | {size} | {e['total']['p50']:.1f} ms | {e['total']['p90']:.1f} ms | "
                         f"{e['total']['p99']:.1f} ms | {e['model']['p50']:.1f} ms |")
            s = res["service"]["service_total"]
            L.append(f"| {res['name']} | full service path | {s['p50']:.1f} ms | {s['p90']:.1f} ms | {s['p99']:.1f} ms | |")
        L += ["", "Target: < 50 ms per window. Excludes the controller polling interval (2 s).", "",
              "![Latency](latency_histogram.png)", ""]

    rep = load(f"{R}/replay_report.json")
    if rep:
        L += ["## 5. Live pipeline on held-out InSDN flows (replay)", "",
              "`scripts/replay_flows.py` feeds held-out flows to the live IDS as OpenFlow flow stats (2 s polls, 10 s window):",
              "benign → attack mixed with benign → benign, for each attack type.", "",
              "| Attack | Windows flagged | Predicted types | Attacker IPs blocked | False blocks | Rule actions |",
              "|---|---:|---|---:|---|---|"]
        for name, a in rep["attacks"].items():
            types = ", ".join(f"{k} {v}" for k, v in a["predicted_types"].items())
            L.append(f"| {name} | {a['windows_flagged']} | {types} | {pct(a['attacker_block_coverage'])} | "
                     f"{', '.join(a['false_blocks']) or 'none'} | {', '.join(a['rule_actions'])} |")
        b = rep["benign"]
        L += ["", f"Benign windows: {b['false_attack_windows']} false alerts in {b['windows']} "
              f"(FPR {pct(b['false_positive_rate'])}). Source IPs blocked: {rep['blocking']['src_ips_blocked']}, of which "
              f"benign-only: {', '.join(rep['blocking']['false_blocks']) or 'none'} (out of {rep['blocking']['benign_only_ips_seen']} benign-only IPs seen). "
              f"Latency p50 {rep['latency_ms']['p50']:.1f} ms, p99 {rep['latency_ms']['p99']:.1f} ms.", ""]

    mininet_order = [
        ("mininet_mitigation_initial_insdn_model", "6.1 First live test: InSDN-only model (1 DoS run, before the label fix)"),
        ("mininet_detection_before_finetune", "6.2 Detection on held-out lab runs: InSDN-only model"),
        ("mininet_detection_after_finetune", "6.3 Detection on held-out lab runs: fine-tuned model"),
        ("mininet_mitigation_finetuned_v1", "6.4 Live mitigation, fine-tuned model, first version (max host score, kbit/s meters, no persistence)"),
        ("mininet_mitigation_persistence_fix", "6.5 Live mitigation, + mean host score, 2-window persistence, packet/s meters"),
        ("mininet_mitigation_final", "6.6 Live mitigation, final system (+ rule priorities by severity, per-switch rule tracking)"),
    ]
    mininet = [(f"{R}/{stem}.json", title) for stem, title in mininet_order if os.path.exists(f"{R}/{stem}.json")]
    if mininet:
        L += ["## 6. Mininet (live SDN) results", "",
              "Lab: 2 OpenFlow switches, 6 hosts (2 attackers, 1 web server/victim, 3 benign clients), os-ken controller",
              "polling flow stats every 2 s, IDS window 10 s. Benign traffic runs throughout; each run is 25 s benign,",
              "45 s attack, 40 s benign. The fine-tuned model was trained on separate lab runs (split by run).", ""]
        ft = load(f"{R}/finetuned_on_insdn_test.json")
        if ft and gnn:
            b0, b1 = gnn["test"], ft
            L += ["Fine-tuning did not cost InSDN performance:", "",
                  "| InSDN test | Window F1 | FPR | Attack-type macro-F1 (major) | Host F1 |", "|---|---:|---:|---:|---:|",
                  f"| InSDN-only model | {f(b0['window']['binary']['f1'])} | {f(b0['window']['binary']['false_positive_rate'])} | "
                  f"{f(b0['window']['multiclass']['major_macro_f1'])} | {f(b0['host']['f1'])} |",
                  f"| Fine-tuned model | {f(b1['window']['binary']['f1'])} | {f(b1['window']['binary']['false_positive_rate'])} | "
                  f"{f(b1['window']['multiclass']['major_macro_f1'])} | {f(b1['host']['f1'])} |", ""]
        for path, title in mininet:
            m = load(path)
            name = title
            if "summary" in m:  # mitigation report
                s = m["summary"]
                ttm = s["time_to_mitigation_s"]
                drop = s["drop_rate"]
                L += [f"### {name}", "",
                      f"- Attack runs mitigated: {s['mitigated_runs']}/{s['attack_runs']}",
                      f"- Attack runs in which a benign host was blocked: {s.get('attack_runs_with_false_blocks', '—')}",
                      f"- Benign traffic to the victim kept after mitigation: {pct(s.get('benign_traffic_kept'))} averaged over runs "
                      "(varies strongly by attack type, see the last column)",
                      f"- Time to mitigation: mean {f(ttm['mean'], 1) if ttm else '—'} s, p90 {f(ttm['p90'], 1) if ttm else '—'} s "
                      "(detection needs the flows to reach the controller: 2 s polls, 10 s window)",
                      f"- Attack traffic dropped: mean {pct(drop['mean']) if drop else '—'}, runs meeting ≥ 70 %: "
                      f"{drop['runs_meeting_70pct'] if drop else 0}/{s['mitigated_runs']}",
                      f"- Benign hosts blocked: {', '.join(s['false_blocks']) or 'none'}; benign-only runs with rules: {s['benign_runs_with_rules']}", "",
                      "| Scenario | Runs | Mitigated | Mean drop rate | Mean time to mitigation | Benign traffic to victim kept |",
                      "|---|---:|---:|---:|---:|---:|"]
                for sc, v in s["by_scenario"].items():
                    L.append(f"| {sc} | {v['runs']} | {v['mitigated']} | {pct(v['mean_drop_rate'])} | "
                             f"{f(v['mean_time_to_mitigation_s'], 1)} s | {pct(v.get('benign_traffic_kept'))} |")
                L.append("")
                if path.endswith("mininet_mitigation_final.json") and mitigation_timeline_plot(args.session, f"{R}/mitigation_timeline.png"):
                    L += ["![Mitigation timeline](mitigation_timeline.png)", "",
                          "Reading the plot: **Probe** — the first full scan (a ~3 s burst) completes before the rule lands",
                          "(detection needs ~17 s); the rate limit throttles the repeated scans that follow. **DDoS** — victim",
                          "protection caps traffic to the victim at 200 packets/s, but spoofed flood packets take most of that",
                          "budget, so legitimate clients get very little service during the attack (see the benign column). Protecting",
                          "legitimate clients during a spoofed flood needs SYN cookies/proxying, which is outside this project.", ""]
            else:  # model evaluation on Mininet graphs
                w, h = m["window"], m["host"]
                L += [f"### {name}", "",
                      f"Checkpoint `{m['checkpoint']}` on `{m['graph_path']}` ({m['split']}, {m['graphs']} windows):",
                      f"window F1 {f(w['binary']['f1'])}, window FPR {f(w['binary']['false_positive_rate'])}, "
                      f"attack-type macro-F1 (major) {f(w['multiclass'].get('major_macro_f1'))}, host F1 {f(h['f1'])} "
                      f"(precision {f(h['precision'])}, recall {f(h['recall'])}).", ""]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
