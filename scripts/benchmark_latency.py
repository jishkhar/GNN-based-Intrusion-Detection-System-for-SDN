#!/usr/bin/env python3
"""Latency benchmark for the live detection path (TODO M7.3).

Times each stage (feature extraction -> graph tensors -> GNN forward ->
decision) on windows of real held-out InSDN flows of increasing size, on CPU
and GPU, plus the full service path (collector + engine + classifier +
mitigation) that a controller upload goes through.

    python scripts/benchmark_latency.py --runs 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.replay_flows import holdout_records, to_entries  # noqa: E402

STAGES = ("features", "graph", "model", "decision", "total")


def percentiles(values: list[float]) -> dict:
    return {
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def bench_engine(engine, windows: dict, runs: int) -> dict:
    out = {}
    for size, records in windows.items():
        for _ in range(5):
            engine.predict(records)  # warm-up (allocator, cuDNN autotune)
        samples = {s: [] for s in STAGES}
        for _ in range(runs):
            lat = engine.predict(records).latency_ms
            for s in STAGES:
                samples[s].append(lat[s])
        out[str(size)] = {"records": len(records), **{s: percentiles(v) for s, v in samples.items()}}
        print(f"  {size:>5} records: total p50={out[str(size)]['total']['p50']:.2f} ms "
              f"p90={out[str(size)]['total']['p90']:.2f} p99={out[str(size)]['total']['p99']:.2f} "
              f"(model p50={out[str(size)]['model']['p50']:.2f})")
    return out


def bench_service(model_path: str, device: str, records, runs: int, window_seconds: float) -> dict:
    from inference.service import IDSService, ServiceConfig

    cfg = ServiceConfig(model_path=model_path, device=device, log_dir=tempfile.mkdtemp(prefix="ids_bench_"),
                        window_seconds=window_seconds, mitigation_enabled=False)
    service = IDSService(cfg)
    keys = records["flow_key"].drop_duplicates().to_numpy()
    now, per_poll = time.time(), 60
    latencies = []
    for i in range(runs + 5):
        chunk = keys[(i * per_poll) % len(keys):(i * per_poll) % len(keys) + per_poll]
        entries = to_entries(records[records["flow_key"].isin(chunk)])
        result = service.process_flows(entries, now=now)
        if i >= 5:
            latencies.append(result["latency_ms"])
        now += 2.0
    return {"flows_per_poll": per_poll, "service_total": percentiles(latencies)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", default="models/gat_ids.pt")
    parser.add_argument("--input-glob", default="data/insdn/cleaned/*_cleaned.csv")
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 250, 500, 1000, 2000, 5000])
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--out", default="results/phase2/latency_report.json")
    parser.add_argument("--plot", default="results/phase2/latency_histogram.png")
    args = parser.parse_args()

    import torch

    from inference.inference_engine import InferenceEngine

    records = holdout_records(args.input_glob, 0.15)
    # Mixed windows: benign and attack records interleaved, like a live network.
    shuffled = records.sample(frac=1.0, random_state=0).reset_index(drop=True)
    windows = {size: shuffled.iloc[:size] for size in args.sizes}

    report = {"target_ms": 50, "runs": args.runs, "devices": {}}
    for device in args.devices:
        if device == "cuda" and not torch.cuda.is_available():
            continue
        name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
        print(f"[{device}] {name}")
        engine = InferenceEngine(args.model_path, device=device)
        report["devices"][device] = {
            "name": name,
            "engine": bench_engine(engine, windows, args.runs),
            "service": bench_service(args.model_path, device, records, args.runs, 10.0),
        }
        s = report["devices"][device]["service"]["service_total"]
        print(f"  service path: p50={s['p50']:.2f} p90={s['p90']:.2f} p99={s['p99']:.2f} ms")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved -> {args.out}")

    if args.plot and report["devices"]:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        for device, res in report["devices"].items():
            sizes = [int(k) for k in res["engine"]]
            ax.plot(sizes, [res["engine"][str(k)]["total"]["p90"] for k in sizes], marker="o", label=f"{device} p90")
            ax.plot(sizes, [res["engine"][str(k)]["total"]["p50"] for k in sizes], marker=".", ls="--",
                    label=f"{device} p50")
        ax.axhline(50, color="red", ls=":", label="50 ms target")
        ax.set_xscale("log")
        ax.set_xlabel("flow records in window")
        ax.set_ylabel("detection latency (ms)")
        ax.set_title("GNN-IDS detection latency")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"Saved -> {args.plot}")


if __name__ == "__main__":
    main()
