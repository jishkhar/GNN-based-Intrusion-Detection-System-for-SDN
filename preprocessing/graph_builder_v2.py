"""Phase 2 graph builder: real host-to-host graphs with node labels.

Pipeline per cleaned CSV (InSDN or CICIDS2017 TrafficLabelling):

1. parse timestamps and stable-sort flows by time,
2. convert rows to unidirectional OpenFlow-style flow records,
3. cut sliding windows (``count`` = N consecutive flows, ``time`` = W seconds),
4. split windows chronologically per (file, class) into train / val / test,
5. optionally overlay attack windows with a benign window from the same split,
   so attackers must be found among normal hosts,
6. build one PyG ``Data`` per window.

Each ``Data`` carries ``x``, ``edge_index``, ``edge_attr``, ``y`` (binary),
``y_multi`` (class id, see ``preprocessing.labels``), ``node_y`` (1 = host sent
attack traffic), ``edge_y`` / ``edge_multi`` (per-record labels, for flow-level
baselines), ``edge_first`` (record not seen in an earlier window of the split),
``edge_reverse`` (backward direction of a bidirectional flow), ``node_ips`` and
split metadata. The output is a dict
``{"train": [...], "val": [...], "test": [...], "meta": {...}}``.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from common.config import parse_args_with_config, run_metadata
from preprocessing.graph_builder import parse_timestamps
from preprocessing.graph_dataset import time_split_indices
from preprocessing.labels import ATTACK_CLASSES, CLASS_TO_ID, canonical_class
from preprocessing.openflow_features import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    dataset_to_flow_records,
    detect_columns,
    edge_features,
    factorize_nodes,
    node_features,
)

BENIGN_ID = CLASS_TO_ID["Benign"]


@dataclass
class Window:
    """A slice of one file's flow records, before graph construction."""

    records: pd.DataFrame
    source_id: int
    start: float  # seconds (time mode) or first flow index (count mode)
    y_multi: int


def load_records(path: str, label_col: str = "Label", attack_col: str = "Attack") -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    columns = detect_columns(df)
    if attack_col not in df.columns:
        df[attack_col] = df[label_col].map(canonical_class)
    ts = parse_timestamps(df[columns.timestamp]).ffill().bfill()
    order = np.argsort(ts.to_numpy(), kind="stable")
    df = df.iloc[order].reset_index(drop=True)
    ts = ts.iloc[order].reset_index(drop=True)

    records = dataset_to_flow_records(df, columns, timestamps=ts)
    records["attack_id"] = records[attack_col].map(CLASS_TO_ID).fillna(CLASS_TO_ID["Other"]).astype(np.int64)
    # Only the forward direction marks its source as an attacker; the reverse
    # record of an attack flow is the victim replying.
    records["is_attack"] = records["attack_id"] != BENIGN_ID
    return records


def window_label(records: pd.DataFrame, min_attack_frac: float) -> int:
    fwd = records[~records["is_reverse"]]
    attack = fwd[fwd["is_attack"]]
    if len(fwd) == 0 or len(attack) / len(fwd) < min_attack_frac:
        return BENIGN_ID
    return int(Counter(attack["attack_id"].tolist()).most_common(1)[0][0])


def cut_windows(
    records: pd.DataFrame,
    source_id: int,
    mode: str,
    size: float,
    stride: float,
    min_attack_frac: float,
    min_flows: int = 2,
) -> list[Window]:
    windows: list[Window] = []
    if mode == "count":
        flow_idx = records["flow_idx"].to_numpy()
        n_flows = int(flow_idx.max()) + 1 if len(flow_idx) else 0
        size_i, stride_i = int(size), int(stride)
        # records are sorted by flow_idx, so each window is a contiguous slice.
        starts = range(0, max(1, n_flows - size_i + 1), stride_i)
        bounds = np.searchsorted(flow_idx, [(s, s + size_i) for s in starts])
        for s, (lo, hi) in zip(starts, bounds):
            w = records.iloc[lo:hi]
            if w["flow_idx"].nunique() >= min_flows:
                windows.append(Window(w, source_id, float(s), window_label(w, min_attack_frac)))
    elif mode == "time":
        ts = pd.to_datetime(records["timestamp"])
        secs = ((ts - ts.min()).dt.total_seconds()).to_numpy()
        order = np.argsort(secs, kind="stable")
        secs_sorted = secs[order]
        t, end = 0.0, float(secs_sorted[-1]) if len(secs_sorted) else 0.0
        while t <= end:
            lo, hi = np.searchsorted(secs_sorted, [t, t + size], side="left")
            if hi - lo > 0:
                w = records.iloc[np.sort(order[lo:hi])]
                if w["flow_idx"].nunique() >= min_flows:
                    windows.append(Window(w, source_id, t, window_label(w, min_attack_frac)))
            t += stride
    else:
        raise ValueError(f"Unknown window mode: {mode}")
    return windows


def build_graph(records: pd.DataFrame, y_multi: int, seen: set | None = None) -> Data:
    src, dst, ips = factorize_nodes(records)
    n = len(ips)
    x = node_features(records, src, dst, n)
    e = edge_features(records)

    fwd_attack = (records["is_attack"] & ~records["is_reverse"]).to_numpy()
    node_y = np.zeros(n, dtype=np.int64)
    node_y[src[fwd_attack]] = 1

    keys = list(zip(records["source_id"].to_numpy(), records["flow_idx"].to_numpy(), records["is_reverse"].to_numpy()))
    if seen is None:
        edge_first = np.ones(len(keys), dtype=bool)
    else:
        edge_first = np.array([k not in seen for k in keys], dtype=bool)
        seen.update(keys)

    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(np.stack([src, dst])).long(),
        edge_attr=torch.from_numpy(e),
        y=torch.tensor([int(y_multi != BENIGN_ID)], dtype=torch.long),
        y_multi=torch.tensor([y_multi], dtype=torch.long),
        node_y=torch.from_numpy(node_y),
        edge_y=torch.from_numpy(records["is_attack"].to_numpy().astype(np.int64)),
        edge_multi=torch.from_numpy(records["attack_id"].to_numpy().astype(np.int64)),
        edge_first=torch.from_numpy(edge_first),
        edge_reverse=torch.from_numpy(records["is_reverse"].to_numpy().astype(bool)),
    )
    data.num_nodes = n
    data.node_ips = ips
    return data


def overlay(attack: Window, benign: Window) -> Window:
    """Merge a benign window into an attack window (both keep their own labels)."""
    merged = pd.concat([benign.records, attack.records], ignore_index=True)
    return Window(merged, attack.source_id, attack.start, attack.y_multi)


def run(
    input_glob: str,
    output_path: str,
    window_mode: str = "count",
    window_size: float = 100,
    window_stride: float = 25,
    stride_overrides: dict | None = None,
    min_attack_frac: float = 0.1,
    overlay_prob: float = 0.5,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
    metadata: dict | None = None,
) -> dict:
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No cleaned files found by: {input_glob}")
    rng = np.random.default_rng(seed)
    stride_overrides = stride_overrides or {}

    windows: list[Window] = []
    for source_id, path in enumerate(files):
        name = os.path.basename(path)
        records = load_records(path)
        records["source_id"] = source_id
        stride = stride_overrides.get(name, window_stride)
        file_windows = cut_windows(records, source_id, window_mode, window_size, stride, min_attack_frac)
        counts = Counter(ATTACK_CLASSES[w.y_multi] for w in file_windows)
        print(f"{name}: {records['flow_idx'].nunique()} flows -> {len(file_windows)} windows {dict(counts)}")
        windows.extend(file_windows)

    # Chronological split inside each (file, class) stream; the gap stops
    # overlapping windows from sharing flows across split boundaries.
    gap = int(math.ceil(window_size / max(window_stride, 1))) if window_mode == "count" else int(
        math.ceil(window_size / max(window_stride, 1e-9))
    )
    groups = [(w.source_id, w.y_multi) for w in windows]
    times = [w.start for w in windows]
    split_idx = dict(zip(("train", "val", "test"), time_split_indices(groups, times, test_size, val_size, gap)))

    out: dict = {}
    for split, idx in split_idx.items():
        split_windows = [windows[i] for i in idx]
        benign = [w for w in split_windows if w.y_multi == BENIGN_ID]
        seen: set = set()
        graphs = []
        n_overlaid = 0
        for w in split_windows:
            if w.y_multi != BENIGN_ID and benign and rng.random() < overlay_prob:
                w = overlay(w, benign[rng.integers(len(benign))])
                n_overlaid += 1
            g = build_graph(w.records, w.y_multi, seen)
            g.source_id = torch.tensor([w.source_id], dtype=torch.long)
            g.window_start = torch.tensor([w.start], dtype=torch.float64)
            graphs.append(g)
        out[split] = graphs
        counts = Counter(ATTACK_CLASSES[int(g.y_multi)] for g in graphs)
        print(f"[{split}] {len(graphs)} graphs ({n_overlaid} overlaid with benign) {dict(counts)}")

    all_graphs = out["train"] + out["val"] + out["test"]
    out["meta"] = {
        "files": [os.path.basename(f) for f in files],
        "classes": ATTACK_CLASSES,
        "node_feature_names": NODE_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "window_mode": window_mode,
        "window_size": window_size,
        "window_stride": window_stride,
        "stride_overrides": stride_overrides,
        "min_attack_frac": min_attack_frac,
        "overlay_prob": overlay_prob,
        "split_gap": gap,
        "stats": {
            "graphs": len(all_graphs),
            "mean_nodes": float(np.mean([g.num_nodes for g in all_graphs])),
            "max_nodes": int(max(g.num_nodes for g in all_graphs)),
            "mean_edges": float(np.mean([g.edge_index.shape[1] for g in all_graphs])),
            "max_edges": int(max(g.edge_index.shape[1] for g in all_graphs)),
            "attacker_node_frac": float(
                torch.cat([g.node_y for g in all_graphs]).float().mean().item()
            ),
        },
        "run": metadata or {},
    }
    print("stats:", out["meta"]["stats"])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(out, output_path)
    print(f"Saved -> {output_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 2 topology graphs with node labels.")
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--window-mode", choices=["count", "time"], default="count")
    parser.add_argument("--window-size", type=float, default=100)
    parser.add_argument("--window-stride", type=float, default=25)
    parser.add_argument("--min-attack-frac", type=float, default=0.1)
    parser.add_argument("--overlay-prob", type=float, default=0.5)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(stride_overrides={})
    args, _ = parse_args_with_config(parser, "graph_builder_v2")
    run(
        input_glob=args.input_glob,
        output_path=args.output_path,
        window_mode=args.window_mode,
        window_size=args.window_size,
        window_stride=args.window_stride,
        stride_overrides=args.stride_overrides,
        min_attack_frac=args.min_attack_frac,
        overlay_prob=args.overlay_prob,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        metadata=run_metadata(args),
    )


if __name__ == "__main__":
    main()
