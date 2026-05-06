from __future__ import annotations

import argparse
import glob
import math
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from preprocessing.feature_extractor import add_basic_node_features, select_edge_feature_columns


def _find_col(df: pd.DataFrame, preferred: list[str]) -> str:
    cmap = {c.lower(): c for c in df.columns}
    for p in preferred:
        if p.lower() in cmap:
            return cmap[p.lower()]
    raise ValueError(f"Could not find any of {preferred}")


def _to_timestamp_series(df: pd.DataFrame, ts_col: str) -> pd.Series:
    if ts_col not in df.columns:
        return pd.to_datetime(np.arange(len(df)), unit="s")
    s = pd.to_datetime(df[ts_col], errors="coerce")
    if s.isna().all():
        return pd.to_datetime(np.arange(len(df)), unit="s")
    return s.fillna(method="ffill").fillna(method="bfill")


def build_graphs_from_dataframe(
    df: pd.DataFrame,
    label_col: str,
    src_col: str,
    dst_col: str,
    ts_col: str,
    window_seconds: int = 5,
    step_seconds: int = 1,
    max_graphs: int | None = None,
) -> list[Data]:
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' missing")

    timestamp = _to_timestamp_series(df, ts_col)
    df = df.copy()
    df["_ts"] = timestamp
    df = df.sort_values("_ts").reset_index(drop=True)

    edge_feature_cols = select_edge_feature_columns(df, exclude_columns=[label_col])

    start = df["_ts"].min()
    end = df["_ts"].max()

    # Guardrail for very large files: adapt step to cap number of windows/graphs.
    if max_graphs is not None and max_graphs > 0:
        total_span_seconds = max(1, int((end - start).total_seconds()))
        estimated_windows = max(1, total_span_seconds // max(1, step_seconds))
        if estimated_windows > max_graphs:
            adaptive_step = max(step_seconds, int(math.ceil(total_span_seconds / max_graphs)))
            print(
                f"Adjusting step_seconds from {step_seconds} to {adaptive_step} "
                f"to cap windows near {max_graphs}."
            )
            step_seconds = adaptive_step

    graphs: list[Data] = []
    t = start
    while t <= end:
        t_end = t + timedelta(seconds=window_seconds)
        w = df[(df["_ts"] >= t) & (df["_ts"] < t_end)]
        if w.empty:
            t = t + timedelta(seconds=step_seconds)
            continue

        src_vals = w[src_col].astype(str)
        dst_vals = w[dst_col].astype(str)
        nodes = sorted(set(src_vals.tolist() + dst_vals.tolist()))

        node_features_df = add_basic_node_features(
            w,
            src_col=src_col,
            dst_col=dst_col,
            bytes_col=("Flow Bytes/s" if "Flow Bytes/s" in w.columns else None),
            duration_col=("Flow Duration" if "Flow Duration" in w.columns else None),
        )
        node_features_df = node_features_df.set_index("node").reindex(nodes).fillna(0.0)
        x = torch.tensor(node_features_df.values, dtype=torch.float32)

        src_codes = pd.Categorical(src_vals, categories=nodes).codes
        dst_codes = pd.Categorical(dst_vals, categories=nodes).codes
        edge_index_np = np.stack([src_codes, dst_codes], axis=0)
        edge_index = torch.from_numpy(edge_index_np).to(torch.long).contiguous()

        if edge_feature_cols:
            edge_attr_np = w[edge_feature_cols].to_numpy(dtype=np.float32, copy=False)
        else:
            edge_attr_np = np.zeros((len(w), 0), dtype=np.float32)
        edge_attr_t = torch.from_numpy(edge_attr_np)

        window_labels = w[label_col].astype(int).to_numpy()
        attack_count = int(window_labels.sum())
        benign_count = int(len(window_labels) - attack_count)
        label = int(attack_count > benign_count)
        y = torch.tensor([label], dtype=torch.long)

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_t, y=y)
        graph.num_nodes = x.shape[0]
        graphs.append(graph)

        t = t + timedelta(seconds=step_seconds)

    return graphs


def run(
    input_glob: str,
    output_path: str,
    label_col: str = "Label",
    window_seconds: int = 5,
    step_seconds: int = 1,
    max_graphs_per_file: int = 5000,
) -> None:
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No cleaned files found by: {input_glob}")

    all_graphs: list[Data] = []
    for file_path in files:
        df = pd.read_csv(file_path)
        
        # Try to find IP columns first (for packet-level data like InSDN)
        src_col = None
        dst_col = None
        try:
            src_col = _find_col(df, ["Source IP", "src_ip", "Src IP"])
            dst_col = _find_col(df, ["Destination IP", "dst_ip", "Dst IP"])
        except ValueError:
            # If IP columns not found, use flow-based graph construction
            # Create synthetic nodes from port numbers or use flow indices
            src_col = "src_node"
            dst_col = "dst_node"
            
            # Create synthetic source/destination nodes based on ports
            df[src_col] = "src_" + df.get("Source Port", df.index.astype(str)).astype(str)
            df[dst_col] = "dst_" + df.get("Destination Port", df.index.astype(str)).astype(str)
        
        ts_col = _find_col(df, ["Timestamp", "timestamp", "Flow Start Time"]) if any(
            c in {x.lower() for x in df.columns} for c in ["timestamp", "flow start time"]
        ) else "Timestamp"

        graphs = build_graphs_from_dataframe(
            df,
            label_col=label_col,
            src_col=src_col,
            dst_col=dst_col,
            ts_col=ts_col,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
            max_graphs=max_graphs_per_file,
        )
        all_graphs.extend(graphs)
        print(f"{os.path.basename(file_path)} -> {len(graphs)} graph snapshots")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(all_graphs, output_path)
    print(f"Saved {len(all_graphs)} total graphs to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sliding-window graph snapshots from cleaned CSVs.")
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--label-col", default="Label")
    parser.add_argument("--window-seconds", type=int, default=5)
    parser.add_argument("--step-seconds", type=int, default=1)
    parser.add_argument("--max-graphs-per-file", type=int, default=5000)
    args = parser.parse_args()

    run(
        input_glob=args.input_glob,
        output_path=args.output_path,
        label_col=args.label_col,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        max_graphs_per_file=args.max_graphs_per_file,
    )


if __name__ == "__main__":
    main()
