from __future__ import annotations

import argparse
import glob
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
) -> list[Data]:
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' missing")

    timestamp = _to_timestamp_series(df, ts_col)
    df = df.copy()
    df["_ts"] = timestamp
    df = df.sort_values("_ts").reset_index(drop=True)

    edge_feature_cols = select_edge_feature_columns(df)

    start = df["_ts"].min()
    end = df["_ts"].max()

    graphs: list[Data] = []
    t = start
    while t <= end:
        t_end = t + timedelta(seconds=window_seconds)
        w = df[(df["_ts"] >= t) & (df["_ts"] < t_end)]
        if w.empty:
            t = t + timedelta(seconds=step_seconds)
            continue

        nodes = sorted(set(w[src_col].astype(str).tolist() + w[dst_col].astype(str).tolist()))
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        node_features_df = add_basic_node_features(
            w,
            src_col=src_col,
            dst_col=dst_col,
            bytes_col=("Flow Bytes/s" if "Flow Bytes/s" in w.columns else None),
            duration_col=("Flow Duration" if "Flow Duration" in w.columns else None),
        )
        node_features_df = node_features_df.set_index("node").reindex(nodes).fillna(0.0)
        x = torch.tensor(node_features_df.values, dtype=torch.float32)

        edges = []
        edge_attr = []
        for _, row in w.iterrows():
            s_idx = node_to_idx[str(row[src_col])]
            d_idx = node_to_idx[str(row[dst_col])]
            edges.append([s_idx, d_idx])
            edge_attr.append([float(row[c]) if c in w.columns else 0.0 for c in edge_feature_cols])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)

        label = int((w[label_col].astype(int) == 1).any())
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
) -> None:
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No cleaned files found by: {input_glob}")

    all_graphs: list[Data] = []
    for file_path in files:
        df = pd.read_csv(file_path)
        src_col = _find_col(df, ["Source IP", "src_ip", "Src IP"])
        dst_col = _find_col(df, ["Destination IP", "dst_ip", "Dst IP"])
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
    args = parser.parse_args()

    run(
        input_glob=args.input_glob,
        output_path=args.output_path,
        label_col=args.label_col,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
    )


if __name__ == "__main__":
    main()
