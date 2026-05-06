from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _resolve_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    mapping = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in mapping:
            return mapping[c.lower()]
    return None


def select_edge_feature_columns(df: pd.DataFrame, max_features: int = 12) -> list[str]:
    preferred = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Fwd Packet Length Mean",
        "Bwd Packet Length Mean",
        "SYN Flag Count",
        "FIN Flag Count",
        "RST Flag Count",
        "Protocol",
    ]
    selected = [c for c in preferred if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(selected) >= max_features:
        return selected[:max_features]

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in selected]
    selected.extend(numeric_cols[: max(0, max_features - len(selected))])
    return selected


def add_basic_node_features(
    df: pd.DataFrame,
    src_col: str,
    dst_col: str,
    bytes_col: str | None = None,
    duration_col: str | None = None,
) -> pd.DataFrame:
    # Vectorized aggregation to avoid per-node filtering (which is O(n^2) for many rows)
    temp = df.copy()
    temp[src_col] = temp[src_col].astype(str)
    temp[dst_col] = temp[dst_col].astype(str)

    nodes = pd.Index(temp[src_col].tolist() + temp[dst_col].tolist()).unique().tolist()

    # Bytes sent/recv: sum if bytes_col present, otherwise use counts
    if bytes_col and bytes_col in temp.columns:
        bytes_sent_s = temp.groupby(src_col)[bytes_col].sum()
        bytes_recv_s = temp.groupby(dst_col)[bytes_col].sum()
    else:
        bytes_sent_s = temp.groupby(src_col).size()
        bytes_recv_s = temp.groupby(dst_col).size()

    # Fan-out: number of unique destinations per source
    fan_out_s = temp.groupby(src_col)[dst_col].nunique()

    # Average duration per source (if available)
    if duration_col and duration_col in temp.columns:
        avg_duration_s = temp.groupby(src_col)[duration_col].mean()
    else:
        avg_duration_s = pd.Series(dtype=float)

    records = []
    for node in nodes:
        bytes_sent = float(bytes_sent_s.get(node, 0))
        bytes_recv = float(bytes_recv_s.get(node, 0))
        fan_out = int(fan_out_s.get(node, 0))
        avg_duration = float(avg_duration_s.get(node, 0)) if not avg_duration_s.empty else 0.0
        records.append(
            {
                "node": str(node),
                "bytes_sent": bytes_sent,
                "bytes_recv": bytes_recv,
                "fan_out": fan_out,
                "avg_duration": 0.0 if np.isnan(avg_duration) else avg_duration,
            }
        )

    return pd.DataFrame(records)
