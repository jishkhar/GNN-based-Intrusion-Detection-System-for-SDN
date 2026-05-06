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
    records = []
    nodes = pd.Index(df[src_col].astype(str).tolist() + df[dst_col].astype(str).tolist()).unique().tolist()

    for node in nodes:
        sent = df[df[src_col].astype(str) == node]
        recv = df[df[dst_col].astype(str) == node]
        bytes_sent = float(sent[bytes_col].sum()) if bytes_col and bytes_col in df.columns else float(len(sent))
        bytes_recv = float(recv[bytes_col].sum()) if bytes_col and bytes_col in df.columns else float(len(recv))
        fan_out = int(sent[dst_col].nunique())
        avg_duration = (
            float(sent[duration_col].mean())
            if duration_col and duration_col in df.columns and not sent.empty
            else 0.0
        )
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
