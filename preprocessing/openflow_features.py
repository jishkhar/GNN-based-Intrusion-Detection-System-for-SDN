"""OpenFlow-compatible flow records and graph features.

This module is the single source of truth for features, used both offline
(CICIDS2017 / InSDN CSVs) and live (OpenFlow flow-stats replies from the
controller), so the model never sees features at inference time that it could
not have been trained on.

A *flow record* is one unidirectional flow entry, as an OpenFlow switch counts
it: ``src_ip, dst_ip, src_port, dst_port, ip_proto, packets, bytes, duration``
(+ ``timestamp`` and optional ``label`` / ``attack`` columns). CICFlowMeter rows
are bidirectional, so each row becomes a forward record and, if any backward
packets exist, a reverse record.

See ``Docs/feature_schema_phase2.md`` for the full schema and dataset mapping.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RECORD_COLUMNS = [
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "ip_proto",
    "packets",
    "bytes",
    "duration",
]

PROTO_TCP, PROTO_UDP, PROTO_ICMP = 6, 17, 1

# Per-packet L2+L3+L4 header bytes. OpenFlow byte counters include the full
# Ethernet frame, while CICFlowMeter "TotLen" columns count payload only.
_HEADER_BYTES = {PROTO_TCP: 14 + 20 + 20, PROTO_UDP: 14 + 20 + 8}
_DEFAULT_HEADER_BYTES = 14 + 20 + 8

# Duration floor (seconds) so single-packet flows do not get infinite rates.
MIN_DURATION = 1e-3

EDGE_FEATURE_NAMES = [
    "packets",
    "bytes",
    "duration",
    "pkt_rate",
    "byte_rate",
    "mean_pkt_size",
    "proto_tcp",
    "proto_udp",
    "proto_icmp",
    "proto_other",
    "dport_well_known",
    "dport_registered",
    "dport_ephemeral",
    "sport_well_known",
    "sport_registered",
    "sport_ephemeral",
]

NODE_FEATURE_NAMES = [
    "out_flows",
    "in_flows",
    "out_packets",
    "in_packets",
    "out_bytes",
    "in_bytes",
    "out_degree",
    "in_degree",
    "unique_dst_ports",
    "dst_port_entropy",
    "unique_src_ports_in",
    "mean_out_duration",
    "out_tcp_frac",
    "out_udp_frac",
    "out_icmp_frac",
    "out_other_frac",
]

EDGE_DIM = len(EDGE_FEATURE_NAMES)
NODE_DIM = len(NODE_FEATURE_NAMES)


@dataclass(frozen=True)
class DatasetColumns:
    """Column names of a CICFlowMeter-style CSV."""

    src_ip: str
    dst_ip: str
    src_port: str
    dst_port: str
    protocol: str
    timestamp: str
    duration_us: str
    fwd_packets: str
    bwd_packets: str
    fwd_bytes: str
    bwd_bytes: str


INSDN_COLUMNS = DatasetColumns(
    src_ip="Src IP",
    dst_ip="Dst IP",
    src_port="Src Port",
    dst_port="Dst Port",
    protocol="Protocol",
    timestamp="Timestamp",
    duration_us="Flow Duration",
    fwd_packets="Tot Fwd Pkts",
    bwd_packets="Tot Bwd Pkts",
    fwd_bytes="TotLen Fwd Pkts",
    bwd_bytes="TotLen Bwd Pkts",
)

CICIDS_COLUMNS = DatasetColumns(
    src_ip="Source IP",
    dst_ip="Destination IP",
    src_port="Source Port",
    dst_port="Destination Port",
    protocol="Protocol",
    timestamp="Timestamp",
    duration_us="Flow Duration",
    fwd_packets="Total Fwd Packets",
    bwd_packets="Total Backward Packets",
    fwd_bytes="Total Length of Fwd Packets",
    bwd_bytes="Total Length of Bwd Packets",
)


def detect_columns(df: pd.DataFrame) -> DatasetColumns:
    """Pick the column mapping whose columns all exist in ``df``."""
    for mapping in (INSDN_COLUMNS, CICIDS_COLUMNS):
        if all(col in df.columns for col in mapping.__dict__.values()):
            return mapping
    raise ValueError(
        "CSV lacks IP/port/timestamp columns required for topology graphs. "
        "Use InSDN or the CICIDS2017 TrafficLabelling (GeneratedLabelledFlows) CSVs."
    )


def _frame_bytes(payload: np.ndarray, packets: np.ndarray, proto: np.ndarray) -> np.ndarray:
    header = np.full(proto.shape, _DEFAULT_HEADER_BYTES, dtype=np.float64)
    for p, size in _HEADER_BYTES.items():
        header[proto == p] = size
    return np.clip(payload, 0, None) + packets * header


def dataset_to_flow_records(
    df: pd.DataFrame,
    columns: DatasetColumns | None = None,
    timestamps: pd.Series | None = None,
    extra_columns: tuple[str, ...] = ("Label", "Attack"),
    normalize_direction: bool = True,
) -> pd.DataFrame:
    """Convert bidirectional CICFlowMeter rows into unidirectional flow records.

    ``flow_idx`` links both directions back to the source row. ``is_reverse``
    marks the *response* direction (server -> client). CICFlowMeter's "forward"
    direction is whichever side it saw first, which is sometimes the server's
    reply (InSDN has tens of thousands of such victim -> attacker rows). With
    ``normalize_direction`` the side with the lower port is taken as the server,
    so only the initiator of an attack flow is labelled as an attacker. The
    records themselves (and their features) are the same either way.
    Extra columns (labels) are copied to both directions.
    """
    c = columns or detect_columns(df)
    proto = pd.to_numeric(df[c.protocol], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    fwd_pkts = pd.to_numeric(df[c.fwd_packets], errors="coerce").fillna(0).to_numpy(np.float64)
    bwd_pkts = pd.to_numeric(df[c.bwd_packets], errors="coerce").fillna(0).to_numpy(np.float64)
    fwd_payload = pd.to_numeric(df[c.fwd_bytes], errors="coerce").fillna(0).to_numpy(np.float64)
    bwd_payload = pd.to_numeric(df[c.bwd_bytes], errors="coerce").fillna(0).to_numpy(np.float64)
    duration = np.clip(pd.to_numeric(df[c.duration_us], errors="coerce").fillna(0).to_numpy(np.float64), 0, None) / 1e6
    src_port = pd.to_numeric(df[c.src_port], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    dst_port = pd.to_numeric(df[c.dst_port], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    ts = timestamps if timestamps is not None else df[c.timestamp]

    base = {
        "flow_idx": np.arange(len(df)),
        "timestamp": np.asarray(ts),
        "ip_proto": proto,
        "duration": duration,
    }
    fwd = pd.DataFrame(
        {
            **base,
            "src_ip": df[c.src_ip].astype(str).to_numpy(),
            "dst_ip": df[c.dst_ip].astype(str).to_numpy(),
            "src_port": src_port,
            "dst_port": dst_port,
            "packets": fwd_pkts,
            "bytes": _frame_bytes(fwd_payload, fwd_pkts, proto),
            "is_reverse": False,
        }
    )
    rev = pd.DataFrame(
        {
            **base,
            "src_ip": df[c.dst_ip].astype(str).to_numpy(),
            "dst_ip": df[c.src_ip].astype(str).to_numpy(),
            "src_port": dst_port,
            "dst_port": src_port,
            "packets": bwd_pkts,
            "bytes": _frame_bytes(bwd_payload, bwd_pkts, proto),
            "is_reverse": True,
        }
    )
    for col in extra_columns:
        if col in df.columns:
            fwd[col] = df[col].to_numpy()
            rev[col] = df[col].to_numpy()

    if normalize_direction:
        # CICFlowMeter started the flow at the server's reply: swap which record is the response.
        flipped = (src_port < dst_port) & np.isin(proto, [PROTO_TCP, PROTO_UDP])
        fwd["is_reverse"] = flipped
        rev["is_reverse"] = ~flipped
    rev = rev[bwd_pkts > 0]
    records = pd.concat([fwd, rev], ignore_index=True)
    # Keep both directions of a flow adjacent and in capture order.
    return records.sort_values(["flow_idx", "is_reverse"], kind="stable").reset_index(drop=True)


def openflow_stats_to_flow_records(stats: list[dict]) -> pd.DataFrame:
    """Convert parsed OpenFlow flow-stats entries into flow records.

    Each entry needs ``ipv4_src``, ``ipv4_dst``, ``ip_proto``, ``packet_count``,
    ``byte_count`` and ``duration_sec`` (+ optional ``duration_nsec``, L4 ports as
    ``tcp_src``/``udp_src``/``tp_src`` etc.). Counters should already be deltas
    over the collection window when used for windowed detection.
    """
    rows = []
    for s in stats:
        proto = int(s.get("ip_proto", 0) or 0)
        src_port = s.get("tcp_src", s.get("udp_src", s.get("tp_src", 0))) or 0
        dst_port = s.get("tcp_dst", s.get("udp_dst", s.get("tp_dst", 0))) or 0
        rows.append(
            {
                "src_ip": str(s["ipv4_src"]),
                "dst_ip": str(s["ipv4_dst"]),
                "src_port": int(src_port),
                "dst_port": int(dst_port),
                "ip_proto": proto,
                "packets": float(s.get("packet_count", 0)),
                "bytes": float(s.get("byte_count", 0)),
                "duration": float(s.get("duration_sec", 0)) + float(s.get("duration_nsec", 0)) / 1e9,
                "timestamp": s.get("timestamp"),
            }
        )
    return pd.DataFrame(rows, columns=RECORD_COLUMNS + ["timestamp"])


def chunk_records(records: pd.DataFrame, chunk_size: int) -> list[pd.DataFrame]:
    """Split a (live) window into consecutive training-sized graphs.

    Shared by the inference engine and the Mininet labeller, so the model is
    trained on exactly the graph sizes it sees live.
    """
    return [records.iloc[i : i + chunk_size] for i in range(0, len(records), chunk_size)]


def _port_buckets(ports: np.ndarray, has_ports: np.ndarray) -> np.ndarray:
    well_known = (ports < 1024) & has_ports
    registered = (ports >= 1024) & (ports < 49152) & has_ports
    ephemeral = (ports >= 49152) & has_ports
    return np.stack([well_known, registered, ephemeral], axis=1).astype(np.float32)


def edge_features(records: pd.DataFrame) -> np.ndarray:
    """Per-record edge features, shape ``[E, EDGE_DIM]`` (raw, un-normalised)."""
    packets = records["packets"].to_numpy(np.float64)
    nbytes = records["bytes"].to_numpy(np.float64)
    duration = records["duration"].to_numpy(np.float64)
    proto = records["ip_proto"].to_numpy(np.int64)
    safe_duration = np.maximum(duration, MIN_DURATION)

    is_tcp, is_udp, is_icmp = proto == PROTO_TCP, proto == PROTO_UDP, proto == PROTO_ICMP
    is_other = ~(is_tcp | is_udp | is_icmp)
    has_ports = is_tcp | is_udp

    numeric = np.stack(
        [
            packets,
            nbytes,
            duration,
            packets / safe_duration,
            nbytes / safe_duration,
            nbytes / np.maximum(packets, 1.0),
        ],
        axis=1,
    ).astype(np.float32)
    protos = np.stack([is_tcp, is_udp, is_icmp, is_other], axis=1).astype(np.float32)
    dports = _port_buckets(records["dst_port"].to_numpy(np.int64), has_ports)
    sports = _port_buckets(records["src_port"].to_numpy(np.int64), has_ports)
    return np.concatenate([numeric, protos, dports, sports], axis=1)


def _count_unique_pairs(a: np.ndarray, b: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique (a, b) pairs -> (a values, pair counts, unique-b count per a)."""
    pairs, counts = np.unique(np.stack([a, b], axis=1), axis=0, return_counts=True)
    return pairs[:, 0], counts, np.bincount(pairs[:, 0], minlength=n)


def factorize_nodes(records: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Map IPs to contiguous node ids; returns (src ids, dst ids, node IP list)."""
    codes, uniques = pd.factorize(
        np.concatenate([records["src_ip"].to_numpy(str), records["dst_ip"].to_numpy(str)])
    )
    e = len(records)
    return codes[:e].astype(np.int64), codes[e:].astype(np.int64), [str(u) for u in uniques]


def node_features(records: pd.DataFrame, src: np.ndarray, dst: np.ndarray, num_nodes: int) -> np.ndarray:
    """Per-node window aggregates, shape ``[N, NODE_DIM]`` (raw, un-normalised)."""
    n = num_nodes
    packets = records["packets"].to_numpy(np.float64)
    nbytes = records["bytes"].to_numpy(np.float64)
    duration = records["duration"].to_numpy(np.float64)
    proto = records["ip_proto"].to_numpy(np.int64)
    dst_port = records["dst_port"].to_numpy(np.int64)
    src_port = records["src_port"].to_numpy(np.int64)

    out_flows = np.bincount(src, minlength=n).astype(np.float64)
    in_flows = np.bincount(dst, minlength=n).astype(np.float64)
    safe_out = np.maximum(out_flows, 1.0)

    _, _, out_degree = _count_unique_pairs(src, dst, n)
    _, _, in_degree = _count_unique_pairs(dst, src, n)
    port_owner, port_counts, unique_dst_ports = _count_unique_pairs(src, dst_port, n)
    _, _, unique_src_ports_in = _count_unique_pairs(dst, src_port, n)

    # Shannon entropy (bits) of each node's outgoing destination-port distribution.
    p = port_counts / safe_out[port_owner]
    entropy = np.bincount(port_owner, weights=-p * np.log2(p), minlength=n)

    def frac(mask: np.ndarray) -> np.ndarray:
        return np.bincount(src, weights=mask.astype(np.float64), minlength=n) / safe_out

    is_tcp, is_udp, is_icmp = proto == PROTO_TCP, proto == PROTO_UDP, proto == PROTO_ICMP
    features = np.stack(
        [
            out_flows,
            in_flows,
            np.bincount(src, weights=packets, minlength=n),
            np.bincount(dst, weights=packets, minlength=n),
            np.bincount(src, weights=nbytes, minlength=n),
            np.bincount(dst, weights=nbytes, minlength=n),
            out_degree,
            in_degree,
            unique_dst_ports,
            entropy,
            unique_src_ports_in,
            np.bincount(src, weights=duration, minlength=n) / safe_out,
            frac(is_tcp),
            frac(is_udp),
            frac(is_icmp),
            frac(~(is_tcp | is_udp | is_icmp)),
        ],
        axis=1,
    )
    return features.astype(np.float32)
