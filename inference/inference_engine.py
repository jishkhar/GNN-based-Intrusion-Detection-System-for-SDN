"""Real-time GNN inference on a window of live flow records.

Uses the exported TorchScript model (``models/export.py``) and the same feature
code as training (``preprocessing.openflow_features``). A live window can hold
far more flows than a training window (e.g. thousands during a DDoS), so it is
split into training-sized chunks that run as one batch. The window score is the
maximum over chunks; a host's score is its *mean* over the chunks it appears in.
Taking the maximum per host would give a benign host dozens of chances per
window (and more every poll) to be flagged once, which live testing showed
leads to benign clients being blocked.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from models.export import load_exported
from preprocessing.openflow_features import chunk_records, edge_features, factorize_nodes, node_features


@dataclass
class Prediction:
    is_attack: bool
    attack_type: str
    confidence: float  # P(attack) for the window, = 1 - P(Benign)
    class_probabilities: dict
    node_scores: dict  # ip -> P(attacker)
    flagged_ips: list  # hosts above the node threshold, highest first
    victims: list  # most-contacted destinations of flagged hosts
    flagged_flow_counts: dict  # flagged ip -> outgoing flow records in the window
    num_flows: int
    num_hosts: int
    num_chunks: int
    latency_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class InferenceEngine:
    def __init__(self, model_path: str, device: str | None = None, chunk_records: int | None = None,
                 node_aggregation: str = "mean") -> None:
        if node_aggregation not in ("mean", "max"):
            raise ValueError("node_aggregation must be 'mean' or 'max'")
        self.node_aggregation = node_aggregation
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.bundle = load_exported(model_path, str(self.device))
        stats = self.bundle["feature_stats"]
        self.node_mean = torch.tensor(stats["node_mean"], device=self.device)
        self.node_std = torch.tensor(stats["node_std"], device=self.device)
        self.edge_mean = torch.tensor(stats["edge_mean"], device=self.device)
        self.edge_std = torch.tensor(stats["edge_std"], device=self.device)
        self.classes: list[str] = self.bundle["classes"]
        self.benign_index = self.classes.index("Benign")
        self.window_threshold = float(self.bundle["thresholds"]["window"])
        self.node_threshold = float(self.bundle["thresholds"]["node"])
        meta = self.bundle.get("graph_meta", {})
        # A training window of N bidirectional flows has up to 2N unidirectional records.
        self.chunk_records = chunk_records or int(2 * meta.get("window_size", 100))
        self.warmup()

    @staticmethod
    def _signed_log1p(t: torch.Tensor) -> torch.Tensor:
        return torch.sign(t) * torch.log1p(torch.abs(t))

    def warmup(self, rounds: int = 5) -> None:
        """Run synthetic windows of varying size before serving traffic.

        TorchScript's profiling executor optimises the graph during its first
        few calls (~200 ms each); paying that here keeps live p99 latency low.
        """
        rng = np.random.default_rng(0)
        for i in range(rounds):
            n = 20 * (i + 1) ** 2
            records = pd.DataFrame(
                {
                    "src_ip": [f"10.0.{k % 7}.{k % 50}" for k in range(n)],
                    "dst_ip": [f"10.1.0.{k % 5}" for k in range(n)],
                    "src_port": rng.integers(1024, 65535, n),
                    "dst_port": rng.choice([22, 80, 443, 53], n),
                    "ip_proto": rng.choice([6, 17], n),
                    "packets": rng.integers(1, 100, n).astype(float),
                    "bytes": rng.integers(60, 10_000, n).astype(float),
                    "duration": rng.random(n),
                }
            )
            self.predict(records)

    def _chunk_tensors(self, records: pd.DataFrame):
        src, dst, ips = factorize_nodes(records)
        x = node_features(records, src, dst, len(ips))
        e = edge_features(records)
        return x, np.stack([src, dst]), e, ips

    @torch.no_grad()
    def predict(self, records: pd.DataFrame) -> Prediction:
        t0 = time.perf_counter()
        n = len(records)
        if n == 0:
            return Prediction(False, "Benign", 0.0, {}, {}, [], [], {}, 0, 0, 0, {"total": 0.0})

        xs, eis, eas, ip_lists, batch = [], [], [], [], []
        offset = 0
        chunks = chunk_records(records, self.chunk_records)
        for ci, chunk in enumerate(chunks):
            x, ei, ea, ips = self._chunk_tensors(chunk)
            xs.append(x)
            eis.append(ei + offset)
            eas.append(ea)
            ip_lists.append(ips)
            batch.append(np.full(len(ips), ci))
            offset += len(ips)
        t1 = time.perf_counter()

        x = torch.from_numpy(np.concatenate(xs)).to(self.device)
        edge_index = torch.from_numpy(np.concatenate(eis, axis=1)).long().to(self.device)
        edge_attr = torch.from_numpy(np.concatenate(eas)).to(self.device)
        batch_t = torch.from_numpy(np.concatenate(batch)).long().to(self.device)
        x = (self._signed_log1p(x) - self.node_mean) / self.node_std
        edge_attr = (self._signed_log1p(edge_attr) - self.edge_mean) / self.edge_std
        t2 = time.perf_counter()

        graph_logits, node_logits = self.model(x, edge_index, edge_attr, batch_t)
        probs = torch.softmax(graph_logits, dim=1).cpu().numpy()
        node_p = torch.softmax(node_logits, dim=1)[:, 1].cpu().numpy()
        t3 = time.perf_counter()

        attack_p = 1.0 - probs[:, self.benign_index]
        worst = int(attack_p.argmax())
        confidence = float(attack_p[worst])
        chunk_probs = probs[worst]
        attack_only = chunk_probs.copy()
        attack_only[self.benign_index] = -1.0
        is_attack = confidence >= self.window_threshold
        attack_type = self.classes[int(attack_only.argmax())] if is_attack else "Benign"

        all_ips = pd.Series([ip for ips in ip_lists for ip in ips])
        grouped = pd.Series(node_p).groupby(all_ips.values)
        agg = grouped.mean() if self.node_aggregation == "mean" else grouped.max()
        node_scores: dict[str, float] = {str(ip): float(s) for ip, s in agg.items()}
        flagged = sorted(
            (ip for ip, s in node_scores.items() if s >= self.node_threshold), key=lambda ip: -node_scores[ip]
        )
        victims: list[str] = []
        flow_counts: dict[str, int] = {}
        if flagged:
            from_flagged = records[records["src_ip"].isin(flagged)]
            victims = [ip for ip in from_flagged["dst_ip"].value_counts().index[:3] if ip not in flagged]
            flow_counts = {str(k): int(v) for k, v in from_flagged["src_ip"].value_counts().items()}
        t4 = time.perf_counter()

        return Prediction(
            is_attack=is_attack,
            attack_type=attack_type,
            confidence=confidence,
            class_probabilities={c: float(p) for c, p in zip(self.classes, chunk_probs)},
            node_scores=node_scores,
            flagged_ips=flagged,
            victims=victims,
            flagged_flow_counts=flow_counts,
            num_flows=n,
            num_hosts=len(node_scores),
            num_chunks=len(chunks),
            latency_ms={
                "features": (t1 - t0) * 1e3,
                "graph": (t2 - t1) * 1e3,
                "model": (t3 - t2) * 1e3,
                "decision": (t4 - t3) * 1e3,
                "total": (t4 - t0) * 1e3,
            },
        )
