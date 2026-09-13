"""IDS service: collector -> GNN -> classifier -> mitigation, plus dashboard state.

One ``IDSService`` instance lives inside the FastAPI app. Each flow-stats upload
from the controller is processed synchronously (the whole path takes tens of
milliseconds) and the response carries any pending mitigation actions back, so
blocking a host costs no extra controller round-trip.
"""
from __future__ import annotations

import csv
import os
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from controller.flow_collector import FlowCollector
from inference.classifier import ATTACK, BENIGN, AlertClassifier
from inference.inference_engine import InferenceEngine
from mitigation.mitigation_engine import MitigationEngine


@dataclass
class ServiceConfig:
    model_path: str = "models/gat_ids.pt"
    window_seconds: float = 10.0
    attack_threshold: float = 0.85
    suspicious_threshold: float = 0.5
    node_threshold: float | None = None
    cooldown_seconds: float = 30.0
    whitelist: tuple = ()
    log_dir: str = "logs"
    mitigation_enabled: bool = True
    idle_timeout: int = 60
    hard_timeout: int = 300
    victim_rate_pps: int = 200
    scanner_rate_pps: int = 20
    confirm_windows: int = 2
    node_aggregation: str = "mean"
    spoof_threshold: int = 20
    heavy_hitter_flows: int = 5
    record_flows_path: str | None = None  # append every ingested entry (for labelled Mininet data)
    device: str | None = None

    @classmethod
    def from_dict(cls, values: dict | None) -> "ServiceConfig":
        values = dict(values or {})
        if "whitelist" in values:
            values["whitelist"] = tuple(values["whitelist"] or ())
        known = {k: v for k, v in values.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class IDSService:
    def __init__(self, config: ServiceConfig, engine: InferenceEngine | None = None) -> None:
        self.config = config
        self.engine = engine or InferenceEngine(config.model_path, device=config.device,
                                                node_aggregation=config.node_aggregation)
        self.collector = FlowCollector(window_seconds=config.window_seconds)
        self.classifier = AlertClassifier(
            attack_threshold=config.attack_threshold,
            suspicious_threshold=config.suspicious_threshold,
            node_threshold=config.node_threshold,
            whitelist=list(config.whitelist),
            cooldown_seconds=config.cooldown_seconds,
            confirm_windows=config.confirm_windows,
        )
        self.mitigation = MitigationEngine(
            log_dir=config.log_dir,
            idle_timeout=config.idle_timeout,
            hard_timeout=config.hard_timeout,
            victim_rate_pps=config.victim_rate_pps,
            scanner_rate_pps=config.scanner_rate_pps,
            spoof_threshold=config.spoof_threshold,
            heavy_hitter_flows=config.heavy_hitter_flows,
            whitelist=list(config.whitelist),
            enabled=config.mitigation_enabled,
        )
        self._lock = threading.Lock()
        self.alerts: deque = deque(maxlen=500)
        self.events: deque = deque(maxlen=1000)  # (id, dict) for the live SSE feed
        self._event_id = 0
        self.latencies: deque = deque(maxlen=1000)
        self.last_prediction: dict | None = None
        self.last_topology: dict = {"nodes": [], "links": []}
        self.uploads = 0
        self.started_at = time.time()
        self._flow_counts: deque = deque(maxlen=120)  # (timestamp, entries)

    # ----------------------------------------------------------------- ingest
    def process_flows(self, entries: list[dict], removed_cookies: list[int] | None = None, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        t0 = time.perf_counter()
        with self._lock:
            self.uploads += 1
            if removed_cookies:
                self.mitigation.forget_cookies(removed_cookies, now)
            if self.config.record_flows_path:
                self._record(entries, now)
            active = self.collector.ingest(entries, now)
            self._flow_counts.append((now, len(entries)))
            records = self.collector.get_current_window(now)
            prediction = self.engine.predict(records)
            decision = self.classifier.decide(prediction, self.engine.node_threshold, now)
            created = self.mitigation.handle(decision)
            actions = self.mitigation.drain_outbox()
            total_ms = (time.perf_counter() - t0) * 1e3
            latency = {**prediction.latency_ms, "service_total": total_ms}
            self.latencies.append(latency)
            self.last_prediction = {**prediction.to_dict(), "decision": decision.to_dict()}
            self.last_topology = self._topology(records, prediction.node_scores, decision)

            if decision.level != BENIGN:
                alert = {
                    "timestamp": now,
                    "level": decision.level,
                    "attack_type": decision.attack_type,
                    "confidence": decision.confidence,
                    "attackers": decision.attackers[:50],
                    "num_attackers": len(decision.attackers),
                    "new_attackers": decision.new_attackers[:50],
                    "victims": decision.victims,
                    "actions": [a["rule_id"] for a in created],
                    "latency_ms": total_ms,
                }
                if decision.level != ATTACK or decision.is_new_alert:
                    self.alerts.append(alert)
                    self._emit("alert", alert)
            for action in actions:
                self._emit("mitigation", action)
            self._emit(
                "window",
                {
                    "timestamp": now,
                    "active_flows": len(records),
                    "hosts": prediction.num_hosts,
                    "confidence": prediction.confidence,
                    "level": decision.level,
                    "latency_ms": total_ms,
                },
            )
        return {
            "status": "ok",
            "active_flows": active,
            "window_flows": len(records),
            "level": decision.level,
            "attack_type": decision.attack_type,
            "confidence": decision.confidence,
            "actions": actions,
            "latency_ms": total_ms,
        }

    # ------------------------------------------------------------ dashboard
    def _emit(self, kind: str, payload: dict) -> None:
        self._event_id += 1
        self.events.append((self._event_id, {"type": kind, **payload}))

    def events_since(self, last_id: int) -> list[tuple[int, dict]]:
        with self._lock:
            return [(i, e) for i, e in self.events if i > last_id]

    def latency_summary(self) -> dict:
        with self._lock:
            values = [l["service_total"] for l in self.latencies]
            model = [l["model"] for l in self.latencies]
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "p50_ms": float(np.percentile(values, 50)),
            "p90_ms": float(np.percentile(values, 90)),
            "p99_ms": float(np.percentile(values, 99)),
            "model_p50_ms": float(np.percentile(model, 50)),
        }

    def status(self) -> dict:
        now = time.time()
        with self._lock:
            recent = [(t, n) for t, n in self._flow_counts if now - t <= 30]
            span = max(1e-9, (now - recent[0][0])) if len(recent) > 1 else 0
            rate = sum(n for _, n in recent) / span if span else 0.0
            return {
                "uptime_s": now - self.started_at,
                "uploads": self.uploads,
                "tracked_flows": len(self.collector),
                "flow_entries_per_s": rate,
                "alerts": len(self.alerts),
                "active_rules": len(self.mitigation.active_rules(now)),
                "mitigation_enabled": self.mitigation.enabled,
                "window_seconds": self.config.window_seconds,
                "thresholds": {
                    "attack": self.config.attack_threshold,
                    "suspicious": self.config.suspicious_threshold,
                    "node": self.config.node_threshold or self.engine.node_threshold,
                    "model_window": self.engine.window_threshold,
                },
                "last_prediction": self.last_prediction,
            }

    @staticmethod
    def _topology(records, node_scores: dict, decision, max_links: int = 300) -> dict:
        attackers, victims = set(decision.attackers), set(decision.victims)
        nodes = [
            {
                "id": ip,
                "score": round(score, 4),
                "role": "attacker" if ip in attackers else "victim" if ip in victims else "host",
            }
            for ip, score in node_scores.items()
        ]
        links = []
        if len(records):
            grouped = records.groupby(["src_ip", "dst_ip"]).agg(flows=("packets", "size"), packets=("packets", "sum"))
            grouped = grouped.sort_values("packets", ascending=False).head(max_links)
            links = [
                {"source": s, "target": d, "flows": int(r.flows), "packets": float(r.packets)}
                for (s, d), r in grouped.iterrows()
            ]
        return {"nodes": nodes, "links": links}

    def _record(self, entries: list[dict], now: float) -> None:
        path = self.config.record_flows_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fields = ["received_at", "dpid", "ipv4_src", "ipv4_dst", "ip_proto", "tp_src", "tp_dst",
                  "packet_count", "byte_count", "duration_sec", "duration_nsec"]
        new_file = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            for e in entries:
                row = {**e, "received_at": now}
                row.setdefault("tp_src", e.get("tcp_src", e.get("udp_src", 0)))
                row.setdefault("tp_dst", e.get("tcp_dst", e.get("udp_dst", 0)))
                writer.writerow(row)
