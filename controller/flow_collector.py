"""Sliding-window buffer of live OpenFlow flow statistics (runs in the IDS service).

The controller posts raw flow-stats entries (cumulative counters per switch).
The collector:

- de-duplicates the same flow seen on several switches along its path (keeps
  the switch reporting the most packets),
- detects activity from counter deltas between polls (a flow is *active* in a
  window if its packet counter grew, or it is new),
- returns, for the current window, the latest cumulative counters of every
  active flow as flow records (``preprocessing.openflow_features`` schema), so
  live features mean the same thing as the dataset features (flow totals).
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import pandas as pd

from preprocessing.openflow_features import RECORD_COLUMNS, openflow_stats_to_flow_records

FlowKey = tuple  # (src_ip, dst_ip, ip_proto, src_port, dst_port)


def flow_key(entry: dict) -> FlowKey | None:
    if not entry.get("ipv4_src") or not entry.get("ipv4_dst"):
        return None  # table-miss, ARP and mitigation rules carry no flow identity
    src_port = entry.get("tcp_src", entry.get("udp_src", entry.get("tp_src", 0))) or 0
    dst_port = entry.get("tcp_dst", entry.get("udp_dst", entry.get("tp_dst", 0))) or 0
    return (
        str(entry["ipv4_src"]),
        str(entry["ipv4_dst"]),
        int(entry.get("ip_proto", 0) or 0),
        int(src_port),
        int(dst_port),
    )


@dataclass
class _FlowState:
    entry: dict  # latest cumulative stats (from the busiest switch)
    first_seen: float
    last_active: float
    per_switch_packets: dict = field(default_factory=dict)


class FlowCollector:
    def __init__(self, window_seconds: float = 10.0, max_flows: int = 50_000) -> None:
        self.window_seconds = window_seconds
        self.max_flows = max_flows
        self._flows: "OrderedDict[FlowKey, _FlowState]" = OrderedDict()
        self._lock = threading.Lock()
        self.total_entries = 0

    def ingest(self, entries: list[dict], now: float | None = None) -> int:
        """Add one poll's flow-stats entries; returns how many flows were active."""
        now = time.time() if now is None else now
        touched: set = set()
        with self._lock:
            for entry in entries:
                key = flow_key(entry)
                if key is None:
                    continue
                self.total_entries += 1
                dpid = entry.get("dpid", 0)
                packets = int(entry.get("packet_count", 0))
                state = self._flows.get(key)
                if state is None:
                    state = _FlowState(entry=dict(entry), first_seen=now, last_active=now)
                    self._flows[key] = state
                    touched.add(key)
                else:
                    previous = state.per_switch_packets.get(dpid)
                    # None -> first report from this switch; a changed count -> traffic
                    # (or a re-installed rule, whose counter restarted).
                    if previous is None or packets != previous:
                        state.last_active = now
                        touched.add(key)
                    if packets >= int(state.entry.get("packet_count", 0)):
                        state.entry = dict(entry)
                state.per_switch_packets[dpid] = packets
            self._evict(now)
        return len(touched)

    def _evict(self, now: float) -> None:
        horizon = now - self.window_seconds
        stale = [k for k, s in self._flows.items() if s.last_active < horizon]
        for k in stale:
            del self._flows[k]
        while len(self._flows) > self.max_flows:
            self._flows.popitem(last=False)

    def get_current_window(self, now: float | None = None) -> pd.DataFrame:
        """Active flows in the last ``window_seconds``, ordered by first appearance."""
        now = time.time() if now is None else now
        with self._lock:
            self._evict(now)
            states = sorted(self._flows.values(), key=lambda s: s.first_seen)
            entries = [{**s.entry, "timestamp": s.first_seen} for s in states]
        if not entries:
            return pd.DataFrame(columns=RECORD_COLUMNS + ["timestamp"])
        return openflow_stats_to_flow_records(entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._flows)
