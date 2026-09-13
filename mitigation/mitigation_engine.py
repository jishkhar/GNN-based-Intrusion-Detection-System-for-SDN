"""Turn ATTACK decisions into OpenFlow rule actions for the controller.

Actions are controller-agnostic JSON dicts queued in an outbox; the controller
app collects them in the response to its next flow-stats upload and installs
them as FlowMods (``controller/ids_controller.py``)::

    {"op": "add", "rule_id": "r-3", "cookie": 3, "action": "drop" | "rate_limit",
     "match": {"ipv4_src": "10.0.0.1"}, "priority": 65535,
     "idle_timeout": 60, "hard_timeout": 300, "rate_pps": 20}
    {"op": "delete", "rule_id": "r-3", "cookie": 3, "match": {...}}

Policy per attack type (``DEFAULT_POLICY``):

- DDoS / DoS / BruteForce / WebAttack / Other -> drop traffic from the source
- Probe                                         -> rate-limit the source (``scanner_rate_pps``)
- Botnet                                        -> isolate the source (drop all egress)
- DDoS with more than ``spoof_threshold`` sources -> sources are likely
  spoofed, so rate-limit traffic *to the victim* (``victim_rate_pps``) instead
  of blocking each IP. Confirmed attackers sending at least
  ``heavy_hitter_flows`` flows in the window are not one-packet spoofed
  sources, so they are still blocked individually.

Rate limits are in packets per second: flood packets are tiny (a SYN is 54
bytes), so a bandwidth limit lets a 1,000 packet/s flood straight through.

Rules get priorities by severity (drop > rate-limit a source > protect a
victim). Two OpenFlow rules with equal priority and overlapping matches have
undefined precedence: in the lab, a victim-protection rate limit left over from
a DDoS shadowed a later drop rule for a DoS source, so the flood kept flowing.
Victim protection also expires sooner (``victim_hard_timeout``) because it
throttles benign traffic to the victim as well.
Nothing is installed until the classifier has confirmed the attack over
consecutive windows (see ``inference.classifier``).

Every action is appended to ``logs/mitigation_log.jsonl``; SUSPICIOUS decisions
go to ``logs/suspicious_log.jsonl``.
"""
from __future__ import annotations

import itertools
import json
import os
import threading
import time
from dataclasses import dataclass

from inference.classifier import ATTACK, SUSPICIOUS, Decision, Whitelist

DEFAULT_POLICY = {
    "DDoS": "drop",
    "DoS": "drop",
    "Probe": "rate_limit",
    "BruteForce": "drop",
    "WebAttack": "drop",
    "Botnet": "isolate",
    "Other": "drop",
}


@dataclass
class Rule:
    rule_id: str
    cookie: int
    action: str
    match: dict
    attack_type: str
    confidence: float
    created_at: float
    idle_timeout: int
    hard_timeout: int
    rate_pps: int | None = None
    source: str = "auto"  # "auto" (GNN) or "manual" (dashboard/API)

    @property
    def expires_at(self) -> float | None:
        return self.created_at + self.hard_timeout if self.hard_timeout else None

    def to_dict(self) -> dict:
        return {**self.__dict__, "expires_at": self.expires_at}


class MitigationEngine:
    def __init__(
        self,
        log_dir: str = "logs",
        policy: dict | None = None,
        priority: int = 65535,
        idle_timeout: int = 60,
        hard_timeout: int = 300,
        victim_rate_pps: int = 200,
        victim_hard_timeout: int = 120,
        scanner_rate_pps: int = 20,
        spoof_threshold: int = 20,
        heavy_hitter_flows: int = 5,
        max_rules: int = 500,
        whitelist: list[str] | None = None,
        enabled: bool = True,
    ) -> None:
        self.policy = {**DEFAULT_POLICY, **(policy or {})}
        self.priority = priority
        self.idle_timeout = idle_timeout
        self.hard_timeout = hard_timeout
        self.victim_rate_pps = victim_rate_pps
        self.victim_hard_timeout = victim_hard_timeout
        self.scanner_rate_pps = scanner_rate_pps
        self.spoof_threshold = spoof_threshold
        self.heavy_hitter_flows = heavy_hitter_flows
        self.max_rules = max_rules
        self.whitelist = Whitelist(whitelist)
        self.enabled = enabled
        self.log_path = os.path.join(log_dir, "mitigation_log.jsonl")
        self.suspicious_path = os.path.join(log_dir, "suspicious_log.jsonl")
        os.makedirs(log_dir, exist_ok=True)
        self._rules: dict[str, Rule] = {}
        self._outbox: list[dict] = []
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public
    def handle(self, decision: Decision) -> list[dict]:
        """Queue actions for a decision; returns the actions created."""
        if decision.level == SUSPICIOUS:
            self._log(self.suspicious_path, {"event": "suspicious", **decision.to_dict()})
            return []
        if decision.level != ATTACK or not decision.confirmed or not self.enabled:
            return []
        now = decision.timestamp
        with self._lock:
            self._expire(now)
            action = self.policy.get(decision.attack_type, "drop")
            if decision.attack_type == "DDoS" and len(decision.attackers) > self.spoof_threshold:
                targets = [({"ipv4_dst": v}, "rate_limit_victim") for v in decision.victims[:1]]
                heavy = [ip for ip in decision.new_attackers
                         if decision.attacker_flows.get(ip, 0) >= self.heavy_hitter_flows]
                targets += [({"ipv4_src": ip}, action) for ip in heavy]
                reason = (f"{len(decision.attackers)} sources (likely spoofed): rate-limit victim"
                          + (f", block {len(heavy)} heavy hitter(s)" if heavy else ""))
            else:
                targets = [({"ipv4_src": ip}, action) for ip in decision.new_attackers]
                reason = f"policy[{decision.attack_type}] = {action}"
            if not targets:
                return []
            created = []
            for match, act in targets:
                ip = next(iter(match.values()))
                if ip in self.whitelist or self._has_rule(match):
                    continue
                if len(self._rules) >= self.max_rules:
                    self._log(self.log_path, {"event": "rule_limit_reached", "match": match, "ts": now})
                    break
                created.append(self._add(match, act, decision.attack_type, decision.confidence, now, reason, "auto"))
            return created

    def block(self, match: dict, action: str = "drop", reason: str = "manual", now: float | None = None) -> dict:
        if action == "rate_limit" and "ipv4_dst" in match:
            action = "rate_limit_victim"
        now = time.time() if now is None else now
        with self._lock:
            return self._add(match, action, "Manual", 1.0, now, reason, "manual")

    def unblock(self, rule_id: str, reason: str = "manual unblock", now: float | None = None) -> dict | None:
        now = time.time() if now is None else now
        with self._lock:
            rule = self._rules.pop(rule_id, None)
            if rule is None:
                return None
            return self._queue_delete(rule, reason, now)

    def forget_cookies(self, cookies: list[int], now: float | None = None) -> None:
        """Drop rules the switch already removed (e.g. idle timeout, reported by the controller)."""
        now = time.time() if now is None else now
        wanted = set(int(c) for c in cookies)
        with self._lock:
            for rule_id in [rid for rid, r in self._rules.items() if r.cookie in wanted]:
                rule = self._rules.pop(rule_id)
                self._log(self.log_path, {"event": "rule_removed_by_switch", "ts": now, "rule_id": rule_id, "match": rule.match})

    def active_rules(self, now: float | None = None) -> list[dict]:
        now = time.time() if now is None else now
        with self._lock:
            self._expire(now)
            return [r.to_dict() for r in self._rules.values()]

    def drain_outbox(self) -> list[dict]:
        with self._lock:
            out, self._outbox = self._outbox, []
            return out

    def read_log(self, limit: int = 100) -> list[dict]:
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]

    # ----------------------------------------------------------------- helpers
    def _has_rule(self, match: dict) -> bool:
        return any(r.match == match for r in self._rules.values())

    def _add(self, match, action, attack_type, confidence, now, reason, source) -> dict:
        cookie = next(self._ids)
        rate = {"rate_limit": self.scanner_rate_pps, "rate_limit_victim": self.victim_rate_pps}.get(action)
        # drop/isolate > rate-limit a source > protect a victim (see module docstring)
        priority = self.priority - {"rate_limit": 100, "rate_limit_victim": 200}.get(action, 0)
        hard_timeout = self.victim_hard_timeout if action == "rate_limit_victim" else self.hard_timeout
        action = "rate_limit" if action == "rate_limit_victim" else action
        rule = Rule(
            rule_id=f"r-{cookie}",
            cookie=cookie,
            action=action,
            match=match,
            attack_type=attack_type,
            confidence=float(confidence),
            created_at=now,
            idle_timeout=self.idle_timeout,
            hard_timeout=hard_timeout,
            rate_pps=rate,
            source=source,
        )
        self._rules[rule.rule_id] = rule
        cmd = {
            "op": "add",
            "rule_id": rule.rule_id,
            "cookie": cookie,
            # isolate == drop everything the host sends; the controller only needs drop/rate_limit.
            "action": "drop" if action == "isolate" else action,
            "match": match,
            "priority": priority,
            "idle_timeout": rule.idle_timeout,
            "hard_timeout": rule.hard_timeout,
            "rate_pps": rule.rate_pps,
        }
        self._outbox.append(cmd)
        self._log(self.log_path, {"event": "rule_added", "ts": now, "reason": reason, **rule.to_dict()})
        return cmd

    def _queue_delete(self, rule: Rule, reason: str, now: float) -> dict:
        cmd = {"op": "delete", "rule_id": rule.rule_id, "cookie": rule.cookie, "match": rule.match}
        self._outbox.append(cmd)
        self._log(self.log_path, {"event": "rule_removed", "ts": now, "reason": reason, "rule_id": rule.rule_id, "match": rule.match})
        return cmd

    def _expire(self, now: float) -> None:
        # The switch removes rules itself at hard_timeout; just forget them here.
        for rule_id in [rid for rid, r in self._rules.items() if r.expires_at is not None and r.expires_at <= now]:
            rule = self._rules.pop(rule_id)
            self._log(self.log_path, {"event": "rule_expired", "ts": now, "rule_id": rule_id, "match": rule.match})

    @staticmethod
    def _log(path: str, event: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
