"""Turn GNN predictions into alert decisions.

- ``confidence >= attack_threshold``      -> ATTACK (mitigation eligible)
- ``suspicious_threshold <= conf < attack`` -> SUSPICIOUS (logged only)
- otherwise                                -> BENIGN

Whitelisted infrastructure (controller, gateway, DNS ...) is never reported as
an attacker, and a per-host cool-down stops an ongoing attack from raising a
fresh alert every polling cycle.

Mitigation needs persistence: a host becomes a *new attacker* (eligible for
blocking) only after it was flagged in ``confirm_windows`` consecutive ATTACK
windows, and a decision is ``confirmed`` after that many consecutive ATTACK
windows (used for victim protection, where spoofed sources change every
window). A one-window false positive therefore never blocks anyone.
"""
from __future__ import annotations

import ipaddress
import time
from dataclasses import dataclass, field

from inference.inference_engine import Prediction

BENIGN, SUSPICIOUS, ATTACK = "BENIGN", "SUSPICIOUS", "ATTACK"


@dataclass
class Decision:
    level: str
    attack_type: str
    confidence: float
    attackers: list  # flagged, non-whitelisted hosts
    new_attackers: list  # attackers not alerted within the cool-down
    victims: list
    timestamp: float
    suppressed: list = field(default_factory=list)  # whitelisted hosts the model flagged
    attacker_flows: dict = field(default_factory=dict)  # attacker ip -> flows in the window
    confirmed: bool = False  # ATTACK for confirm_windows consecutive windows

    @property
    def is_new_alert(self) -> bool:
        return self.level == ATTACK and bool(self.new_attackers)

    def to_dict(self) -> dict:
        return {**self.__dict__, "is_new_alert": self.is_new_alert}


class Whitelist:
    def __init__(self, entries: list[str] | None = None) -> None:
        self.networks = [ipaddress.ip_network(e, strict=False) for e in (entries or [])]

    def __contains__(self, ip: str) -> bool:
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False
        return any(addr in net for net in self.networks)


def get_attack_sources(node_scores: dict, threshold: float, whitelist: Whitelist | None = None) -> list[str]:
    """Hosts whose attacker score passes ``threshold``, highest first."""
    whitelist = whitelist or Whitelist()
    flagged = [ip for ip, s in node_scores.items() if s >= threshold and ip not in whitelist]
    return sorted(flagged, key=lambda ip: -node_scores[ip])


class AlertClassifier:
    def __init__(
        self,
        attack_threshold: float = 0.85,
        suspicious_threshold: float = 0.5,
        node_threshold: float | None = None,
        whitelist: list[str] | None = None,
        cooldown_seconds: float = 30.0,
        confirm_windows: int = 2,
    ) -> None:
        self.attack_threshold = attack_threshold
        self.suspicious_threshold = suspicious_threshold
        self.node_threshold = node_threshold
        self.whitelist = Whitelist(whitelist)
        self.cooldown_seconds = cooldown_seconds
        self._last_alert: dict[str, float] = {}
        self.confirm_windows = max(1, confirm_windows)
        self._attack_streak = 0
        self._host_streak: dict[str, int] = {}

    def decide(self, prediction: Prediction, node_threshold: float, now: float | None = None) -> Decision:
        now = time.time() if now is None else now
        threshold = self.node_threshold if self.node_threshold is not None else node_threshold
        conf = prediction.confidence
        if conf >= self.attack_threshold and prediction.attack_type != "Benign":
            level = ATTACK
        elif conf >= self.suspicious_threshold:
            level = SUSPICIOUS
        else:
            level = BENIGN

        attackers, new_attackers, suppressed = [], [], []
        if level != BENIGN:
            flagged = get_attack_sources(prediction.node_scores, threshold)
            suppressed = [ip for ip in flagged if ip in self.whitelist]
            attackers = [ip for ip in flagged if ip not in self.whitelist]

        self._attack_streak = self._attack_streak + 1 if level == ATTACK else 0
        current = set(attackers) if level == ATTACK else set()
        self._host_streak = {ip: self._host_streak.get(ip, 0) + 1 for ip in current}
        if level == ATTACK:
            for ip in attackers:
                confirmed_host = self._host_streak[ip] >= self.confirm_windows
                if confirmed_host and now - self._last_alert.get(ip, float("-inf")) >= self.cooldown_seconds:
                    new_attackers.append(ip)
                    self._last_alert[ip] = now

        return Decision(
            level=level,
            attack_type=prediction.attack_type if level != BENIGN else "Benign",
            confidence=conf,
            attackers=attackers,
            new_attackers=new_attackers,
            victims=[v for v in prediction.victims if v not in self.whitelist],
            timestamp=now,
            suppressed=suppressed,
            attacker_flows={ip: prediction.flagged_flow_counts.get(ip, 0) for ip in attackers},
            confirmed=level == ATTACK and self._attack_streak >= self.confirm_windows,
        )

    def reset_cooldown(self, ip: str) -> None:
        self._last_alert.pop(ip, None)
