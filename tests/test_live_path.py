import json
import os
import sys

import pandas as pd
import pytest

from conftest import ROOT, flow_entry
from controller.flow_collector import FlowCollector
from inference.classifier import ATTACK, BENIGN, SUSPICIOUS, AlertClassifier, Whitelist
from inference.inference_engine import InferenceEngine, Prediction
from inference.service import IDSService, ServiceConfig
from mitigation.mitigation_engine import MitigationEngine


def prediction(conf=0.95, attack_type="DoS", scores=None, victims=("10.0.0.4",), flows=None):
    scores = scores or {"10.0.0.1": 0.99, "10.0.0.4": 0.01}
    return Prediction(conf >= 0.5, attack_type, conf, {}, scores, [], list(victims), flows or {}, 10, len(scores), 1)


# -------------------------------------------------------------- collector
def test_collector_dedupes_switches_and_tracks_activity():
    c = FlowCollector(window_seconds=10)
    assert c.ingest([flow_entry("10.0.0.1", "10.0.0.4", packets=5, dpid=1),
                     flow_entry("10.0.0.1", "10.0.0.4", packets=7, dpid=2)], now=0) == 1
    window = c.get_current_window(now=0)
    assert len(window) == 1 and window.iloc[0].packets == 7  # busiest switch wins
    # unchanged counters -> not active; flow ages out of the window
    c.ingest([flow_entry("10.0.0.1", "10.0.0.4", packets=5, dpid=1)], now=5)
    assert len(c.get_current_window(now=12)) == 0
    # growing counters keep it active
    c.ingest([flow_entry("10.0.0.2", "10.0.0.4", packets=1)], now=20)
    c.ingest([flow_entry("10.0.0.2", "10.0.0.4", packets=9)], now=28)
    assert len(c.get_current_window(now=35)) == 1


def test_collector_ignores_non_ip_entries():
    c = FlowCollector()
    assert c.ingest([{"dpid": 1, "packet_count": 3}], now=0) == 0


# ------------------------------------------------------------- classifier
def test_classifier_levels_whitelist_and_cooldown():
    clf = AlertClassifier(attack_threshold=0.85, suspicious_threshold=0.5, whitelist=["10.0.0.0/30"], cooldown_seconds=30,
                          confirm_windows=1)
    assert clf.decide(prediction(conf=0.2, attack_type="Benign"), 0.5, now=0).level == BENIGN
    assert clf.decide(prediction(conf=0.6), 0.5, now=0).level == SUSPICIOUS
    d = clf.decide(prediction(scores={"10.0.0.1": 0.99, "10.0.0.9": 0.97}), 0.5, now=0)
    assert d.level == ATTACK
    assert d.attackers == ["10.0.0.9"] and d.suppressed == ["10.0.0.1"]  # 10.0.0.1 is whitelisted
    assert d.new_attackers == ["10.0.0.9"] and d.is_new_alert
    again = clf.decide(prediction(scores={"10.0.0.9": 0.97}), 0.5, now=10)
    assert again.new_attackers == [] and not again.is_new_alert  # cool-down
    assert clf.decide(prediction(scores={"10.0.0.9": 0.97}), 0.5, now=41).new_attackers == ["10.0.0.9"]


def test_classifier_requires_persistence_before_blocking():
    clf = AlertClassifier(confirm_windows=2, cooldown_seconds=0)
    first = clf.decide(prediction(scores={"10.0.0.1": 0.99, "10.0.0.2": 0.9}), 0.5, now=0)
    assert first.level == ATTACK and not first.confirmed and first.new_attackers == []
    # 10.0.0.2 was a one-window blip; 10.0.0.1 is flagged again
    second = clf.decide(prediction(scores={"10.0.0.1": 0.99, "10.0.0.2": 0.1}), 0.5, now=2)
    assert second.confirmed and second.new_attackers == ["10.0.0.1"]
    # a benign window resets the streaks
    clf.decide(prediction(conf=0.1, attack_type="Benign"), 0.5, now=4)
    assert clf.decide(prediction(scores={"10.0.0.1": 0.99}), 0.5, now=6).new_attackers == []


def test_whitelist_handles_bad_input():
    wl = Whitelist(["192.168.1.1"])
    assert "192.168.1.1" in wl and "192.168.1.2" not in wl and "not-an-ip" not in wl


# ------------------------------------------------------------- mitigation
def attack_decision(clf, **kw):
    return clf.decide(prediction(**kw), 0.5, now=100)


def confirmed_clf(**kw):
    return AlertClassifier(confirm_windows=1, **kw)


def test_mitigation_policy_dedup_and_unblock(tmp_path):
    engine = MitigationEngine(log_dir=str(tmp_path), whitelist=["10.0.0.254"])
    clf = confirmed_clf(cooldown_seconds=0)
    actions = engine.handle(attack_decision(clf, attack_type="DoS"))
    assert actions == engine.drain_outbox()
    assert actions[0]["action"] == "drop" and actions[0]["match"] == {"ipv4_src": "10.0.0.1"}
    assert engine.handle(attack_decision(clf, attack_type="DoS")) == []  # rule already active

    probe = engine.handle(attack_decision(clf, attack_type="Probe", scores={"10.0.0.2": 0.9}))
    assert probe[0]["action"] == "rate_limit" and probe[0]["rate_pps"] == 20
    botnet = engine.handle(attack_decision(clf, attack_type="Botnet", scores={"10.0.0.3": 0.9}))
    assert botnet[0]["action"] == "drop"  # isolate == drop all egress
    assert engine.handle(attack_decision(clf, scores={"10.0.0.254": 0.99})) == []  # whitelisted

    removed = engine.unblock(actions[0]["rule_id"])
    assert removed["op"] == "delete" and removed["cookie"] == actions[0]["cookie"]
    assert len(engine.active_rules(now=100)) == 2
    engine.forget_cookies([probe[0]["cookie"]])
    assert len(engine.active_rules(now=100)) == 1
    events = [json.loads(l)["event"] for l in open(tmp_path / "mitigation_log.jsonl")]
    assert events.count("rule_added") == 3 and "rule_removed" in events and "rule_removed_by_switch" in events


def test_spoofed_ddos_rate_limits_victim(tmp_path):
    engine = MitigationEngine(log_dir=str(tmp_path), spoof_threshold=5)
    scores = {f"7.7.7.{i}": 0.99 for i in range(30)}
    actions = engine.handle(attack_decision(confirmed_clf(), attack_type="DDoS", scores=scores))
    assert len(actions) == 1
    assert actions[0]["match"] == {"ipv4_dst": "10.0.0.4"} and actions[0]["action"] == "rate_limit"
    assert actions[0]["rate_pps"] == 200  # victim protection limit, in packets/s


def test_victim_protection_waits_for_confirmation(tmp_path):
    engine = MitigationEngine(log_dir=str(tmp_path), spoof_threshold=5)
    clf = AlertClassifier(confirm_windows=2)
    first = {f"7.7.7.{i}": 0.99 for i in range(30)}
    second = {f"8.8.8.{i}": 0.99 for i in range(30)}  # spoofed sources change every window
    assert engine.handle(clf.decide(prediction(attack_type="DDoS", scores=first), 0.5, now=0)) == []
    actions = engine.handle(clf.decide(prediction(attack_type="DDoS", scores=second), 0.5, now=2))
    assert [a["match"] for a in actions] == [{"ipv4_dst": "10.0.0.4"}]


def test_spoofed_ddos_still_blocks_heavy_hitters(tmp_path):
    engine = MitigationEngine(log_dir=str(tmp_path), spoof_threshold=5, heavy_hitter_flows=5)
    scores = {**{f"7.7.7.{i}": 0.99 for i in range(30)}, "10.0.0.1": 0.99}
    flows = {**{f"7.7.7.{i}": 1 for i in range(30)}, "10.0.0.1": 40}  # a real DoS source hiding in a DDoS
    actions = engine.handle(attack_decision(confirmed_clf(), attack_type="DDoS", scores=scores, flows=flows))
    assert [a["match"] for a in actions] == [{"ipv4_dst": "10.0.0.4"}, {"ipv4_src": "10.0.0.1"}]


def test_suspicious_is_logged_not_mitigated(tmp_path):
    engine = MitigationEngine(log_dir=str(tmp_path))
    assert engine.handle(AlertClassifier().decide(prediction(conf=0.6), 0.5, now=0)) == []
    assert os.path.exists(tmp_path / "suspicious_log.jsonl")


def test_rules_expire_after_hard_timeout(tmp_path):
    engine = MitigationEngine(log_dir=str(tmp_path), hard_timeout=60)
    engine.handle(attack_decision(confirmed_clf()))
    assert len(engine.active_rules(now=150)) == 1
    assert engine.active_rules(now=161) == []


# ---------------------------------------------------- engine + service + API
def records(n=30):
    return pd.DataFrame({"src_ip": [f"10.0.0.{i % 6}" for i in range(n)], "dst_ip": ["10.0.0.9"] * n,
                         "src_port": range(40000, 40000 + n), "dst_port": [80] * n, "ip_proto": [6] * n,
                         "packets": [5.0] * n, "bytes": [500.0] * n, "duration": [0.5] * n})


def test_inference_engine_chunks_large_windows(tiny_model_path):
    engine = InferenceEngine(tiny_model_path, device="cpu")
    assert engine.chunk_records == 100  # 2 x window_size from the bundle
    p = engine.predict(records(250))
    assert p.num_chunks == 3 and p.num_flows == 250
    assert set(p.node_scores) == {f"10.0.0.{i}" for i in range(6)} | {"10.0.0.9"}
    assert 0.0 <= p.confidence <= 1.0 and abs(sum(p.class_probabilities.values()) - 1) < 1e-4
    assert set(p.latency_ms) >= {"features", "graph", "model", "decision", "total"}
    empty = engine.predict(records(0))
    assert not empty.is_attack and empty.num_flows == 0


def test_host_score_is_mean_over_chunks(tiny_model_path):
    mean_engine = InferenceEngine(tiny_model_path, device="cpu", node_aggregation="mean")
    max_engine = InferenceEngine(tiny_model_path, device="cpu", node_aggregation="max")
    rec = records(250)
    mean_p, max_p = mean_engine.predict(rec), max_engine.predict(rec)
    # the victim appears in every chunk: its mean can never exceed its max
    assert all(mean_p.node_scores[ip] <= max_p.node_scores[ip] + 1e-6 for ip in mean_p.node_scores)


def test_service_round_trip(tiny_model_path, tmp_path):
    cfg = ServiceConfig(model_path=tiny_model_path, device="cpu", log_dir=str(tmp_path), attack_threshold=0.0,
                        suspicious_threshold=0.0, node_threshold=0.0, confirm_windows=1,
                        record_flows_path=str(tmp_path / "flows.csv"))
    svc = IDSService(cfg)
    result = svc.process_flows([flow_entry("10.0.0.1", "10.0.0.4"), flow_entry("10.0.0.2", "10.0.0.4", sport=1)], now=1)
    assert result["window_flows"] == 2
    # thresholds at 0 force an ATTACK decision so the mitigation path is exercised
    assert result["level"] == ATTACK and result["actions"]
    assert svc.alerts and svc.last_topology["nodes"] and svc.latency_summary()["count"] == 1
    assert pd.read_csv(tmp_path / "flows.csv").shape[0] == 2


@pytest.fixture
def api_client(tiny_model_path, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("IDS_DISABLE_LIVE", "1")
    monkeypatch.setenv("IDS_API_KEY", "secret")
    sys.path.insert(0, os.path.join(ROOT, "web"))
    import app as web_app

    web_app.service = IDSService(ServiceConfig(model_path=tiny_model_path, device="cpu", log_dir=str(tmp_path)))
    yield TestClient(web_app.app)
    web_app.service = None


def test_api_flow_upload_and_auth(api_client):
    body = {"flows": [flow_entry("10.0.0.1", "10.0.0.4")]}
    assert api_client.post("/api/flows", json=body).status_code == 401
    r = api_client.post("/api/flows", json=body, headers={"X-API-Key": "secret"})
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert api_client.get("/api/live/status").json()["uploads"] == 1
    assert "nodes" in api_client.get("/api/topology").json()
    assert api_client.get("/api/health").json()["live_ids"] is True


def test_api_manual_block_unblock(api_client):
    h = {"X-API-Key": "secret"}
    rule = api_client.post("/api/mitigations/block", json={"ip": "10.0.0.66"}, headers=h).json()
    assert rule["match"] == {"ipv4_src": "10.0.0.66"}
    assert len(api_client.get("/api/mitigations").json()) == 1
    # the queued rule is delivered to the controller with the next upload
    actions = api_client.post("/api/flows", json={"flows": []}, headers=h).json()["actions"]
    assert actions[0]["rule_id"] == rule["rule_id"]
    assert api_client.post("/api/mitigations/unblock", json={"rule_id": rule["rule_id"]}, headers=h).status_code == 200
    assert api_client.post("/api/mitigations/unblock", json={"rule_id": "r-999"}, headers=h).status_code == 404


def test_rule_priorities_by_severity(tmp_path):
    engine = MitigationEngine(log_dir=str(tmp_path), spoof_threshold=5, victim_hard_timeout=120)
    ddos = engine.handle(attack_decision(confirmed_clf(), attack_type="DDoS", scores={f"7.7.7.{i}": 0.99 for i in range(30)}))
    dos = engine.handle(attack_decision(confirmed_clf(), attack_type="DoS", scores={"10.0.0.1": 0.99}))
    probe = engine.handle(attack_decision(confirmed_clf(), attack_type="Probe", scores={"10.0.0.6": 0.99}))
    # a drop must beat a leftover victim rate limit that also matches the flood
    assert dos[0]["priority"] > probe[0]["priority"] > ddos[0]["priority"]
    assert ddos[0]["hard_timeout"] == 120 and dos[0]["hard_timeout"] == 300
