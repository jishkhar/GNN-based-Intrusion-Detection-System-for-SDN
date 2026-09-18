"""GNN-IDS API and dashboard.

Offline results (Phase 1 + Phase 2 metrics) are always available. The live IDS
(``inference.service.IDSService``) starts when the exported model exists; the
SDN controller then posts flow statistics to ``POST /api/flows`` and receives
mitigation actions in the response.

Environment variables:
    IDS_CONFIG        config file with an ``ids_service`` section (default configs/phase2.yaml)
    IDS_API_KEY       if set, write endpoints require header ``X-API-Key``
    IDS_CORS_ORIGINS  comma-separated allowed origins (default: localhost:3000)
    IDS_LOG_DIR       override ids_service.log_dir
    IDS_RECORD_FLOWS  record every flow-stats entry to this CSV (labelled Mininet data)
    IDS_MITIGATION    0 = detect and alert only, never install rules
"""
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    BASELINE_METRICS_FILE,
    GNN_CLASSIFICATION_REPORT_FILE,
    GNN_METRICS_FILE,
    PHASE2_RESULTS_DIR,
    PIPELINE_DIR,
    PIPELINE_STATUS_FILE,
    PROJECT_ROOT,
    STATIC_DIR,
)

# The IDS service lives in project packages (inference/, controller/, ...).
sys.path.insert(0, str(PROJECT_ROOT))

app = FastAPI(title=API_TITLE, version=API_VERSION, description=API_DESCRIPTION)

_origins = os.environ.get("IDS_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

service = None  # IDSService, created on startup when the model exists
service_error: str | None = None


def _load_service():
    global service, service_error
    from common.config import load_config
    from inference.service import IDSService, ServiceConfig

    config_path = os.environ.get("IDS_CONFIG", str(PROJECT_ROOT / "configs" / "phase2.yaml"))
    cfg = ServiceConfig.from_dict(load_config(config_path).get("ids_service"))
    # Per-session overrides, e.g. when collecting labelled Mininet data.
    cfg.log_dir = os.environ.get("IDS_LOG_DIR", cfg.log_dir)
    cfg.record_flows_path = os.environ.get("IDS_RECORD_FLOWS", cfg.record_flows_path)
    if os.environ.get("IDS_MITIGATION") == "0":  # detect only, e.g. while collecting training data
        cfg.mitigation_enabled = False
    for attr in ("model_path", "log_dir", "record_flows_path"):
        value = getattr(cfg, attr)
        if value and not os.path.isabs(value):
            setattr(cfg, attr, str(PROJECT_ROOT / value))
    if not os.path.exists(cfg.model_path):
        service_error = f"Model not found: {cfg.model_path} (run scripts/run_phase2_training.sh)"
        return
    try:
        service = IDSService(cfg)
    except Exception as exc:  # keep the offline dashboard usable
        service_error = f"{type(exc).__name__}: {exc}"


@app.on_event("startup")
def startup() -> None:
    if os.environ.get("IDS_DISABLE_LIVE") != "1":
        _load_service()


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("IDS_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def require_service():
    if service is None:
        raise HTTPException(status_code=503, detail=service_error or "Live IDS not started")
    return service


def _read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": f"File not found: {path}"}
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON in {path}"}


# ------------------------------------------------------------------ pages
@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/live")
async def live_page():
    return FileResponse(str(STATIC_DIR / "live.html"))


@app.get("/pipeline")
async def pipeline_page():
    return FileResponse(str(STATIC_DIR / "pipeline.html"))


# ------------------------------------------------------- offline results
@app.get("/api/metrics/gnn")
async def get_gnn_metrics():
    """Phase 1 GNN metrics including training history and test results."""
    return _read_json(GNN_METRICS_FILE)


@app.get("/api/metrics/baseline")
async def get_baseline_metrics():
    """Phase 1 baseline metrics (Random Forest, XGBoost)."""
    return _read_json(BASELINE_METRICS_FILE)


@app.get("/api/classification-report")
async def get_classification_report():
    return _read_json(GNN_CLASSIFICATION_REPORT_FILE)


@app.get("/api/metrics/phase2")
async def get_phase2_metrics():
    """Phase 2 GNN, baselines and ablations (whatever has been produced)."""
    out = {}
    for path in sorted(PHASE2_RESULTS_DIR.glob("*.json")):
        out[path.stem] = _read_json(path)
    return out


@app.get("/api/summary")
async def get_summary():
    try:
        gnn = _read_json(GNN_METRICS_FILE)
        baseline = _read_json(BASELINE_METRICS_FILE)
        return {
            "gnn": {
                "accuracy": gnn["test"]["accuracy"],
                "precision": gnn["test"]["precision"],
                "recall": gnn["test"]["recall"],
                "f1": gnn["test"]["f1"],
                "best_epoch": gnn["best_epoch"],
                "best_val_f1": gnn["best_val_f1"],
                "decision_threshold": gnn["decision_threshold"],
            },
            "baseline": {"random_forest": baseline["random_forest"], "xgboost": baseline["xgboost"]},
            "classification": _read_json(GNN_CLASSIFICATION_REPORT_FILE),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "gnn_metrics": GNN_METRICS_FILE.exists(),
        "baseline_metrics": BASELINE_METRICS_FILE.exists(),
        "classification_report": GNN_CLASSIFICATION_REPORT_FILE.exists(),
        "live_ids": service is not None,
        "live_ids_error": service_error,
    }


# ------------------------------------------------ master pipeline progress
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _log_tail(rel_path: str, lines: int) -> list[str]:
    """Last lines of a stage log; progress bars (carriage returns) keep only their final state."""
    path = (PROJECT_ROOT / rel_path).resolve()
    if PIPELINE_DIR.resolve() not in path.parents or not path.is_file():
        return []
    with open(path, "rb") as f:
        f.seek(max(0, path.stat().st_size - 256 * 1024))
        text = f.read().decode("utf-8", errors="replace")
    out = [_ANSI.sub("", line.rsplit("\r", 1)[-1]) for line in text.split("\n")]
    while out and not out[-1].strip():
        out.pop()
    return out[-lines:]


def _dig(data, *keys):
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def _result(label: str, path: Path, value, fmt: str, detail: str = "") -> dict | None:
    if value is None:
        return None
    return {"label": label, "value": fmt.format(value), "detail": detail,
            "file": str(path.relative_to(PROJECT_ROOT)), "mtime": path.stat().st_mtime}


def _key_results(session: str | None) -> list[dict]:
    """Headline numbers from whatever result files exist (each file is optional)."""
    def load(path: Path):
        data = _read_json(path) if path.exists() else None
        return None if not data or "error" in data else data

    results = []
    p1 = GNN_METRICS_FILE
    if (d := load(p1)):
        results.append(_result("Phase 1 GNN attack F1", p1, _dig(d, "test", "f1"), "{:.3f}", "CICIDS2017, binary"))
    gnn_path, base_path = PHASE2_RESULTS_DIR / "gnn_v2_gat.json", PHASE2_RESULTS_DIR / "baselines_v2.json"
    gnn, base = load(gnn_path), load(base_path)
    if gnn:
        ours = _dig(gnn, "test", "window", "multiclass", "major_macro_f1")
        detail = "InSDN, main classes"
        if base:
            scores = [_dig(base, m, "window", "multiclass", "major_macro_f1") for m in ("random_forest", "xgboost")]
            scores = [s for s in scores if s is not None]
            if scores:
                detail += f"; best baseline {max(scores):.3f}"
        results.append(_result("Phase 2 attack-type macro-F1", gnn_path, ours, "{:.3f}", detail))
        results.append(_result("Phase 2 attacker-host F1", gnn_path, _dig(gnn, "test", "host", "f1"), "{:.3f}", "per-host head"))
    replay_path = PHASE2_RESULTS_DIR / "replay_report.json"
    if (d := load(replay_path)):
        attacks = d.get("attacks") or {}
        detected = sum(1 for a in attacks.values() if a.get("detected"))
        results.append(_result("Replay: attacks detected", replay_path, f"{detected}/{len(attacks)}", "{}",
                               f"benign false-positive rate {100 * (_dig(d, 'benign', 'false_positive_rate') or 0):.1f} %"))
    lat_path = PHASE2_RESULTS_DIR / "latency_report.json"
    if (d := load(lat_path)) and d.get("devices"):
        device, rep = next(iter(d["devices"].items()))
        results.append(_result("Detection latency p90", lat_path, _dig(rep, "service", "service_total", "p90"),
                               "{:.1f} ms", f"full service path, {device}"))
    if session:
        mit_path = PROJECT_ROOT / session / "mitigation_report.json"
        if (d := load(mit_path)) and (s := d.get("summary")):
            results.append(_result("Live demo: attack traffic dropped", mit_path, _dig(s, "drop_rate", "mean"), "{:.0%}",
                                   f"{_dig(s, 'drop_rate', 'runs_meeting_70pct')}/{s.get('mitigated_runs')} runs ≥ 70 %, "
                                   f"{s.get('attack_runs_with_false_blocks', 0)} runs blocking a benign host"))
            results.append(_result("Live demo: time to mitigation", mit_path, _dig(s, "time_to_mitigation_s", "p50"),
                                   "{:.1f} s", "median from attack start"))
    return [r for r in results if r]


@app.get("/api/pipeline")
def pipeline_status(stage: str | None = None, lines: int = 120):
    """Progress of the latest ``scripts/run_all.sh`` run, with a stage's log tail and headline results."""
    status = _read_json(PIPELINE_STATUS_FILE)
    if "error" in status:
        return {"state": "none", "message": "No pipeline run yet. Start one with: bash scripts/run_all.sh",
                "results": _key_results(None), "server_time": time.time()}
    stages = status.get("stages", [])
    for s in stages:
        s["outputs"] = [
            {"path": p, "exists": (PROJECT_ROOT / p).exists(),
             "mtime": (PROJECT_ROOT / p).stat().st_mtime if (PROJECT_ROOT / p).exists() else None}
            for p in s.get("outputs", [])
        ]
    started = [s for s in stages if s.get("state") != "pending"]
    focus = next((s for s in stages if s["id"] == stage), None) or \
        next((s for s in stages if s["id"] == status.get("current")), None) or (started[-1] if started else None)
    status["log_stage"] = focus["id"] if focus else None
    status["log_tail"] = _log_tail(focus["log"], max(1, min(lines, 500))) if focus else []
    status["results"] = _key_results(status.get("session"))
    status["server_time"] = time.time()
    return status


# ------------------------------------------------------------- live IDS
class FlowUpload(BaseModel):
    flows: list[dict] = Field(default_factory=list, description="OpenFlow flow-stats entries")
    removed_cookies: list[int] = Field(default_factory=list, description="IDS rule cookies removed by switches")
    controller: str | None = None


class BlockRequest(BaseModel):
    ip: str
    direction: str = Field(default="src", pattern="^(src|dst)$")
    action: str = Field(default="drop", pattern="^(drop|rate_limit)$")
    reason: str = "manual block from dashboard"


class UnblockRequest(BaseModel):
    rule_id: str
    reason: str = "manual unblock from dashboard"


@app.post("/api/flows", dependencies=[Depends(require_api_key)])
def post_flows(upload: FlowUpload):
    """Controller uploads one polling cycle; the response carries mitigation actions."""
    return require_service().process_flows(upload.flows, upload.removed_cookies)


@app.get("/api/alerts")
def get_alerts(limit: int = 100):
    return list(require_service().alerts)[-limit:][::-1]


@app.get("/api/mitigations")
def get_mitigations():
    return require_service().mitigation.active_rules()


@app.get("/api/mitigations/log")
def get_mitigation_log(limit: int = 100):
    return require_service().mitigation.read_log(limit)[::-1]


@app.post("/api/mitigations/block", dependencies=[Depends(require_api_key)])
def block(req: BlockRequest):
    match = {"ipv4_src" if req.direction == "src" else "ipv4_dst": req.ip}
    return require_service().mitigation.block(match, req.action, req.reason)


@app.post("/api/mitigations/unblock", dependencies=[Depends(require_api_key)])
def unblock(req: UnblockRequest):
    svc = require_service()
    cmd = svc.mitigation.unblock(req.rule_id, req.reason)
    if cmd is None:
        raise HTTPException(status_code=404, detail=f"No active rule {req.rule_id}")
    ip = next(iter(cmd["match"].values()))
    svc.classifier.reset_cooldown(ip)
    return cmd


@app.get("/api/topology")
def get_topology():
    return require_service().last_topology


@app.get("/api/live/status")
def live_status():
    return require_service().status()


@app.get("/api/live/latency")
def live_latency():
    return require_service().latency_summary()


@app.get("/api/live/stream")
async def live_stream(request: Request, last_id: int = 0):
    """Server-Sent Events: alerts, mitigation actions and per-window summaries."""
    svc = require_service()

    async def events():
        cursor = last_id
        while not await request.is_disconnected():
            for event_id, event in svc.events_since(cursor):
                cursor = event_id
                yield f"id: {event_id}\ndata: {json.dumps(event, default=str)}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(events(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3000)
