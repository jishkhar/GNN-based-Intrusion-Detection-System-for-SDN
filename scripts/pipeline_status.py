#!/usr/bin/env python3
"""Progress file for ``scripts/run_all.sh``, shown live on the dashboard at ``/pipeline``.

The master script calls this after every step; the web app reads the file on
each request (``GET /api/pipeline``). Writes are atomic (temp file + rename), so
the dashboard never sees a half-written file.

    pipeline_status.py FILE init --run-id ID --log-dir DIR --stage "id|Title|Description|out1,out2" ...
    pipeline_status.py FILE start ID
    pipeline_status.py FILE done ID [--note=TEXT]
    pipeline_status.py FILE fail ID [--note=TEXT]
    pipeline_status.py FILE skip ID [--note=TEXT]  # --note=... so notes like "--skip-training" parse
    pipeline_status.py FILE set KEY VALUE          # top-level field, e.g. session, dashboard_url
    pipeline_status.py FILE finish done|failed|interrupted [--note=TEXT]
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save(path: str, status: dict) -> None:
    status["updated"] = time.time()
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".status-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    os.replace(tmp, path)


def find_stage(status: dict, stage_id: str) -> dict:
    for stage in status["stages"]:
        if stage["id"] == stage_id:
            return stage
    raise SystemExit(f"unknown stage: {stage_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    sub = parser.add_subparsers(dest="cmd", required=True)
    init = sub.add_parser("init")
    init.add_argument("--run-id", required=True)
    init.add_argument("--log-dir", required=True)
    init.add_argument("--stage", action="append", default=[], help="id|Title|Description|out1,out2")
    init.add_argument("--options", default="", help="command-line options of the run, for display")
    for name in ("start", "done", "fail", "skip"):
        p = sub.add_parser(name)
        p.add_argument("stage")
        p.add_argument("--note", default="")
    setter = sub.add_parser("set")
    setter.add_argument("key")
    setter.add_argument("value")
    finish = sub.add_parser("finish")
    finish.add_argument("state", choices=["done", "failed", "interrupted"])
    finish.add_argument("--note", default="")
    args = parser.parse_args()

    now = time.time()
    if args.cmd == "init":
        stages = []
        for i, spec in enumerate(args.stage, start=1):
            sid, title, description, outputs = (spec.split("|") + ["", "", ""])[:4]
            stages.append({
                "id": sid, "title": title, "description": description,
                "outputs": [o for o in outputs.split(",") if o],
                "state": "pending", "started": None, "finished": None, "note": "",
                "log": os.path.join(args.log_dir, f"{i:02d}_{sid}.log"),
            })
        save(args.file, {"run_id": args.run_id, "state": "running", "started": now, "finished": None,
                         "options": args.options, "current": None, "note": "", "stages": stages})
        return

    status = load(args.file)
    if args.cmd == "set":
        status[args.key] = args.value
    elif args.cmd == "finish":
        status["state"], status["finished"], status["note"] = args.state, now, args.note
        status["current"] = None
        for stage in status["stages"]:  # a stage cut short by an interrupt
            if stage["state"] == "running":
                stage["state"], stage["finished"], stage["note"] = "failed", now, args.note or args.state
    else:
        stage = find_stage(status, args.stage)
        if args.cmd == "start":
            stage["state"], stage["started"], stage["finished"] = "running", now, None
            status["current"] = stage["id"]
        else:
            stage["state"] = {"done": "done", "fail": "failed", "skip": "skipped"}[args.cmd]
            stage["finished"] = now
            if status.get("current") == stage["id"]:
                status["current"] = None
        stage["note"] = args.note
    save(args.file, status)


if __name__ == "__main__":
    main()
