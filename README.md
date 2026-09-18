# GNN-Based Intrusion Detection System for SDN

**Siddaganga Institute of Technology · Batch B24 · AY 2025-26**
Abhishek · Ashish Kumar Bhagat · Jishnu Khargharia · Vaishnavi A Hachadad

A graph neural network watches the flow tables of an SDN, detects attacks, identifies the attacking
hosts, and has the controller block them automatically.

```
Mininet hosts ─ Open vSwitch (OpenFlow 1.3) ─ os-ken controller ──flow stats──► IDS (FastAPI + GNN)
                        ▲                                       ◄──rules───────  │ live dashboard
                        └──────────── drop / rate-limit rules ───────────────────┘
```

## What it does

- **Topology graphs from flows.** Each window of traffic becomes a graph: hosts are nodes, flows are
  edges, and features use only what OpenFlow switches report (so the same model runs offline and live).
- **Multi-task GAT.** One head gives the attack type of the window (DDoS, DoS, Probe, BruteForce, …),
  another gives a per-host attacker score, which tells the mitigation engine *whom* to block.
- **Automated mitigation.** Per-attack policy (drop, rate-limit with meters, isolate). Spoofed DDoS is
  handled by rate-limiting the victim. Rules expire, infrastructure can be whitelisted, and everything is
  audit-logged.
- **Live SDN integration.** An os-ken controller app polls flow stats every 2 s and installs the rules
  returned by the IDS. A Docker lab runs Mininet with scripted benign and attack traffic.
- **Dashboard.** Offline evaluation at `/`, live monitor at `/live` (traffic graph, alerts, rules,
  manual block/unblock).

Results: [results/phase2/final_results.md](results/phase2/final_results.md).

## Repository layout

| Path | Contents |
|---|---|
| `preprocessing/` | cleaning, label mapping, OpenFlow feature schema, graph builders |
| `baselines/` | Random Forest / XGBoost (Phase 1 and the Phase 2 fair comparison) |
| `models/` | GNN definitions, training, evaluation, TorchScript export |
| `inference/` | live inference engine, alert classifier, IDS service |
| `mitigation/` | mitigation policy, OpenFlow rule actions, audit log |
| `controller/` | os-ken/Ryu controller app, launcher, flow collector |
| `topology/` | Mininet topology |
| `scripts/` | pipelines, lab demo, attack injection, replay, benchmarks, reports |
| `web/` | FastAPI app and dashboards |
| `docker/` | SDN lab image (Mininet + OVS + os-ken + attack tools) |
| `configs/` | `phase1.yaml`, `phase2.yaml` (all scripts read `--config`) |
| `tests/` | pytest suite |
| `Docs/` | architecture, task lists, feature schema, label scheme, lab setup, design decisions, EDA |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r web/requirements.txt
```

Datasets (not in git): put the InSDN CSVs in `data/insdn/raw/`. Optionally, the CICIDS2017
*TrafficLabelling* CSVs (with IP columns) go in a new folder and use the same pipeline.

## Run

Everything in one command, with progress at http://127.0.0.1:3000/pipeline:

```bash
bash scripts/run_all.sh            # add --quick to skip ablations, --skip-training to reuse models
```

Or step by step:

```bash
# 1. Clean the data
python -m preprocessing.clean_data --input-glob "data/insdn/raw/*.csv" --output-dir data/insdn/cleaned

# 2. Offline pipeline: graphs -> baselines -> GNN -> ablations -> TorchScript export
PYTHON=.venv/bin/python bash scripts/run_phase2_training.sh          # add --skip-ablations for speed

# 3. Tests
python -m pytest tests -q

# 4. Live pipeline without Mininet (held-out flows replayed as OpenFlow stats)
python scripts/replay_flows.py
python scripts/benchmark_latency.py

# 5. Live SDN lab (Docker) — see Docs/sdn_lab_setup.md
docker build -t gnn-ids-lab -f docker/Dockerfile.sdn-lab docker/
KEEP_IDS=1 bash scripts/run_phase2_demo.sh            # then open http://127.0.0.1:3000/live

# 6. Results report
python scripts/make_phase2_report.py
```

Dashboard only: `bash web/run.sh`, then http://localhost:3000.

## Documentation

- [Architecture](Docs/GNN_IDS_Architecture.md) · [Phase 2 tasks](Docs/tasks_phase2.md)
- [Feature schema](Docs/feature_schema_phase2.md) · [Label scheme](Docs/label_scheme.md) · [EDA](Docs/eda_summary.md)
- [SDN lab setup](Docs/sdn_lab_setup.md) · [Design decisions](Docs/design_decisions.md)
- [Execution guide](Execution.md) · [Presentation notes](explain.md)
