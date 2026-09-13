# Execution Guide (Phase 1 MVP + Dashboard)

This document explains how to run the Phase 1 MVP end-to-end for the GNN-based IDS project and open the web dashboard for visualizing the generated results.

---

## 1) Prerequisites

- OS: Linux (recommended)
- Python: 3.10+
- Browser: Chrome, Firefox, Edge, or any modern browser
- Dataset CSVs available locally:
  - CICIDS2017 CSV files
  - InSDN CSV files

---

## 2) Project location

From terminal, move to project root:

```bash
cd /home/nyx/Projects/GNN-based-Intrusion-Detection-System-for-SDN
```

---

## 3) Create and activate environment

### Option A: venv + pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r web/requirements.txt
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate gnn-ids-phase1
pip install -r web/requirements.txt
```

---

## 4) Place datasets in correct folders

Put raw CSV files here:

- `data/cicids2017/raw/`
- `data/insdn/raw/`

Example expected structure:

- `data/cicids2017/raw/*.csv`
- `data/insdn/raw/*.csv`

Do not place cleaned files manually; the pipeline will generate them.

---

## 5) Quick checks before full run

### 5.1 Check script syntax

```bash
bash -n scripts/run_phase1_pipeline.sh
```

### 5.2 Verify raw files are present

```bash
ls data/cicids2017/raw/*.csv
ls data/insdn/raw/*.csv
```

---

## 6) Run complete Phase 1 pipeline

```bash
bash scripts/run_phase1_pipeline.sh
```

This executes in order:
1. Clean CICIDS raw data
2. Clean InSDN raw data
3. Train/evaluate baselines (RandomForest + XGBoost if installed)
4. Build graph snapshots
5. Train GNN (edge-aware GAT)
6. Evaluate GNN

The pipeline must finish successfully before the dashboard can show final metrics.

---

## 7) Run the frontend dashboard

The frontend is a FastAPI + Bootstrap + Chart.js dashboard located in `web/`. It reads the generated JSON files from `results/`, so run the ML pipeline first.

### 7.1 Start the dashboard from the project root

```bash
source .venv/bin/activate
bash web/run.sh
```

Alternative direct command:

```bash
source .venv/bin/activate
cd web
python -m uvicorn app:app --host 0.0.0.0 --port 3000 --reload
```

### 7.2 Open in browser

```text
http://localhost:3000
```

### 7.3 Useful API checks

Open these URLs in the browser, or use `curl`:

```bash
curl http://localhost:3000/api/health
curl http://localhost:3000/api/summary
```

Expected health response should show the result files as available:

```json
{
  "status": "healthy",
  "gnn_metrics": true,
  "baseline_metrics": true,
  "classification_report": true
}
```

Stop the server with `Ctrl+C`.

---

## 8) Outputs generated

After successful run, check:

- Cleaned data:
  - `data/cicids2017/cleaned/*_cleaned.csv`
  - `data/insdn/cleaned/*_cleaned.csv`
  - `data/*/cleaned/cleaning_report.json`

- Graph dataset:
  - `data/graphs/cicids_graphs.pt`

- Baseline results:
  - `results/baseline_metrics.json`
  - `results/baseline_confusion_matrix.png`

- GNN results:
  - `models/checkpoints/best_gat.pt`
  - `results/gnn_metrics.json`
  - `results/gnn_classification_report.json`
  - `results/gnn_confusion_matrix.png`

- Frontend dashboard:
  - `web/app.py`
  - `web/static/index.html`
  - `web/static/css/style.css`
  - `web/static/js/app.js`

The dashboard does not create a database. It loads these files at runtime:

- `results/gnn_metrics.json`
- `results/baseline_metrics.json`
- `results/gnn_classification_report.json`

---

## 9) Run modules individually (optional)

If debugging is needed, run steps one by one.

### 9.1 Clean CICIDS data

```bash
python -m preprocessing.clean_data \
  --input-glob "data/cicids2017/raw/*.csv" \
  --output-dir "data/cicids2017/cleaned" \
  --label-column "Label"
```

### 9.2 Clean InSDN data

```bash
python -m preprocessing.clean_data \
  --input-glob "data/insdn/raw/*.csv" \
  --output-dir "data/insdn/cleaned" \
  --label-column "Label"
```

### 9.3 Run baselines

```bash
python -m baselines.train_baselines \
  --input-glob "data/cicids2017/cleaned/*.csv" \
  --label-col "Label" \
  --metrics-out "results/baseline_metrics.json" \
  --cm-out "results/baseline_confusion_matrix.png"
```

### 9.4 Build graphs

```bash
python -m preprocessing.graph_builder \
  --input-glob "data/cicids2017/cleaned/*.csv" \
  --output-path "data/graphs/cicids_graphs.pt" \
  --label-col "Label" \
  --window-seconds 5 \
  --step-seconds 1
```

### 9.5 Train GNN

```bash
python -m models.train_gnn \
  --graph-path "data/graphs/cicids_graphs.pt" \
  --checkpoint-path "models/checkpoints/best_gat.pt" \
  --metrics-path "results/gnn_metrics.json" \
  --batch-size 128 \
  --epochs 40
```

### 9.6 Evaluate GNN

```bash
python -m models.evaluate \
  --graph-path "data/graphs/cicids_graphs.pt" \
  --checkpoint-path "models/checkpoints/best_gat.pt" \
  --report-out "results/gnn_classification_report.json" \
  --cm-out "results/gnn_confusion_matrix.png"
```

### 9.7 Run only the dashboard

Use this when the JSON result files already exist and you only want to view them:

```bash
bash web/run.sh
```

---

## 10) Common issues and fixes

- **No files matched error**
  - Ensure raw CSV files exist in `data/*/raw/`.
- **xgboost missing**
  - Install via `pip install xgboost`.
  - Pipeline still runs RandomForest even if XGBoost is unavailable.
- **torch-geometric install issues**
  - Use the same Python version (3.10 recommended).
  - Install `torch` first, then `torch-geometric`.
- **Out-of-memory during training**
  - Reduce epochs or graph volume.
  - Start with fewer raw CSVs for initial validation.
- **Dashboard shows missing file errors**
  - Run `bash scripts/run_phase1_pipeline.sh` first.
  - Confirm these files exist: `results/gnn_metrics.json`, `results/baseline_metrics.json`, `results/gnn_classification_report.json`.
- **FastAPI or uvicorn missing**
  - Run `pip install -r web/requirements.txt`.
- **Port 3000 already in use**
  - Stop the existing process or run manually on another port:
    ```bash
    cd web
    python -m uvicorn app:app --host 0.0.0.0 --port 3001 --reload
    ```
  - Then open `http://localhost:3001`.

---

## 11) MVP completion checklist

- [ ] Pipeline runs without failure
- [ ] Baseline metrics generated
- [ ] Graph dataset generated
- [ ] GNN model trained and checkpoint saved
- [ ] GNN report + confusion matrix generated
- [ ] Dashboard dependencies installed
- [ ] Dashboard starts at `http://localhost:3000`
- [ ] `/api/health` returns all result files as `true`
- [ ] Charts and metric cards load in browser
- [ ] Results ready for Phase 1 guide review

---

# Phase 2: Topology GNN, Live SDN Integration and Mitigation

Phase 2 replaces the synthetic CICIDS graphs with real host-to-host graphs (InSDN), adds a per-host
attacker head, and connects the model to a live SDN (os-ken controller + Mininet) that blocks attackers
automatically. Every script accepts `--config configs/phase2.yaml`; command-line flags override it.

## 12) Extra prerequisites

- Docker, with your user in the `docker` group (for the Mininet lab), or an Ubuntu 22.04 VM
- `pip install -r requirements.txt -r web/requirements.txt` (pins exact versions; includes pytest)

## 13) Offline pipeline

```bash
# InSDN must be cleaned with the Phase 2 cleaner (keeps the attack type, fixes "Normal" = benign)
python -m preprocessing.clean_data --input-glob "data/insdn/raw/*.csv" --output-dir data/insdn/cleaned

PYTHON=.venv/bin/python bash scripts/run_phase2_training.sh      # ~70 min with ablations
PYTHON=.venv/bin/python bash scripts/run_phase2_training.sh --skip-ablations   # ~15 min
```

| Step | Command | Output |
|---|---|---|
| Graphs | `python -m preprocessing.graph_builder_v2 --config configs/phase2.yaml` | `data/graphs/insdn_v2.pt` |
| Baselines | `python -m baselines.train_baselines_v2 --config configs/phase2.yaml` | `results/phase2/baselines_v2.json` |
| GNN | `python -m models.train_gnn_v2 --config configs/phase2.yaml` | `models/checkpoints/gnn_v2_gat.pt`, `results/phase2/gnn_v2_gat.json` |
| Ablations | `--conv-type gcn/sage`, `--no-edge-features`, other window sizes | `results/phase2/ablation_*.json` |
| Export | `python -m models.export --config configs/phase2.yaml` | `models/gat_ids.pt` (+ `.json` bundle) |
| EDA | `python scripts/eda_insdn.py` | `Docs/eda_summary.md`, `results/phase2/eda/` |

## 14) Tests and live-path checks (no Mininet needed)

```bash
python -m pytest tests -q                     # unit + integration tests
python scripts/replay_flows.py                # held-out flows through the live IDS
python scripts/benchmark_latency.py           # latency per window size, CPU and GPU
```

## 15) Live SDN lab

Full details: `Docs/sdn_lab_setup.md`.

```bash
docker build -t gnn-ids-lab -f docker/Dockerfile.sdn-lab docker/     # once
KEEP_IDS=1 bash scripts/run_phase2_demo.sh                           # all scenarios, dashboard stays up
```

Start order used by the script (and for a manual demo):
1. IDS API + dashboard on the host: `cd web && IDS_API_KEY=<key> python -m uvicorn app:app --port 3000`
2. Controller in the lab container: `python3 controller/run_controller.py --api-key <key>`
3. Mininet + traffic: `python3 scripts/inject_attack.py --scenario ddos`
4. Watch http://127.0.0.1:3000/live: alert, rule installed, attack traffic drops.

Environment variables of the IDS API: `IDS_API_KEY`, `IDS_CONFIG`, `IDS_LOG_DIR`, `IDS_RECORD_FLOWS`,
`IDS_MITIGATION=0` (detect only), `IDS_CORS_ORIGINS`.

## 16) Fine-tuning on lab traffic

```bash
COLLECT=1 REPEAT=5 SCENARIO=all SESSION=data/mininet/collect1 bash scripts/run_phase2_demo.sh   # ~40 min
bash scripts/finetune_on_lab.sh data/mininet/collect1                                          # ~15 min
```

`run_phase2_training.sh` exports the InSDN-only model to `models/gat_ids_insdn_only.pt` and never
overwrites `models/gat_ids.pt` (the live model); `finetune_on_lab.sh` installs the fine-tuned model there.

## 17) Results report

```bash
python scripts/make_phase2_report.py          # -> results/phase2/final_results.md
```

## 18) Phase 2 checklist

- [ ] `pytest` passes
- [ ] `results/phase2/final_results.md` generated
- [ ] Lab `pingall` works through the IDS controller
- [ ] Demo: attack detected, rule installed, attack traffic dropped ≥ 70 %
- [ ] No benign host blocked in benign-only runs
