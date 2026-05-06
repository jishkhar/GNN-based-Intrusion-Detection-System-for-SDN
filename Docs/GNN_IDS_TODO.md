# GNN-Based IDS for SDN — Project TODO List
**Siddaganga Institute of Technology | Batch B24 | AY 2025-26**  
**Team:** Abhishek · Ashish Kumar Bhagat · Jishnu Khargharia · Vaishnavi A Hachadad

> **How to read this file:**  
> Milestones are ordered by dependency — complete them in sequence.  
> Each subtask has a checkbox. Tick as you go.  
> Estimated effort is marked as 🟢 Easy · 🟡 Medium · 🔴 Hard

---

## Milestone 0 — Project Setup & Environment
> **Goal:** Everyone on the team can run code on the same reproducible environment.  
> **Estimated Duration:** 3–4 days

- [ ] 🟢 Create a shared GitHub repository with branch protection on `main`
- [ ] 🟢 Define branching strategy: `main` → `dev` → `feature/xxx`
- [ ] 🟢 Create `README.md` with project overview and setup instructions
- [ ] 🟢 Set up `.gitignore` for Python, Jupyter, and dataset folders
- [ ] 🟡 Create `requirements.txt` with pinned versions:
  - [ ] `torch >= 2.0`, `torch-geometric`, `torch-scatter`, `torch-sparse`
  - [ ] `ryu`, `networkx`, `flask`, `scikit-learn`, `xgboost`
  - [ ] `pandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`
  - [ ] `mlflow` (optional, for experiment tracking)
- [ ] 🟢 Create `environment.yml` for Conda (cross-platform reproducibility)
- [ ] 🟡 Install and verify **Mininet** on a Linux machine (Ubuntu 20.04 / 22.04 recommended)
  - [ ] Run `sudo mn --test pingall` to verify base install
  - [ ] Install Mininet Python API: `pip install mininet`
- [ ] 🟡 Install and verify **Ryu SDN Framework**
  - [ ] `pip install ryu`
  - [ ] Verify: `ryu-manager --version`
- [ ] 🟢 Set up project directory structure as defined in architecture doc
- [ ] 🟢 Create a shared task board (GitHub Projects / Notion) and assign milestone owners

---

## Milestone 1 — Dataset Acquisition & Exploratory Data Analysis (EDA)
> **Goal:** Understand the datasets deeply before writing any model code.  
> **Estimated Duration:** 5–7 days

### 1.1 Dataset Download
- [ ] 🟢 Download **CICIDS2017** dataset from the official CIC website
  - URL: `https://www.unb.ca/cic/datasets/ids-2017.html`
  - Files needed: Monday–Friday CSVs (CICFlowMeter-generated)
- [ ] 🟢 Download **InSDN** dataset
  - Search: "InSDN dataset intrusion detection SDN"
- [ ] 🟢 Verify file integrity (check SHA checksums if provided)
- [ ] 🟢 Store raw files in `data/cicids2017/raw/` and `data/insdn/raw/`
- [ ] 🟢 **Do NOT commit datasets to Git** — add to `.gitignore`

### 1.2 Data Understanding
- [ ] 🟡 Open `notebooks/eda.ipynb` and load all CSVs using `pandas`
- [ ] 🟡 Print shape, dtypes, and `df.describe()` for each file
- [ ] 🟡 Identify and document all **feature columns** (there are ~80 in CICIDS2017)
- [ ] 🟡 Check for and document:
  - [ ] Missing values (`df.isnull().sum()`)
  - [ ] Infinite values (`np.isinf(df).sum()`)
  - [ ] Duplicate rows
- [ ] 🟡 Plot **class distribution** — count of each attack label vs. BENIGN
- [ ] 🟡 Identify the top-10 most discriminative features using feature importance from a quick Random Forest fit
- [ ] 🟢 Document EDA findings in a `docs/eda_summary.md` file

### 1.3 Data Cleaning
- [ ] 🟡 Replace `inf` values with `NaN`, then impute with column median
- [ ] 🟡 Drop duplicate rows
- [ ] 🟡 Drop columns with >40% missing values
- [ ] 🟡 Standardize label names (e.g., strip whitespace, unify casing)
- [ ] 🟡 Save cleaned CSVs to `data/cicids2017/cleaned/` and `data/insdn/cleaned/`

---

## Milestone 2 — Baseline Models (XGBoost / Random Forest)
> **Goal:** Establish the benchmark F1-scores that GNN must beat by 15–25%.  
> **Estimated Duration:** 4–5 days

- [ ] 🟢 Open `notebooks/baseline_comparison.ipynb`
- [ ] 🟡 Select feature subset: use top-30 features from EDA (drop raw IPs, timestamps)
- [ ] 🟡 Encode labels: map attack type strings → integer class IDs
- [ ] 🟡 Apply **stratified train/val/test split** (70% / 15% / 15%)
- [ ] 🟡 Apply **z-score normalization** (`StandardScaler`) — fit only on train set
- [ ] 🔴 Handle class imbalance:
  - [ ] Compute class weights: `sklearn.utils.class_weight.compute_class_weight`
  - [ ] Apply to XGBoost via `scale_pos_weight` or sample weights
- [ ] 🟡 Train **Random Forest** classifier
  - [ ] `n_estimators=200`, `max_depth=None`, `class_weight='balanced'`
  - [ ] Evaluate: Accuracy, Precision, Recall, **F1-macro**, ROC-AUC
- [ ] 🟡 Train **XGBoost** classifier
  - [ ] Use `XGBClassifier` with `use_label_encoder=False`, `eval_metric='mlogloss'`
  - [ ] Evaluate same metrics
- [ ] 🟡 Generate and save **confusion matrices** for both models
- [ ] 🟢 Record baseline F1 scores in `docs/baseline_results.md`
  - This becomes the benchmark GNN must beat
- [ ] 🟢 Repeat on **InSDN** dataset for cross-dataset validation

---

## Milestone 3 — Feature Extraction & Graph Construction Pipeline
> **Goal:** Convert raw flow records into PyTorch Geometric graph objects.  
> **Estimated Duration:** 7–10 days  
> ⚠️ This is the most critical engineering milestone — get it right before touching the GNN.

### 3.1 Feature Extractor (`preprocessing/feature_extractor.py`)
- [ ] 🟡 Define the **node feature vector** schema:
  - [ ] Total bytes sent, total bytes received
  - [ ] Unique destination count (fan-out degree)
  - [ ] Average flow duration
  - [ ] Port entropy (Shannon entropy of destination ports)
  - [ ] Protocol distribution (fraction TCP / UDP / ICMP)
- [ ] 🟡 Define the **edge feature vector** schema:
  - [ ] Packet rate (packets/sec)
  - [ ] Byte rate (bytes/sec)
  - [ ] Flow inter-arrival time
  - [ ] TCP flag distribution (SYN ratio, FIN ratio, RST ratio)
  - [ ] Protocol type (one-hot: TCP=0, UDP=1, ICMP=2)
- [ ] 🟡 Write `extract_node_features(flow_df, window_df)` function
- [ ] 🟡 Write `extract_edge_features(flow_row)` function
- [ ] 🟢 Write unit tests for both functions with a small synthetic DataFrame

### 3.2 Graph Builder (`preprocessing/graph_builder.py`)
- [ ] 🔴 Implement **sliding window** over the dataset:
  - [ ] Configurable window size `W` (default: 5 seconds) and step size `S` (default: 1 second)
  - [ ] For each window: filter flows by `flow_start_time` in `[t, t+W]`
- [ ] 🔴 For each window, construct a `networkx.DiGraph`:
  - [ ] Nodes = unique IP addresses in the window
  - [ ] Edges = flows (src_ip → dst_ip), attributed with edge features
  - [ ] Node attributes = aggregated node features for that window
- [ ] 🔴 Convert `networkx` graph → `torch_geometric.data.Data` object:
  - [ ] `x` = node feature matrix `[num_nodes × node_feat_dim]`
  - [ ] `edge_index` = `[2 × num_edges]` (COO format)
  - [ ] `edge_attr` = edge feature matrix `[num_edges × edge_feat_dim]`
  - [ ] `y` = graph-level label (0=benign, 1=attack) derived from majority label in window
  - [ ] `node_y` = per-node label (for node-level loss)
- [ ] 🟡 Save all graph objects as a list: `torch.save(graph_list, 'data/graphs/cicids2017_graphs.pt')`
- [ ] 🟢 Print stats: number of graphs, avg nodes per graph, avg edges, class distribution
- [ ] 🟢 Visualize 2–3 sample graphs using `networkx.draw()` — benign vs. attack

### 3.3 Dataset Class (`preprocessing/graph_dataset.py`)
- [ ] 🟡 Create a `torch_geometric.data.Dataset` subclass: `SDNGraphDataset`
  - [ ] `__len__`: return number of graph snapshots
  - [ ] `__getitem__`: return `Data` object by index
- [ ] 🟡 Implement stratified split: `train_dataset`, `val_dataset`, `test_dataset`
- [ ] 🟡 Create `DataLoader` objects from each split using `torch_geometric.loader.DataLoader`
- [ ] 🟢 Verify a single batch loads correctly (check shapes of `x`, `edge_index`, `y`)

---

## Milestone 4 — GNN Model Implementation & Training
> **Goal:** Implement GAT model, train it, and beat the XGBoost baseline.  
> **Estimated Duration:** 8–12 days

### 4.1 Model Architecture (`models/gat_model.py`)
- [ ] 🔴 Implement `GATIntrustionDetector` class inheriting `torch.nn.Module`:
  - [ ] **GAT Layer 1:** `GATConv(in_channels=node_feat_dim, out_channels=64, heads=8, edge_dim=edge_feat_dim)`
  - [ ] **ELU activation + Dropout(0.3)**
  - [ ] **GAT Layer 2:** `GATConv(64*8, 128, heads=8, concat=False)`
  - [ ] **ELU activation**
  - [ ] **Node-level head:** `Linear(128, num_classes)` → per-node logits
  - [ ] **Graph-level head:** `global_mean_pool` → `Linear(128, num_classes)` → graph logits
- [ ] 🟡 Implement `forward(data)` returning both node and graph predictions
- [ ] 🟢 Instantiate model and print parameter count: `sum(p.numel() for p in model.parameters())`
- [ ] 🟢 Run a single forward pass with a dummy `Data` object — verify output shapes

### 4.2 Training Loop (`models/train.py`)
- [ ] 🔴 Implement training loop with:
  - [ ] **Weighted cross-entropy loss** (use class weights from Milestone 2)
  - [ ] Combined loss: `loss = α * node_loss + (1-α) * graph_loss` (try α=0.5)
  - [ ] **Adam optimizer** (`lr=0.001`, `weight_decay=1e-4`)
  - [ ] **ReduceLROnPlateau** scheduler (patience=5, factor=0.5)
  - [ ] **Early stopping** (patience=10 on val F1)
  - [ ] Save best model checkpoint: `torch.save(model.state_dict(), 'models/best_model.pt')`
- [ ] 🟡 Log per-epoch metrics: train loss, val loss, val F1, val precision, val recall
- [ ] 🟡 Save training curves to `results/training_curves.png`
- [ ] 🟢 (Optional) Integrate **MLflow**: `mlflow.log_metric()` for all tracked values

### 4.3 Evaluation (`models/evaluate.py`)
- [ ] 🟡 Load best checkpoint and run on test set
- [ ] 🟡 Compute and print:
  - [ ] F1-score (macro and per-class)
  - [ ] Precision, Recall
  - [ ] ROC-AUC (one-vs-rest)
  - [ ] Confusion matrix
- [ ] 🟡 Save confusion matrix plot to `results/gnn_confusion_matrix.png`
- [ ] 🔴 **Key comparison:** create a side-by-side table:
  - | Model | F1-Macro | Precision | Recall | AUC |
  - | Random Forest | x | x | x | x |
  - | XGBoost | x | x | x | x |
  - | GAT-IDS (ours) | x | x | x | x |
- [ ] 🟡 Repeat evaluation on **InSDN** test set (cross-dataset generalization check)
- [ ] 🟡 Specifically evaluate on **coordinated attack subsets only** — this is where the 15-25% F1 gain must show

### 4.4 Model Export (`models/export.py`)
- [ ] 🟡 Export trained model to **TorchScript** for inference:
  - [ ] `scripted = torch.jit.script(model); scripted.save('models/gat_ids.pt')`
- [ ] 🟢 Verify loaded TorchScript model gives identical outputs as original

---

## Milestone 5 — Real-Time Inference Engine
> **Goal:** Wrap the trained model in a fast inference pipeline.  
> **Estimated Duration:** 4–6 days

### 5.1 Inference Engine (`inference/inference_engine.py`)
- [ ] 🟡 Load TorchScript model at startup
- [ ] 🟡 Implement `InferenceEngine` class with method:
  - `predict(graph: Data) -> dict` returning `{node_scores, graph_label, confidence, latency_ms}`
- [ ] 🔴 Implement **end-to-end latency benchmark**:
  - [ ] Time: feature extraction → graph build → model forward pass → decision
  - [ ] Assert total < 50ms on 100 consecutive runs
  - [ ] Log p50, p90, p99 latencies
- [ ] 🟢 If latency > 50ms, optimize:
  - [ ] Move model to GPU if available (`model.to('cuda')`)
  - [ ] Reduce GAT attention heads
  - [ ] Profile with `torch.profiler`

### 5.2 Classifier (`inference/classifier.py`)
- [ ] 🟡 Implement threshold-based decision: `confidence > θ (default: 0.85) → ATTACK`
- [ ] 🟡 Map graph-level and node-level predictions to attack type labels
- [ ] 🟡 Implement `get_attack_sources(node_scores, node_ips) -> List[str]` — returns flagged source IPs
- [ ] 🟢 Write unit tests with mocked model outputs

---

## Milestone 6 — SDN Controller Integration (Ryu)
> **Goal:** Connect the IDS engine to Ryu and enable live flow monitoring.  
> **Estimated Duration:** 7–10 days  
> ⚠️ Highest integration risk — start Mininet testing early.

### 6.1 Ryu Application (`controller/ryu_app.py`)
- [ ] 🔴 Create a custom Ryu app inheriting `app_manager.RyuApp`:
  - [ ] Register for `EventOFPSwitchFeatures` (handle new switch connections)
  - [ ] Register for `EventOFPFlowStatsReply` (receive flow statistics)
  - [ ] Install default table-miss flow entry on switch connect
- [ ] 🔴 Implement flow statistics polling:
  - [ ] Every 2 seconds: send `OFPFlowStatsRequest` to all connected switches
  - [ ] Parse reply into a list of flow dicts (src_ip, dst_ip, port, protocol, bytes, packets, duration)
- [ ] 🟡 Expose flow data via a **thread-safe queue** for the IDS engine to consume
- [ ] 🟢 Test: run `ryu-manager controller/ryu_app.py` and verify it connects to a Mininet switch

### 6.2 Flow Collector (`controller/flow_collector.py`)
- [ ] 🟡 Implement `FlowCollector` that reads from the Ryu queue
- [ ] 🟡 Accumulate flows into a sliding window buffer (deque of last `W` seconds of flows)
- [ ] 🟡 Expose `get_current_window() -> pd.DataFrame` for the feature extractor

### 6.3 Ryu ↔ IDS REST Bridge (`api/app.py`)
- [ ] 🟡 Create a Flask app with endpoints:
  - [ ] `POST /flows` — Ryu pushes flow data here
  - [ ] `GET /alerts` — IDS engine polls for pending alerts
  - [ ] `POST /mitigate` — Mitigation engine pushes flow rules here
- [ ] 🟢 Add basic API key auth header check (security hygiene)
- [ ] 🟢 Run Flask on `localhost:5000` and verify with `curl` tests

---

## Milestone 7 — Mitigation Engine
> **Goal:** Automatically generate and push OpenFlow rules to block attack traffic.  
> **Estimated Duration:** 4–5 days

### 7.1 Rule Generator (`mitigation/mitigation_engine.py`)
- [ ] 🔴 Implement `MitigationEngine` class:
  - [ ] `generate_rule(src_ip, attack_type) -> OFPFlowMod`
  - [ ] Priority: `65535` (highest, overrides all other rules)
  - [ ] Action: `DROP`
  - [ ] Idle timeout: `60` seconds (auto-expire stale rules)
- [ ] 🟡 Map attack type to mitigation action:
  - [ ] `DDoS` → DROP all flows from `src_ip`
  - [ ] `PortScan` → Rate-limit (set max packet rate via meter)
  - [ ] `LateralMovement` → Block all egress from `src_ip`
- [ ] 🟡 Push rules via Ryu REST API: `POST http://localhost:8080/stats/flowentry/add`
- [ ] 🟡 Implement **mitigation log**: append each action to `logs/mitigation_log.jsonl`

### 7.2 Validation
- [ ] 🔴 In Mininet: inject a known attack (e.g., `hping3` DDoS from h1 to h3)
- [ ] 🔴 Verify the pipeline triggers mitigation and drops traffic within 2 round-trips
- [ ] 🟡 Measure: time from first malicious flow detected → flow rule installed on switch
- [ ] 🟡 Capture with `tcpdump` before/after mitigation to confirm drop rate ≥ 70%

---

## Milestone 8 — End-to-End Integration & System Testing
> **Goal:** All components working together in a live Mininet environment.  
> **Estimated Duration:** 7–10 days

### 8.1 Mininet Topology Setup
- [ ] 🟡 Write `topology/sdn_topology.py` using Mininet Python API:
  - [ ] 2 OpenFlow switches (s1, s2)
  - [ ] 4–6 hosts (h1–h6) distributed across switches
  - [ ] Link s1 ↔ s2 with configurable bandwidth
- [ ] 🟡 Verify topology with `sudo python topology/sdn_topology.py`
- [ ] 🟢 Test basic connectivity: `pingall` from Mininet CLI

### 8.2 Attack Traffic Injection
- [ ] 🟡 Script DDoS attack: `hping3 -S --flood -V -p 80 <victim_ip>` from attacker host
- [ ] 🟡 Script port scan: `nmap -sS <target_ip>` from attacker host
- [ ] 🟡 Write `scripts/inject_attack.py` to automate attack scenarios programmatically
- [ ] 🟡 Capture ground-truth traffic with `tcpdump -i s1-eth1 -w capture.pcap`

### 8.3 Full Pipeline Test
- [ ] 🔴 Start all components in order:
  1. [ ] `sudo mn --custom topology/sdn_topology.py --controller remote`
  2. [ ] `ryu-manager controller/ryu_app.py`
  3. [ ] `python api/app.py` (Flask bridge)
  4. [ ] `python inference/inference_engine.py` (polling mode)
  5. [ ] Inject attack from Mininet CLI
- [ ] 🔴 Confirm end-to-end: attack injected → GNN detects → mitigation rule pushed → traffic drops
- [ ] 🟡 Measure and log total pipeline latency for 20 attack runs
- [ ] 🟡 Confirm p90 latency < 50ms

### 8.4 Stress Testing
- [ ] 🟡 Simulate simultaneous benign + attack traffic
- [ ] 🟡 Measure false positive rate: how often benign flows trigger mitigation
- [ ] 🟡 Target: false positive rate < 5%

---

## Milestone 9 — Results, Analysis & Documentation
> **Goal:** Compile all results, write analysis, and prepare submission materials.  
> **Estimated Duration:** 5–7 days

### 9.1 Results Compilation
- [ ] 🟢 Collect all metric outputs into `results/final_results.md`:
  - [ ] Baseline vs. GNN F1 comparison table
  - [ ] Per-attack-type breakdown (which attacks improved most?)
  - [ ] Latency measurements (mean, p50, p90, p99)
  - [ ] Mitigation effectiveness (% traffic dropped, time-to-mitigation)
- [ ] 🟢 Generate final plots:
  - [ ] Training curves (loss + F1 over epochs)
  - [ ] Confusion matrices (baseline vs. GNN)
  - [ ] Latency histogram
  - [ ] ROC curves (one per class)

### 9.2 Ablation Studies (if time permits)
- [ ] 🟡 Compare GAT vs. GCN vs. GraphSAGE on same data
- [ ] 🟡 Compare window sizes: W=1s vs. W=3s vs. W=5s — effect on F1 and latency
- [ ] 🟡 Evaluate with and without edge features — how much do they matter?

### 9.3 Final Documentation
- [ ] 🟢 Update `README.md` with:
  - [ ] Full setup and run instructions
  - [ ] Architecture overview (link to architecture doc)
  - [ ] Final result numbers
- [ ] 🟢 Write `docs/design_decisions.md` — justify every major choice
- [ ] 🟢 Ensure all notebooks are cleaned and re-run top-to-bottom
- [ ] 🟢 Tag final release on GitHub: `git tag v1.0.0`

---

## Milestone 10 — Presentation & Demo
> **Goal:** A live, convincing demonstration and strong report.  
> **Estimated Duration:** 3–4 days

- [ ] 🟢 Prepare 15–20 slide deck covering:
  - [ ] Problem statement (why tabular ML fails)
  - [ ] Architecture walkthrough
  - [ ] Key results (F1 improvement, latency, mitigation)
  - [ ] Live demo plan
- [ ] 🔴 Prepare a **live demo script**:
  - [ ] Pre-start all services (Mininet, Ryu, Flask, Inference Engine)
  - [ ] Show benign traffic → no alert
  - [ ] Inject DDoS → show GNN alert + mitigation firing in terminal
  - [ ] Show `mitigation_log.jsonl` live update
- [ ] 🟡 Have a **pre-recorded video backup** of the demo in case of Mininet issues
- [ ] 🟢 Rehearse demo at least twice with the full team

---

## Task Assignment Template

| Milestone | Owner | Status | ETA |
|---|---|---|---|
| M0 — Setup | All | ⬜ Not Started | |
| M1 — EDA | Vaishnavi | ⬜ Not Started | |
| M2 — Baselines | Ashish | ⬜ Not Started | |
| M3 — Graph Pipeline | Jishnu | ⬜ Not Started | |
| M4 — GNN Model | Abhishek + Jishnu | ⬜ Not Started | |
| M5 — Inference Engine | Jishnu | ⬜ Not Started | |
| M6 — Ryu Integration | Abhishek | ⬜ Not Started | |
| M7 — Mitigation | Ashish | ⬜ Not Started | |
| M8 — Integration Testing | All | ⬜ Not Started | |
| M9 — Results & Docs | Vaishnavi | ⬜ Not Started | |
| M10 — Presentation | All | ⬜ Not Started | |

> **Status legend:** ⬜ Not Started · 🟦 In Progress · ✅ Done · 🔴 Blocked

---

## Critical Path (Must Not Slip)

```
M0 (Setup)
  └──► M1 (EDA)
         └──► M2 (Baselines) ──────────────────────────────────────────┐
         └──► M3 (Graph Pipeline)                                       │
                └──► M4 (GNN Training) ◄──── needs baseline F1 target ─┘
                       └──► M5 (Inference Engine)
                              └──► M8 (Integration Testing) ◄── M6 (Ryu) ◄── M7 (Mitigation)
                                     └──► M9 (Results)
                                            └──► M10 (Presentation)
```

**M3 (Graph Pipeline)** is the single biggest risk — if graph construction is slow or buggy, everything downstream is delayed. **Start M3 as early as possible.**

---

*Generated from GNN_IDS_Architecture.md · Batch B24 · AY 2025-26*
