# Phase 2 Task List
## GNN-Based Intrusion Detection System for SDN

**Project:** Major Project — 6th Semester, Batch B24  
**Purpose:** Step-by-step implementation plan for Phase 2 — turning the offline MVP into a live, topology-aware SDN IDS with automated mitigation  
**References:** `Docs/GNN_IDS_Architecture.md`, `Docs/GNN_IDS_TODO.md` (Milestones 5–10), `Docs/tasks_phase1.md`

---

## How to use this file

- Follow the milestones in order. Milestone 0 blocks everything else.
- Milestone 5 (SDN environment) has no dependency on the ML work — start it in parallel with Milestone 1.
- Each task includes an expected output so progress is easy to verify.
- File paths in `code` are the intended locations for new or changed code.
- Phase 2 ends when an attack injected in Mininet is detected by the GNN and blocked automatically, with measured latency and false-positive rate.

---

## Phase 1 Starting Point (summary)

Already implemented: data cleaning, RF/XGBoost baselines, sliding-window graph builder, graph-level GAT classifier with threshold tuning, evaluation script, end-to-end pipeline script, and a read-only FastAPI + Chart.js results dashboard.

Known gaps that Phase 2 must fix first:
- The CICIDS2017 files in use (`MachineLearningCVE` variant) have **no IP or timestamp columns**, so graphs were built from synthetic nodes (`src_<row index>` → `dst_<port>`) and row-index "time". The graphs carry no real host-to-host topology.
- InSDN was cleaned but **every row is labelled 1 (attack)** because its benign label is `Normal`, not `BENIGN`.
- The model has **no node-level output**, so it cannot say which IP to block.
- Features come from CICFlowMeter (~78 columns); a live OpenFlow controller cannot produce most of them.

---

## Milestone 0 — Phase 1 Carry-over Fixes
**Goal:** Remove bugs and methodology issues before building anything new on top.

### 0.1 Bug fixes
- [x] Fix `normalize_label()` in `preprocessing/clean_data.py` to treat both `benign` and `normal` as class 0
- [x] Re-run cleaning on InSDN and verify the class distribution (expect ~68k benign / ~275k attack)
- [x] Replace `fillna(method="ffill")` / `fillna(method="bfill")` in `preprocessing/graph_builder.py` with `.ffill().bfill()` (the old form fails on pandas 3.x)
- [x] Parse timestamps with an explicit format (InSDN uses `12/1/2020 1:14`) instead of relying on inference

**Expected output:** Correctly labelled InSDN data and a graph builder that runs on files with real timestamps.

### 0.2 Configuration and reproducibility
- [x] Make every script (`train_baselines`, `graph_builder`, `train_gnn`, `evaluate`) read `configs/*.yaml` instead of hardcoded values
- [x] Create `configs/phase2.yaml` for the new pipeline
- [x] Pin exact package versions in `requirements.txt` (currently only lower bounds)
- [x] Record the git commit hash and config in every metrics JSON

**Expected output:** Any result can be reproduced from a config file and a commit hash.

### 0.3 Evaluation methodology
- [x] Replace the random graph split in `preprocessing/graph_dataset.py` with a **time-ordered split** (train on earlier windows, test on later windows) to stop neighbouring windows leaking between train and test
- [x] Save GNN probabilities during evaluation and report ROC-AUC and PR-AUC
- [x] Save the training-curve plot to `results/training_curves.png`

**Expected output:** Honest, leakage-free metrics comparable to the baselines.

---

## Milestone 1 — Data v2: Real Topology and SDN-Compatible Features
**Goal:** Train on data where nodes are real hosts, using only features that a live SDN controller can supply.

### 1.1 Acquire datasets with IP and time columns
- [ ] Download the CICIDS2017 **`TrafficLabelling` / GeneratedLabelledFlows** CSVs (include `Flow ID`, `Source IP`, `Source Port`, `Destination IP`, `Timestamp`, `Protocol`) *(needs the dataset download (see status notes))*
- [x] Handle non-UTF-8 characters in some CICIDS label strings (read with `encoding="latin-1"` and normalise)
- [ ] Check the timestamp resolution in both datasets (minute-level timestamps need longer windows or ordering within the minute) *(done for InSDN; CICIDS2017 labelled files not available)*
- [ ] Store in `data/cicids2017_labelled/raw/` and keep the old variant for baseline reference *(needs the dataset download)*

**Expected output:** Two datasets (CICIDS2017 labelled + InSDN) with real IPs and timestamps.

### 1.2 Define the OpenFlow-compatible feature set
- [x] List the fields OpenFlow 1.3 flow stats provide: match fields (`ipv4_src`, `ipv4_dst`, `ip_proto`, `tcp/udp_src`, `tcp/udp_dst`), `packet_count`, `byte_count`, `duration_sec/nsec`
- [x] Define derived features: packets/s, bytes/s, mean packet size, protocol one-hot, destination-port bucket (well-known / registered / ephemeral), per-poll deltas
- [x] Map each derived feature to its equivalent column in CICIDS2017 and InSDN
- [x] Write the schema to `Docs/feature_schema_phase2.md`
- [x] Implement `preprocessing/openflow_features.py` so the **same function** is used for offline datasets and live controller data

**Expected output:** One feature definition shared by training and live inference (no train/serve mismatch).

### 1.3 Labels
- [x] Define a multi-class label map shared across datasets: `Benign`, `DDoS`, `DoS`, `Probe/PortScan`, `BruteForce`, `WebAttack`, `Botnet`, `Other`
- [x] Keep the binary label as a secondary target
- [x] Derive **node labels**: a host is `attacker` if it sources any attack flow in the window, `victim` if it only receives attack flows, otherwise `benign`
- [x] Write the label scheme to `Docs/label_scheme.md`

**Expected output:** Graph-level, node-level, and multi-class labels available for every window.

### 1.4 EDA (carried over from Phase 1)
- [x] Create `notebooks/eda.ipynb`: class distribution, per-attack host fan-out/fan-in, window size statistics *(done as `scripts/eda_insdn.py` (Jupyter not installed))*
- [x] Plot sample benign vs. attack graphs (e.g. a DDoS star vs. a port-scan fan-out)
- [x] Summarise findings in `Docs/eda_summary.md`

**Expected output:** Evidence that attack types have distinct graph structure — the core justification for a GNN.

---

## Milestone 2 — Graph Pipeline v2
**Goal:** Build real host-to-host graph snapshots with node labels.

### 2.1 Graph builder rewrite
- [x] Nodes = unique IPs in the window; edges = flows (src IP → dst IP) with OpenFlow-compatible edge features
- [x] Node features: bytes sent/received (actual byte counts, not `Flow Bytes/s`), packets sent/received, fan-out, fan-in, **destination-port entropy**, protocol mix, mean flow duration
- [x] Attach `node_y`, `node_ip` (for mapping predictions back to hosts), `y` (binary), `y_multi` (multi-class) to each `Data` object
- [x] Make window and step size configurable; test W ∈ {10s, 30s, 60s} given timestamp resolution
- [x] Remove the synthetic-node fallback, or keep it behind an explicit flag with a warning

**Expected output:** `data/graphs/{cicids,insdn}_v2.pt` with real topology and node labels.

### 2.2 Performance and scale
- [x] Replace the per-window DataFrame filter with a single sorted pass (two-pointer or `searchsorted`) so graph building scales to millions of flows
- [x] Use every row (no `max_graphs_per_file` step inflation that skips ~95% of flows)
- [ ] Save as a PyG `InMemoryDataset` (or sharded files) instead of one large pickled list *(not done: graphs are saved as one dict of lists (fast enough at this size))*

**Expected output:** Full-dataset graph generation in minutes, not hours.

### 2.3 Validation
- [x] Print graph statistics: graph count, mean/max nodes and edges, class balance per split
- [x] Unit test: a 6-flow synthetic DataFrame produces the expected nodes, edges, and labels

**Expected output:** Verified graph snapshots ready for training.

---

## Milestone 3 — Baselines v2 (Fair Comparison)
**Goal:** Make baseline vs. GNN numbers directly comparable.

- [x] Retrain RF and XGBoost on the **same OpenFlow-compatible features** and the **same time-based split** as the GNN
- [x] Add a train/validation/test split to the baselines (currently 80/20 with no validation)
- [x] Report baselines at two levels: per-flow, and per-window (aggregate flow predictions → window label) so they can be compared to the GNN's graph output
- [x] Add a per-host baseline (tabular host features → attacker/benign) to compare with the GNN node head
- [x] Evaluate on **coordinated-attack subsets** (DDoS, PortScan, Botnet) separately — this is where the 15–25% F1 gain is claimed
- [x] Write `Docs/baseline_results.md` *(baseline results are section 2 of `results/phase2/final_results.md`)*

**Expected output:** A fair comparison table at flow, window, and host level.

---

## Milestone 4 — GNN Model v2
**Goal:** A model that classifies the window, identifies attacker hosts, and names the attack type.

### 4.1 Architecture
- [x] Add a **node-level head** to `models/gat_model.py` → per-node attacker probability *(new model in `models/gnn_v2.py`; Phase 1 model kept)*
- [x] Switch the graph head to multi-class output
- [x] `forward()` returns `(graph_logits, node_logits)`
- [x] Verify with a dummy `Data` object and print the parameter count

**Expected output:** A multi-task GAT with node and graph outputs.

### 4.2 Training
- [x] Combined loss: `α · node_loss + (1 − α) · graph_loss` (start with α = 0.5, tune)
- [x] Class-weighted cross-entropy for both heads
- [x] Add a `ReduceLROnPlateau` scheduler
- [x] Early stopping on macro-F1 across both tasks *(mean of window F1, host F1 and major-class macro-F1)*
- [ ] (Optional) MLflow logging *(not done (optional))*

**Expected output:** `models/checkpoints/best_gat_v2.pt` with node and graph metrics.

### 4.3 Evaluation
- [ ] Per-class precision, recall, F1; macro and weighted averages; one-vs-rest ROC-AUC *(all done except per-class one-vs-rest ROC-AUC; binary ROC-AUC and PR-AUC reported)*
- [x] Node-level metrics: attacker-detection precision/recall
- [x] Multi-class confusion matrix
- [ ] Cross-dataset generalisation: train on CICIDS2017 → test on InSDN, and the reverse *(blocked on the CICIDS2017 TrafficLabelling download)*
- [x] Final comparison table: RF vs. XGBoost vs. GAT-IDS (overall + coordinated-attack subsets)

**Expected output:** `results/gnn_v2_metrics.json` and `results/final_comparison.md`.

### 4.4 Ablations
- [x] GAT vs. GCN vs. GraphSAGE on identical data
- [x] Window size: 10s vs. 30s vs. 60s — effect on F1 and latency *(InSDN timestamps are minute-level, so tested as 50 / 100 / 200-flow windows)*
- [x] With vs. without edge features
- [ ] (Optional) Temporal extension: GRU over the last T = 5 snapshots for slow scans *(not done (optional))*

**Expected output:** An ablation table justifying each design choice.

### 4.5 Export
- [x] Create `models/export.py`: export to TorchScript (`models/gat_ids.pt`)
- [x] Bundle normalisation stats, label map, and decision thresholds with the exported model
- [x] Verify the exported model gives identical outputs to the original on 100 test graphs

**Expected output:** A self-contained deployable model artifact.

---

## Milestone 5 — SDN Environment Setup
**Goal:** A working Mininet + controller lab. Start in parallel with Milestone 1.

### 5.1 Environment
- [x] Set up an **Ubuntu 22.04 VM or privileged Docker container** for Mininet + Open vSwitch (the dev machine runs Arch with Python 3.14; Mininet needs root and a supported distro) *(Docker: `docker/Dockerfile.sdn-lab`)*
- [x] Choose the controller: **os-ken** (maintained fork of Ryu) or Ryu in a separate Python 3.8/3.9 environment (Ryu is unmaintained and fails on modern Python) *(os-ken 2.8.1 in the lab image)*
- [x] Keep the controller and the IDS engine (PyTorch) in separate processes/environments, connected over REST
- [x] Install attack and traffic tools: `hping3`, `nmap`, `iperf3`, `tcpdump`
- [x] Verify: `sudo mn --test pingall` and controller start-up succeed
- [x] Document the setup in `Docs/sdn_lab_setup.md` (or add a `docker/` folder with a Dockerfile)

**Expected output:** Any team member can bring up the lab from the documentation.

### 5.2 Topology
- [x] Create `topology/sdn_topology.py`: 2 OpenFlow 1.3 switches (s1, s2), 6 hosts, configurable link bandwidth
- [x] Assign roles: attacker hosts, victim/server hosts, benign clients
- [x] Verify with `pingall` against the remote controller

**Expected output:** A reproducible test topology.

---

## Milestone 6 — Controller App and Flow Collection
**Goal:** Stream per-flow statistics from switches into the IDS.

### 6.1 Controller application (`controller/ids_controller.py`)
- [x] Handle `EventOFPSwitchFeatures`: install the table-miss entry
- [x] Implement **L3/L4 reactive forwarding** that installs flows matching `ipv4_src`, `ipv4_dst`, `ip_proto`, and L4 ports (a plain `simple_switch_13` only matches MAC addresses, so per-IP flow stats would be missing)
- [x] Poll `OFPFlowStatsRequest` on all switches every 2 seconds
- [x] Parse `EventOFPFlowStatsReply` into flow dicts: src/dst IP, ports, protocol, packets, bytes, duration, switch ID
- [x] Compute per-poll deltas (flow stats counters are cumulative)
- [x] Push flow batches to the IDS engine (`POST /flows`)

**Expected output:** Live flow records arriving at the IDS every 2 seconds.

### 6.2 Flow collector (`controller/flow_collector.py`)
- [x] Thread-safe sliding-window buffer (deque) holding the last W seconds of flows
- [x] Deduplicate flows seen on multiple switches along the path
- [x] `get_current_window() -> pd.DataFrame` using the Milestone 1.2 feature schema

**Expected output:** Windows of live flows in the same format as training data.

### 6.3 Mininet data collection (sim-to-real gap)
- [x] Script labelled traffic runs: benign (`iperf3`, HTTP, ping), DDoS (`hping3 --flood`), port scan (`nmap -sS`), brute force
- [x] Record the controller's flow stats with ground-truth labels to `data/mininet/`
- [x] Evaluate the dataset-trained model on Mininet traffic; fine-tune if performance drops

**Expected output:** A small self-generated SDN dataset and evidence that the model transfers to live traffic.

---

## Milestone 7 — Inference Engine and Classifier
**Goal:** Fast, live predictions on each window.

### 7.1 Inference engine (`inference/inference_engine.py`)
- [x] Load the TorchScript model, normalisation stats, and label map at start-up
- [x] `InferenceEngine.predict(window_df) -> dict` returning `{graph_label, attack_type, confidence, node_scores, flagged_ips, latency_ms}`
- [x] Polling loop: every 2s → get window → build graph → predict → emit alert
- [x] GPU if available, CPU fallback

**Expected output:** A running engine producing one prediction per window.

### 7.2 Classifier (`inference/classifier.py`)
- [x] Threshold decision (confidence > θ, default 0.85 → ATTACK); `0.5 < confidence ≤ θ` → SUSPICIOUS (log only)
- [x] `get_attack_sources(node_scores, node_ips) -> list[str]`
- [x] Whitelist protection: never flag the controller, gateway, or configured infrastructure IPs
- [x] Alert de-duplication/cool-down so the same host is not re-flagged every window

**Expected output:** Clean, actionable alerts with source IPs and attack types.

### 7.3 Latency benchmark (`scripts/benchmark_latency.py`)
- [x] Time each stage: feature extraction → graph build → forward pass → decision
- [x] 100+ consecutive runs; report p50 / p90 / p99
- [x] Target: **p90 < 50 ms**; if exceeded, profile with `torch.profiler`, reduce heads/hidden size

**Expected output:** `results/latency_report.json` and a latency histogram.

---

## Milestone 8 — Mitigation Engine
**Goal:** Automatically block or throttle attack sources.

### 8.1 Rule generation (`mitigation/mitigation_engine.py`)
- [x] `MitigationEngine.mitigate(src_ip, attack_type, confidence)`
- [x] Rules at priority 65535, idle timeout 60s (auto-expire) *(priorities by severity: drop 65535 > rate-limit 65435 > victim protection 65335)*
- [x] Map attack type → action:
  - [x] DDoS / DoS → DROP all flows from `src_ip`
  - [x] Port scan / probe → rate-limit with an OpenFlow meter (fallback: DROP if the switch lacks meter support)
  - [x] Lateral movement / botnet → block all egress from `src_ip`
- [x] Install rules via the controller (`POST /mitigate` handled by the controller app, or `ofctl_rest` `/stats/flowentry/add`) *(rules travel in the response to the controller's `POST /api/flows`)*
- [x] Manual unblock / rollback endpoint for false positives

**Expected output:** Attack sources blocked on the switch within one polling cycle.

### 8.2 Audit log
- [x] Append every action to `logs/mitigation_log.jsonl` (timestamp, IP, attack type, confidence, action, rule ID, expiry)
- [x] Log SUSPICIOUS (below-threshold) events separately

**Expected output:** A complete, timestamped record of all mitigation decisions.

---

## Milestone 9 — IDS API and Live Dashboard
**Goal:** Connect all components and make live activity visible.

### 9.1 API (extend `web/app.py`, FastAPI)
- [x] `POST /flows` — controller pushes flow batches
- [x] `GET /alerts` — recent alerts
- [x] `POST /mitigate`, `POST /unblock` — mitigation actions
- [x] `GET /topology` — current hosts, links, and flagged nodes
- [x] API-key header check on write endpoints; restrict CORS (currently `*`)
- [x] Update docs that still say "Flask" to FastAPI

**Expected output:** One API surface connecting controller, engine, and dashboard.

### 9.2 Live dashboard (extend `web/static/`)
- [x] Live alert feed via WebSocket or Server-Sent Events
- [x] Topology view with attacker/victim nodes highlighted in real time
- [x] Mitigation log table with active rules and expiry countdown
- [x] Live charts: flows/s, alerts/min, inference latency *(chart of confidence + active flows; latency and entries/s as live figures; no alerts/min chart)*
- [x] Keep the existing Phase 1 results pages as an "Offline Evaluation" tab

**Expected output:** A dashboard suitable for the live demo.

---

## Milestone 10 — End-to-End Integration and System Testing
**Goal:** Prove the full loop works under realistic conditions.

### 10.1 Attack injection
- [x] `scripts/inject_attack.py`: automated DDoS, port scan, and brute-force scenarios using the Mininet Python API
- [x] Capture ground truth with `tcpdump` on switch interfaces *(captured on the victim's link)*

### 10.2 Full pipeline test
- [x] Start order script (`scripts/run_phase2_demo.sh`): Mininet → controller → IDS API → inference engine → dashboard
- [x] Confirm: attack injected → GNN detects → rule installed → traffic drops
- [ ] 20 runs per attack type; log detection time and time-to-mitigation *(3 runs per attack type in the final evaluation (plus 29 collection runs))*

### 10.3 Targets to measure
- [ ] Attack traffic dropped **≥ 70%** within 2 controller round-trips (compare `tcpdump` before/after) *(≥ 70 % met in 12/12 runs, mean 95 %; timing not met: median 4.5 s, bounded by 2 s polling + 2-window confirmation)*
- [x] End-to-end detection latency **p90 < 50 ms** (excluding the polling interval)
- [x] False-positive rate **< 5%** under mixed benign + attack traffic
- [x] Stress test: simultaneous benign load plus multiple attackers *(DDoS from 2 attackers under continuous benign load; flood controller-saturation effects documented)*

**Expected output:** `results/system_test_report.md` with all measurements.

---

## Milestone 11 — Testing, Code Quality, and Documentation
**Goal:** A maintainable, reviewable codebase.

### 11.1 Tests (`tests/` is currently empty)
- [x] Unit tests: label normalisation, OpenFlow feature extraction, graph builder, classifier threshold logic, mitigation rule generation (mocked controller)
- [x] Integration test: synthetic flow batch → `/flows` → alert → mitigation call (mocked)
- [x] Run tests with `pytest`; optionally add a GitHub Actions workflow

### 11.2 Documentation
- [x] Update `README.md` with Phase 2 setup, architecture, and run instructions
- [x] Write `Docs/design_decisions.md` (why GAT, why OpenFlow-only features, why time-based split, why os-ken/Ryu)
- [x] Update `Execution.md` with the live-demo run order
- [x] Update `explain.md` with Phase 2 results and new Q&A

**Expected output:** Tested code and documentation that matches the implementation.

---

## Milestone 12 — Results and Review Readiness
**Goal:** Final results and a convincing demo.

- [x] `results/final_results.md`: baseline vs. GNN tables, per-attack breakdown, ablations, latency, mitigation effectiveness, false-positive rate
- [x] Final plots: training curves, confusion matrices, ROC curves, latency histogram, before/after mitigation traffic *(all in `results/phase2/`)*
- [x] Demo script: benign traffic → no alert; inject DDoS → alert + rule + traffic drop; show `mitigation_log.jsonl`
- [ ] Record a backup demo video *(team task)*
- [ ] Slide deck (15–20 slides) and at least two full-team rehearsals *(team task)*
- [ ] Tag the release: `git tag v2.0.0` *(waiting for the team to review and commit)*

**Expected output:** A complete Phase 2 submission.

---

## Status (2026-09-11)

Implemented, tested (43 pytest tests) and evaluated end to end; results in
`results/phase2/final_results.md`. Headline numbers:

- Attack-type macro-F1 on InSDN: GAT 0.993 vs XGBoost/RF 0.970; binary detection ≥ 0.997 for all models.
- Live Mininet evaluation (final system): 12/12 attack runs mitigated with ≥ 70 % of attack traffic dropped
  (mean 95 %), 0 benign hosts blocked, 0 rules in benign-only runs, median time to mitigation 4.5 s.
- Detection latency p90 ≈ 10 ms per window.

Open items:
1. **CICIDS2017 TrafficLabelling CSVs** (with IP columns) for the cross-dataset test: download from
   https://www.unb.ca/cic/datasets/ids-2017.html (registration form) → `GeneratedLabelledFlows.zip`,
   unzip into `data/cicids2017_labelled/raw/`. The pipeline already supports the column names.
2. Team: slides, backup demo video (`KEEP_IDS=1 bash scripts/run_phase2_demo.sh`), rehearsals.
3. Review and commit the `phase2` branch, then tag `v2.0.0`.
4. Known limitations: time to mitigation is seconds (2 s polling + confirmation), not "2 round-trips";
   during a spoofed DDoS, legitimate clients keep only ~3 % of their traffic to the rate-limited victim;
   the first burst of a port scan completes before mitigation (detection ~17 s).

---

## Recommended Order of Execution

```
M0 (Fixes) ──► M1 (Data v2) ──► M2 (Graphs v2) ──► M3 (Baselines v2)
                                      │
                                      └──► M4 (Model v2) ──► M7 (Inference) ──┐
                                                                             ├──► M10 (Integration) ──► M12 (Results)
M5 (SDN lab, in parallel) ──► M6 (Controller + collector) ──► M8 (Mitigation) ┘
                                                  M9 (API + dashboard) ───────┘
M11 (Tests + docs) runs continuously alongside every milestone.
```

---

## Suggested Team Split

| Milestone | Suggested owner | Status |
|---|---|---|
| M0 — Carry-over fixes | Jishnu | ✅ Done |
| M1 — Data v2 + EDA | Vaishnavi | 🟦 In Progress |
| M2 — Graph pipeline v2 | Jishnu | ✅ Done |
| M3 — Baselines v2 | Ashish | ✅ Done |
| M4 — GNN model v2 | Abhishek + Jishnu | ✅ Done |
| M5 — SDN lab setup | Abhishek | ✅ Done |
| M6 — Controller + collector | Abhishek | ✅ Done |
| M7 — Inference engine | Jishnu | ✅ Done |
| M8 — Mitigation | Ashish | ✅ Done |
| M9 — API + live dashboard | Vaishnavi | ✅ Done |
| M10 — Integration testing | All | ✅ Done |
| M11 — Tests + docs | All | ✅ Done |
| M12 — Results + demo | All | 🟦 In Progress |

> **Status legend:** ⬜ Not Started · 🟦 In Progress · ✅ Done · 🔴 Blocked

---

## Phase 2 Done Criteria

Phase 2 can be considered complete when all of the following are true:
- Graphs use real host IPs and the model outputs per-node attacker scores and attack types
- GNN and baselines are compared on the same features and the same time-based split, including coordinated-attack subsets
- The model runs on live flow stats from the controller using the same feature code as training
- An attack injected in Mininet is detected and blocked automatically
- Latency, drop rate, and false-positive rate are measured against the targets
- Tests pass and documentation matches the implementation

---

## Top Risks

| Risk | Mitigation |
|---|---|
| Model trained on dataset features fails on live OpenFlow stats | Use one shared OpenFlow-only feature module (M1.2) and fine-tune on Mininet data (M6.3) |
| Ryu/Mininet will not install on the dev machine | Use an Ubuntu 22.04 VM/Docker and os-ken; start M5 in week 1 |
| Coarse (minute-level) timestamps weaken windowed graphs | Longer windows, ordering within the minute, and live data where timing is exact |
| GNN does not beat baselines overall | Focus the claim on coordinated attacks and attacker identification (node head), which tabular per-flow models cannot do directly |
| False positives block legitimate hosts | Whitelist, cool-down, auto-expiring rules, and a manual unblock endpoint |
