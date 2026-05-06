# GNN-Based Intrusion Detection System for Software-Defined Networks
## Architecture Document

**Project:** Major Project — 6th Semester, Batch B24  
**Institute:** Siddaganga Institute of Technology, Tumakuru  
**Team:** Abhishek (1SI23CI002) · Ashish Kumar Bhagat (1SI23CI009) · Jishnu Khargharia (1SI23CI017) · Vaishnavi A Hachadad (1SI23CI056)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Component Breakdown](#3-component-breakdown)
4. [Data Pipeline](#4-data-pipeline)
5. [GNN Model Architecture](#5-gnn-model-architecture)
6. [SDN Integration Layer](#6-sdn-integration-layer)
7. [Mitigation Engine](#7-mitigation-engine)
8. [Technology Stack](#8-technology-stack)
9. [Dataset & Training Strategy](#9-dataset--training-strategy)
10. [Performance Targets](#10-performance-targets)
11. [Deployment Architecture](#11-deployment-architecture)

---

## 1. System Overview

Traditional IDS solutions operate on tabular, per-flow feature vectors — making them blind to **coordinated, multi-flow attack patterns** such as distributed DDoS and lateral movement. This project addresses that gap by treating SDN traffic as a **dynamic graph**, where nodes represent network entities (hosts, switches) and edges represent traffic flows. A Graph Neural Network (GNN) learns structural and temporal attack patterns that tabular ML methods like XGBoost, Random Forest, and SVM fundamentally cannot capture.

### Core Problem

| Challenge | Traditional IDS | GNN-IDS (This Project) |
|---|---|---|
| Coordinated low-volume attacks | ❌ Missed (no topology context) | ✅ Detected via graph structure |
| SDN controller as single point of failure | ❌ Unmonitored | ✅ Integrated monitoring |
| Real-time mitigation | ❌ Manual intervention | ✅ Automated within 2 round-trips |
| Feature engineering overhead | High | Low (GNN learns representations) |

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SDN NETWORK PLANE                           │
│                                                                     │
│   [Host A] ──┐                              ┌── [Host D]           │
│   [Host B] ──┤──► [OpenFlow Switch 1] ──────┤── [Host E]           │
│   [Host C] ──┘         │                   └── [Host F]            │
│                         │                                           │
│                  [OpenFlow Switch 2]                                │
│                         │                                           │
└─────────────────────────┼───────────────────────────────────────────┘
                          │  OpenFlow Protocol (Flow Stats)
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SDN CONTROLLER (Ryu / ONOS)                    │
│                                                                     │
│   ┌─────────────────┐        ┌──────────────────────────────────┐  │
│   │  Flow Stat      │        │       REST / Northbound API      │  │
│   │  Collector      │◄──────►│  (Exposes flow data to IDS)      │  │
│   └────────┬────────┘        └──────────────────────────────────┘  │
│            │                                       ▲                │
└────────────┼───────────────────────────────────────┼────────────────┘
             │  Raw Flow Records                      │ Mitigation Commands
             ▼                                        │
┌─────────────────────────────────────────────────────────────────────┐
│                     GNN-IDS ENGINE                                 │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────────┐  │
│  │   Feature    │──►│   Graph      │──►│    GNN Inference       │  │
│  │  Extraction  │   │  Builder     │   │    (PyTorch Geometric)  │  │
│  └──────────────┘   └──────────────┘   └───────────┬────────────┘  │
│                                                     │               │
│                                         ┌───────────▼────────────┐  │
│                                         │  Anomaly Classifier    │  │
│                                         │  (Attack / Benign)     │  │
│                                         └───────────┬────────────┘  │
│                                                     │               │
│                                         ┌───────────▼────────────┐  │
│                                         │  Mitigation Engine     │  │
│                                         │  (Flow Rule Generator) │  │
│                                         └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Breakdown

### 3.1 Flow Stat Collector
- Polls OpenFlow switches via the **Ryu/ONOS REST API** at fixed intervals
- Collects: source/destination IP & port, protocol, byte count, packet count, flow duration, flags
- Streams records to the Feature Extraction module in near real-time

### 3.2 Feature Extraction Module
Transforms raw flow records into node and edge features for graph construction.

**Node Features (per host/switch):**
- Total bytes sent/received
- Unique destination count (fan-out)
- Average flow duration
- Port entropy (indicator of scanning behaviour)

**Edge Features (per flow):**
- Packet rate, byte rate
- Flow inter-arrival time
- TCP flag distribution (SYN, FIN, RST ratios)
- Protocol type (one-hot encoded)

### 3.3 Dynamic Graph Builder
- Constructs a **timestamped graph snapshot** for each sliding time window (configurable: 1–5 seconds)
- Nodes = network entities (identified by IP)
- Edges = active flows between entities in the window
- Maintains a **temporal buffer** of last N snapshots for sequence-aware GNN variants

### 3.4 GNN Inference Engine
- Loads a pre-trained GNN model (see Section 5)
- Runs forward pass on the current graph snapshot
- Outputs a **per-node attack probability score** and a **graph-level label** (attack / benign)
- Target: inference latency < 50ms per flow batch

### 3.5 Anomaly Classifier
- Applies a threshold on node-level scores to flag suspicious entities
- Uses **multi-class classification** to distinguish attack types:
  - DDoS (volumetric)
  - Port Scan / Probing
  - Lateral Movement
  - Benign

### 3.6 Mitigation Engine
- Receives flagged flows and attack-source IPs from the classifier
- Generates **OpenFlow DROP / REDIRECT rules** via controller REST API
- Target: drop 70% of attack traffic within **2 round-trips** (~<100ms)
- Logs all mitigation actions with timestamps for audit

---

## 4. Data Pipeline

```
Raw PCAP / Flow Logs
        │
        ▼
┌───────────────────┐
│  Preprocessing    │   ← CICFlowMeter / custom parser
│  & Normalization  │   ← Handle missing values, z-score normalization
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Graph            │   ← Sliding window: W seconds
│  Construction     │   ← Nodes: IPs,  Edges: flows
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Train / Val /    │   ← 70% / 15% / 15% split
│  Test Split       │   ← Stratified by attack type
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  GNN Training     │   ← PyTorch Geometric
│  (Offline)        │   ← Cross-entropy loss + class weighting
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Model Export     │   ← TorchScript / ONNX for inference
│  & Versioning     │
└───────────────────┘
```

### Datasets Used

| Dataset | Description | Attack Types |
|---|---|---|
| **CICIDS2017** | Canadian Institute for Cybersecurity benchmark | DoS, DDoS, Brute Force, Web Attacks, Infiltration |
| **InSDN** | SDN-specific intrusion dataset | DDoS, Probe, R2L, U2R |

---

## 5. GNN Model Architecture

### 5.1 Model Choice: Graph Attention Network (GAT) + Temporal Component

```
Input Graph Snapshot
        │
        ▼
┌──────────────────────────────────────────┐
│          GAT Layer 1                     │
│  Multi-head attention (K=8 heads)        │
│  Node embedding: 64-dim                  │
│  Edge features injected via concat       │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│          GAT Layer 2                     │
│  Multi-head attention (K=8 heads)        │
│  Node embedding: 128-dim                 │
│  Dropout: 0.3 (training only)            │
└────────────────┬─────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌──────────────┐   ┌──────────────┐
│  Node-level  │   │  Graph-level │
│  MLP Head    │   │  Readout     │
│  (per-flow   │   │  (global     │
│   scoring)   │   │   mean pool) │
└──────┬───────┘   └──────┬───────┘
       │                  │
       ▼                  ▼
 Node Attack         Graph Attack
  Probability          Label
  [0.0 – 1.0]      [Benign / Attack]
```

### 5.2 Temporal Extension (Optional Phase 2)
- Stack last **T = 5 graph snapshots** and pass through a **GRU / Temporal GNN** layer
- Captures attack progression patterns (e.g., slow port scan over multiple windows)

### 5.3 Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (lr = 0.001) |
| Loss Function | Weighted Cross-Entropy (to handle class imbalance) |
| Batch Size | 32 graph snapshots |
| Epochs | 100 (with early stopping, patience = 10) |
| Evaluation Metric | F1-score (macro), Precision, Recall, ROC-AUC |

---

## 6. SDN Integration Layer

### 6.1 Controller: Ryu (Primary) / ONOS (Alternative)

```
GNN-IDS Engine
      │
      │  REST API calls
      ▼
┌─────────────────────────────────┐
│       Ryu SDN Controller        │
│                                 │
│  ┌─────────────────────────┐   │
│  │  IDS Ryu App Module     │   │
│  │  (custom Python app)    │   │
│  │  - Listens for alerts   │   │
│  │  - Generates flow mods  │   │
│  └──────────┬──────────────┘   │
└─────────────┼───────────────────┘
              │  OpenFlow 1.3 FlowMod
              ▼
        Network Switches
        (flow table update)
```

### 6.2 Flow Monitoring Sequence

```
Switch          Controller (Ryu)       GNN-IDS Engine
  │                    │                     │
  │── FlowStats ──────►│                     │
  │                    │── REST Poll ────────►│
  │                    │◄── Graph Snapshot ──│
  │                    │    + Attack Alert    │
  │◄── FlowMod DROP ──│                     │
  │   (attack flow)    │                     │
```

---

## 7. Mitigation Engine

### 7.1 Mitigation Decision Logic

```
Attack Alert Received
        │
        ▼
  Confidence > θ?  (threshold, default: 0.85)
        │
   YES  │  NO
        │  └──► Log as suspicious, continue monitoring
        ▼
  Identify source IPs / flows
        │
        ▼
  Generate OpenFlow rules:
    - Priority: 65535 (highest)
    - Action: DROP
    - Idle timeout: 60s (auto-expire)
        │
        ▼
  Push via Ryu REST API
        │
        ▼
  Log mitigation event
  (timestamp, source, confidence, action)
```

### 7.2 Mitigation Targets

| Attack Type | Mitigation Action |
|---|---|
| DDoS (volumetric) | Drop flows from flagged source IPs |
| Port Scan / Probing | Rate-limit or redirect to honeypot |
| Lateral Movement | Isolate source host (block all egress) |

---

## 8. Technology Stack

| Layer | Technology |
|---|---|
| **Network Emulation** | Mininet (virtual SDN topology) |
| **SDN Controller** | Ryu Framework / ONOS |
| **Flow Feature Extraction** | CICFlowMeter / custom Python parser |
| **GNN Framework** | PyTorch Geometric (PyG) |
| **Model Training** | PyTorch, scikit-learn (baseline comparison) |
| **Baseline Models** | XGBoost, Random Forest, SVM |
| **Graph Library** | NetworkX (graph construction), PyG (GNN ops) |
| **API / Integration** | Flask REST API (IDS ↔ Controller bridge) |
| **Experiment Tracking** | MLflow / Weights & Biases (optional) |
| **Visualization** | Matplotlib, Seaborn (metrics), Gephi (graph viz) |
| **Language** | Python 3.10+ |

---

## 9. Dataset & Training Strategy

### 9.1 Class Imbalance Handling
- Attack flows are far less frequent than benign flows in both CICIDS2017 and InSDN
- Strategies:
  - **Class-weighted loss function** (primary)
  - **SMOTE on node features** (secondary, applied offline)
  - **Oversampling rare attack types** during graph construction

### 9.2 Baseline Comparison Plan

| Model | Input Type | Expected F1 |
|---|---|---|
| Random Forest | Tabular flow features | ~0.85 |
| XGBoost | Tabular flow features | ~0.88 |
| **GNN-IDS (ours)** | Dynamic graphs | **Target: 0.93+** |

The 15–25% F1 improvement target applies specifically on **multi-stage / coordinated attack subsets** of CICIDS2017 where tabular models degrade the most.

---

## 10. Performance Targets

| Metric | Target |
|---|---|
| F1-Score Improvement | +15–25% over XGBoost on coordinated attacks |
| Inference Latency | < 50ms per flow batch |
| Attack Traffic Dropped | ≥ 70% within 2 controller round-trips |
| False Positive Rate | < 5% (critical for production environments) |

### 10.1 Latency Budget Breakdown

```
Flow stat collection:     ~5ms
Feature extraction:       ~5ms
Graph construction:       ~10ms
GNN forward pass:         ~20ms
Classifier decision:      ~2ms
Mitigation API call:      ~8ms
─────────────────────────────
Total:                   ~50ms  ✓
```

---

## 11. Deployment Architecture

### 11.1 Mininet Emulation Setup

```
┌──────────────────────────────────────────────────┐
│                  Linux Host Machine              │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │          Mininet Virtual Network           │  │
│  │  h1 ─── s1 ─── s2 ─── h3                 │  │
│  │  h2 ──────────────┘    h4                 │  │
│  └────────────────────────┬───────────────────┘  │
│                            │ OpenFlow             │
│  ┌─────────────────────────▼───────────────────┐  │
│  │         Ryu Controller (localhost)          │  │
│  └─────────────────────────┬───────────────────┘  │
│                            │ REST API             │
│  ┌─────────────────────────▼───────────────────┐  │
│  │    GNN-IDS Engine (Flask + PyTorch)         │  │
│  └─────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

### 11.2 Module Directory Structure

```
gnn-ids/
├── controller/
│   ├── ryu_app.py              # Custom Ryu application
│   └── flow_collector.py       # Flow stat polling
├── preprocessing/
│   ├── feature_extractor.py    # Raw flow → feature vectors
│   └── graph_builder.py        # Feature vectors → PyG graphs
├── models/
│   ├── gat_model.py            # GAT architecture definition
│   ├── train.py                # Training loop
│   └── evaluate.py             # Metrics and baseline comparison
├── inference/
│   ├── inference_engine.py     # Real-time inference wrapper
│   └── classifier.py           # Threshold-based classifier
├── mitigation/
│   └── mitigation_engine.py    # OpenFlow rule generator
├── api/
│   └── app.py                  # Flask REST API (IDS ↔ Controller)
├── data/
│   ├── cicids2017/             # Dataset (raw + preprocessed)
│   └── insdn/
├── notebooks/
│   ├── eda.ipynb               # Exploratory data analysis
│   └── baseline_comparison.ipynb
├── tests/
└── requirements.txt
```

---

## Appendix: Key Design Decisions

**Why GAT over GCN?**  
Graph Attention Networks learn *which neighbors matter* via attention weights — critical in traffic graphs where not all connected flows are equally relevant to an attack classification. GCN treats all neighbors equally.

**Why sliding window graphs over static snapshots?**  
Low-volume coordinated attacks (e.g., slow DDoS, lateral movement) unfold over time. A sliding window captures temporal flow evolution without requiring the entire network history in memory.

**Why Ryu over ONOS?**  
Ryu is Python-native, lightweight, and has simpler REST API integration for research prototypes. ONOS is kept as an alternative for production-scale evaluation.

**Why Mininet for emulation?**  
Mininet provides a full SDN emulation environment on a single machine with realistic OpenFlow behavior, making it ideal for controlled attack injection and mitigation validation without physical hardware.

---

*Document version 1.0 — Academic Year 2025-26*
