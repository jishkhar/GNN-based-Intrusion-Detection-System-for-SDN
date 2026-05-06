# GNN-Based Intrusion Detection System for Software-Defined Networks
### Major Project Documentation — 6th Semester | Batch ID: B24
**Institution:** Siddaganga Institute of Technology, Tumakuru  
**Department:** Computer Science and Engineering  
**Academic Year:** 2025–26

---

## Team Members

| Name | USN |
|------|-----|
| Abhishek | 1SI23CI002 |
| Ashish Kumar Bhagat | 1SI23CI009 |
| Jishnu Khargharia | 1SI23CI017 |
| Vaishnavi A Hachadad | 1SI23CI056 |

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Project Objectives](#project-objectives)
3. [Expected Outcomes](#expected-outcomes)
4. [Full Architecture Overview](#full-architecture-overview)
5. [Tech Stack](#tech-stack)
6. [Workflow & Data Pipeline](#workflow--data-pipeline)
7. [Module-by-Module Build Guide](#module-by-module-build-guide)
8. [Dataset Details](#dataset-details)
9. [GNN Model Design](#gnn-model-design)
10. [SDN Integration](#sdn-integration)
11. [Evaluation Metrics](#evaluation-metrics)
12. [Project Timeline & Milestones](#project-timeline--milestones)

---

## Problem Statement

Traditional Intrusion Detection Systems (IDS) struggle with **coordinated, low-volume attacks** such as:
- Distributed DDoS
- Lateral movement attacks

These attacks **evade signature-based rules** by exploiting **network topology and flow relationships** — something tabular ML models like **Random Forest** or **SVM** fundamentally cannot capture.

### Why SDN Makes This Harder

Software-Defined Networking (SDN) centralizes control into a controller, which introduces new vulnerabilities:
- **Single Point of Failure**: The controller itself becomes a high-value target.
- **Stealthy Probing**: Flow tables can be probed without triggering traditional thresholds.

### Why Existing Graph Methods Fall Short

- Require heavy manual **feature engineering**.
- Cannot meet **real-time inference constraints** (<50ms).
- Limited **production deployability**.

---

## Project Objectives

1. **Develop a GNN-based IDS** that models SDN traffic as dynamic graphs to detect coordinated low-volume attacks missed by traditional tabular ML.
2. **Achieve 15–25% F1-score improvement** over XGBoost baselines on public datasets: **CICIDS2017** and **InSDN**.
3. **Integrate with SDN controller** (Ryu/ONOS) for automated mitigation — dropping **70% of attack traffic within 2 round-trips**.
4. **Real-time inference latency under 50ms** per flow batch using **Mininet** emulation.

---

## Expected Outcomes

| Outcome | Target |
|--------|--------|
| F1-score improvement over XGBoost | 15–25% on CICIDS2017/InSDN |
| Inference latency per flow batch | < 50ms |
| Attack traffic dropped by SDN controller | 70% within 2 round-trips |

---

## Full Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     NETWORK LAYER (Mininet)                 │
│  Hosts ──► Switches ──► SDN Controller (Ryu / ONOS)        │
└────────────────────────┬────────────────────────────────────┘
                         │ Raw Packet/Flow Data
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              FLOW COLLECTOR & GRAPH BUILDER                 │
│  OpenFlow Stats / sFlow / NetFlow                           │
│  CICFlowMeter → Feature Extraction → Graph Construction     │
│  Nodes = IPs, Edges = Flows, Edge Features = Flow Stats     │
└────────────────────────┬────────────────────────────────────┘
                         │ Dynamic Graph (PyG / DGL format)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    GNN INFERENCE ENGINE                     │
│  GraphSAGE / GAT / GCN layers                               │
│  Node Embedding → Edge Classification → Attack Score        │
│  Batch inference: <50ms per flow batch                      │
└────────────────────────┬────────────────────────────────────┘
                         │ Attack Label + Confidence Score
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               MITIGATION & RESPONSE MODULE                  │
│  SDN Controller REST API                                    │
│  Flow Rule Injection: DROP / REDIRECT malicious flows       │
│  Target: 70% attack traffic blocked within 2 round-trips    │
└─────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. Network Emulation Layer
- **Tool:** Mininet
- Simulates real SDN topology: hosts, switches, links
- Connects to real SDN controllers (Ryu/ONOS)
- Used for integration testing and latency benchmarking

#### 2. Flow Collection & Graph Construction
- **Tools:** CICFlowMeter, sFlow-RT, or custom OpenFlow stat polling
- Collects per-flow features: duration, byte count, packet count, inter-arrival time, flags
- Builds a **dynamic graph** every T seconds (sliding window):
  - **Nodes** = IP addresses (src/dst)
  - **Edges** = individual flows
  - **Edge Features** = flow statistics vector

#### 3. GNN Inference Engine
- **Framework:** PyTorch Geometric (PyG) or DGL
- Architecture: GraphSAGE or Graph Attention Network (GAT)
- Input: Graph snapshot with node/edge features
- Output: Per-edge label (benign / attack type)
- Trained offline, deployed as a real-time inference server (FastAPI / gRPC)

#### 4. SDN Controller Integration
- **Controllers:** Ryu (Python) or ONOS (Java)
- REST API calls to push OpenFlow rules
- Mitigation: `DROP` rules for flagged src IPs or flow patterns
- Target: 2 round-trips (detection → rule push → effect)

---

## Tech Stack

### Core ML / Graph
| Component | Tool/Library |
|-----------|-------------|
| GNN Framework | PyTorch Geometric (PyG) or DGL |
| Deep Learning | PyTorch |
| Baseline Models | XGBoost, scikit-learn |
| Data Processing | pandas, NumPy |
| Graph Visualization | NetworkX, Gephi |

### Network / SDN
| Component | Tool/Library |
|-----------|-------------|
| Network Emulation | Mininet |
| SDN Controller | Ryu (recommended) or ONOS |
| Flow Protocol | OpenFlow 1.3 |
| Flow Feature Extraction | CICFlowMeter |
| Packet Capture | Scapy, Wireshark/tshark |

### Deployment / Serving
| Component | Tool/Library |
|-----------|-------------|
| Model Serving | FastAPI + Uvicorn |
| Containerization | Docker |
| Monitoring | Prometheus + Grafana (optional) |

### Datasets
| Dataset | Purpose |
|---------|---------|
| CICIDS2017 | Primary training/evaluation |
| InSDN | SDN-specific attack scenarios |

### Development
| Component | Tool |
|-----------|------|
| Language | Python 3.10+ |
| Version Control | Git + GitHub |
| Experiment Tracking | MLflow or Weights & Biases |
| Notebooks | Jupyter |

---

## Workflow & Data Pipeline

### Step-by-Step End-to-End Flow

```
[Raw Dataset / Live Traffic]
        │
        ▼
[Preprocessing]
  - Parse PCAP or CSV (CICIDS2017/InSDN)
  - Label encoding (attack types)
  - Normalize flow features
        │
        ▼
[Graph Construction]
  - Time-windowed snapshots (e.g., 5s windows)
  - Build graph: G = (V, E, X_node, X_edge)
  - V = unique IPs in window
  - E = flows between IPs
  - X_edge = [duration, bytes, packets, flags...]
        │
        ▼
[GNN Model Training]
  - Split: 70% train, 15% val, 15% test
  - Model: GraphSAGE or GAT
  - Loss: Binary or multi-class cross-entropy
  - Optimizer: Adam (lr=1e-3)
  - Early stopping on validation F1
        │
        ▼
[Evaluation]
  - Compare vs XGBoost baseline
  - Metrics: F1, Precision, Recall, AUC
  - Target: +15–25% F1 over XGBoost
        │
        ▼
[Inference Server]
  - Load trained model
  - Accept live graph snapshots via REST API
  - Return attack labels + confidence
  - Benchmark latency: must be <50ms
        │
        ▼
[SDN Controller (Ryu)]
  - Poll inference server
  - On attack detection: push DROP flow rule
  - Monitor mitigation effectiveness
```

---

## Module-by-Module Build Guide

### Module 1: Dataset Preparation

```python
# Download CICIDS2017
# Source: https://www.unb.ca/cic/datasets/ids-2017.html
# Files: Monday.pcap through Friday.pcap + CSV labels

import pandas as pd

df = pd.read_csv("CICIDS2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
df.columns = df.columns.str.strip()
df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

# Drop NaN, Inf
df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
df.dropna(inplace=True)
```

### Module 2: Graph Construction

```python
import networkx as nx
import torch
from torch_geometric.data import Data

def build_graph_snapshot(flows_df):
    """Convert a window of flows into a PyG graph."""
    nodes = list(set(flows_df['src_ip'].tolist() + flows_df['dst_ip'].tolist()))
    node_index = {ip: i for i, ip in enumerate(nodes)}
    
    edge_index = []
    edge_attr = []
    edge_labels = []
    
    feature_cols = ['duration', 'total_fwd_packets', 'total_bwd_packets',
                    'fwd_packet_length_mean', 'flow_bytes_s', 'flow_packets_s']
    
    for _, row in flows_df.iterrows():
        src = node_index[row['src_ip']]
        dst = node_index[row['dst_ip']]
        edge_index.append([src, dst])
        edge_attr.append(row[feature_cols].values.tolist())
        edge_labels.append(row['label'])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = torch.tensor(edge_labels, dtype=torch.long)
    
    # Node features: degree-based or learned embeddings
    x = torch.ones((len(nodes), 1))  # placeholder; enrich as needed
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
```

### Module 3: GNN Model (GraphSAGE)

```python
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class GNN_IDS(nn.Module):
    def __init__(self, node_in, edge_in, hidden, num_classes):
        super().__init__()
        self.conv1 = SAGEConv(node_in, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + edge_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        
        # For each edge, concat src+dst node embeddings + edge features
        src, dst = edge_index
        edge_repr = torch.cat([x[src], x[dst], edge_attr], dim=1)
        return self.edge_mlp(edge_repr)
```

### Module 4: Training Loop

```python
from torch_geometric.loader import DataLoader

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Setup
model = GNN_IDS(node_in=1, edge_in=6, hidden=64, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    loss = train(model, train_loader, optimizer, criterion)
    print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
```

### Module 5: FastAPI Inference Server

```python
from fastapi import FastAPI
import torch, time

app = FastAPI()
model = GNN_IDS(...)
model.load_state_dict(torch.load("gnn_ids.pt"))
model.eval()

@app.post("/predict")
async def predict(flow_window: dict):
    graph = build_graph_snapshot_from_dict(flow_window)
    
    start = time.time()
    with torch.no_grad():
        logits = model(graph)
        preds = logits.argmax(dim=1).tolist()
    latency_ms = (time.time() - start) * 1000
    
    return {"predictions": preds, "latency_ms": latency_ms}
```

### Module 6: Ryu SDN Controller Integration

```python
# ryu_ids_app.py
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, set_ev_cls
import requests

class IDSMitigationApp(app_manager.RyuApp):
    
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        flows = self.extract_flows(ev.msg.body)
        
        # Call GNN inference server
        response = requests.post("http://localhost:8000/predict", json=flows)
        predictions = response.json()['predictions']
        
        # Mitigate detected attacks
        for i, pred in enumerate(predictions):
            if pred == 1:  # Attack
                self.add_drop_rule(ev.msg.datapath, flows[i]['src_ip'])
    
    def add_drop_rule(self, datapath, src_ip):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(ipv4_src=src_ip)
        actions = []  # Empty = DROP
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=100,
                                match=match, instructions=inst)
        datapath.send_msg(mod)
```

### Module 7: Mininet Emulation & Testing

```python
# topo_test.py
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.topo import SingleSwitchTopo

topo = SingleSwitchTopo(k=4)
net = Mininet(topo=topo, controller=RemoteController('c0', ip='127.0.0.1'))
net.start()

# Simulate attack: host h1 floods h4
h1, h4 = net.get('h1', 'h4')
h1.cmd('hping3 -S --flood -V -p 80 ' + h4.IP() + ' &')

# Check if Ryu drops traffic
net.pingAll()
net.stop()
```

---

## Dataset Details

### CICIDS2017
- **Source:** Canadian Institute for Cybersecurity
- **URL:** https://www.unb.ca/cic/datasets/ids-2017.html
- **Attacks covered:** DDoS, PortScan, Brute Force, Botnet, Web Attacks, Infiltration
- **Format:** PCAP + pre-extracted CSV (CICFlowMeter features)
- **Size:** ~2.8 million flows

### InSDN
- **Purpose:** SDN-specific, complements CICIDS2017
- **Attacks:** OpenFlow-specific attacks, controller flooding, topology poisoning
- **Use:** Validates SDN-specific detection capability

---

## GNN Model Design

### Why GraphSAGE?
- Inductive: works on unseen nodes (new IPs) without retraining
- Scalable: samples fixed-size neighborhoods, efficient for large graphs
- Outperforms GCN on dynamic/heterogeneous graphs

### Why GAT (alternative)?
- Attention mechanism weighs edge importance
- Better for sparse, high-variance traffic graphs
- Higher accuracy but slightly more compute

### Recommended Architecture

```
Input Graph (dynamic window)
    │
    ▼
SAGEConv(1 → 64) + ReLU
    │
    ▼
SAGEConv(64 → 64) + ReLU
    │
    ▼
Edge MLP: [h_src || h_dst || edge_feat] → Linear(128+6, 64) → ReLU → Linear(64, 2)
    │
    ▼
Softmax → [P(benign), P(attack)]
```

---

## Evaluation Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| F1-Score | 2 × (P × R) / (P + R) | +15–25% over XGBoost |
| Precision | TP / (TP + FP) | High (minimize false positives) |
| Recall | TP / (TP + FN) | High (catch all attacks) |
| AUC-ROC | Area under ROC curve | > 0.95 |
| Inference Latency | ms per batch | < 50ms |
| Mitigation Rate | % attack traffic dropped | ≥ 70% |

### Baseline: XGBoost on Tabular Features
Train XGBoost on the same feature set (without graph structure) to establish the baseline F1 before comparing with GNN results.

---

## Project Timeline & Milestones

| Week | Milestone |
|------|-----------|
| 1–2 | Dataset download, preprocessing, EDA |
| 3–4 | Graph construction pipeline + visualization |
| 5–6 | XGBoost baseline training & evaluation |
| 7–9 | GNN model design, training, hyperparameter tuning |
| 10–11 | FastAPI inference server + latency benchmarking |
| 12–13 | Ryu SDN integration + Mininet emulation |
| 14 | End-to-end integration testing |
| 15 | Final evaluation, report writing |
| 16 | Presentation preparation |

---

## Directory Structure (Recommended)

```
gnn-ids-sdn/
├── data/
│   ├── cicids2017/          # Raw + processed CSVs
│   └── insdn/
├── graphs/
│   ├── build_graph.py       # Graph construction logic
│   └── visualize.py
├── models/
│   ├── gnn_model.py         # GraphSAGE / GAT definition
│   ├── baseline_xgb.py      # XGBoost baseline
│   └── train.py
├── server/
│   ├── app.py               # FastAPI inference server
│   └── Dockerfile
├── sdn/
│   ├── ryu_app.py           # Ryu controller app
│   └── mininet_topo.py      # Mininet topology scripts
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_graph_viz.ipynb
│   └── 03_results.ipynb
├── requirements.txt
└── README.md
```

---

## Requirements (`requirements.txt`)

```
torch>=2.0
torch-geometric
torch-scatter
torch-sparse
dgl
xgboost
scikit-learn
pandas
numpy
networkx
fastapi
uvicorn
requests
scapy
matplotlib
seaborn
mlflow
jupyter
```

---

## Quick Start

```bash
# 1. Clone and setup environment
git clone https://github.com/yourteam/gnn-ids-sdn.git
cd gnn-ids-sdn
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Preprocess CICIDS2017
python data/preprocess.py --dataset cicids2017

# 3. Build graphs
python graphs/build_graph.py --window 5

# 4. Train XGBoost baseline
python models/baseline_xgb.py

# 5. Train GNN
python models/train.py --model graphsage --epochs 50

# 6. Start inference server
uvicorn server.app:app --port 8000

# 7. Start Mininet emulation (separate terminal, run as root)
sudo python sdn/mininet_topo.py

# 8. Start Ryu controller (separate terminal)
ryu-manager sdn/ryu_app.py
```

---

*Documentation generated for Batch B24 — Siddaganga Institute of Technology, Tumakuru*
