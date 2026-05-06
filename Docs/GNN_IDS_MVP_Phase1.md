# GNN-Based Intrusion Detection System for SDN
## Phase 1 MVP Document

**Project:** Major Project — 6th Semester, Batch B24  
**Institute:** Siddaganga Institute of Technology, Tumakuru  
**Team:** Abhishek (1SI23CI002) · Ashish Kumar Bhagat (1SI23CI009) · Jishnu Khargharia (1SI23CI017) · Vaishnavi A Hachadad (1SI23CI056)

---

## 1. MVP Summary

This project proposes an intrusion detection system for Software-Defined Networks (SDN) that models traffic as a graph and applies a Graph Neural Network (GNN) to detect coordinated attacks that are difficult to identify using traditional tabular machine learning models.

The **Phase 1 MVP** focuses on proving the core idea end-to-end on offline datasets:
- collect and clean SDN intrusion data,
- build graph snapshots from flow records,
- train a GNN classifier,
- compare it with baseline ML models,
- and prepare the system for later live SDN integration.

---

## 2. Problem Statement

Traditional IDS approaches rely on per-flow or per-packet features and often miss attacks that only become visible through relationships between multiple flows, hosts, or switches. In SDN, this is especially important because the controller and flow table behavior can be exploited by coordinated scans, DDoS patterns, and lateral movement.

A graph-based IDS is better suited because it can learn:
- communication patterns between hosts,
- temporal flow relationships,
- neighborhood behavior in the network,
- and structural indicators of attack propagation.

---

## 3. MVP Goal

The goal of the MVP is to demonstrate that the proposed GNN-based approach is feasible and beneficial before building the full live SDN mitigation system.

### What the MVP must prove
- Flow records can be converted into graph snapshots.
- A GNN can classify benign vs. attack traffic from graph representations.
- The graph approach can be compared against standard baselines such as Random Forest and XGBoost.
- The pipeline can later be extended to Mininet and Ryu for real-time SDN deployment.

---

## 4. Phase 1 Scope

### Included in the MVP
- Dataset acquisition and preprocessing
- Exploratory data analysis (EDA)
- Baseline model training
- Graph construction pipeline
- Initial GNN model training
- Metric comparison and result documentation

### Out of scope for Phase 1
- Live Mininet deployment
- Ryu controller integration
- Automated OpenFlow rule injection
- Real-time mitigation engine
- Full production optimization

---

## 5. Proposed MVP Architecture

```text
Raw Dataset (CICIDS2017 / InSDN)
        │
        ▼
Preprocessing & Cleaning
        │
        ▼
Flow Feature Extraction
        │
        ▼
Graph Construction (sliding window)
        │
        ▼
GNN Training & Inference
        │
        ▼
Attack / Benign Classification
        │
        ▼
Baseline Comparison & Result Report
```

### MVP modules

#### 5.1 Data Preparation
- Download and inspect CICIDS2017 and InSDN datasets.
- Clean missing, infinite, and duplicate values.
- Standardize labels.
- Split data into training, validation, and test sets.

#### 5.2 Graph Builder
- Convert each flow window into a graph.
- Nodes represent hosts or network entities.
- Edges represent flows between nodes.
- Attach flow statistics as edge features.

#### 5.3 GNN Classifier
- Train a GNN on graph snapshots.
- Predict whether a graph window is benign or attack-prone.
- Use graph-level output for MVP evaluation.

#### 5.4 Baseline Comparison
- Train Random Forest and XGBoost on the same cleaned flow data.
- Compare precision, recall, F1-score, and ROC-AUC.

---

## 6. Dataset Plan

### Primary datasets
| Dataset | Purpose |
|---|---|
| CICIDS2017 | Primary training and evaluation dataset |
| InSDN | SDN-specific validation dataset |

### Why these datasets
- CICIDS2017 provides a widely used intrusion detection benchmark.
- InSDN is closer to the SDN setting and helps validate domain relevance.

---

## 7. Feature Plan

### Flow-level features
- duration
- packet count
- byte count
- packet rate
- byte rate
- flow direction statistics
- TCP flag ratios
- protocol type

### Graph-level representation
- **Nodes:** IP addresses or network entities
- **Edges:** communication flows
- **Edge attributes:** flow statistics and protocol indicators
- **Windowing:** sliding time window for graph snapshots

---

## 8. Model Plan

### Baseline models
- Random Forest
- XGBoost

### GNN model
- GAT or GraphSAGE as the primary graph model
- Optional temporal extension in later phase

### Initial success criteria
- A working graph-based classifier pipeline
- Clear metric comparison against baselines
- Evidence that the GNN can capture network structure better than tabular models on coordinated attack patterns

---

## 9. Evaluation Plan

The MVP will be evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

### Expected outcome
The GNN should be competitive with the baseline models and show stronger performance on attack patterns where topology matters.

---

## 10. Deliverables for Phase 1

- Cleaned and documented datasets
- EDA summary
- Baseline model results
- Graph construction pipeline
- Initial GNN prototype
- Comparison report in markdown/PDF form
- Short presentation-ready summary for guide review

---

## 11. Phase 1 Milestone Timeline

| Week | Work Item |
|---|---|
| Week 1 | Dataset download, setup, and EDA |
| Week 2 | Baseline models and metric recording |
| Week 3 | Graph construction pipeline |
| Week 4 | GNN training and evaluation report |

---

## 12. Risks and Mitigation

| Risk | Mitigation |
|---|---|
| Dataset cleaning takes longer than expected | Start with a smaller subset and validate pipeline early |
| Graph construction is slow or unstable | Prototype with simple sliding windows first |
| GNN underperforms baselines initially | Tune graph features, window size, and class weights |
| Live SDN integration becomes too large for Phase 1 | Keep integration as Phase 2 scope |

---

## 13. Phase 1 MVP Definition

At the end of Phase 1, the project should demonstrate:
1. A reproducible preprocessing pipeline.
2. A valid graph representation of SDN traffic.
3. A trained GNN classifier with measurable performance.
4. Baseline comparison against standard ML methods.
5. A clear path to extend the system into live SDN deployment.

---

## 14. Conclusion

This MVP validates the core research idea: SDN intrusion detection can benefit from graph-based learning because attack behavior is not always visible in isolated flow records. By completing the offline pipeline first, the team can present a strong Phase 1 evaluation and then move confidently toward real-time SDN integration in the next phase.
