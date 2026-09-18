# Guide Explanation for Project Output Data

This document is written as a detailed explanation you can use while presenting the generated output files to your guide. The goal is to explain not only the numbers, but also what they mean, why they matter, and what limitations should be honestly mentioned.

It has two parts:

- **Phase 1** (sections 1–23): the offline binary GNN on CICIDS2017-style flow data.
- **Phase 2** (sections P1–P21): real host-to-host graphs, a multi-task GNN that names the attack type
  and the attacking hosts, and a live SDN (os-ken controller + Mininet) that blocks attackers
  automatically. If you have limited time, present Phase 2 and use Phase 1 as background.

## 1. Short Presentation Summary

You can start with this:

> This project implements a Phase 1 offline Intrusion Detection System for Software Defined Networks using graph-based learning. The raw CICIDS2017 and InSDN flow data is cleaned, converted into binary labels where BENIGN is class `0` and attack traffic is class `1`, then evaluated using both traditional machine learning baselines and a Graph Attention Network model. The GNN converts network flows into sliding-window graph snapshots and classifies each graph as benign or attack. The final GNN achieved around `99.80%` test accuracy and an attack-class F1-score of around `99.40%`.

The most important point to communicate is:

> The model is not just classifying isolated rows. The GNN attempts to learn relationships between network entities and flows over short time windows, which is more aligned with SDN traffic behavior than a purely tabular classifier.

## 2. What The Output Files Represent

The main generated outputs are inside the `results/` folder.

| File | Purpose |
|---|---|
| `results/gnn_metrics.json` | Main training and test metrics for the GNN model. It includes validation performance, selected threshold, class balance, and final test results. |
| `results/gnn_classification_report.json` | Detailed class-wise report for the final GNN evaluation. It shows precision, recall, F1-score, and support for benign and attack classes. |
| `results/gnn_confusion_matrix.png` | Visual confusion matrix showing correct and incorrect predictions. |
| `results/baseline_metrics.json` | Results of traditional ML models such as Random Forest and XGBoost. These are used as comparison baselines. |
| `results/baseline_confusion_matrix.png` | Confusion matrix for the baseline model output. |

You can explain that the JSON files are machine-readable result summaries, while the PNG files are useful for visual presentation in the dashboard or report.

## 3. Dataset Label Meaning

The project currently uses binary classification:

| Label | Meaning |
|---|---|
| `0` | BENIGN / normal traffic |
| `1` | ATTACK / malicious traffic |

So, when the classification report shows class `0`, it refers to benign traffic. When it shows class `1`, it refers to attack traffic.

This is an important point to explain clearly because the output files only show numeric labels.

## 4. Pipeline Explanation

The complete Phase 1 pipeline follows this order:

1. Raw network traffic CSV files are loaded from the dataset folders.
2. Data cleaning is applied to remove invalid values, normalize labels, and prepare numeric features.
3. Baseline machine learning models are trained on tabular flow features.
4. Cleaned flow records are converted into graph snapshots.
5. A Graph Attention Network is trained on those graph snapshots.
6. The trained GNN is evaluated on the test split.
7. Metrics and confusion matrices are saved in the `results/` folder.
8. The dashboard reads these result files and displays them visually.

The important technical idea is that network data is converted from rows into graphs. Each graph is a short time-window view of network activity.

## 5. How The Graphs Are Built

The graph builder uses sliding time windows.

Current graph settings:

| Setting | Value |
|---|---|
| Window size | `5` seconds |
| Step size | `1` second |
| Output graph file | `data/graphs/cicids_graphs.pt` |

This means the system takes a 5-second window of traffic, builds a graph from it, then moves forward by 1 second and builds the next graph. This creates overlapping graph snapshots of traffic behavior.

In each graph:

| Graph Component | Meaning |
|---|---|
| Nodes | Network entities such as source/destination IPs where available, or synthetic source/destination nodes based on flow information when IP columns are missing. |
| Edges | Network flows between source and destination nodes. |
| Edge features | Flow-level properties such as duration, packet counts, byte rates, packet length statistics, flags, and other CICIDS-style features. |
| Graph label | `0` if the window is mostly benign, `1` if attack flows are the majority in that window. |

The graph label is assigned using majority voting inside the window:

> If attack flows are more than benign flows in a window, the graph is labeled as attack. Otherwise, it is labeled as benign.

This should be explained as a Phase 1 design choice. It converts individual flow labels into graph-level labels.

## 6. Why Use A GNN Here?

Traditional models such as Random Forest and XGBoost treat each flow mostly as an independent tabular record. That is useful and often very strong, but network attacks are not always isolated single-flow events.

A GNN is useful because it can model:

| Network Behavior | Why It Matters |
|---|---|
| Communication relationships | Attack behavior may appear through interactions between hosts, not only individual flow values. |
| Local traffic structure | Nodes and edges can show patterns such as many connections to one destination, repeated attempts, or abnormal communication paths. |
| Temporal windows | Sliding windows allow the model to classify a short period of traffic instead of one row at a time. |
| Edge-aware learning | The model can use flow attributes on edges while learning graph structure. |

The model used here is a Graph Attention Network, or GAT. Attention helps the model learn which connections or parts of the graph are more important for classification.

## 7. GNN Model Architecture Explanation

The GNN model is an edge-aware GAT classifier.

High-level architecture:

1. Node features are passed through the first GAT layer.
2. The second GAT layer refines node embeddings.
3. Global mean pooling and global max pooling summarize the full graph.
4. Edge features are also encoded when available.
5. The graph-level representation is passed into a final classifier.
6. The model outputs two logits: one for benign and one for attack.

Important configuration values:

| Parameter | Value |
|---|---|
| Hidden dimension | `64` |
| Attention heads | `4` |
| Dropout | `0.25` |
| Number of classes | `2` |
| Batch size | `128` |
| Epochs requested | `40` |
| Best epoch | `30` |

The best model was selected based on validation F1-score, not just training accuracy.

## 8. Training Class Distribution

From `results/gnn_metrics.json`:

| Class | Count |
|---|---:|
| Benign, class `0` | `23195` |
| Attack, class `1` | `4641` |

This shows class imbalance. The benign class is much larger than the attack class.

Approximate training distribution:

| Class | Approximate Share |
|---|---:|
| Benign | `83.3%` |
| Attack | `16.7%` |

To handle this, class weights were used:

| Class | Weight |
|---|---:|
| Benign | `0.7746` |
| Attack | `1.7317` |

How to explain this:

> Since attack samples are fewer than benign samples, the model could otherwise become biased toward predicting benign. To reduce that bias, the attack class is given a higher loss weight. This means mistakes on attack samples are penalized more strongly during training.

## 9. Validation Results

From `results/gnn_metrics.json`:

| Metric | Value |
|---|---:|
| Best validation F1 | `0.99546` |
| Best validation loss | `0.00947` |
| Best epoch | `30` |
| Selected decision threshold | `0.81` |

How to explain this:

> The model performed best on validation data at epoch 30. The selected validation F1-score was about 99.55%, and the validation loss was very low. This indicates that the model learned a strong separation between benign and attack graph patterns.

Mention that the best epoch is chosen from validation performance. This is important because it shows the result is not selected only from the test data.

## 10. Decision Threshold Explanation

The final selected decision threshold is:

```text
0.81
```

Normally, binary classifiers often use `0.5` as the threshold. That means:

> If the model thinks attack probability is at least 50%, classify it as attack.

In this project, the threshold was tuned on validation data, and the best threshold became `0.81`. That means:

> The model classifies a graph as attack only when the attack probability is at least 81%.

Why this matters:

| Lower Threshold | Higher Threshold |
|---|---|
| More sensitive to attacks | More conservative about attack predictions |
| May catch slightly more attacks | Reduces false alarms |
| Can increase false positives | Can slightly increase false negatives |

Here, threshold tuning improved the final F1-score by creating a better precision-recall balance.

Comparison from the GNN output:

| Evaluation Mode | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Default threshold `0.5` | `0.99732` | `0.98999` | `0.99397` | `0.99198` |
| Tuned threshold `0.81` | `0.99799` | `0.99496` | `0.99296` | `0.99396` |

How to explain this:

> With threshold tuning, recall became very slightly lower, but precision improved. This means the final model produced fewer false alarms while still detecting almost all attacks. Because F1-score balances precision and recall, the tuned threshold gives the better overall result.

## 11. Final GNN Test Results

From `results/gnn_metrics.json`, final test performance is:

| Metric | Value | Percentage |
|---|---:|---:|
| Accuracy | `0.997988` | `99.80%` |
| Precision, attack class | `0.994965` | `99.50%` |
| Recall, attack class | `0.992965` | `99.30%` |
| F1-score, attack class | `0.993964` | `99.40%` |

How to explain each metric:

| Metric | Meaning In This Project |
|---|---|
| Accuracy | Out of all graph snapshots, how many were classified correctly. |
| Precision | Out of all graphs predicted as attack, how many were truly attack. |
| Recall | Out of all actual attack graphs, how many were detected by the model. |
| F1-score | Balanced score combining precision and recall. |

For intrusion detection, recall is important because missed attacks are dangerous. Precision is also important because too many false alarms make the system noisy and less useful.

## 12. Classification Report Explanation

From `results/gnn_classification_report.json`:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| `0` Benign | `0.99859` | `0.99899` | `0.99879` | `4970` |
| `1` Attack | `0.99496` | `0.99296` | `0.99396` | `995` |
| Accuracy |  |  | `0.99799` | `5965` |
| Macro avg | `0.99678` | `0.99598` | `0.99638` | `5965` |
| Weighted avg | `0.99799` | `0.99799` | `0.99799` | `5965` |

Support means how many test samples belong to that class.

So the test set contains:

| Class | Number of Test Graphs |
|---|---:|
| Benign | `4970` |
| Attack | `995` |
| Total | `5965` |

How to explain class-wise results:

> For benign traffic, the model achieved about 99.88% F1-score, meaning it is very accurate at recognizing normal behavior. For attack traffic, it achieved about 99.40% F1-score, meaning it also detects malicious windows very strongly despite attack samples being fewer than benign samples.

## 13. Macro Average vs Weighted Average

The report contains both macro average and weighted average.

| Average Type | Meaning |
|---|---|
| Macro average | Simple average across classes. Both benign and attack are treated equally. |
| Weighted average | Average weighted by number of samples in each class. Larger classes influence the score more. |

In this project:

| Average Type | F1-score |
|---|---:|
| Macro average F1 | `0.99638` |
| Weighted average F1 | `0.99799` |

How to explain this:

> Weighted F1 is slightly higher because the benign class has more samples and is classified extremely well. Macro F1 is useful because it gives equal importance to benign and attack classes. Since macro F1 is also very high, the model is not performing well only because of the majority benign class.

## 14. Confusion Matrix Interpretation

Using the final classification report and prediction counts, the GNN confusion matrix can be explained as:

| Actual Class | Predicted Benign | Predicted Attack |
|---|---:|---:|
| Actual Benign | `4965` | `5` |
| Actual Attack | `7` | `988` |

This means:

| Case | Count | Meaning |
|---|---:|---|
| True benign predicted benign | `4965` | Correct normal traffic predictions |
| Benign predicted as attack | `5` | False alarms |
| Attack predicted as benign | `7` | Missed attacks |
| True attack predicted attack | `988` | Correct attack detections |

Total test samples:

```text
5965
```

Total incorrect predictions:

```text
5 + 7 = 12
```

How to explain this:

> Out of 5965 test graph snapshots, only 12 were misclassified. There were 5 false alarms and 7 missed attacks. This gives a very strong detection result, especially because the attack class is the minority class.

## 15. Baseline Model Comparison

From `results/baseline_metrics.json`:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | `0.99895` | `0.99564` | `0.99799` | `0.99682` | `0.99995` |
| XGBoost | `0.99903` | `0.99635` | `0.99779` | `0.99707` | `0.99996` |
| GNN | `0.99799` | `0.99496` | `0.99296` | `0.99396` | Not reported |

Important honest explanation:

> In the current Phase 1 results, the traditional baselines, especially XGBoost, achieve slightly higher tabular metrics than the GNN. This does not make the GNN result weak. It means that the selected CICIDS-style flow features are already very strong for classical machine learning. The value of the GNN is that it introduces graph-based traffic modeling, which is more suitable for SDN relationship analysis and can be extended toward topology-aware or live SDN detection in later phases.

Do not claim that the GNN outperformed the baselines. A better statement is:

> The GNN achieved comparable high performance while modeling network traffic as graph snapshots, which supports the feasibility of graph-based IDS for SDN.

## 16. Why Baselines Can Perform Slightly Better

You can explain:

1. CICIDS-style datasets contain strong engineered flow features.
2. Random Forest and XGBoost are very effective on tabular data.
3. The GNN adds graph structure, but Phase 1 graph construction may not yet capture full SDN topology.
4. Some CICIDS files may not contain real IP topology columns, so synthetic nodes are used when needed.
5. A GNN often becomes more valuable when relationship structure is rich, such as live switch-host-controller SDN traffic graphs.

This makes your explanation balanced and mature.

## 17. Dashboard Explanation

The dashboard is a FastAPI, Bootstrap, and Chart.js frontend.

It reads:

```text
results/gnn_metrics.json
results/baseline_metrics.json
results/gnn_classification_report.json
```

The dashboard does not retrain the model. It only visualizes already generated results.

How to explain this:

> The dashboard acts as a result visualization layer. After the ML pipeline generates metrics, the dashboard loads the JSON files and displays the performance summaries, comparisons, and reports in a more understandable form.

## 18. Main Points To Tell Your Guide

Use these as your core speaking points:

1. The project implements an offline Phase 1 SDN IDS pipeline.
2. Labels are binary: benign is `0`, attack is `1`.
3. The data is cleaned and converted into graph snapshots.
4. Each graph represents a short traffic window.
5. The GNN uses graph attention to learn important traffic relationships.
6. The final GNN achieved `99.80%` accuracy.
7. The attack-class F1-score is `99.40%`.
8. The confusion matrix shows only `12` mistakes out of `5965` test graphs.
9. Traditional baselines are slightly higher, especially XGBoost.
10. The GNN is still valuable because it supports graph-based and SDN-oriented intrusion detection.

## 19. Suggested Explanation Script

You can say:

> First, I cleaned the raw network traffic datasets and converted the original labels into binary labels, where benign traffic is represented as 0 and all attack traffic is represented as 1. After cleaning, I trained baseline models like Random Forest and XGBoost to establish a traditional machine learning comparison.

> Then I converted the cleaned traffic into graph snapshots using a sliding window approach. Each graph represents a 5-second interval of network traffic, and the window moves by 1 second. Nodes represent network entities or synthetic flow-based nodes, while edges represent traffic flows with their flow-level features.

> The GNN model used is an edge-aware Graph Attention Network. It learns from both the node relationships and the edge features. The graph is classified as benign or attack based on the learned graph-level representation.

> The final GNN test accuracy is approximately 99.80%. For the attack class, precision is approximately 99.50%, recall is approximately 99.30%, and F1-score is approximately 99.40%. This means the model detects almost all attacks while producing very few false alarms.

> From the confusion matrix, out of 5965 test graphs, 4965 benign graphs were correctly classified, 988 attack graphs were correctly detected, 5 benign graphs were falsely marked as attacks, and 7 attack graphs were missed. So the model made only 12 total mistakes.

> The baseline models also performed very strongly. XGBoost achieved slightly higher metrics than the GNN. This is expected because the CICIDS-style flow features are highly suitable for tabular models. However, the GNN is important because it models traffic as a graph, which is closer to how SDN networks are structured and can be extended in future phases for topology-aware live detection.

## 20. Limitations To Mention

It is good to mention limitations honestly:

| Limitation | Explanation |
|---|---|
| Offline only | Phase 1 uses stored CSV datasets, not live SDN controller traffic. |
| Binary classification | The current model detects benign vs attack, not individual attack types. |
| Graph construction is approximate | If real IP columns are missing, synthetic nodes are created from flow information. |
| Dataset-specific performance | Very high metrics may depend on CICIDS/InSDN feature patterns. |
| Baselines slightly outperform GNN | XGBoost performs slightly better in the current tabular metric comparison. |
| No live mitigation yet | The model detects attacks but does not yet push SDN rules to block traffic. |

This helps show that you understand the project deeply and are not overclaiming.

Phase 2 addressed most of these limitations: live controller traffic, multi-class detection, real
host-to-host graphs and automated mitigation (see P2 and P3–P12).

## 21. Future Work

The list below was the plan at the end of Phase 1. Items 1–5, 8 and 9 were implemented in Phase 2.
Items 6 and 7 are still open.

Good future improvements to mention:

1. Extend binary classification to multi-class attack classification.
2. Integrate the trained model with a live SDN controller such as Ryu.
3. Use Mininet to generate live SDN traffic.
4. Add real-time flow collection from OpenFlow switches.
5. Build topology-aware graphs using switches, hosts, and controller information.
6. Add explainability, such as showing which nodes or edges influenced the GNN decision.
7. Evaluate cross-dataset generalization, for example training on CICIDS and testing on InSDN.
8. Compare inference latency to check whether the model can operate in real time.
9. Add mitigation actions such as blocking malicious flows or rate-limiting suspicious hosts.

## 22. Possible Questions And Answers

### Q1. Why did you use GNN instead of only Random Forest or XGBoost?

Because network traffic has relationship structure. In SDN, hosts, switches, and flows interact with each other. A GNN can learn from this structure, while Random Forest and XGBoost mainly learn from independent tabular rows. Even though baselines are slightly stronger in Phase 1, the GNN provides a better foundation for topology-aware SDN intrusion detection.

### Q2. What does precision mean here?

Precision answers:

> Of all the graph windows predicted as attack, how many were actually attack?

In the final GNN result, attack precision is about `99.50%`, meaning false alarms are very low.

### Q3. What does recall mean here?

Recall answers:

> Of all the actual attack graph windows, how many did the model detect?

In the final GNN result, attack recall is about `99.30%`, meaning the model missed very few attacks.

### Q4. Why is F1-score important?

F1-score balances precision and recall. In intrusion detection, only accuracy can be misleading because benign traffic is usually more common. F1-score gives a better view of detection quality, especially for the attack class.

### Q5. Why is the decision threshold 0.81 instead of 0.5?

The threshold was tuned on validation data to maximize F1-score. A threshold of `0.81` made the model more conservative about predicting attacks, which reduced false alarms while still keeping recall very high.

### Q6. What does the confusion matrix show?

It shows correct and incorrect predictions. For the final GNN:

```text
4965 benign samples were correctly classified.
988 attack samples were correctly detected.
5 benign samples became false alarms.
7 attack samples were missed.
```

This means only 12 out of 5965 test graphs were wrong.

### Q7. Why are the metrics so high?

The cleaned CICIDS-style flow features are very informative, and the dataset has strong patterns between benign and attack traffic. Also, both baseline models and the GNN are trained on structured features that are useful for classification. However, the result should still be validated further on unseen datasets or live traffic before claiming real-world deployment readiness.

### Q8. Did the GNN outperform the baseline models?

No. In the current output, XGBoost and Random Forest are slightly higher than the GNN in standard metrics. The correct conclusion is that the GNN achieved comparable high performance while enabling graph-based SDN traffic modeling.

### Q9. What is the main contribution of this phase?

The main contribution is building a complete working Phase 1 pipeline:

```text
raw dataset -> cleaning -> baselines -> graph construction -> GNN training -> evaluation -> dashboard
```

This proves that graph-based intrusion detection can be implemented and evaluated end to end.

### Q10. What will you improve next?

The next phase should focus on live SDN integration, topology-aware graph construction, multi-class attack detection, and real-time mitigation through the SDN controller.

(This is what Phase 2 went on to implement; see the Phase 2 part below.)

## 23. Final Conclusion

The output data shows that the Phase 1 system is working successfully. The GNN model performs very strongly, with approximately `99.80%` accuracy and `99.40%` attack-class F1-score. The confusion matrix shows only `12` errors out of `5965` test graph snapshots.

The baseline models are slightly stronger in raw metric comparison, which should be stated honestly. However, the GNN result is important because it demonstrates graph-based intrusion detection, which is a better conceptual fit for SDN environments where traffic relationships and topology matter.

The best way to present the result is:

> The current phase validates the feasibility of using graph neural networks for SDN intrusion detection. The model achieves very high detection performance on offline datasets, and the next step is to extend it into live SDN traffic monitoring and mitigation.

---

# Phase 2 — Explanation for the Guide

Full numbers: `results/phase2/final_results.md`. Design reasoning: `Docs/design_decisions.md`.

## P1. One-paragraph summary

> In Phase 2 the IDS became a working SDN security system. Traffic windows are turned into real
> host-to-host graphs using only features an OpenFlow switch can report, so the same model runs on
> datasets and on live switches. A multi-task Graph Attention Network predicts the attack type of each
> window and an attacker score for every host. The live system runs on an os-ken SDN controller with
> Mininet: the controller streams flow statistics to the IDS every 2 seconds, the IDS detects the attack,
> identifies the attacking hosts and sends back OpenFlow rules that drop or rate-limit them.

## P2. What changed from Phase 1 and why (say this first)

| Phase 1 issue | What we found | Phase 2 fix |
|---|---|---|
| Graphs had no real topology | The CICIDS2017 files used have no IP columns; nodes were made up from row numbers and ports | Real host graphs from InSDN (IPs, ports, timestamps) |
| InSDN labels were wrong | Only `BENIGN` was treated as benign; InSDN uses `Normal`, so 100 % of InSDN was "attack" | Shared label map; unknown labels are never benign |
| Scores were inflated | Random split of overlapping windows leaks near-duplicates into the test set | Time-ordered split with a gap (Phase 1 F1: 0.994 → 0.983 when fixed) |
| Features couldn't run live | 78 CICFlowMeter features; a switch reports ~6 counters | OpenFlow-only features, one shared module |
| No "who to block" | Only a window label | Per-host attacker head |
| Victims labelled as attackers | InSDN has victim→attacker rows with attack labels (27 % of one capture) | Direction normalisation (lower port = server) |

This is a strong point to make: we found and fixed problems in our own Phase 1 results before building on them.

## P3. End-to-end architecture

```
  SDN lab (Docker container)                            IDS (host machine, web/app.py)
┌────────────────────────────────┐                   ┌──────────────────────────────────┐
│ Mininet hosts h1..h6           │                   │ FlowCollector    10 s window     │
│      │                         │  flow stats       │        ▼                         │
│ Open vSwitch s1 ═ s2 (OF 1.3)  │  every 2 s        │ InferenceEngine  TorchScript GNN │
│      ▲  │                      │ ───────────────►  │        ▼                         │
│ rules│  │stats                 │  POST /api/flows  │ AlertClassifier  thresholds      │
│      │  ▼                      │                   │        ▼                         │
│ os-ken controller              │ ◄───────────────  │ MitigationEngine policy + audit  │
│ table 0: IDS rules             │  rules in the     │                                  │
│ table 1: forwarding            │  HTTP reply       │ Dashboards: /  and  /live        │
└────────────────────────────────┘                   └──────────────────────────────────┘
```

One polling cycle, step by step:

1. The **controller** (`controller/ids_controller.py`) asks every switch for its flow statistics every
   2 s and POSTs them to the IDS at `/api/flows` (with an `X-API-Key` header).
2. The **flow collector** (`controller/flow_collector.py`) removes duplicates (the same flow is seen on
   both switches), keeps flows that were active in the last 10 s and converts them to flow records.
3. The **inference engine** (`inference/inference_engine.py`) builds a host graph from those records,
   computes features and runs the GNN. The output is an attack probability and an attack type for the
   window, plus an attacker score for every host.
4. The **alert classifier** (`inference/classifier.py`) decides BENIGN / SUSPICIOUS / ATTACK and which
   hosts are confirmed attackers.
5. The **mitigation engine** (`mitigation/mitigation_engine.py`) turns an ATTACK decision into OpenFlow
   rule actions (drop / rate-limit), which go back **in the same HTTP response**.
6. The controller installs the rules on the switches immediately, so blocking costs no extra round-trip.

The whole IDS side of one cycle (steps 2–5) takes about 10 ms.

How to explain it:

> The switches already count packets and bytes for every flow. The controller collects those counters
> every two seconds and sends them to our IDS. The IDS turns the last ten seconds of traffic into a graph
> of hosts, the GNN says whether there is an attack, what kind it is and which hosts are responsible,
> and the IDS answers with the OpenFlow rules to block them. The controller installs those rules right
> away.

## P4. Data, labels and graph windows

**Dataset.** InSDN (`Normal_data.csv`, `OVS.csv`, `metasploitable-2.csv`). Unlike the CICIDS2017 files
we had, it has real IP addresses, ports and timestamps, and it was captured on an SDN testbed.

**Cleaning** (`preprocessing/clean_data.py`) keeps two label columns: `Label` (binary, as in Phase 1)
and `Attack` (the attack class).

**Eight canonical classes** (`preprocessing/labels.py`, full mapping in `Docs/label_scheme.md`):

| ID | Class | InSDN label |
|---:|---|---|
| 0 | Benign | `Normal` |
| 1 | DDoS | `DDoS` |
| 2 | DoS | `DoS` |
| 3 | Probe | `Probe` |
| 4 | BruteForce | `BFA` |
| 5 | WebAttack | `Web-Attack` |
| 6 | Botnet | `BOTNET` |
| 7 | Other | `U2R` |

An unknown label always maps to `Other`, never to Benign, so a new attack name can never become a
benign training example.

**Graph windows** (`preprocessing/graph_builder_v2.py`, settings in `configs/phase2.yaml`):

| Setting | Offline (InSDN) | Live (Mininet) |
|---|---|---|
| Window | 100 consecutive flows | every flow active in the last 10 s |
| Stride | 25 flows (10 for the benign file, to balance classes) | a new window every 2 s poll |
| Why | InSDN timestamps are mostly minute-level, so time windows would give very few graphs | real clock available |

A live window can hold thousands of flows (for example during a DDoS). It is cut into chunks of 200
records, about the size of a training window, and the chunks run as one batch.

**Three kinds of labels per window:**

| Label | Meaning |
|---|---|
| `y_multi` (window) | Most frequent attack class if at least 10 % of the window's flows are attacks, otherwise Benign |
| `y` (window) | `y_multi != Benign` |
| `node_y` (host) | 1 if the host **started** at least one attack flow in the window. Victims are 0 |

**Two tricks that make the data realistic:**

- **Benign overlay.** In InSDN, benign and attack traffic come from different captures, so raw windows
  are "pure". Half of the attack windows get a benign window merged in, so the model has to find the
  attackers among normal hosts.
- **Time-ordered split.** Each (file, class) stream is split 70 / 15 / 15 in time order, with a gap so
  that no flow appears in two splits. Result: **12,458 train / 2,645 validation / 2,677 test** windows.

## P5. OpenFlow-only features

An OpenFlow 1.3 switch reports only a few counters per flow entry: packets, bytes, duration and the
match fields (IPs, ports, protocol). Phase 1's 78 CICFlowMeter features can't be computed live, so
Phase 2 uses **only features a switch can give**. They are defined in one module,
`preprocessing/openflow_features.py`, which both the dataset pipeline and the live IDS call. The model
therefore sees exactly the same features in training and in production.

| Level | Count | Features |
|---|---:|---|
| Edge (one flow) | 16 | packets, bytes, duration, packet rate, byte rate, mean packet size, protocol one-hot (TCP/UDP/ICMP/other), source and destination port bucket (well-known / registered / ephemeral) |
| Node (one host in the window) | 16 | flows / packets / bytes sent and received, fan-out (distinct destinations), fan-in (distinct sources), distinct destination ports, **destination-port entropy** (scan indicator), distinct incoming source ports, mean outgoing flow duration, protocol mix of outgoing flows |

All features go through `sign(x)·log(1+|x|)` (bytes span 8 orders of magnitude) and are then z-scored
with the **training split's** mean and standard deviation. Those statistics are saved inside the model
file, so the live system normalises exactly as in training.

Full schema: `Docs/feature_schema_phase2.md`.

## P6. The multi-task GNN model

Code: `models/gnn_v2.py` (`MultiTaskGNN`). Size: about 65,000 parameters.

```
node features (16) ──► node MLP ─┐
edge features (16) ──► edge MLP ─┼─► mean of outgoing / incoming edge embeddings per node ─► node input (64)
                                 │
                                 ▼
             2 × [ GATConv (4 heads × 16, uses edge embeddings in attention)
                   + residual + LayerNorm + ELU + dropout 0.2 ]
                                 │
                 ┌───────────────┴───────────────────┐
                 ▼                                   ▼
   Node head (per host)                 Graph head (per window)
   → P(host is an attacker)             mean+max pool of nodes and edges
                                        → 8 attack classes
```

Key design points (details in `Docs/design_decisions.md`, D8–D12):

- **Two heads, one model.** The graph head says *whether* and *what* (attack type). The node head says
  *who* (which host to block). Mitigation needs both.
- **Messages go both ways.** Traffic is directed, but a pure attacker (for example a spoofed DDoS source)
  has no incoming edges and would never receive information about its victim. Every edge also gets a
  reversed copy, with a direction flag added to the edge features.
- **Edge features inside attention.** GAT uses the flow features to decide which neighbours matter.
  Every model variant also starts each host from the average of its flows, so the GCN / GraphSAGE
  ablations differ only in the convolution layer.
- **Window attack probability = 1 − P(Benign)**, i.e. the probability of all attack classes combined.

## P7. Training, evaluation and export

Code: `models/train_gnn_v2.py`, `baselines/train_baselines_v2.py`, `models/export.py`.

| Setting | Value |
|---|---|
| Loss | `0.5 × node loss + 0.5 × graph loss` (both cross-entropy) |
| Class weights | square-root inverse frequency, capped at 10 (so rare classes don't dominate) |
| Optimiser | AdamW, learning rate 0.001, weight decay 0.0001 |
| Batch size / epochs | 64 / up to 60, early stopping after 10 epochs without improvement |
| Model selection | mean of validation window F1, host F1 and attack-type macro-F1 (major classes) |
| Thresholds | window and host thresholds tuned on the validation split, stored with the model |

**Fair baselines.** Random Forest (200 trees) and XGBoost (300 trees) use the **same features and the
same split**. They score each flow and average per window. For hosts, they get the same per-host
features as the GNN. All models use the same metric code (`common/metrics.py`).

**Ablations.** GCN, GraphSAGE, GAT without edge features, and window sizes 50 and 200.

**Export.** The best checkpoint is exported as one **TorchScript** file with an embedded JSON bundle
(normalisation statistics, class names, thresholds, feature names). The live engine loads that one
file, so the model can never be paired with the wrong statistics. The export is checked to give the
same output as the original model on 100 test graphs.

| File | What it is |
|---|---|
| `models/gat_ids_insdn_only.pt` | trained on InSDN only |
| `models/gat_ids.pt` | **the live model**: fine-tuned on InSDN + lab traffic (P12) |

The whole offline pipeline is one command: `bash scripts/run_phase2_training.sh`.

## P8. Live detection: controller, collector, inference, alerts

**Controller** (`controller/ids_controller.py`, started with `controller/run_controller.py`):

- os-ken 2.8.1 (the maintained fork of Ryu), OpenFlow 1.3.
- **Two flow tables.** Table 0 holds only IDS rules (drop, or meter + continue). Table 1 does forwarding.
  IDS rules therefore never appear in the statistics used for detection, and unblocking is one delete.
- **Forwarding by 5-tuple.** Each IPv4 flow gets its own forwarding entry, so the switch keeps
  per-flow counters. A plain MAC-learning switch would give no per-host statistics. `--match-mode
  host_pair` gives coarser entries when a flood fills the flow table.
- Polls every 2 s. **Fail-open:** if the IDS is down, the network keeps forwarding.

**Inference engine** (`inference/inference_engine.py`): features → chunks of 200 records → GNN. The
window score is the **maximum** over chunks (one attacking chunk is enough). A host's score is the
**mean** over the chunks it appears in, so one noisy chunk can't get a benign host blocked. Five
warm-up calls run at start-up, which brought p99 latency from about 208 ms to about 11 ms.

**Alert classifier** (`inference/classifier.py`):

| Rule | Value |
|---|---|
| ATTACK (mitigation allowed) | window confidence ≥ 0.85 |
| SUSPICIOUS (logged only) | 0.5 ≤ confidence < 0.85 |
| Attacker host | host score ≥ node threshold (tuned on validation, 0.56 in the live model) |
| Persistence | a host is blocked only after being flagged in **2 consecutive** ATTACK windows |
| Cool-down | the same host is not re-alerted within 30 s |
| Whitelist | infrastructure IPs / CIDRs are never reported as attackers |

All of these live in the `ids_service` section of `configs/phase2.yaml`.

## P9. Mitigation engine

Code: `mitigation/mitigation_engine.py`. It turns decisions into controller-independent JSON rule
actions. The controller turns them into OpenFlow FlowMods (drop) or meters (rate limits).

| Attack type | Action |
|---|---|
| DoS, BruteForce, WebAttack, Other | **drop** all traffic from the attacker |
| Probe | **rate-limit** the scanner to 20 packets/s |
| Botnet | **isolate** the host (drop all its outgoing traffic) |
| DDoS, ≤ 20 sources | drop each source |
| DDoS, > 20 sources (spoofed) | **rate-limit traffic to the victim** to 200 packets/s; still drop any source with ≥ 5 flows in the window (a real host hiding in the flood) |

Safety rails:

- Rate limits are in **packets per second**, not bits per second: a SYN is only 54 bytes, so a bandwidth
  limit let the flood straight through.
- **Priorities by severity:** drop 65535 > rate-limit a source 65435 > protect a victim 65335. Equal
  priorities let a leftover rule shadow a newer drop rule.
- **Rules expire:** 60 s idle, 300 s maximum (victim protection: 120 s).
- Manual **block / unblock** from the dashboard (API-key protected).
- Every action is appended to `logs/mitigation_log.jsonl`. SUSPICIOUS decisions go to
  `logs/suspicious_log.jsonl`.
- `IDS_MITIGATION=0` runs the IDS in detect-only mode.

## P10. SDN lab and attack scenarios

**Lab** (`docker/Dockerfile.sdn-lab`): a privileged Ubuntu 22.04 container with Mininet, Open vSwitch,
os-ken, hping3, nmap, iperf3, tcpdump and curl. It uses host networking, so the controller inside it
reaches the IDS at `127.0.0.1:3000`. The IDS itself runs on the host in the normal Python environment.

**Topology** (`topology/sdn_topology.py`): two OpenFlow 1.3 switches, six hosts, 100 Mbit/s links.

```
h1 (attacker)  ─┐                    ┌─ h4 (web server / victim)
h2 (client)    ─┼── s1 ══════ s2 ────┼─ h5 (client / iperf server)
h3 (client)    ─┘                    └─ h6 (attacker 2)
```

**Scenarios** (`scripts/inject_attack.py`). Each run is benign warm-up → attack → benign cool-down, with
benign background traffic (clients polling the web server, an iperf stream) throughout:

| Scenario | Attacker | Tool / pattern |
|---|---|---|
| `ddos` | h1 + h6 | `hping3` SYN flood to h4:80 with random (spoofed) source IPs |
| `dos` | h1 | `hping3` SYN flood to h4:80 |
| `probe` | h6 | `nmap` SYN scan of ports 1–2000 on h4, repeated |
| `bruteforce` | h1 | loop of short HTTP login requests to h4:2121 |
| `benign` | — | background traffic only (for false-positive checks) |

Each run writes a ground-truth JSON file (attack type, attacker IPs, victim, start/end, spoofed or not)
and a packet capture at the victim.

**Demo script** (`scripts/run_phase2_demo.sh`) runs everything and then:

1. measures mitigation from the captures (`scripts/analyze_mitigation.py`): time to mitigation, how
   much attack traffic was dropped, false blocks;
2. labels the recorded flows with the ground truth (`scripts/label_mininet_flows.py`);
3. evaluates the model on those lab graphs.

## P11. Dashboard and API

The FastAPI app (`web/app.py`) serves both the dashboards and the live IDS.

| Page | Shows |
|---|---|
| `/` | Offline results: Phase 1 metrics and Phase 2 comparison (GNN vs baselines, ablations) |
| `/live` | Live monitor: status (BENIGN / SUSPICIOUS / ATTACK), active flows and hosts, active rules and alerts, detection latency p90, live **traffic graph** of hosts, attack-confidence timeline, predicted type and flagged hosts, alert table, active-rule table, manual block / unblock form, mitigation audit log |

Main API endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /api/flows` | controller uploads flow stats; the reply carries mitigation rules (API key) |
| `GET /api/alerts`, `/api/mitigations`, `/api/mitigations/log` | alerts, active rules, audit log |
| `POST /api/mitigations/block`, `/unblock` | manual control (API key) |
| `GET /api/topology`, `/api/live/status`, `/api/live/latency`, `/api/live/stream` | data for the live page |
| `GET /api/health`, `/api/summary`, `/api/metrics/phase2` | health check and offline results |

## P12. Fine-tuning, tests and output files

**Fine-tuning on lab traffic.** The InSDN-only model blocked benign lab hosts (P14), so lab traffic
was recorded with mitigation switched off (`COLLECT=1 bash scripts/run_phase2_demo.sh`), labelled, and
added to training with 2× oversampling (`scripts/finetune_on_lab.sh`). Runs are split whole into
train / validation / test, so the test runs are completely unseen. The result is installed as
`models/gat_ids.pt`.

**Tests.** `python -m pytest tests -q` covers the graph builder, preprocessing, the Mininet tools and
the full live path (collector → GNN → classifier → mitigation → API). The tests use synthetic data and a
tiny generated model, so they need no dataset or GPU. They run on GitHub Actions on every push.

**Phase 2 output files:**

| File | Purpose |
|---|---|
| `results/phase2/final_results.md` | **The main report**: every table in this document, generated by `scripts/make_phase2_report.py` |
| `results/phase2/gnn_v2_gat.json`, `baselines_v2.json` | GNN and baseline test metrics |
| `results/phase2/ablation_*.json` | GCN, GraphSAGE, no-edge-features, window 50 / 200 |
| `results/phase2/gnn_v2_gat_ft.json` | fine-tuned model metrics |
| `results/phase2/confusion_matrices.png`, `roc_windows.png`, `gnn_v2_gat_curves.png` | plots for the report |
| `results/phase2/replay_report.json` | replay test (P14 step 1) |
| `results/phase2/latency_report.json`, `latency_histogram.png` | latency benchmark |
| `results/phase2/mininet_mitigation_*.json`, `mitigation_timeline.png` | live mitigation runs |
| `results/phase2/mininet_detection_{before,after}_finetune.json` | lab detection before / after fine-tuning |
| `Docs/eda_summary.md`, `results/phase2/eda/` | dataset analysis |

## P13. Offline results (InSDN test split, same features and split for all models)

| | Random Forest | XGBoost | GAT-IDS |
|---|---:|---:|---:|
| Attack detected in window (F1) | 0.998 | 0.998 | 0.999 |
| Attack type, macro-F1 (classes with ≥ 20 test windows) | 0.970 | 0.970 | **0.993** |
| DDoS / DoS / Probe F1 | 0.972 / 0.961 / 0.951 | 0.973 / 0.961 / 0.950 | **0.997 / 0.979 / 0.996** |
| Attacker host F1 (per-host features) | 0.994 | 0.995 | **0.997** |
| Attacker host F1 (max of flow scores) | 0.996 | **0.999** | — |

How to explain it:
- *Detecting* an attack is easy on InSDN: every model is at 0.998–0.999. The difference is in
  *naming* the attack: the GNN makes about 75 % fewer attack-type errors on the main classes. Those are
  the coordinated/structural attacks (DDoS star, scan fan-out), where the graph helps.
- For identifying attacker hosts the models are roughly equal; XGBoost on flow scores is slightly
  better. Say so. The GNN's advantage is one model doing both tasks, and it runs within 10 ms.
- The GNN is **worse on rare classes** (BruteForce 0.48, WebAttack/Botnet 0.0 on 1–9 test windows):
  a flow model gets every one of ~1,400 flows as a training sample, the GNN only ~50 windows. The
  binary detector still flags these windows as attacks.
- The original "+15–25 % F1" target is not meaningful when baselines already score 0.97; the honest
  framing is error reduction on coordinated attacks.

Ablations: GAT gives the best attack-type score (0.993) vs. GCN 0.984, GraphSAGE 0.911 and GAT without
edge features 0.981, so attention and edge features both help. GraphSAGE has the best host F1 (0.998).

## P14. From dataset to live network

1. **Replay test** (held-out InSDN flows through the live code path): every attack detected in all
   windows, all real attacker IPs blocked, DDoS handled by rate-limiting the victim (its sources are
   spoofed), 0 false alerts in 55 benign windows, 0 benign hosts blocked, latency p99 ≈ 12 ms.
2. **First Mininet test** (InSDN-only model): the attack was dropped 100 % within ~6 s, **but benign
   hosts, including the web server, were blocked**. On lab traffic the InSDN-only model flagged 98.7 % of
   benign windows. Benign traffic in our lab looks nothing like InSDN's benign capture.
3. **Fine-tuning** on 29 labelled Mininet runs (mitigation off while recording, split by run): benign
   false-positive rate on held-out lab runs 98.7 % → **1.7 %**, attack-type macro-F1 0.16 → **0.994**,
   while InSDN performance stayed the same (0.991 macro-F1, 0 % FPR). This is the domain-gap lesson:
   a model is only as good as its match to the network it protects.

4. **Live mitigation, three iterations** (3 repeats × benign/DDoS/DoS/Probe/BruteForce, mitigation on):

| Version | Attack runs ≥ 70 % dropped | Mean drop | Runs blocking a benign host | Rules in benign runs |
|---|---:|---:|---:|---:|
| Fine-tuned model, first version | 5/12 | 50 % | 9/12 | 1/3 |
| + mean host score, 2-window persistence, packet/s meters | 6/12 | 66 % | 1/12 | 0/3 |
| **+ rule priorities, per-switch rule tracking (final)** | **12/12** | **95 %** | **0/12** | **0/3** |

Final per attack: DoS 100 %, BruteForce 100 %, Probe 99 %, DDoS 80 % (victim rate-limited to 200 pps
by design). Median time to mitigation 4.5 s (Probe 17 s).

Be upfront about two things visible in `results/phase2/mitigation_timeline.png`:
- **Probe**: the first full scan (a ~3 s burst) is over before the rule lands; the rate limit stops the
  repeated scans that follow, not the first one.
- **DDoS**: benign clients keep only ~3 % of their traffic to the victim during the flood (DoS, Probe
  and BruteForce: 96–100 %+). Rate-limiting the victim protects the server from overload, but spoofed
  packets use up most of the allowance. Serving legitimate clients during a spoofed flood needs SYN
  cookies or a SYN proxy — a known limit of IP-based mitigation.

What each fix was for:
- *Benign clients blocked*: a host was scored by its *worst* chunk; with ~20 chunks per window and a
  window every 2 s, rare mistakes became certain. Now: mean over chunks + flagged in 2 windows in a row.
- *Rate limits did nothing*: they were in kbit/s, and a SYN flood is 54-byte packets; now packets/s.
- *A DoS kept flowing despite a correct drop rule*: a leftover DDoS victim rate-limit had the same
  OpenFlow priority and won the tie. Now drop > rate-limit > victim protection.

## P15. Things we discovered during integration (good discussion points)

- **The controller is a DoS target.** With 5-tuple forwarding rules, a SYN flood with random source
  ports creates a new flow entry per packet: flow tables reached 60,000 entries, stats polls slowed from
  2 s to 5 s and a later port scan was starved. This is the known SDN controller-saturation problem; a
  coarser match mode (`--match-mode host_pair`) and early mitigation reduce it.
- **Spoofed DDoS can't be blocked by source IP.** 400+ one-flow sources in a window: the engine
  rate-limits traffic to the victim instead, but still blocks any source with many flows (a real DoS
  host hiding in the flood).
- **Fail-open.** If the IDS is down, the network keeps forwarding.
- **Latency.** p90 ≈ 10 ms per window; 37 ms even for a 5,000-flow window on the laptop GPU.
  The first three calls took ~200 ms (TorchScript warm-up), now paid at start-up.

## P16. Limitations to state

- One real SDN dataset (InSDN) plus our own lab; the CICIDS2017 files with IP columns were not
  available for a cross-dataset test.
- Lab attacks are tool-generated (hping3, nmap, curl loops) on a 6-host topology.
- Rare attack types have too little window-level data.
- Detection time is dominated by the 2 s polling interval and the 10 s window, not by the model.
- Mitigation by IP cannot stop spoofed floods at the source.

## P17. Likely questions

**Q. Why not just use XGBoost?** It is as good at *detecting* and at scoring hosts from flows. The GNN
is clearly better at naming coordinated attacks and gives window type and host scores in one model.
Both could run side by side; our comparison is honest about this.

**Q. How do you know which host to block?** The node head scores every host; only hosts above a
validation-tuned threshold, not whitelisted and not in cool-down, are blocked, with rules that expire.

**Q. What if it blocks the wrong host?** Rules expire (60 s idle / 300 s max), there's a dashboard
unblock button, infrastructure can be whitelisted, and every action is in the audit log. This is
exactly what happened in our first live test, which is why we fine-tuned and measure false blocks.

**Q. Is 1.7 % false positives acceptable?** It is per 200-flow chunk of lab traffic; the architecture
target was < 5 %. What matters operationally is blocking: in the final live evaluation no benign host
was blocked in any run, and no rule was installed during benign-only traffic.

**Q. Why does mitigation take seconds, not the "2 round-trips" in the architecture?** The switch only
reports statistics every 2 s, the model looks at a 10 s window, and we deliberately wait for 2 attack
windows before blocking (that removed all false blocks). The model itself answers in ~10 ms. Faster
detection would need faster polling or packet sampling, which costs controller load.

**Q. Why InSDN and not CICIDS2017 in Phase 2?** The CICIDS2017 files we had (*MachineLearningCVE*)
have no IP, port or timestamp columns, so no real host graph can be built from them. InSDN has all three
and was captured on an SDN. The CICIDS2017 *TrafficLabelling* files would plug into the same pipeline.

**Q. Why only OpenFlow features when CICFlowMeter features are richer?** A live controller only gets
packet and byte counters, duration and match fields from the switch. A model trained on features the
controller can't produce would not run in the network. The same feature code is used for training and
live detection.

**Q. How does the GNN find the attacker in a DDoS where the sources are spoofed?** It doesn't need to
block each source. When a DDoS window has more than 20 sources, the mitigation engine rate-limits traffic
*to the victim* instead, and still blocks any source with 5 or more flows (a real host).

**Q. What happens if the IDS crashes?** The controller keeps forwarding traffic (fail-open), and
existing rules expire on their own. The network never goes down because of the IDS.

**Q. Why a Docker container for Mininet?** Mininet needs root and a supported Linux distribution. The
container (Ubuntu 22.04) runs Mininet, Open vSwitch and the controller, while the IDS runs on the host in
the normal Python environment.

## P18. Main points to tell your guide (Phase 2)

1. Phase 2 turned the offline model into a working SDN security system: detect → identify → block.
2. We first fixed three Phase 1 problems: fake topology, wrong InSDN labels and a leaky random split.
3. Graphs are real host-to-host graphs, and features use only what an OpenFlow switch reports.
4. One multi-task GAT gives the attack type of the window and an attacker score for every host.
5. Offline, detection is saturated for every model (≈ 0.999). The GNN names the attack type best
   (macro-F1 0.993 vs 0.970), i.e. about 75 % fewer attack-type errors on the main classes.
6. The live loop: controller → flow stats every 2 s → IDS → rules in the reply, in about 10 ms of
   IDS time.
7. Mitigation is per attack type (drop, rate-limit, isolate, victim protection), with priorities, expiry,
   whitelist, manual unblock and an audit log.
8. The InSDN-only model blocked benign lab hosts. Fine-tuning on lab traffic fixed it (false-positive
   rate 98.7 % → 1.7 %).
9. Final live result: 12/12 attack runs with at least 70 % of attack traffic dropped (mean 95 %), no
   benign host blocked, no rules in benign-only runs, median time to mitigation 4.5 s.
10. Honest limits: one real dataset, rare classes weak, spoofed DDoS still hurts legitimate clients,
    detection time bound by polling.

## P19. Suggested explanation script (Phase 2)

You can say:

> In Phase 2 we started by auditing our Phase 1 results. We found that the CICIDS2017 files had no IP
> addresses, so our graphs had no real topology; InSDN's benign label was misread, so all of InSDN was
> labelled as attack; and a random split of overlapping windows inflated our scores. We fixed all three
> before building anything new.

> We then rebuilt the graphs from InSDN. Each graph is a window of traffic where nodes are hosts and
> edges are flows. We used only features that an OpenFlow switch can report, like packet and byte
> counts and duration, so that exactly the same model can run on a live switch.

> The model is a multi-task Graph Attention Network. One head predicts the type of attack in the window,
> the other gives every host a probability of being an attacker. That second head is what makes
> automatic blocking possible, because it tells us whom to block.

> Offline, every model detects attacks almost perfectly, so the difference is in naming the attack: our
> GNN reaches a macro-F1 of 0.993 against 0.970 for XGBoost, especially on coordinated attacks like DDoS
> and port scans.

> For the live system we built an os-ken SDN controller and a Mininet lab in Docker. Every two seconds
> the controller sends flow statistics to our IDS. The IDS builds the graph, runs the GNN, and sends back
> OpenFlow rules to drop or rate-limit the attackers, all in about ten milliseconds.

> Our first live test failed in an instructive way: the attack was blocked, but so were benign hosts,
> because lab traffic looks different from InSDN. We fine-tuned on labelled lab traffic and added
> safety rules like mean host scores and two-window confirmation. In the final evaluation, every attack
> run had at least 70 % of its traffic dropped, 95 % on average, and no benign host was ever blocked.

## P20. Phase 2 contribution in one line

```text
InSDN → OpenFlow features → host graphs → multi-task GAT → TorchScript
      → os-ken controller + Mininet → live detection → automatic OpenFlow mitigation → dashboard
```

## P21. Final conclusion (both phases)

Phase 1 showed that a GNN can detect attacks from graph snapshots of network traffic. Phase 2 made
that result trustworthy and useful. The evaluation was fixed (real topology, correct labels, no
leakage). The model now names the attack and the attacker. And the system runs inside a real SDN
control loop, where it blocks attacks automatically and has not blocked a benign host.

The best way to present the result is:

> The GNN is not dramatically better than XGBoost at *detecting* attacks. Both are near perfect on
> InSDN. Its value is that one graph model names coordinated attacks more accurately and identifies the
> attacking hosts, fast enough (about 10 ms) to drive automatic mitigation in a live SDN. The biggest
> lesson was the domain gap: a model is only as good as its match to the network it protects, which is
> why fine-tuning on the target network was essential.
