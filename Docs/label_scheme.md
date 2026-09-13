# Phase 2 Label Scheme

**Code:** `preprocessing/labels.py` (class mapping), `preprocessing/graph_builder_v2.py` (window and node
labels), `scripts/label_mininet_flows.py` (live Mininet data).

## 1. Canonical classes

| ID | Class | CICIDS2017 labels | InSDN labels |
|---:|---|---|---|
| 0 | Benign | `BENIGN` | `Normal` |
| 1 | DDoS | `DDoS` | `DDoS` |
| 2 | DoS | `DoS Hulk`, `DoS GoldenEye`, `DoS slowloris`, `DoS Slowhttptest`, `Heartbleed` | `DoS` |
| 3 | Probe | `PortScan` | `Probe` |
| 4 | BruteForce | `FTP-Patator`, `SSH-Patator`, `Web Attack – Brute Force` | `BFA` |
| 5 | WebAttack | `Web Attack – XSS`, `Web Attack – Sql Injection` | `Web-Attack` |
| 6 | Botnet | `Bot` | `BOTNET` |
| 7 | Other | `Infiltration` | `U2R` |

- Matching ignores case, whitespace and punctuation, so CICIDS2017's mis-encoded dash in
  `Web Attack � XSS` still maps correctly.
- **Unknown labels map to `Other`, never to Benign**, so a new attack name cannot silently become a
  benign training example.
- The binary label is `class != Benign`.
- The cleaner writes both: `Label` (binary, backwards compatible with Phase 1) and `Attack` (class name).

**Phase 1 bug fixed:** Phase 1 treated only `benign` as benign, so all 68,423 InSDN `Normal` rows were
labelled as attacks.

## 2. Window (graph) labels

- `y_multi`: if at least 10 % of the window's initiator-direction flow records are attacks, the most
  frequent attack class; otherwise Benign.
- `y`: `y_multi != Benign`.
- Windows with a benign window overlaid keep the attack window's label.

## 3. Host (node) labels — who to block

`node_y = 1` if the host **initiated** at least one attack flow in the window. Hosts that only receive
attack traffic (victims) or reply to it are 0.

"Initiated" uses the direction normalisation from `Docs/feature_schema_phase2.md`: InSDN labels flows by
capture period, and CICFlowMeter sometimes starts a flow at the server's reply, so many rows run
*victim → attacker* with the attack label. Without correction the victim would be labelled an attacker
(and the mitigation engine would learn to block servers). The lower-port side is taken as the server;
this corrects 27 % of metasploitable-2 rows, 5 % of OVS rows and 20 % of benign rows.

For spoofed DDoS every spoofed source initiates one flow, so every spoofed IP is an attacker node. The
mitigation engine handles this case separately (rate-limit the victim).

## 4. Flow (edge) labels

`edge_y` / `edge_multi` carry each record's label for the flow-level baselines. `edge_first` marks the first
time a record appears in a split (overlapping windows repeat flows), so baselines train on each flow once.

## 5. Live Mininet data

`scripts/inject_attack.py` writes ground truth per run (attack type, attacker IPs, victim, start/end,
spoofed or not). During `[start, end + window]`, a flow is an attack if it goes from an attacker IP to the
victim (for spoofed DDoS: from any IP outside the topology), and the victim's replies are attack flows
marked as responses. Runs are split whole into train/val/test.

## 6. Known limitations

- WebAttack, Botnet and Other have 1–9 test windows each in InSDN; their window-level scores are not
  statistically meaningful. Metrics report `major_macro_f1` (classes with ≥ 20 test windows) next to the
  all-class macro-F1.
- The 10 % window threshold means a window with a handful of attack flows among benign traffic is
  labelled Benign.
