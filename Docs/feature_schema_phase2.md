# Phase 2 Feature Schema (OpenFlow-compatible)

**Code:** `preprocessing/openflow_features.py` is the only place features are defined. The dataset
pipeline (`graph_builder_v2.py`) and the live IDS (`inference/inference_engine.py`) both call it, so the
model sees the same features in training and in production.

## Why restrict features

Phase 1 used ~78 CICFlowMeter columns (inter-arrival times, flag counts, bulk rates, …). A live SDN
controller cannot produce those: OpenFlow flow statistics only give counters per flow-table entry. A
model trained on CICFlowMeter features would have nothing to run on in the live network. Phase 2
therefore uses only what an OpenFlow 1.3 switch reports, plus quantities derived from it.

## 1. Flow record (one unidirectional flow)

| Field | OpenFlow source (live) | CICFlowMeter source (datasets) |
|---|---|---|
| `src_ip`, `dst_ip` | match `ipv4_src`, `ipv4_dst` | `Src IP`/`Dst IP` (InSDN), `Source IP`/`Destination IP` (CICIDS2017) |
| `src_port`, `dst_port` | match `tcp_src`/`udp_src`, `tcp_dst`/`udp_dst` | `Src Port`, `Dst Port` |
| `ip_proto` | match `ip_proto` | `Protocol` |
| `packets` | `packet_count` | forward: `Tot Fwd Pkts`; reverse: `Tot Bwd Pkts` |
| `bytes` | `byte_count` (whole Ethernet frames) | `TotLen Fwd/Bwd Pkts` (payload) **+ packets × header size** (54 B TCP, 42 B UDP/other) |
| `duration` | `duration_sec + duration_nsec/1e9` | `Flow Duration` / 10⁶ (µs → s) |

**Bidirectional rows.** CICFlowMeter rows cover both directions; OpenFlow counts each direction as a
separate flow entry. Each dataset row becomes a forward record and, if it has backward packets, a
reverse record with source/destination swapped.

**Direction normalisation.** CICFlowMeter's "forward" is whichever side it saw first, which is sometimes
the server's reply. The side with the lower port is taken as the server, and the record *from* the
server is marked as the response (`is_reverse`). This only affects labels (who is an attacker), not
features. See `Docs/label_scheme.md`.

**Live counters.** Flow-stats counters are cumulative. The collector (`controller/flow_collector.py`)
uses the latest cumulative values (the same "flow totals so far" meaning as the dataset columns) and
uses the change between polls only to decide whether a flow is active.

## 2. Edge features (per flow record, 16)

| # | Name | Definition |
|---|---|---|
| 1 | `packets` | packet count |
| 2 | `bytes` | byte count (frames) |
| 3 | `duration` | seconds |
| 4 | `pkt_rate` | packets / max(duration, 1 ms) |
| 5 | `byte_rate` | bytes / max(duration, 1 ms) |
| 6 | `mean_pkt_size` | bytes / packets |
| 7–10 | `proto_tcp`, `proto_udp`, `proto_icmp`, `proto_other` | protocol one-hot |
| 11–13 | `dport_well_known`, `dport_registered`, `dport_ephemeral` | destination port bucket (<1024, 1024–49151, ≥49152); all 0 for non-TCP/UDP |
| 14–16 | `sport_well_known`, `sport_registered`, `sport_ephemeral` | source port bucket |

The model also receives a direction flag (0 = original edge, 1 = reversed copy used for message passing).

## 3. Node features (per host per window, 16)

| # | Name | Definition |
|---|---|---|
| 1–2 | `out_flows`, `in_flows` | flow records sent / received |
| 3–4 | `out_packets`, `in_packets` | packets sent / received |
| 5–6 | `out_bytes`, `in_bytes` | bytes sent / received |
| 7 | `out_degree` | distinct destination hosts (fan-out) |
| 8 | `in_degree` | distinct source hosts (fan-in) |
| 9 | `unique_dst_ports` | distinct destination ports contacted |
| 10 | `dst_port_entropy` | Shannon entropy (bits) of the host's destination-port distribution (scanning indicator) |
| 11 | `unique_src_ports_in` | distinct source ports of incoming flows |
| 12 | `mean_out_duration` | mean duration of outgoing flows |
| 13–16 | `out_tcp_frac`, `out_udp_frac`, `out_icmp_frac`, `out_other_frac` | protocol mix of outgoing flows |

## 4. Normalisation

Features are heavy-tailed (bytes range over 8 orders of magnitude). Both node and edge features are
transformed as `sign(x)·log(1+|x|)` and then z-scored with the mean/std of the **training split only**.
The statistics are stored in the checkpoint and in the exported model bundle
(`models/gat_ids.json`), so the live engine applies exactly the training transform.

## 5. Graph windows

- **Offline (InSDN):** timestamps are mostly minute-resolution, so a window is 100 consecutive flows in
  capture order (stride 25; 10 for the benign capture to balance classes).
- **Live:** a window is every flow active in the last 10 s. Windows larger than a training window are
  split into chunks of 200 records (2 × 100 flows), run as one batch; the window score is the maximum
  over chunks and each host keeps its highest score.
