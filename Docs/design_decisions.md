# Design Decisions (Phase 2)

Each entry gives the decision, the reason, and the evidence or trade-off. Measured numbers are in
`results/phase2/final_results.md`.

## Data

**D1. InSDN is the primary dataset; the CICIDS2017 files in the repo are not used for graphs.**
The CICIDS2017 files in `data/cicids2017/raw` are the *MachineLearningCVE* variant, which has no IP,
source-port or timestamp columns. Phase 1 therefore built graphs from made-up nodes (`src_<row index>` →
`dst_<port>`), which carry no host-to-host topology. InSDN has real IPs, ports and timestamps and was
captured on an SDN testbed. The CICIDS2017 *TrafficLabelling* CSVs (with IPs) plug into the same
pipeline (`preprocessing/openflow_features.CICIDS_COLUMNS`) once downloaded.

**D2. Only OpenFlow-derivable features.** A live OpenFlow controller reports packet/byte counters,
duration and match fields per flow entry, not CICFlowMeter's ~78 statistics. Training on features the
controller can't produce would make the model unusable live. One module computes features for both the
datasets and the live collector. See `Docs/feature_schema_phase2.md`.

**D3. Count-based windows offline, time-based windows live.** InSDN timestamps are mostly minute-level,
so 5-second windows would give only a handful of graphs per file. Offline windows are 100 consecutive
flows (capture order). Live windows are 10 s of flows, split into training-sized chunks when larger.

**D4. Time-ordered split, per (file, class), with a gap.** Sliding windows overlap. A random split puts
neighbouring, almost identical windows into both train and test and inflates scores (the Phase 1 GNN
drops from F1 0.994 to 0.983 with a time split). Splitting each class stream chronologically keeps every
class in every split, and the gap guarantees no flow is shared (unit-tested).

**D5. Benign overlay.** InSDN's benign and attack traffic come from different captures, so raw windows
are pure. Without mixing, "is there an attack" is trivial and "which host is the attacker" is barely
tested. Half of the attack windows get a benign window from the *same split* merged in.

**D6. Direction-normalised host labels.** InSDN labels flows by capture period, and CICFlowMeter
sometimes starts a flow at the server's reply, so victims appear as sources of attack flows (27% of
metasploitable-2 rows). Treating the lower-port side as the server fixes the attacker labels. Left as
is, the node head is trained to flag servers that answer attack traffic, which is the worst host to block.

**D7. Report major-class macro-F1.** WebAttack, Botnet and Other have 1–9 test windows. An F1 from
one window is noise and dominated the all-class macro average (0.66 vs 0.99). Both figures are reported;
model selection uses the major-class figure.

## Model

**D8. Multi-task GNN: graph head (attack type) + node head (attacker host).** The window label tells
*whether* and *what*; mitigation needs *who*. A per-host head is what makes automated blocking possible.
Loss = 0.5·node + 0.5·graph, with class weights (square-root inverse frequency, capped at 10).

**D9. Bidirectional message passing with a direction flag.** Attack traffic is directed, and a pure
attacker (e.g. a spoofed DDoS source) has no incoming edges, so it would never receive messages from
its victim. Reversed copies of all edges are added with a flag feature.

**D10. GAT as the main conv.** GAT uses edge features inside attention and learns which neighbours
matter. GCN and GraphSAGE are kept as ablations. Every variant also gets edge information through
mean-pooled incoming/outgoing edge embeddings, so ablations differ only in the conv layer.

**D11. Thresholds tuned on validation; decision = 1 − P(Benign).** The binary window score is the
probability mass on all attack classes. Window and node thresholds are chosen on the validation split
and stored with the model.

**D12. TorchScript export with an embedded bundle.** The live engine loads one file containing weights,
normalisation stats, class names, thresholds and feature names, so it can't be paired with the wrong
statistics. Verified bit-identical to the eager model on 100 test graphs. Conv-specific branches are
TorchScript constants so GCN/SAGE variants also export.

## Live system

**D13. Controller pushes stats; mitigation rides on the response.** The controller POSTs flow stats every
2 s and installs whatever rules come back. No extra round-trip, no server inside the controller, and the
IDS host needs no access to the switches. If the IDS is unreachable, the network keeps forwarding
(fail-open; a security product that breaks the network when it's down won't be deployed).

**D14. Two flow tables.** Table 0 holds only IDS rules (drop, or meter + continue); table 1 holds
forwarding. IDS rules then never show up in the statistics used for detection, and unblocking is one
delete by cookie that doesn't touch forwarding.

**D15. L3/L4 forwarding entries.** A MAC-learning switch has one entry per MAC pair and would give no
per-flow or per-port statistics. Forwarding entries match the 5-tuple, which mirrors dataset flows.
Trade-off: more flow-table entries and packet-ins during floods (`--match-mode host_pair` reduces this).

**D16. os-ken instead of Ryu.** Ryu is unmaintained and doesn't install on current Python. os-ken is
the OpenStack-maintained fork with the same OpenFlow API (the base class is renamed; `run_controller.py`
provides the missing `osken-manager` launcher).

**D17. Mitigation policy per attack type, with safety rails.**
- DoS/BruteForce/WebAttack → drop the source; Probe → rate-limit (a scanner may share an address with
  legitimate use); Botnet → drop all egress.
- Rate limits are in **packets per second** (victim 200 pps, scanner 20 pps). The first version used
  512 kbit/s, which a SYN flood (54-byte packets, 1,000 pps ≈ 430 kbit/s) passes almost untouched:
  measured DDoS reduction was only 31 %.
- DDoS with more than 20 sources → the sources are likely spoofed, so rate-limit traffic *to the
  victim*. Hosts with ≥ 5 flows in the window are still blocked individually (a real DoS source hiding in
  a DDoS; found in the replay test).
- Rule priorities by severity: drop 65535 > rate-limit a source 65435 > protect a victim 65335. With
  equal priorities, a leftover victim rate-limit shadowed a later drop rule for a DoS source (undefined
  precedence for overlapping equal-priority rules); DoS reduction went from 39 % to 100 % once fixed.
- Victim protection expires after 120 s. It throttles benign traffic to the victim too: in the live
  evaluation legitimate clients kept only ~3 % of their traffic during a spoofed DDoS. Accepted as a
  limitation; IP-based rules cannot tell spoofed from real sources (SYN cookies/proxy would).
- Each switch holds its own copy of a rule; a rule counts as removed only when all copies are gone
  (a copy on a switch the attack never crosses idles out early).
- Rules expire (60 s idle, 300 s hard), whitelisted infrastructure is never blocked, a per-host
  cool-down prevents rule storms, there is a manual unblock, and every action is audit-logged.
- ATTACK needs confidence ≥ 0.85; 0.5–0.85 is SUSPICIOUS (logged only).

**D18b. Mean host score over chunks + persistence before blocking.** The first live evaluation
with the fine-tuned model blocked benign clients in most attack runs, although offline the per-host
false-positive rate was only 0.38 %. Live, a host is scored in ~20 chunks per window and a new
window every 2 s, so taking the *maximum* gives it hundreds of chances to cross the threshold once.
Fixes: a host's score is its mean over its chunks, a host must be flagged in 2 consecutive windows
before it can be blocked, and victim protection needs 2 consecutive DDoS windows. Cost: about one
extra polling interval (2 s) of detection delay.

**D18. Warm-up at start.** TorchScript's profiling executor optimises during the first calls (~200 ms
each). Running five synthetic windows at start-up moved live p99 latency from 208 ms to 11 ms.

**D19. Docker lab.** Mininet needs root and a supported distro; the development machine runs Arch.
A privileged Ubuntu 22.04 container with host networking runs Mininet, Open vSwitch and the controller,
while the IDS stays in the normal Python environment on the host.

**D20. Fine-tuning on Mininet traffic.** The InSDN-trained model blocked benign hosts in the first live
test. Benign lab traffic (three clients polling one web server, an iperf stream) differs from InSDN's
benign capture. Labelled lab traffic is collected with mitigation off, split by run, and added to
training with oversampling.
