# SDN Lab Setup (Mininet + Open vSwitch + os-ken)

The lab runs in a privileged **Docker container** (Ubuntu 22.04 with Mininet 2.3, Open vSwitch 2.17,
os-ken 2.8.1, hping3, nmap, iperf3, tcpdump). The IDS API runs on the host, in the project's Python
environment. `--network host` lets the container's controller reach the IDS at `127.0.0.1:3000`.

This setup has been tested on the reference machine (Arch Linux, kernel 6.19, Docker, user in the
`docker` group).

```
┌──────────────────────────── host ────────────────────────────┐
│  IDS API + dashboard (web/app.py, :3000, PyTorch model)       │
│        ▲ POST /api/flows (every 2 s)  │ rules in the response │
│  ┌─────┴──────────── lab container (privileged) ───────────┐  │
│  │ controller/run_controller.py (os-ken, OpenFlow :6653)    │  │
│  │ Mininet: h1..h6, s1, s2 (Open vSwitch, OpenFlow 1.3)     │  │
│  │ scripts/inject_attack.py (benign traffic + attacks)      │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

## 1. One-time setup

```bash
# Docker access without sudo (log out and back in afterwards)
sudo usermod -aG docker "$USER"

# Build the lab image (~2 minutes)
docker build -t gnn-ids-lab -f docker/Dockerfile.sdn-lab docker/

# Train and export the model if models/gat_ids.pt does not exist yet
PYTHON=.venv/bin/python bash scripts/run_phase2_training.sh --skip-ablations
```

**Kernel module.** Open vSwitch's kernel datapath needs the `openvswitch` module on the **host**. Most
distributions load it on demand. If the lab reports `no openvswitch kernel module`, either load it:

```bash
sudo modprobe openvswitch
```

or run everything with the userspace datapath (slower, no module needed): `DATAPATH=user`.

## 2. Check the lab works

```bash
docker run --rm --privileged --network host -v "$PWD":/work gnn-ids-lab bash -c '
  python3 controller/run_controller.py > /tmp/ctrl.log 2>&1 &
  sleep 4
  python3 topology/sdn_topology.py --test'
```

Expected: `*** Results: 0% dropped (30/30 received)`.

## 3. Run the end-to-end demo

```bash
bash scripts/run_phase2_demo.sh                         # benign + DDoS + DoS + Probe + BruteForce
SCENARIO=dos DURATION=60 bash scripts/run_phase2_demo.sh
KEEP_IDS=1 SCENARIO=ddos bash scripts/run_phase2_demo.sh   # keep the dashboard up afterwards
```

The script:
1. starts the IDS API on the host with a random API key,
2. starts the controller and Mininet in the container and runs the scenario (benign warm-up → attack →
   benign cool-down), capturing packets at the victim,
3. analyses mitigation (time to mitigation, drop rate, false blocks),
4. labels the recorded flows into graphs (`mininet_v2.pt`),
5. evaluates the model on those graphs.

Everything for a run goes into `data/mininet/<timestamp>/`: `runs/*.json` (ground truth), `runs/*.pcap`,
`flows.csv`, `logs/mitigation_log.jsonl`, `mitigation_report.json`, `transfer_eval.json`.

Open the dashboard at http://127.0.0.1:3000/live while it runs.

## 4. Collect training data from the lab

```bash
COLLECT=1 REPEAT=5 SCENARIO=all bash scripts/run_phase2_demo.sh
```

`COLLECT=1` turns mitigation off (blocks would change the traffic being recorded) and varies the attack
packet rate between runs. Runs are split whole: the 4th run of each scenario → test, the 5th → validation.

## 5. Manual control

```bash
docker run -it --rm --privileged --network host -v "$PWD":/work gnn-ids-lab
# inside the container:
python3 controller/run_controller.py --ids-url http://127.0.0.1:3000/api/flows --api-key <key> &
python3 topology/sdn_topology.py            # Mininet CLI: h1 hping3 -S -p 80 -i u2000 10.0.0.4
```

Controller options: `--poll-interval` (default 2 s), `--match-mode 5tuple|host_pair`,
`--rate-limit-as-drop` (for switches without meter support).

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `IDS unreachable` in `controller.log` | The IDS API isn't running on the host, or the container isn't using `--network host`. |
| `pingall` drops packets | Check `controller.log` for `switch … connected`; make sure nothing else is listening on 6653. |
| Switches fail to start (datapath errors in `/var/log/openvswitch/ovs-vswitchd.log`) | Use `DATAPATH=user`, or `sudo modprobe openvswitch` on the host. |
| Rate limits have no effect | The switch has no meter support: use `--rate-limit-as-drop`. |
| Files in `data/mininet/` owned by root | The demo script restores ownership; after manual runs use `sudo chown -R $USER data/mininet`. |

## 7. Without Docker (Ubuntu 22.04 VM)

```bash
sudo apt install mininet openvswitch-switch hping3 nmap iperf3 tcpdump python3-pip
pip3 install os-ken==2.8.1 requests
# on the VM, with the IDS reachable at <host-ip>:3000:
python3 controller/run_controller.py --ids-url http://<host-ip>:3000/api/flows &
sudo python3 scripts/inject_attack.py --scenario all
```
Start the IDS on the host with `--host 0.0.0.0` and set `IDS_API_KEY`.
