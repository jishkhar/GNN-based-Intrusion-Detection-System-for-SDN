#!/usr/bin/env python3
"""Run a labelled attack scenario in the Mininet lab.

Starts the topology, runs benign background traffic, launches the attack, and
writes ground truth (attacker IPs, victim, start/end times) to
``data/mininet/runs/<timestamp>_<scenario>.json``. A packet capture on the
victim link is saved next to it for mitigation analysis
(``scripts/analyze_mitigation.py``).

Run inside the SDN lab VM with the controller and IDS already running::

    sudo python3 scripts/inject_attack.py --scenario ddos --duration 60
    sudo python3 scripts/inject_attack.py --scenario benign --duration 120   # benign-only data
    sudo python3 scripts/inject_attack.py --scenario all                     # every attack in turn

Scenarios: benign, ddos, dos, probe, bruteforce, all (= benign + every attack).
``--repeat N`` runs the scenario N times (for 20-run latency statistics);
``--vary-rate`` changes the hping3 packet rate between runs (more varied training data).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mininet.log import info, setLogLevel  # noqa: E402

from topology.sdn_topology import HOSTS, build_network  # noqa: E402

VICTIM = "h4"
ATTACKS = {
    # hping3 SYN flood from random (spoofed) sources, from both attacker hosts.
    "ddos": {
        "type": "DDoS",
        "hosts": ["h1", "h6"],
        "cmd": "hping3 -S -p 80 --rand-source -i u{interval_us} {victim_ip}",
        "spoofed": True,
    },
    # Single-source SYN flood (hping3 increments the source port per packet).
    "dos": {"type": "DoS", "hosts": ["h1"], "cmd": "hping3 -S -p 80 -i u{interval_us} {victim_ip}", "spoofed": False},
    # SYN scan of the first 2000 ports, repeated for the whole attack phase.
    "probe": {
        "type": "Probe",
        "hosts": ["h6"],
        "cmd": "sh -c 'while true; do nmap -sS -T4 -Pn -p 1-2000 {victim_ip} >/dev/null; done'",
        "spoofed": False,
    },
    # Many short TCP sessions against one service port (a login brute-force pattern).
    "bruteforce": {
        "type": "BruteForce",
        "hosts": ["h1"],
        "cmd": "sh -c 'while true; do curl -s -m 1 -o /dev/null http://{victim_ip}:2121/login?u=admin\\&p=$RANDOM; done'",
        "spoofed": False,
    },
}


def start_benign(net) -> None:
    """Web server + iperf server, and clients generating steady mixed traffic."""
    server, iperf_server = net.get(VICTIM), net.get("h5")
    server.cmd("mkdir -p /tmp/www && echo ok > /tmp/www/index.html")
    server.cmd("cd /tmp/www && python3 -m http.server 80 >/dev/null 2>&1 &")
    server.cmd("cd /tmp/www && python3 -m http.server 2121 >/dev/null 2>&1 &")
    iperf_server.cmd("iperf3 -s -D >/dev/null 2>&1")
    time.sleep(1)
    web_ip, iperf_ip = HOSTS[VICTIM]["ip"], HOSTS["h5"]["ip"]
    for name in ("h2", "h3", "h5"):
        net.get(name).cmd(
            f"sh -c 'while true; do curl -s -o /dev/null http://{web_ip}/; sleep $(awk \"BEGIN{{srand(); print 0.5+rand()*2}}\"); done' &"
        )
    net.get("h2").cmd(f"iperf3 -c {iperf_ip} -t 100000 -b 2M >/dev/null 2>&1 &")
    net.get("h3").cmd(f"ping -i 0.5 {iperf_ip} >/dev/null 2>&1 &")


def stop_all(net) -> None:
    for host in net.hosts:
        host.cmd("pkill -f hping3; pkill -f nmap; pkill -f curl; pkill -f iperf3; pkill -f 'ping -i'; pkill -f http.server")
        host.cmd("pkill -f 'while true'")


def run_attack(net, name: str, duration: float, interval_us: int) -> dict:
    spec = ATTACKS[name]
    victim_ip = HOSTS[VICTIM]["ip"]
    cmd = spec["cmd"].format(victim_ip=victim_ip, interval_us=interval_us)
    start = time.time()
    for host in spec["hosts"]:
        net.get(host).cmd(f"{cmd} >/dev/null 2>&1 &")
    info(f"*** {name}: {spec['hosts']} -> {victim_ip} for {duration}s\n")
    time.sleep(duration)
    for host in spec["hosts"]:
        net.get(host).cmd("pkill -f hping3; pkill -f nmap; pkill -f 'while true'; pkill -f curl")
    end = time.time()
    return {
        "scenario": name,
        "attack_type": spec["type"],
        "attacker_hosts": spec["hosts"],
        "attacker_ips": [HOSTS[h]["ip"] for h in spec["hosts"]],
        "spoofed_sources": spec["spoofed"],
        "victim_ip": victim_ip,
        "start": start,
        "end": end,
        "command": cmd,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=["benign", *ATTACKS, "all"], required=True)
    parser.add_argument("--duration", type=float, default=60, help="attack duration (s)")
    parser.add_argument("--warmup", type=float, default=30, help="benign-only time before the attack (s)")
    parser.add_argument("--cooldown", type=float, default=30, help="benign-only time after the attack (s)")
    parser.add_argument("--interval-us", type=int, default=2000, help="hping3 packet interval (µs)")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--vary-rate", action="store_true", help="scale --interval-us by 0.5-5x per run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--controller-ip", default="127.0.0.1")
    parser.add_argument("--datapath", choices=["kernel", "user"], default="kernel")
    parser.add_argument("--out-dir", default="data/mininet/runs")
    parser.add_argument("--no-capture", action="store_true")
    args = parser.parse_args()

    setLogLevel("info")
    os.makedirs(args.out_dir, exist_ok=True)
    scenarios = ["benign", *ATTACKS] if args.scenario == "all" else [args.scenario]
    rng = random.Random(args.seed)
    net = build_network(args.controller_ip, datapath=args.datapath)
    net.start()
    try:
        net.pingAll()  # populate MAC tables before measuring anything
        for run in range(args.repeat):
            for name in scenarios:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                base = os.path.join(args.out_dir, f"{stamp}_{name}_{run}")
                victim = net.get(VICTIM)
                if not args.no_capture:
                    victim.cmd(f"tcpdump -i {VICTIM}-eth0 -nn -tt -w {base}.pcap >/dev/null 2>&1 &")
                start_benign(net)
                info(f"*** warmup {args.warmup}s (benign only)\n")
                time.sleep(args.warmup)
                record = {"run": run, "benign_start": time.time() - args.warmup, "hosts": HOSTS}
                if name == "benign":
                    time.sleep(args.duration)
                    record.update({"scenario": "benign", "attack_type": "Benign", "attacker_ips": []})
                else:
                    interval = int(args.interval_us * rng.choice([0.5, 1, 2, 5])) if args.vary_rate else args.interval_us
                    record.update(run_attack(net, name, args.duration, interval))
                    record["interval_us"] = interval
                time.sleep(args.cooldown)
                record["benign_end"] = time.time()
                stop_all(net)
                if not args.no_capture:
                    victim.cmd("pkill -f tcpdump")
                    record["pcap"] = base + ".pcap"
                with open(base + ".json", "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2)
                info(f"*** ground truth -> {base}.json\n")
                time.sleep(5)
    finally:
        stop_all(net)
        net.stop()


if __name__ == "__main__":
    main()
