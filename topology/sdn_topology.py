#!/usr/bin/env python3
"""Two-switch Mininet topology for the GNN-IDS lab.

    h1 (attacker)  ─┐                      ┌─ h4 (web server / victim)
    h2 (client)    ─┼── s1 ════════ s2 ────┼─ h5 (client / iperf server)
    h3 (client)    ─┘                      └─ h6 (attacker 2)

Run (inside the SDN lab VM, controller already started)::

    sudo python3 topology/sdn_topology.py --controller-ip 127.0.0.1          # CLI
    sudo python3 topology/sdn_topology.py --test                             # pingall and exit

Attack scenarios are in ``scripts/inject_attack.py``.
"""
from __future__ import annotations

import argparse
from functools import partial

from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.topo import Topo

HOSTS = {
    "h1": {"ip": "10.0.0.1", "switch": "s1", "role": "attacker"},
    "h2": {"ip": "10.0.0.2", "switch": "s1", "role": "client"},
    "h3": {"ip": "10.0.0.3", "switch": "s1", "role": "client"},
    "h4": {"ip": "10.0.0.4", "switch": "s2", "role": "server"},
    "h5": {"ip": "10.0.0.5", "switch": "s2", "role": "client"},
    "h6": {"ip": "10.0.0.6", "switch": "s2", "role": "attacker"},
}


class IDSTopo(Topo):
    def build(self, host_bw: float = 100, core_bw: float = 100):  # noqa: D401 - Mininet API
        s1 = self.addSwitch("s1", protocols="OpenFlow13")
        s2 = self.addSwitch("s2", protocols="OpenFlow13")
        self.addLink(s1, s2, bw=core_bw)
        for name, spec in HOSTS.items():
            host = self.addHost(name, ip=spec["ip"] + "/24", mac="00:00:00:00:00:0" + name[1])
            self.addLink(host, spec["switch"], bw=host_bw)


def build_network(controller_ip: str = "127.0.0.1", controller_port: int = 6653, host_bw: float = 100,
                  core_bw: float = 100, datapath: str = "kernel") -> Mininet:
    """``datapath="user"`` runs OVS in userspace, for hosts/containers without the openvswitch kernel module."""
    net = Mininet(
        topo=IDSTopo(host_bw=host_bw, core_bw=core_bw),
        switch=partial(OVSSwitch, datapath=datapath),
        link=TCLink,
        controller=None,
        autoSetMacs=False,
    )
    net.addController("c0", controller=RemoteController, ip=controller_ip, port=controller_port)
    return net


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--controller-ip", default="127.0.0.1")
    parser.add_argument("--controller-port", type=int, default=6653)
    parser.add_argument("--host-bw", type=float, default=100, help="Mbit/s per host link")
    parser.add_argument("--core-bw", type=float, default=100, help="Mbit/s on the s1-s2 link")
    parser.add_argument("--datapath", choices=["kernel", "user"], default="kernel")
    parser.add_argument("--test", action="store_true", help="run pingall and exit")
    args = parser.parse_args()

    setLogLevel("info")
    net = build_network(args.controller_ip, args.controller_port, args.host_bw, args.core_bw, args.datapath)
    net.start()
    try:
        if args.test:
            loss = net.pingAll()
            info(f"*** pingall loss: {loss}%\n")
        else:
            CLI(net)
    finally:
        net.stop()


if __name__ == "__main__":
    main()
