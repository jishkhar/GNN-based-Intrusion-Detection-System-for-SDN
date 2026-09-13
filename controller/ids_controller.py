"""IDS-aware OpenFlow 1.3 controller app for os-ken (preferred) or Ryu.

Run inside the SDN lab (see Docs/sdn_lab_setup.md)::

    python3 controller/run_controller.py --ids-url http://<ids-host>:3000/api/flows
    # or, with Ryu installed: IDS_API_URL=... ryu-manager controller/ids_controller.py

Pipeline on every switch:

- **table 0 (IDS)**: mitigation rules from the IDS at priority 65535
  (drop, or meter + continue); a table-miss entry sends everything to table 1.
- **table 1 (forwarding)**: reactive L2 learning, but each IPv4 flow is
  installed with an L3/L4 match (5-tuple by default) so the switch keeps
  per-flow counters. A plain MAC-learning switch would give the IDS no
  per-host statistics. Table-miss sends packets to the controller.

Every ``IDS_POLL_INTERVAL`` seconds the app requests flow stats from table 1 on
all switches and POSTs them to the IDS. The HTTP response contains mitigation
actions, which are installed immediately (no extra round-trip). If the IDS is
unreachable the network keeps forwarding (fail-open).

Configuration (environment variables):
    IDS_API_URL             default http://127.0.0.1:3000/api/flows
    IDS_API_KEY             sent as X-API-Key if set
    IDS_POLL_INTERVAL       seconds between stats polls (default 2)
    IDS_MATCH_MODE          5tuple (default) | host_pair  (src, dst, proto, dst port)
    IDS_FLOW_IDLE_TIMEOUT   idle timeout of forwarding flows (default 30)
    IDS_RATE_LIMIT_AS_DROP  1 = install drop instead of meters (switches without meter support)
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

try:  # os-ken is the maintained fork of Ryu; the OpenFlow APIs match, the base class is renamed.
    from os_ken.base import app_manager
    from os_ken.controller import ofp_event
    from os_ken.controller.handler import CONFIG_DISPATCHER, DEAD_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
    from os_ken.lib import hub
    from os_ken.lib.packet import ether_types, ethernet, in_proto, ipv4, packet, tcp, udp
    from os_ken.ofproto import ofproto_v1_3
except ImportError:  # pragma: no cover - depends on the lab environment
    from ryu.base import app_manager
    from ryu.controller import ofp_event
    from ryu.controller.handler import CONFIG_DISPATCHER, DEAD_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
    from ryu.lib import hub
    from ryu.lib.packet import ether_types, ethernet, in_proto, ipv4, packet, tcp, udp
    from ryu.ofproto import ofproto_v1_3

IDS_TABLE, FORWARD_TABLE = 0, 1
IDS_PRIORITY = 65535
# os-ken renamed Ryu's base class; accept either.
BaseApp = getattr(app_manager, "OSKenApp", None) or getattr(app_manager, "RyuApp")

IDS_COOKIE_TAG = 0x1D5 << 48  # marks rules installed on behalf of the IDS
IDS_COOKIE_MASK = 0xFFFF << 48
FORWARD_PRIORITY = 10


def is_ids_cookie(cookie: int) -> bool:
    return (cookie & IDS_COOKIE_MASK) == IDS_COOKIE_TAG


def stat_to_entry(dpid: int, stat) -> dict | None:
    """OFPFlowStats -> flow-stats entry understood by the IDS (None for non-IP rules)."""
    match = stat.match
    if match.get("ipv4_src") is None or match.get("ipv4_dst") is None:
        return None
    entry = {
        "dpid": dpid,
        "ipv4_src": match.get("ipv4_src"),
        "ipv4_dst": match.get("ipv4_dst"),
        "ip_proto": match.get("ip_proto", 0),
        "packet_count": stat.packet_count,
        "byte_count": stat.byte_count,
        "duration_sec": stat.duration_sec,
        "duration_nsec": stat.duration_nsec,
    }
    for field in ("tcp_src", "tcp_dst", "udp_src", "udp_dst"):
        if match.get(field) is not None:
            entry[field] = match.get(field)
    return entry


class IDSController(BaseApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_url = os.environ.get("IDS_API_URL", "http://127.0.0.1:3000/api/flows")
        self.api_key = os.environ.get("IDS_API_KEY")
        self.poll_interval = float(os.environ.get("IDS_POLL_INTERVAL", "2"))
        self.match_mode = os.environ.get("IDS_MATCH_MODE", "5tuple")
        self.flow_idle_timeout = int(os.environ.get("IDS_FLOW_IDLE_TIMEOUT", "30"))
        self.rate_limit_as_drop = os.environ.get("IDS_RATE_LIMIT_AS_DROP") == "1"
        self.mac_to_port: dict = {}
        self.datapaths: dict = {}
        self.pending_entries: list = []
        self.removed_cookies: list = []
        self.active_ids_rules: dict = {}  # rule cookie -> add command (re-applied to new switches)
        self.rule_switches: dict = {}  # rule cookie -> dpids that still hold the rule
        self.monitor_thread = hub.spawn(self._monitor)

    # ------------------------------------------------------------ switches
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        parser, ofp = dp.ofproto_parser, dp.ofproto
        # table 0: everything not blocked continues to forwarding
        self._flow_mod(dp, IDS_TABLE, 0, parser.OFPMatch(), [parser.OFPInstructionGotoTable(FORWARD_TABLE)])
        # table 1: unknown traffic goes to the controller
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        self._flow_mod(
            dp, FORWARD_TABLE, 0, parser.OFPMatch(), [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        )
        self.datapaths[dp.id] = dp
        for cookie, cmd in self.active_ids_rules.items():
            self._apply_add(dp, cmd)
            self.rule_switches.setdefault(cookie, set()).add(dp.id)
        self.logger.info("switch %016x connected", dp.id)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER and dp.id is not None:
            self.datapaths[dp.id] = dp
        elif ev.state == DEAD_DISPATCHER and dp.id in self.datapaths:
            del self.datapaths[dp.id]

    def _flow_mod(self, dp, table, priority, match, instructions, idle=0, hard=0, cookie=0, flags=0, buffer_id=None):
        parser, ofp = dp.ofproto_parser, dp.ofproto
        kwargs = dict(
            datapath=dp,
            table_id=table,
            priority=priority,
            match=match,
            instructions=instructions,
            idle_timeout=idle,
            hard_timeout=hard,
            cookie=cookie,
            flags=flags,
        )
        if buffer_id is not None and buffer_id != ofp.OFP_NO_BUFFER:
            kwargs["buffer_id"] = buffer_id
        dp.send_msg(parser.OFPFlowMod(**kwargs))

    # ---------------------------------------------------------- forwarding
    def _ip_match(self, parser, in_port, pkt):
        ip = pkt.get_protocol(ipv4.ipv4)
        fields = {"in_port": in_port, "eth_type": ether_types.ETH_TYPE_IP, "ipv4_src": ip.src, "ipv4_dst": ip.dst,
                  "ip_proto": ip.proto}
        l4 = pkt.get_protocol(tcp.tcp) or pkt.get_protocol(udp.udp)
        if l4 is not None:
            prefix = "tcp" if ip.proto == in_proto.IPPROTO_TCP else "udp"
            if self.match_mode == "5tuple":
                fields[f"{prefix}_src"] = l4.src_port
            fields[f"{prefix}_dst"] = l4.dst_port
        return parser.OFPMatch(**fields)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        parser, ofp = dp.ofproto_parser, dp.ofproto
        in_port = msg.match["in_port"]
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        if eth is None or eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        table = self.mac_to_port.setdefault(dp.id, {})
        table[eth.src] = in_port
        out_port = table.get(eth.dst, ofp.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofp.OFPP_FLOOD and eth.ethertype == ether_types.ETH_TYPE_IP:
            match = self._ip_match(parser, in_port, pkt)
            inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
            self._flow_mod(
                dp, FORWARD_TABLE, FORWARD_PRIORITY, match, inst,
                idle=self.flow_idle_timeout, flags=ofp.OFPFF_SEND_FLOW_REM, buffer_id=msg.buffer_id,
            )
            if msg.buffer_id != ofp.OFP_NO_BUFFER:
                return  # the flow mod released the buffered packet

        data = msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None
        dp.send_msg(parser.OFPPacketOut(datapath=dp, buffer_id=msg.buffer_id, in_port=in_port, actions=actions, data=data))

    # -------------------------------------------------------- statistics
    def _monitor(self):
        while True:
            for dp in list(self.datapaths.values()):
                parser = dp.ofproto_parser
                dp.send_msg(parser.OFPFlowStatsRequest(dp, table_id=FORWARD_TABLE))
            hub.sleep(self.poll_interval / 2)  # let replies arrive
            entries, self.pending_entries = self.pending_entries, []
            removed, self.removed_cookies = self.removed_cookies, []
            if entries or removed:
                self._post({"flows": entries, "removed_cookies": removed, "controller": "ids_controller"})
            hub.sleep(self.poll_interval / 2)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        dpid = ev.msg.datapath.id
        for stat in ev.msg.body:
            if stat.priority == 0 or is_ids_cookie(stat.cookie):
                continue
            entry = stat_to_entry(dpid, stat)
            if entry is not None:
                self.pending_entries.append(entry)

    @set_ev_cls(ofp_event.EventOFPFlowRemoved, MAIN_DISPATCHER)
    def flow_removed_handler(self, ev):
        msg = ev.msg
        if is_ids_cookie(msg.cookie):
            rule_cookie = msg.cookie & ~IDS_COOKIE_MASK
            # Each switch holds its own copy; a copy on a switch the attack never crosses idles out
            # early. The rule is gone only when every copy is.
            holders = self.rule_switches.get(rule_cookie, set())
            holders.discard(msg.datapath.id)
            if not holders:
                self.rule_switches.pop(rule_cookie, None)
                self.active_ids_rules.pop(rule_cookie, None)
                self.removed_cookies.append(rule_cookie)
            return
        # A forwarding flow expired: report its final counters so they aren't lost.
        entry = stat_to_entry(msg.datapath.id, msg)
        if entry is not None:
            self.pending_entries.append(entry)

    # ------------------------------------------------------ IDS exchange
    def _post(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(self.api_url, data=body, method="POST", headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("X-API-Key", self.api_key)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode())
        except (urllib.error.URLError, OSError, ValueError) as exc:
            self.logger.warning("IDS unreachable (%s); forwarding continues", exc)
            return
        if result.get("level") not in (None, "BENIGN"):
            self.logger.warning(
                "IDS %s: %s (confidence %.2f)", result["level"], result.get("attack_type"), result.get("confidence", 0)
            )
        for cmd in result.get("actions", []):
            self.apply_action(cmd)

    def apply_action(self, cmd: dict) -> None:
        for dp in list(self.datapaths.values()):
            if cmd["op"] == "add":
                self._apply_add(dp, cmd)
            elif cmd["op"] == "delete":
                self._apply_delete(dp, cmd)
        if cmd["op"] == "add":
            self.active_ids_rules[cmd["cookie"]] = cmd
            self.rule_switches[cmd["cookie"]] = set(self.datapaths)
        else:
            self.active_ids_rules.pop(cmd["cookie"], None)
            self.rule_switches.pop(cmd["cookie"], None)
        self.logger.warning("IDS rule %s %s %s %s", cmd["op"], cmd["rule_id"], cmd.get("action", ""), cmd["match"])

    def _apply_add(self, dp, cmd: dict) -> None:
        parser, ofp = dp.ofproto_parser, dp.ofproto
        match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, **cmd["match"])
        cookie = IDS_COOKIE_TAG | int(cmd["cookie"])
        if cmd["action"] == "rate_limit" and not self.rate_limit_as_drop:
            meter_id = int(cmd["cookie"]) & 0xFFFF or 1
            # Packets/s by default: flood packets are tiny, so a kbit/s limit lets them through.
            if cmd.get("rate_pps"):
                rate, flags = int(cmd["rate_pps"]), ofp.OFPMF_PKTPS | ofp.OFPMF_BURST
            else:
                rate, flags = int(cmd.get("rate_kbps") or 512), ofp.OFPMF_KBPS | ofp.OFPMF_BURST
            bands = [parser.OFPMeterBandDrop(rate=rate, burst_size=max(1, rate // 10))]
            dp.send_msg(parser.OFPMeterMod(dp, command=ofp.OFPMC_ADD, flags=flags, meter_id=meter_id, bands=bands))
            inst = [parser.OFPInstructionMeter(meter_id), parser.OFPInstructionGotoTable(FORWARD_TABLE)]
        else:
            inst = []  # no instructions == drop
        self._flow_mod(
            dp, IDS_TABLE, int(cmd.get("priority", IDS_PRIORITY)), match, inst,
            idle=int(cmd.get("idle_timeout", 60)), hard=int(cmd.get("hard_timeout", 300)),
            cookie=cookie, flags=ofp.OFPFF_SEND_FLOW_REM,
        )

    def _apply_delete(self, dp, cmd: dict) -> None:
        parser, ofp = dp.ofproto_parser, dp.ofproto
        cookie = IDS_COOKIE_TAG | int(cmd["cookie"])
        dp.send_msg(
            parser.OFPFlowMod(
                datapath=dp, table_id=IDS_TABLE, command=ofp.OFPFC_DELETE, cookie=cookie,
                cookie_mask=0xFFFFFFFFFFFFFFFF, out_port=ofp.OFPP_ANY, out_group=ofp.OFPG_ANY,
                match=parser.OFPMatch(),
            )
        )
        meter_id = int(cmd["cookie"]) & 0xFFFF or 1
        dp.send_msg(parser.OFPMeterMod(dp, command=ofp.OFPMC_DELETE, flags=0, meter_id=meter_id, bands=[]))
