#!/usr/bin/env python3
"""Start the IDS controller app.

os-ken 4.x ships no ``osken-manager`` command, so this launcher does what
``ryu-manager`` does: patch for green threads, parse the controller config and
run the app together with the OpenFlow handler. Falls back to Ryu if os-ken is
not installed.

    python3 controller/run_controller.py --ids-url http://127.0.0.1:3000/api/flows
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ofp-port", type=int, default=6653, help="OpenFlow listen port")
    parser.add_argument("--ids-url", default=os.environ.get("IDS_API_URL", "http://127.0.0.1:3000/api/flows"))
    parser.add_argument("--api-key", default=os.environ.get("IDS_API_KEY"))
    parser.add_argument("--poll-interval", type=float, default=float(os.environ.get("IDS_POLL_INTERVAL", "2")))
    parser.add_argument("--match-mode", choices=["5tuple", "host_pair"], default=os.environ.get("IDS_MATCH_MODE", "5tuple"))
    parser.add_argument("--rate-limit-as-drop", action="store_true", help="for switches without meter support")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.environ["IDS_API_URL"] = args.ids_url
    os.environ["IDS_POLL_INTERVAL"] = str(args.poll_interval)
    os.environ["IDS_MATCH_MODE"] = args.match_mode
    if args.api_key:
        os.environ["IDS_API_KEY"] = args.api_key
    if args.rate_limit_as_drop:
        os.environ["IDS_RATE_LIMIT_AS_DROP"] = "1"
    sys.path.insert(0, ROOT)

    try:
        from os_ken.lib import hub

        hub.patch(thread=False)
        from os_ken import cfg
        from os_ken.base.app_manager import AppManager
        import os_ken.controller.controller  # noqa: F401  (registers OpenFlow CLI options)

        project, handler = "os_ken", "os_ken.controller.ofp_handler"
    except ImportError:
        from ryu.lib import hub

        hub.patch(thread=False)
        from ryu import cfg
        from ryu.base.app_manager import AppManager
        import ryu.controller.controller  # noqa: F401

        project, handler = "ryu", "ryu.controller.ofp_handler"

    import logging

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg.CONF(args=[f"--ofp-tcp-listen-port={args.ofp_port}"], project=project)
    print(f"[{project}] OpenFlow on :{args.ofp_port} -> IDS {args.ids_url} every {args.poll_interval}s")
    AppManager.run_apps([handler, "controller.ids_controller"])


if __name__ == "__main__":
    main()
