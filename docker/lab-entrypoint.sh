#!/bin/bash
# Start Open vSwitch inside the lab container, then run the given command.
set -e
mkdir -p /var/run/openvswitch /var/log/openvswitch
if [ ! -f /etc/openvswitch/conf.db ]; then
  ovsdb-tool create /etc/openvswitch/conf.db /usr/share/openvswitch/vswitch.ovsschema
fi
ovsdb-server --remote=punix:/var/run/openvswitch/db.sock \
  --remote=db:Open_vSwitch,Open_vSwitch,manager_options \
  --pidfile --detach --log-file
ovs-vsctl --no-wait init
ovs-vswitchd --pidfile --detach --log-file
echo "[lab] Open vSwitch $(ovs-vsctl --version | head -1 | awk '{print $NF}') running"
if lsmod 2>/dev/null | grep -q '^openvswitch'; then
  echo "[lab] kernel datapath available"
else
  echo "[lab] no openvswitch kernel module: use the userspace datapath (--datapath user)"
fi
exec "$@"
