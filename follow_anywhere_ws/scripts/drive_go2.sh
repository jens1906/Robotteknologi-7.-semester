#!/usr/bin/env bash
# Drive the physical Go2 with the keyboard (WASD + arrows).
#
# Usage: scripts/drive_go2.sh [NIC]
#   NIC  Ethernet interface connected to the Go2. Defaults to $GO2_NIC, then the
#        interface holding a 192.168.123.x address, then the first enx*/en* link.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# ROS 2 Lyrical: $ROS_SETUP if set, else the binary install, else a source build in ~/ros2_lyrical.
if [ -z "${ROS_SETUP:-}" ]; then
    ROS_SETUP="/opt/ros/lyrical/setup.bash"
    [ -f "$ROS_SETUP" ] || ROS_SETUP="$HOME/ros2_lyrical/install/setup.bash"
fi
GO2_IP="${GO2_IP:-192.168.123.161}"

detect_nic() {
    local nic
    nic="$(ip -br -4 addr | awk '$3 ~ /^192\.168\.123\./ {print $1; exit}')"
    [ -n "$nic" ] || nic="$(ip -br link | awk '$1 ~ /^(enx|en|eth)/ {print $1; exit}')"
    echo "$nic"
}

NIC="${1:-${GO2_NIC:-$(detect_nic)}}"
if [ -z "$NIC" ]; then
    echo "No Ethernet interface found. Pass it as an argument: $0 <nic>" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$ROS_SETUP"

if [ ! -f "$WS/install/setup.bash" ]; then
    echo "Workspace not built yet, running colcon build..."
    (cd "$WS" && colcon build)
fi
# shellcheck disable=SC1091
source "$WS/install/setup.bash"

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"$NIC\" priority=\"default\" multicast=\"default\"/></Interfaces></General></Domain></CycloneDDS>"

echo "Interface: $NIC"
if ! ping -c 1 -W 1 "$GO2_IP" >/dev/null 2>&1; then
    echo "Warning: Go2 ($GO2_IP) does not answer ping on $NIC. Check cable and the laptop's 192.168.123.99 address." >&2
fi

exec ros2 run manual_controller keyboard_teleop
