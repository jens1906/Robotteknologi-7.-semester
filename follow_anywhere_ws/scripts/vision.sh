#!/usr/bin/env bash
# Run the vision pipeline (YOLO + tracker + leader following) on the RealSense or on a dataset.
#
# Usage: scripts/vision.sh [MODE] [PATH] [launch args...]
#   (no mode)        live RealSense camera
#   data [PATH]      play a video file or image folder as the camera, print ID-stability numbers
#                    at the end. PATH defaults to data/mot17_02
#   compare [PATH]   data mode twice: ByteTrack (before) and botsort_tuned.yaml (after), then both
#                    summaries together
#   score [opts]     run the detector over the MOT17 sequences in data/, 3 times per tracker, and print
#                    a table of ground-truth scores (IDF1, ID switches, IDs per person, fps).
#                    Options go to scripts/score_tracking.py, e.g. --seqs 02 13 --runs 1
#                    --variant "keep10=occlusion_keep_s:=10"
#   build            only (re)build the vision packages
# Extra arguments go to `ros2 launch vision vision.launch.py`, e.g. tracker_cfg:=bytetrack.yaml
# loop:=true video_width:=960.
#
# The vision packages are rebuilt automatically when a source file is newer than the install, or when
# the install was built without the .venv python (the installed scripts run with the python that
# built them, and onnxruntime lives in the venv).
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_SETUP="/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
DEFAULT_DATA="data/mot17_02"

mode="camera"
case "${1:-}" in
    data|compare|build|camera|score) mode="$1"; shift ;;
    -h|--help) sed -n '2,/^set -/p' "${BASH_SOURCE[0]}" | sed '$d;s/^# \{0,1\}//'; exit 0 ;;
esac

data_path="$DEFAULT_DATA"
if [ "$mode" = data ] || [ "$mode" = compare ]; then
    if [ $# -gt 0 ] && [[ "$1" != *:=* ]]; then data_path="$1"; shift; fi
fi

set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
set -u
cd "$WS"

build() {
    local py="python3"
    [ -x "$WS/.venv/bin/python3" ] && py="$WS/.venv/bin/python3"
    echo "Building vision packages with $py ..."
    "$py" -m colcon build --packages-select vision_interfaces vision
}

stale() {
    [ ! -f install/vision/lib/vision/detector ] && return 0
    # a plain `colcon build` rewrites the scripts to run with /usr/bin/python3, which has no onnxruntime
    if [ -x "$WS/.venv/bin/python3" ] \
        && [ "$(head -n 1 install/vision/lib/vision/detector)" != "#!$WS/.venv/bin/python3" ]; then
        return 0
    fi
    [ -n "$(find src/vision src/vision_interfaces -type f ! -path '*/__pycache__/*' \
        -newer install/vision/lib/vision/detector -print -quit)" ]
}

if [ "$mode" = build ] || stale; then build; fi
[ "$mode" = build ] && exit 0

set +u
# shellcheck disable=SC1091
source "$WS/install/setup.bash"
set -u
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"      # OpenCV's Qt has no wayland plugin

launch=(ros2 launch vision vision.launch.py)

case "$mode" in
    camera)
        exec "${launch[@]}" "$@"
        ;;
    score)
        exec python3 "$WS/scripts/score_tracking.py" "$@"
        ;;
    data)
        [ -e "$data_path" ] || { echo "Not found: $data_path (video file or image folder)" >&2; exit 1; }
        exec "${launch[@]}" use_dataset:=true monitor:=true "video_path:=$data_path" "$@"
        ;;
    compare)
        [ -e "$data_path" ] || { echo "Not found: $data_path (video file or image folder)" >&2; exit 1; }
        tmp="$(mktemp -d)"
        for name in "before (ByteTrack)" "after (botsort_tuned.yaml)"; do
            extra=()
            [[ "$name" == before* ]] && extra=(tracker_cfg:=bytetrack.yaml)
            echo "=== $name on $data_path ==="
            "${launch[@]}" use_dataset:=true monitor:=true "video_path:=$data_path" \
                "${extra[@]}" "$@" > "$tmp/${name%% *}.log" 2>&1 || true
        done
        for name in before after; do
            echo
            echo "##### $name"
            sed -n '/=== track_monitor summary/,/MOT tracks/p' "$tmp/$name.log" \
                | sed 's/^\[track_monitor-2\] //' | grep -v '^$' || echo "(no summary, see $tmp/$name.log)"
        done
        ;;
esac
