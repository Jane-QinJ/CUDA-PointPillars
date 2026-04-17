#!/bin/bash
set -euo pipefail

POINTS_TOPIC="${1:-/kitti/velo/pointcloud}"
MARKER_TOPIC="${2:-/pointpillar/markers}"
LATENCY_TOPIC="${3:-/pointpillar/inference_latency_ms}"
OUT_DIR="${4:-/tmp/pointpillar_benchmark}"
INTERVAL_MS="${INTERVAL_MS:-500}"
DURATION_SEC="${DURATION_SEC:-20}"

mkdir -p "${OUT_DIR}"

source /opt/ros/noetic/setup.bash
source "${HOME}/catkin_ws/devel/setup.bash"

echo "Benchmark output dir: ${OUT_DIR}"
echo "Points topic: ${POINTS_TOPIC}"
echo "Marker topic: ${MARKER_TOPIC}"
echo "Latency topic: ${LATENCY_TOPIC}"
echo "Duration: ${DURATION_SEC}s"

tegrastats --interval "${INTERVAL_MS}" --logfile "${OUT_DIR}/tegrastats.log" &
TEGRA_PID=$!

timeout "${DURATION_SEC}" rostopic hz "${POINTS_TOPIC}" > "${OUT_DIR}/points_hz.log" 2>&1 &
POINTS_PID=$!
timeout "${DURATION_SEC}" rostopic hz "${MARKER_TOPIC}" > "${OUT_DIR}/markers_hz.log" 2>&1 &
MARKERS_PID=$!
timeout "${DURATION_SEC}" rostopic hz "${LATENCY_TOPIC}" > "${OUT_DIR}/latency_hz.log" 2>&1 &
LAT_PID=$!
timeout "${DURATION_SEC}" rostopic echo -p "${LATENCY_TOPIC}" > "${OUT_DIR}/latency.csv" 2>/dev/null &
LAT_CSV_PID=$!

sleep "${DURATION_SEC}"

kill "${TEGRA_PID}" 2>/dev/null || true
wait "${TEGRA_PID}" 2>/dev/null || true
wait "${POINTS_PID}" 2>/dev/null || true
wait "${MARKERS_PID}" 2>/dev/null || true
wait "${LAT_PID}" 2>/dev/null || true
wait "${LAT_CSV_PID}" 2>/dev/null || true

echo "Benchmark complete."
echo "Saved files:"
echo "  ${OUT_DIR}/tegrastats.log"
echo "  ${OUT_DIR}/points_hz.log"
echo "  ${OUT_DIR}/markers_hz.log"
echo "  ${OUT_DIR}/latency_hz.log"
echo "  ${OUT_DIR}/latency.csv"
