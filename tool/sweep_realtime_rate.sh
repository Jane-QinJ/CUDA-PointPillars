#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POINTS_TOPIC="${POINTS_TOPIC:-/kitti/velo/pointcloud}"
MARKER_TOPIC="${MARKER_TOPIC:-/pointpillar/markers}"
LATENCY_TOPIC="${LATENCY_TOPIC:-/pointpillar/inference_latency_ms}"
FRAME_ID="${FRAME_ID:-velo_link}"
DATA_PATH="${DATA_PATH:-${REPO_DIR}/data}"
RATES="${RATES:-5 10 15 20 25 30 35 40}"
DURATION_SEC="${DURATION_SEC:-12}"
OUT_ROOT="${OUT_ROOT:-/tmp/pointpillar_rate_sweep}"

source /opt/ros/noetic/setup.bash
source "${HOME}/catkin_ws/devel/setup.bash"

mkdir -p "${OUT_ROOT}"

echo "Rate sweep output: ${OUT_ROOT}"
echo "Rates: ${RATES}"

for rate in ${RATES}; do
  RUN_DIR="${OUT_ROOT}/${rate}hz"
  mkdir -p "${RUN_DIR}"
  echo
  echo "=== Testing ${rate} Hz ==="

  python3 "${REPO_DIR}/tool/publish_kitti_pointcloud.py" \
    --data "${DATA_PATH}" \
    --topic "${POINTS_TOPIC}" \
    --frame-id "${FRAME_ID}" \
    --rate "${rate}" \
    --loop > "${RUN_DIR}/publisher.log" 2>&1 &
  PUB_PID=$!

  sleep 2

  DURATION_SEC="${DURATION_SEC}" \
  "${REPO_DIR}/tool/benchmark_ros_runtime.sh" \
    "${POINTS_TOPIC}" "${MARKER_TOPIC}" "${LATENCY_TOPIC}" "${RUN_DIR}" \
    > "${RUN_DIR}/benchmark.log" 2>&1 || true

  kill "${PUB_PID}" 2>/dev/null || true
  wait "${PUB_PID}" 2>/dev/null || true

  echo "--- input rate ---"
  tail -n 5 "${RUN_DIR}/points_hz.log" || true
  echo "--- output rate ---"
  tail -n 5 "${RUN_DIR}/markers_hz.log" || true
  echo "--- latency rate ---"
  tail -n 5 "${RUN_DIR}/latency_hz.log" || true
done

echo
echo "Sweep complete. Review logs under ${OUT_ROOT}"
