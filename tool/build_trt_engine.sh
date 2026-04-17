#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

TRTEXEC=${TRTEXEC:-}
if [ -z "$TRTEXEC" ] && [ -n "${TensorRT_Bin:-}" ] && [ -x "${TensorRT_Bin}/trtexec" ]; then
    TRTEXEC="${TensorRT_Bin}/trtexec"
fi
if [ -z "$TRTEXEC" ] && command -v trtexec >/dev/null 2>&1; then
    TRTEXEC=$(command -v trtexec)
fi

if [ -z "$TRTEXEC" ] || [ ! -x "$TRTEXEC" ]; then
    echo "trtexec not found."
    echo "Install TensorRT's trtexec binary or export TRTEXEC=/abs/path/to/trtexec before running this script."
    exit 1
fi

ONNX_PATH=${1:-./model/custom/pointpillar.onnx}
ENGINE_PATH=${2:-./model/runtime/pointpillar.plan}
PLUGIN_PATH=${3:-build/libpointpillar_core.so}
LOG_PATH=${4:-./model/runtime/pointpillar.trtexec.log}

mkdir -p "$(dirname "$ENGINE_PATH")"
mkdir -p "$(dirname "$LOG_PATH")"

"$TRTEXEC" \
    --onnx="$ONNX_PATH" \
    --fp16 \
    --plugins="$PLUGIN_PATH" \
    --saveEngine="$ENGINE_PATH" \
    --inputIOFormats=fp16:chw,int32:chw,int32:chw \
    --verbose \
    --dumpLayerInfo \
    --dumpProfile \
    --separateProfileRun \
    --profilingVerbosity=detailed \
    > "$LOG_PATH" 2>&1

echo "TensorRT engine saved to: $ENGINE_PATH"
echo "trtexec log saved to: $LOG_PATH"
