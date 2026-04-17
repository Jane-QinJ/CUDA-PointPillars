#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

find_first_existing_dir() {
    for path in "$@"; do
        if [ -n "$path" ] && [ -d "$path" ]; then
            echo "$path"
            return 0
        fi
    done
    return 1
}

find_first_existing_file() {
    for path in "$@"; do
        if [ -n "$path" ] && [ -f "$path" ]; then
            echo "$path"
            return 0
        fi
    done
    return 1
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
LOCAL_TRT_ROOT="${PROJECT_ROOT}/third_party/TensorRT-8.6.1.6"

# export CUDA_VISIBLE_DEVICES=2

export DEBUG_PRECISION=${DEBUG_PRECISION:-fp16}
export DEBUG_DATA=${DEBUG_DATA:-data}
export USE_Python=${USE_Python:-OFF}

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDA_Bin=$(find_first_existing_dir \
    "${CUDA_Bin}" \
    "${CUDA_HOME}/bin" \
    /usr/local/cuda/bin \
    /usr/local/cuda-12.6/bin \
    /usr/local/cuda-11.1/bin)
export CUDA_Inc=$(find_first_existing_dir \
    "${CUDA_Inc}" \
    "${CUDA_HOME}/include" \
    /usr/local/cuda/include \
    /usr/local/cuda-12.6/include \
    /usr/local/cuda-11.1/include)
export CUDA_Lib=$(find_first_existing_dir \
    "${CUDA_Lib}" \
    "${CUDA_HOME}/targets/x86_64-linux/lib" \
    "${CUDA_HOME}/lib64" \
    /usr/local/cuda/targets/x86_64-linux/lib \
    /usr/local/cuda/lib64 \
    /usr/local/cuda-12.6/targets/x86_64-linux/lib \
    /usr/local/cuda-11.1/targets/x86_64-linux/lib)

export TensorRT_Inc=$(find_first_existing_dir \
    "${TensorRT_Inc}" \
    "${LOCAL_TRT_ROOT}/include" \
    /usr/local/cuda-11.1/targets/x86_64-linux/include \
    /usr/include/x86_64-linux-gnu \
    /usr/include)
export TensorRT_Lib=$(find_first_existing_dir \
    "${TensorRT_Lib}" \
    "${LOCAL_TRT_ROOT}/lib" \
    /usr/local/cuda-11.1/targets/x86_64-linux/lib \
    /usr/lib/x86_64-linux-gnu \
    /usr/lib)
export TensorRT_Bin=$(find_first_existing_dir \
    "${TensorRT_Bin}" \
    "${LOCAL_TRT_ROOT}/bin" \
    /usr/src/tensorrt/bin \
    /usr/local/bin \
    /usr/bin)

export CUDNN_Lib=$(find_first_existing_dir \
    "${CUDNN_Lib}" \
    "${CUDA_HOME}/lib64" \
    /usr/local/cuda/lib64 \
    /usr/local/cuda-12.6/lib64 \
    /usr/local/cuda-11.1/lib64)
export CUDNN_INCLUDE_DIR=$(find_first_existing_dir \
    "${CUDNN_INCLUDE_DIR}" \
    "${CUDA_HOME}/include" \
    /usr/local/cuda/include \
    /usr/local/cuda-12.6/include \
    /usr/local/cuda-11.1/include)

export ConfigurationStatus=Failed

TRTEXEC_PATH=$(find_first_existing_file \
    "${TensorRT_Bin}/trtexec" \
    "${LOCAL_TRT_ROOT}/bin/trtexec" \
    /usr/src/tensorrt/bin/trtexec \
    /usr/local/bin/trtexec \
    /usr/bin/trtexec)
NVCC_PATH=$(find_first_existing_file \
    "${CUDA_Bin}/nvcc" \
    /usr/local/cuda/bin/nvcc \
    /usr/local/cuda-12.6/bin/nvcc \
    /usr/local/cuda-11.1/bin/nvcc)

if [ -z "${NVCC_PATH}" ]; then
    echo "Can not find nvcc. Please set CUDA_Bin/CUDA_HOME correctly."
    return
fi

if [ -z "${TRTEXEC_PATH}" ]; then
    echo "Can not find trtexec. TensorRT runtime headers/libs exist, but the engine builder binary is not installed."
    echo "You can still compile the project, but ONNX -> .plan conversion will fail until trtexec is available."
else
    export TensorRT_Bin=$(dirname "${TRTEXEC_PATH}")
fi

export CUDA_Bin=$(dirname "${NVCC_PATH}")
export CUDA_HOME=$(cd "${CUDA_Bin}/.." && pwd)

echo "=========================================================="
echo "||  PRECISION: $DEBUG_PRECISION"
echo "||  DATA: $DEBUG_DATA"
echo "||  USEPython: $USE_Python"
echo "||"
echo "||  TensorRT Include: $TensorRT_Inc"
echo "||  TensorRT Lib: $TensorRT_Lib"
echo "||  TensorRT Bin: ${TensorRT_Bin:-<missing>}"
echo "||  CUDA HOME: $CUDA_HOME"
echo "||  CUDA Bin: $CUDA_Bin"
echo "||  CUDA Include: $CUDA_Inc"
echo "||  CUDA Lib: $CUDA_Lib"
echo "||  CUDNN Lib: $CUDNN_Lib"
echo "=========================================================="

BuildDirectory=$(pwd)/build

if [ "$USE_Python" = "ON" ]; then
    export Python_Inc=$(python3 -c "import sysconfig;print(sysconfig.get_path('include'))")
    export Python_Lib=$(python3 -c "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))")
    export Python_Soname=$(python3 -c "import sysconfig;import re;print(re.sub('.a', '.so', sysconfig.get_config_var('LIBRARY')))")
    echo "Find Python_Inc: $Python_Inc"
    echo "Find Python_Lib: $Python_Lib"
    echo "Find Python_Soname: $Python_Soname"
fi

export PATH=${TensorRT_Bin:+$TensorRT_Bin:}$CUDA_Bin:$PATH
export LD_LIBRARY_PATH=$TensorRT_Lib:$CUDA_Lib:$CUDNN_Lib:$BuildDirectory:$LD_LIBRARY_PATH
export PYTHONPATH=$BuildDirectory:$PYTHONPATH

if [ -f "tool/cudasm.sh" ]; then
    echo "Try to get the current device SM"
    . "tool/cudasm.sh"
    if [ -n "$cudasm" ]; then
        echo "Current CUDA SM: $cudasm"
        export CUDASM=$cudasm
    else
        echo "Current CUDA SM: unavailable, fallback value will be used by CMake"
    fi
fi

export ConfigurationStatus=Success
echo "Configuration done!"
