# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 2.8.7)
project(pointpillar)
set(CMAKE_BUILD_PYTHON $ENV{USE_Python})
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

find_package(CUDA REQUIRED)

find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h NvInferPlugin.h
    HINTS
        $ENV{TensorRT_Inc}
        /usr/local/cuda-11.1/targets/x86_64-linux/include
        /usr/include/x86_64-linux-gnu
        /usr/include
)

find_library(TENSORRT_LIB
    NAMES nvinfer libnvinfer.so
    HINTS
        $ENV{TensorRT_Lib}
        /usr/local/cuda-11.1/targets/x86_64-linux/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib
)

if(NOT TENSORRT_INCLUDE_DIR)
    message(FATAL_ERROR "TensorRT headers not found. Please set TensorRT_Inc or install TensorRT headers.")
endif()

if(NOT TENSORRT_LIB)
    message(FATAL_ERROR "TensorRT library not found. Please set TensorRT_Lib or install TensorRT runtime.")
endif()

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT lib: ${TENSORRT_LIB}")

set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3")
set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++14 -O0 -g")

set(CUDA_NVCC_FLAGS_RELEASE "-Werror=all-warnings -Xcompiler -std=c++14,-Wextra,-Wall,-Wno-deprecated-declarations,-O3")
set(CUDA_NVCC_FLAGS_DEBUG   "-std=c++14 -O0 -g")

set(CUDA_ARCH "$ENV{CUDASM}")
if(NOT CUDA_ARCH)
    set(CUDA_ARCH "86")
endif()
message(STATUS "Using CUDA SM architecture: ${CUDA_ARCH}")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}
)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    src
	src/pointpillar
)

link_directories(
    build
)

file(GLOB_RECURSE CORE_FILES 
    src/pointpillar/*.cu 
    src/pointpillar/*.cpp
    src/common/tensor.cu
    src/common/tensorrt.cpp
)

####################### libpointpillar_core.so ##########################
cuda_add_library(pointpillar_core SHARED 
  	${CORE_FILES}
)

target_link_libraries(pointpillar_core
	${TENSORRT_LIB}
    ${CUDA_CUDART_LIBRARY}
)
#######################################################################


####################### pointpillar program #############################
cuda_add_executable(pointpillar 
	src/main.cpp
)

target_link_libraries(pointpillar
	pointpillar_core
)
#######################################################################
