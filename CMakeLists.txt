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

set(CUDA_INC_DIR $ENV{CUDA_Inc})
set(CUDA_LIB_DIR $ENV{CUDA_Lib})
set(TENSORRT_INC_DIR $ENV{TensorRT_Inc})
set(TENSORRT_LIB_DIR $ENV{TensorRT_Lib})
set(CUDA_SM $ENV{CUDASM})

if(NOT CUDA_INC_DIR)
    set(CUDA_INC_DIR /usr/local/cuda-11.4/targets/aarch64-linux/include)
endif()

if(NOT CUDA_LIB_DIR)
    set(CUDA_LIB_DIR /usr/local/cuda-11.4/targets/aarch64-linux/lib)
endif()

if(NOT TENSORRT_INC_DIR)
    set(TENSORRT_INC_DIR /usr/include/aarch64-linux-gnu)
endif()

if(NOT TENSORRT_LIB_DIR)
    set(TENSORRT_LIB_DIR /lib/aarch64-linux-gnu)
endif()

if(NOT CUDA_SM)
    if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(CUDA_SM 87)
        message(STATUS "CUDASM not set, defaulting to SM ${CUDA_SM} for Jetson Orin")
    else()
        message(FATAL_ERROR "CUDASM is not set. Export CUDASM or source tool/environment.sh before building.")
    endif()
endif()

find_package(CUDA REQUIRED)
find_package(catkin QUIET COMPONENTS roscpp roslib sensor_msgs visualization_msgs)

set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3")
set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++14 -O0 -g")

set(CUDA_NVCC_FLAGS_RELEASE "-Werror=all-warnings -Xcompiler -std=c++14,-Wextra,-Wall,-Wno-deprecated-declarations,-O3")
set(CUDA_NVCC_FLAGS_DEBUG   "-std=c++14 -O0 -g")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
    -gencode arch=compute_${CUDA_SM},code=compute_${CUDA_SM}
)

include_directories(
    ${CUDA_INC_DIR}
    ${TENSORRT_INC_DIR}
    src
	src/pointpillar
)

if(catkin_FOUND)
    catkin_package(
        INCLUDE_DIRS src src/pointpillar
        LIBRARIES pointpillar_core
        CATKIN_DEPENDS roscpp roslib sensor_msgs visualization_msgs
    )

    include_directories(
        ${catkin_INCLUDE_DIRS}
    )
endif()

link_directories(
    ${CUDA_LIB_DIR}
    ${TENSORRT_LIB_DIR}
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
	libnvinfer.so
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

if(catkin_FOUND)
    ################### pointpillar_ros_node program ###################
    add_executable(pointpillar_ros_node
        src/ros/pointpillar_ros_node.cpp
    )

    target_link_libraries(pointpillar_ros_node
        pointpillar_core
        ${catkin_LIBRARIES}
    )

    install(TARGETS pointpillar_core pointpillar_ros_node
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )

    install(DIRECTORY launch/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
    )
    ###################################################################
else()
    message(STATUS "catkin not found, skipping ROS node build")
endif()
