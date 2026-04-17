cmake_minimum_required(VERSION 3.0.2)
project(pointpillar_ros)

find_package(CUDA REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  visualization_msgs
)

set(REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

catkin_package()

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${REPO_ROOT}/src
  ${REPO_ROOT}/src/pointpillar
  ${REPO_ROOT}/src/common
)

link_directories(
  ${REPO_ROOT}/build
)

add_executable(pointpillar_ros_node src/pointpillar_ros_node.cpp)

target_link_libraries(pointpillar_ros_node
  pointpillar_core
  ${catkin_LIBRARIES}
  ${CUDA_CUDART_LIBRARY}
)
