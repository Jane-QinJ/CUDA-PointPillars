# PointPillars Inference with TensorRT

This repository contains sources and model for [pointpillars](https://arxiv.org/abs/1812.05784) inference using TensorRT.

Overall inference has below phases:

- Voxelize points cloud into 10-channel features
- Run TensorRT engine to get detection feature
- Parse detection feature and apply NMS

## Prerequisites

### Prepare Model && Data

We provide a [Dockerfile](docker/Dockerfile) to ease environment setup. Please execute the following command to build the docker image after nvidia-docker installation:
```
cd docker && docker build . -t pointpillar
```
We can then run the docker with the following command: 
```
nvidia-docker run --rm -ti -v /home/$USER/:/home/$USER/ --net=host --rm pointpillar:latest
```
For model exporting, please run the following command to clone pcdet repo and install custom CUDA extensions:
```
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet && git checkout 846cf3e && python3 setup.py develop
```
Download [PTM](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view) to ckpts/, then use below command to export ONNX model:
```
python3 tool/export_onnx.py --ckpt ckpts/pointpillar_7728.pth --out_dir model
```
Use below command to evaluate on kitti dataset, follow [Evaluation on Kitti](tool/eval/README.md) to get more detail for dataset preparation.
```
sh tool/evaluate_kitti_val.sh
```

### Setup Runtime Environment

- Nvidia Jetson Orin + CUDA 11.4 + cuDNN 8.9.0 + TensorRT 8.6.11

## Compile && Run

```shell
sudo apt-get install git-lfs && git lfs install
git clone https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars.git
cd CUDA-PointPillars && . tool/environment.sh
mkdir build && cd build
cmake .. && make -j$(nproc)
cd ../ && sh tool/build_trt_engine.sh
cd build && ./pointpillar ../data/ ../data/ --timer
```

## ROS Launch Quick Start

The ROS 1 wrapper can be built directly from this repository. The commands
below were smoke-tested with `pointpillar_custom_pedestrian.launch` on
ROS Noetic with RViz disabled (`open_rviz:=false`).

### 1. Build the ROS node

```shell
source /opt/ros/noetic/setup.bash
cd CUDA-PointPillars
. tool/environment.sh
mkdir -p build
cd build
cmake ..
make -j$(nproc) pointpillar_ros_node
```

### 2. Launch the node

For the default KITTI-style 3-class launch file:

```shell
source /opt/ros/noetic/setup.bash
cd CUDA-PointPillars
export ROS_PACKAGE_PATH=$PWD:$ROS_PACKAGE_PATH
source build/devel/setup.bash
. tool/environment.sh
roslaunch pointpillar pointpillar_ros.launch open_rviz:=false
```

For the included custom single-class Pedestrian example:

```shell
source /opt/ros/noetic/setup.bash
cd CUDA-PointPillars
export ROS_PACKAGE_PATH=$PWD:$ROS_PACKAGE_PATH
source build/devel/setup.bash
. tool/environment.sh
roslaunch pointpillar pointpillar_custom_pedestrian.launch open_rviz:=false
```

Set `open_rviz:=true` if you want RViz to open automatically.

### 3. Publish sample point clouds

The ROS node subscribes to `sensor_msgs/PointCloud2`, so you can drive it with
the included KITTI publisher script:

```shell
source /opt/ros/noetic/setup.bash
cd CUDA-PointPillars
source build/devel/setup.bash
python3 tool/publish_kitti_pointcloud.py _rate:=10.0
```

If you use `pointpillar_ros.launch`, keep the default topic
`/kitti/velo/pointcloud`. If you use
`pointpillar_custom_pedestrian.launch`, either publish to `/points_raw` or
override the launch arg:

```shell
roslaunch pointpillar pointpillar_custom_pedestrian.launch \
  open_rviz:=false \
  points_topic:=/kitti/velo/pointcloud
```

### 4. Check outputs

- Markers: `/pointpillar/markers`
- Structured detections: `/pointpillar/detections`
- Inference latency: `/pointpillar/inference_latency_ms`

## FP16 Performance && Metrics

Average perf in FP16 on the training set(7481 instances) of KITTI dataset.

```
| Function(unit:ms) | Orin   |
| ----------------- | ------ |
| Voxelization      | 0.18   |
| Backbone & Head   | 4.87   |
| Decoder & NMS     | 1.79   |
| Overall           | 6.84   |
```

3D moderate metrics on the validation set(3769 instances) of KITTI dataset.

```
|                   | Car@R11 | Pedestrian@R11 | Cyclist@R11  | 
| ----------------- | --------| -------------- | ------------ |
| CUDA-PointPillars | 77.00   | 52.50          | 62.26        |
| OpenPCDet         | 77.28   | 52.29          | 62.68        |
```

## ROS Real-Time Benchmark on Jetson Orin

Measured on a Jetson Orin using the ROS 1 wrapper in this repository, with
KITTI-format point clouds published as `sensor_msgs/PointCloud2` to the
`/kitti/velo/pointcloud` topic and PointPillars detections published on
`/pointpillar/markers`.

### Summary

| Test | Result |
| ---- | ------ |
| Standalone warmed-up throughput | about 48 FPS |
| ROS callback latency | about 19.5 to 23.0 ms |
| Sustained real-time ROS input rate | about 49 Hz |
| Saturation starts | about 50 Hz |

### Interpretation

- The ROS node keeps up in real time through about 49 Hz on Jetson Orin.
- At 50 Hz input and above, output rate begins to lag behind input rate.
- Near the real-time limit, `tegrastats` showed `GR3D_FREQ` around 93% to 99%.

### Reproduce

Build and launch the ROS node:

```shell
source /opt/ros/noetic/setup.bash
cd ~/catkin_ws
catkin_make --pkg pointpillar
source devel/setup.bash
roslaunch pointpillar pointpillar_ros.launch fixed_frame:=velo_link
```

Run a live ROS benchmark with `tegrastats` and `rostopic hz`:

```shell
bash tool/benchmark_ros_runtime.sh
```

Sweep controlled input rates using the included KITTI `.bin` files:

```shell
bash tool/sweep_realtime_rate.sh
```

The sweep uses [tool/publish_kitti_pointcloud.py](tool/publish_kitti_pointcloud.py)
to publish KITTI point clouds at controlled rates and records the results under
`/tmp/pointpillar_rate_sweep*`.

## Using a Custom-Trained Model

This repo's ONNX export/graph-surgery (`tool/export_onnx.py`, `tool/modify_onnx.py`) and CUDA/TensorRT runtime are written for OpenPCDet's PointPillars architecture, but the point cloud range, voxel size, class count and anchor layout are all derived from your training cfg rather than hardcoded to KITTI. If you trained your own PointPillars checkpoint in OpenPCDet (same model, different data/classes/weights), do the following.

**The ONNX export step needs `pcdet` installed and only needs plain PyTorch (no TensorRT/Jetson hardware), so it's normally easiest to run on the machine you trained on** (copy `tool/export_onnx.py` and `tool/modify_onnx.py` into that OpenPCDet checkout's `tools/` directory, or point `--cfg_file`/`--ckpt` at this repo if `pcdet` is installed here):

1. **Export ONNX** from your checkpoint and its training cfg:
   ```shell
   python3 tool/export_onnx.py --cfg_file <your_cfg>.yaml --ckpt <your_checkpoint>.pth --data_path <dir_with_a_few_.bin_files> --out_dir model
   ```
   `<your_cfg>.yaml` is the exact cfg used to train the checkpoint. `export_onnx.py` reads `CLASS_NAMES`, `DATA_CONFIG.POINT_CLOUD_RANGE`, the `transform_points_to_voxels` processor's `VOXEL_SIZE`, and `MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG` from it to size the exported graph correctly — no manual shape editing needed. `--data_path` just needs a directory with a few `.bin` point cloud files (any point cloud works; it's only used to construct the dataset object, not for the export itself).

   Copy the resulting `model/pointpillar.onnx` onto the Jetson (the TensorRT engine build below must run on the Jetson itself — engines aren't portable across GPUs).

2. **Build the TensorRT engine** (on the Jetson):
   ```shell
   sh tool/build_trt_engine.sh
   ```

3. **Point the ROS node at your model's config.** `src/ros/pointpillar_ros_node.cpp` reads `VoxelizationParameter` (point cloud range, voxel size) and `PostProcessParameter` (anchors, anchor bottom heights, direction offset) from ROS params, falling back to the KITTI 3-class defaults if unset — no recompiling needed for a different range/voxel size/class count, as long as it's **at most 6 anchors and 3 classes** (fixed-size arrays in `PostProcessParameter`; a model with more than that needs those array sizes bumped in `src/pointpillar/lidar-postprocess.hpp`). See `launch/pointpillar_custom_pedestrian.launch` for a fully worked example (single-class `Pedestrian` model, non-KITTI range/voxel size), copied straight from that model's training cfg.

### Detection topic (locations + distances)

The ROS node publishes structured detections on `~detections_topic` (default `/pointpillar/detections`, type `pointpillar/Detection3DArray`), in addition to the `visualization_msgs/MarkerArray` used for RViz. Each `pointpillar/Detection3D` in the array carries:

- `label_id` / `label_name` — class index and name, resolved from the `~class_names` ROS param
- `score` — detection confidence
- `x`, `y`, `z` — box center in `header.frame_id`
- `length`, `width`, `height`, `yaw` — box size and heading
- `distance` — Euclidean range from the sensor origin to the box center (`sqrt(x^2+y^2+z^2)`)

Relevant launch params (see `launch/pointpillar_ros.launch` and `launch/pointpillar_custom_pedestrian.launch`):

| Param | Default | Purpose |
| ----- | ------- | ------- |
| `detections_topic` | `/pointpillar/detections` | Topic name for `Detection3DArray` |
| `class_names` | `["Car", "Pedestrian", "Cyclist"]` | Must match your cfg's `CLASS_NAMES` order |
| `point_cloud_range` | KITTI's `[0, -39.68, -3, 69.12, 39.68, 1]` | `[xmin, ymin, zmin, xmax, ymax, zmax]`, from your cfg's `DATA_CONFIG.POINT_CLOUD_RANGE` |
| `voxel_size` | KITTI's `[0.16, 0.16, 4.0]` | From your cfg's `transform_points_to_voxels.VOXEL_SIZE` |
| `anchors` | KITTI's 6 anchors (`[w, l, h, rot]` each) | Flattened list, one `[w, l, h, rot]` per anchor in your cfg's `ANCHOR_GENERATOR_CONFIG` |
| `anchor_bottom_heights` | KITTI's `[-1.78, -0.6, -0.6]` | One value per class, same order as `class_names` |
| `dir_offset` | `0.78539` | From your cfg's `DENSE_HEAD.DIR_OFFSET` |
| `score_thresh` | `-1.0` (use compiled default `0.1`) | Overrides `PostProcessParameter::score_thresh` |
| `nms_thresh` | `-1.0` (use compiled default `0.01`) | Overrides `PostProcessParameter::nms_thresh` |

## Note

- Voxelization has random output since GPU processes all points simultaneously while points selection for a voxel is random.

## References

- [Detecting Objects in Point Clouds with NVIDIA CUDA-Pointpillars](https://developer.nvidia.com/blog/detecting-objects-in-point-clouds-with-cuda-pointpillars/)
- [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)
