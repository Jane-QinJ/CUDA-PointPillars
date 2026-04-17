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

## Custom Dataset Workflow

This repo now includes a custom PointPillars pipeline for:

- training/export config: [cfgs/custom_models/pointpillar.yaml](/home/firo/workspace/CUDA-PointPillars/cfgs/custom_models/pointpillar.yaml)
- custom dataset config: [cfgs/dataset_configs/custom_dataset.yaml](/home/firo/workspace/CUDA-PointPillars/cfgs/dataset_configs/custom_dataset.yaml)
- ONNX export output: `model/runtime_onnx/pointpillar.onnx`
- TensorRT engine output: `model/runtime/pointpillar.plan`

### 1. Export ONNX

OpenPCDet export was validated inside the Docker image:

```bash
docker build -f docker/Dockerfile . -t cudapointpillar:latest

docker run --gpus all --rm -ti \
  --net=host --ipc=host \
  -v /home/$USER/workspace/CUDA-PointPillars:/workspace/CUDA-PointPillars \
  cudapointpillar:latest
```

Inside the container:

```bash
cd /workspace/CUDA-PointPillars/OpenPCDet
python3 -m pip install -r requirements.txt
python3 setup.py develop

cd /workspace/CUDA-PointPillars
export PYTHONPATH=/workspace/CUDA-PointPillars/OpenPCDet:$PYTHONPATH

python3 tool/export_onnx.py \
  --cfg_file cfgs/custom_models/pointpillar.yaml \
  --ckpt ckpts/checkpoint_epoch_133.pth \
  --out_dir model/runtime_onnx \
  --data_path data
```

### 2. Build TensorRT Engine

TensorRT engine generation is done on the host with the local TensorRT tar package under `third_party/TensorRT-8.6.1.6`:

```bash
cd /home/$USER/workspace/CUDA-PointPillars

unset TensorRT_Inc TensorRT_Lib TensorRT_Bin
export TensorRT_Inc=$PWD/third_party/TensorRT-8.6.1.6/include
export TensorRT_Lib=$PWD/third_party/TensorRT-8.6.1.6/lib
export TensorRT_Bin=$PWD/third_party/TensorRT-8.6.1.6/bin

. tool/environment.sh

bash tool/build_trt_engine.sh \
  ./model/runtime_onnx/pointpillar.onnx \
  ./model/runtime/pointpillar.plan
```

### 3. Offline Runtime

The native runtime is aligned with the custom config and loads:

- engine: `model/runtime/pointpillar.plan`
- point cloud range: `[-10, -20, -1, 30, 20, 3]`
- voxel size: `[0.05, 0.05, 4]`
- class: `Pedestrian`

Build and run:

```bash
cd /home/$USER/workspace/CUDA-PointPillars
. tool/environment.sh
mkdir -p build
cd build
cmake ..
make -j$(nproc)
./pointpillar ../data/ ../data/ --timer
```

### Performance Notes

Current custom-model measurements on the laptop branch:

- offline end-to-end runtime: about `31~33 ms / frame`
- offline end-to-end throughput: about `30~32 FPS`
- TensorRT engine-only runtime (`trtexec` total): about `9.25 ms / frame`
- TensorRT engine-only throughput: about `108 FPS`

Observed offline breakdown:

- voxelization: about `0.05 ms`
- backbone + head: about `8~10 ms`
- decoder + NMS: about `22~24 ms`

The current bottleneck is the CUDA decoder + NMS stage rather than the TensorRT backbone.

## ROS1 Runtime

A ROS1 wrapper package is included in [ros1/pointpillar_ros](/home/firo/workspace/CUDA-PointPillars/ros1/pointpillar_ros).

Features:

- subscribes to `/velodyne_points`
- filters points by `point_cloud_range`
- publishes filtered cloud on `/pointpillar/filtered_points`
- publishes detection markers on `/pointpillar/detections`
- shows `class_name + euclidean_distance` above each box
- supports configurable `min_confidence` and `nms_thresh`
- can print moving-average runtime FPS in ROS log

### Build

```bash
mkdir -p ~/catkin_ws/src
ln -sfn /home/$USER/workspace/CUDA-PointPillars/ros1/pointpillar_ros ~/catkin_ws/src/pointpillar_ros

cd ~/catkin_ws
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```

### Launch

```bash
cd /home/$USER/workspace/CUDA-PointPillars
. tool/environment.sh
source ~/catkin_ws/devel/setup.bash
roslaunch pointpillar_ros pointpillar_with_rviz.launch
```

Main ROS config:

- [ros1/pointpillar_ros/config/pointpillar_custom.yaml](/home/firo/workspace/CUDA-PointPillars/ros1/pointpillar_ros/config/pointpillar_custom.yaml)
- [ros1/pointpillar_ros/config/pointpillar.rviz](/home/firo/workspace/CUDA-PointPillars/ros1/pointpillar_ros/config/pointpillar.rviz)

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

## Note

- Voxelization has random output since GPU processes all points simultaneously while points selection for a voxel is random.

## References

- [Detecting Objects in Point Clouds with NVIDIA CUDA-Pointpillars](https://developer.nvidia.com/blog/detecting-objects-in-point-clouds-with-cuda-pointpillars/)
- [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)
