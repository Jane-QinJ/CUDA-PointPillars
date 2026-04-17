FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    ca-certificates \
    build-essential \
    cmake \
    git \
    git-lfs \
    wget \
    curl \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN python3 -m pip install --upgrade pip

WORKDIR /workspace/CUDA-PointPillars
COPY . /workspace/CUDA-PointPillars

ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_Bin=/usr/local/cuda/bin
ENV CUDA_Inc=/usr/local/cuda/include
ENV CUDA_Lib=/usr/local/cuda/targets/x86_64-linux/lib
ENV CUDNN_Lib=/usr/local/cuda/lib64
ENV CUDNN_INCLUDE_DIR=/usr/local/cuda/include
ENV PATH=${CUDA_Bin}:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_Lib}:${CUDNN_Lib}:/workspace/CUDA-PointPillars/build:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/workspace/CUDA-PointPillars/build
ENV USE_Python=OFF
ENV TORCH_CUDA_ARCH_LIST=8.6
ENV FORCE_CUDA=1

RUN git lfs install

RUN python3 -m pip install --upgrade setuptools wheel ninja

RUN python3 -m pip install \
    numpy==1.24.4 \
    llvmlite==0.41.1 \
    numba==0.58.1 \
    easydict \
    pyyaml \
    scikit-image \
    tqdm \
    tensorboardX \
    SharedArray \
    pillow \
    opencv-python \
    onnx==1.15.0 \
    onnxsim==0.4.35

RUN python3 -m pip install \
    torch==1.10.0+cu113 \
    torchvision==0.11.1+cu113 \
    torchaudio==0.10.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN python3 -m pip install \
    onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com

RUN cd OpenPCDet && python3 -m pip install -r requirements.txt && python3 setup.py develop

CMD ["/bin/bash"]

# Build from repo root:
# docker build -f docker/Dockerfile . -t cudapointpillar:latest
#
# Run:
# docker run --gpus all --rm -ti -v /home/$USER/:/home/$USER/ --net=host --ipc=host cudapointpillar:latest
