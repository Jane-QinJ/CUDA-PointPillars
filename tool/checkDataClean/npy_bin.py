import numpy as np

npy = np.load("/home/firo/Documents/workspace/CUDA-PointPillars/data/npy/000000.npy")  # shape should be (N, 4)
print("Shape:", npy.shape)
print("Dtype:", npy.dtype)

assert npy.shape[1] == 4, "Point cloud must be Nx4"
assert npy.dtype == np.float32, "Point cloud must be float32"

npy.tofile("/home/firo/Documents/workspace/CUDA-PointPillars/data/custom_bin/000000.bin")
# Shape: (15313, 4)
# Dtype: float32
