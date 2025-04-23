# import numpy as np

# bin_data = np.fromfile("custom_bin/000000.bin", dtype=np.float32).reshape(-1, 4)  # or your specific shape
# npy_data = np.load("npy/000000.npy")

# print(np.allclose(bin_data, npy_data))  # True = format is same
# False

import numpy as np

# Load binary data
bin_path = "data/cleanBin/000001.bin"
npy_path = "data/npy/000001.npy"

# Adjust dtype and shape to match your data format (common: float32, shape: (-1, 4))
bin_data = np.fromfile(bin_path, dtype=np.float32)
print(f"BIN shape before reshape: {bin_data.shape}")

try:
    bin_data = bin_data.reshape(-1, 4)
except:
    print("Error reshaping bin_data. Check if total elements are multiple of 4.")
    exit()

# Load npy
npy_data = np.load(npy_path)
print(f"NPY shape: {npy_data.shape}")

# Check shape match
if bin_data.shape != npy_data.shape:
    print("Shape mismatch!")
    exit()

# Check for NaNs or Infs
print("Checking for NaNs/Infs...")
if np.isnan(bin_data).any() or np.isinf(bin_data).any():
    print("BIN data has NaNs or Infs.")
if np.isnan(npy_data).any() or np.isinf(npy_data).any():
    print("NPY data has NaNs or Infs.")

# Check actual values
close = np.allclose(bin_data, npy_data, equal_nan=True)
print("Data match:", close)
