import numpy as np
with open("data/custom_bin/000000.bin", "rb") as f:
    data = np.frombuffer(f.read(), dtype=np.float32)
    print("Total floats:", len(data))
    print("First 5 points:", data[:20].reshape(-1, 4))  # 打印前5个点
    small = data[:4000]  # data 是你 bin 读取出的 numpy array
    small.tofile("small.bin")

# Total floats: 61252
# First 5 points: [[ 3.8190942  -1.4576064  -1.0953223  13.        ]
#  [ 3.8240507  -1.460263    0.07145015 10.        ]
#  [ 3.8155599  -1.4577836  -0.94299495 13.        ]
#  [ 3.8228686  -1.4605759   0.21447276 12.        ]
#  [ 3.8272166  -1.4630029  -0.7964368  16.        ]]