import numpy as np
data1 = np.load(r"C:\Users\79152\Desktop\3rdtopic\StableD\GRC2\generateRSSI_2025-04-24_12-49-23\X_data.npy")  # 形状 (1, 28, 28, 4)
print(data1.shape)

data2= np.load(r"C:\Users\79152\Desktop\3rdtopic\StableD\GRC2\generateRSSI_2025-02-13_12-51-49\weights_16_experiments\excluded_1_1_1_1\untrained\context_1_1_1_1\samples.npy")  # 形状 (1, 28, 28, 4)
print(data2.shape)