import itertools
import numpy as np
import torch
np.set_printoptions(threshold=np.inf)  # 设置 NumPy 不限制打印数量
combinations = list(itertools.product([0, 1], repeat=4))

# 每个组合重复10次
expanded_data = np.repeat(combinations, 10, axis=0)
print(expanded_data)