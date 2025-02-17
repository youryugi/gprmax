import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_5 = r"C:\Users\79152\Desktop\3rdtopic\writerConference\accuracy_results_5.csv"
file_10 = r"C:\Users\79152\Desktop\3rdtopic\writerConference\accuracy_results_10.csv"
file_15 = r"C:\Users\79152\Desktop\3rdtopic\writerConference\accuracy_results_15.csv"

df_5 = pd.read_csv(file_5)
df_10 = pd.read_csv(file_10)
df_15 = pd.read_csv(file_15)

# 确保数据按照文件名排序
df_5.sort_values("File Name", inplace=True)
df_10.sort_values("File Name", inplace=True)
df_15.sort_values("File Name", inplace=True)

# 提取数据
file_names = df_5["File Name"]
accuracy_5 = df_5["Accuracy Ratio"]
accuracy_10 = df_10["Accuracy Ratio"]
accuracy_15 = df_15["Accuracy Ratio"]

# 画柱状图，调整图形尺寸
plt.figure(figsize=(13, 4))  # 宽 14，高 5，确保图形长扁

bar_width = 0.3  # 柱状图的宽度
x = range(len(file_names))

# 颜色设置
colors = ["blue", "orange", "green"]

plt.bar(x, accuracy_5, width=bar_width, color=colors[0], label="ETR 5%", alpha=0.7)
plt.bar([i + bar_width for i in x], accuracy_10, width=bar_width, color=colors[1], label="ETR 10%", alpha=0.7)
plt.bar([i + 2 * bar_width for i in x], accuracy_15, width=bar_width, color=colors[2], label="ETR 15%", alpha=0.7)

# 让用户自己定义横坐标
custom_xticks = [
    "[0,0,0,0]", "[0,0,0,1]", "[0,0,1,0]", "[0,0,1,1]",
    "[0,1,0,0]", "[0,1,0,1]", "[0,1,1,0]", "[0,1,1,1]",
    "[1,0,0,0]", "[1,0,0,1]", "[1,0,1,0]", "[1,0,1,1]",
    "[1,1,0,0]", "[1,1,0,1]", "[1,1,1,0]", "[1,1,1,1]"
]  # 你可以修改这部分来定义自己的横坐标

# 设置横坐标
plt.xticks([i + bar_width for i in x], custom_xticks, rotation=30,fontsize=14)  # 旋转角度 0，确保横坐标水平
plt.yticks(fontsize=14)  # 将纵坐标字体大小调整为14

# 设置标签和标题

plt.ylabel("Accuracy Ratio",fontsize=14)
plt.title("Comparison of Accuracy Ratios for Different Error Tolerance Rates",fontsize=14)
plt.legend(fontsize=14, loc="upper left", bbox_to_anchor=(1, 1))


# 调整图形布局，避免标签重叠
plt.tight_layout()

# 显示图表
plt.show()
