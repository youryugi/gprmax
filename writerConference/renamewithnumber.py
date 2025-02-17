import os
import re

# 指定目标文件夹路径
folder_path = r"C:\Users\79152\Desktop\3rdtopic\writerConference\truedata"  # 替换为你的文件夹路径

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)

    # 确保是文件而不是文件夹
    if os.path.isfile(file_path):
        # 使用正则提取文件名中的数字部分
        numbers = re.findall(r'\d+', file_name)

        if numbers:
            new_file_name = ''.join(numbers) + os.path.splitext(file_name)[-1]
            new_file_path = os.path.join(folder_path, new_file_name)

            # 重命名文件
            os.rename(file_path, new_file_path)
            print(f"重命名: {file_name} -> {new_file_name}")
        else:
            print(f"跳过: {file_name}（无数字）")
