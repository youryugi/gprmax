import os
import pandas as pd
import re

#这个不会去除重复的行
# 设定文件夹路径
folder_path = r'C:\Users\79152\Desktop\3rdtopic\demo0309\close0309'

# 获取所有xlsx文件的文件名
xlsx_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]


# 按文件名中的数字顺序对文件进行排序
def get_file_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')


xlsx_files.sort(key=get_file_number)

# 用于存储所有文件的数据
all_data = []

# 遍历所有xlsx文件
for file in xlsx_files:
    file_path = os.path.join(folder_path, file)

    # 读取xlsx文件，从第二行开始读取，同时忽略第一列
    df = pd.read_excel(file_path, header=1).iloc[:, 1:]

    # 提取列名为 ytest1, ytest2, ytest3, ytest4 的列
    selected_columns = ['ytest1-2.4g', 'ytest2-2.4g', 'ytest3-2.4g', 'ytest4-2.4g']
    df_selected = df[selected_columns]

    # 删除包含空单元格的行
    df_selected.dropna(inplace=True)

    # 获取文件名中的数字并减去1作为标签
    file_number = get_file_number(file)
    label = file_number - 1

    # 在最左边插入标签列
    df_selected.insert(0, 'Label', label)

    # 将数据添加到列表中
    all_data.append(df_selected)

# 将所有数据竖向拼接
final_data = pd.concat(all_data, ignore_index=True)
# 获取文件夹名称作为输出文件的前缀
folder_name = os.path.basename(os.path.normpath(folder_path))
output_filename = f'{folder_name}_all.xlsx'

# 输出到新的xlsx文件
final_data.to_excel(output_filename, index=False)

"""代码说明：
1. 使用 `os.listdir()` 遍历指定文件夹中的所有xlsx文件。
2. 使用 `re` 库对文件名中的数字进行提取和排序，确保按数字顺序拼接。
3. 逐个读取文件，使用 `pandas` 读取Excel，忽略第一行和第一列，提取指定列名的数据。
4. 使用 `dropna()` 删除包含空单元格的行。
5. 获取文件名中的数字减去1，作为标签列插入到最左侧。
6. 使用 `pandas.concat()` 将所有处理后的数据竖向拼接，最后保存到新的Excel文件中。
"""