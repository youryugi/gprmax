import sys
from pywifi import PyWiFi, const, Profile
import pandas as pd
import time
from datetime import datetime

# 最终使用的是这个代码来收集的
def scan_wifi_networks(location_number):
    wifi = PyWiFi()  # 创建一个无线对象
    ifaces = wifi.interfaces()[0]  # 获取第一个无线网卡

    # 初始化一个空列表，用于后续存储每次扫描的结果
    scans = []
    ssids = {}  # 用来存储 MAC 地址对应的 SSID

    for _ in range(40):
        ifaces.scan()  # 扫描网络
        time.sleep(0.25)  # 等待 0.2 秒以获取扫描结果
        results = ifaces.scan_results()  # 获取扫描结果

        # 创建一个空字典来存储这次扫描的结果
        scan_data = {}

        for network in results:
            # 以 MAC 地址为键，信号强度为值
            scan_data[network.bssid] = network.signal
            # 存储或更新 MAC 地址对应的 SSID
            ssids[network.bssid] = network.ssid

        # 将这次扫描的结果存储为 DataFrame，然后添加到列表中
        scans.append(pd.DataFrame([scan_data]))

    # 使用 concat 方法合并所有单次扫描的 DataFrame
    all_scans = pd.concat(scans, ignore_index=True)

    # 在 DataFrame 顶部添加一行 SSID
    ssid_row = pd.DataFrame([ssids])  # 创建包含 SSID 的 DataFrame
    all_scans = pd.concat([ssid_row, all_scans], ignore_index=True)

    # 将所有扫描结果写入 Excel 文件
    #current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 将所有扫描结果写入 Excel 文件，文件名包含当前时间和传入的参数
    filename = f'wifi_scans_{location_number}.xlsx'
    all_scans.to_excel(filename)
    # 打印 DataFrame 的内容到终端
    if len(all_scans.columns) > 5:
        print(all_scans.iloc[:, :5])
    else:
        print(all_scans)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        location_number = sys.argv[1]
    else:
        location_number = "default"

    scan_wifi_networks(location_number)
