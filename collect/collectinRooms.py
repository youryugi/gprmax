import sys
from pywifi import PyWiFi, const, Profile
import pandas as pd
import time
from datetime import datetime
import threading
import tkinter as tk


# 最终使用的是这个代码来收集的
def scan_wifi_networks(location_number, stop_event):
    wifi = PyWiFi()  # 创建一个无线对象
    ifaces = wifi.interfaces()[0]  # 获取第一个无线网卡

    # 初始化一个空列表，用于后续存储每次扫描的结果
    scans = []
    ssids = {}  # 用来存储 MAC 地址对应的 SSID

    try:
        while not stop_event.is_set():
            ifaces.scan()  # 扫描网络
            time.sleep(0.1)  # 等待 0.1 秒以获取扫描结果
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

    except KeyboardInterrupt:
        stop_event.set()

    # 使用 concat 方法合并所有单次扫描的 DataFrame
    all_scans = pd.concat(scans, ignore_index=True)

    # 在 DataFrame 顶部添加一行 SSID
    ssid_row = pd.DataFrame([ssids])  # 创建包含 SSID 的 DataFrame
    all_scans = pd.concat([ssid_row, all_scans], ignore_index=True)

    # 将所有扫描结果写入 Excel 文件
    filename = f'closeRandom_{location_number}.xlsx'
    all_scans.to_excel(filename)
    # 打印 DataFrame 的内容到终端
    if len(all_scans.columns) > 5:
        print(all_scans.iloc[:, :5])
    else:
        print(all_scans)


def start_scan():
    location_number = location_entry.get() if location_entry.get() else "default"
    stop_event.clear()
    threading.Thread(target=scan_wifi_networks, args=(location_number, stop_event)).start()


def stop_scan():
    stop_event.set()


if __name__ == "__main__":
    stop_event = threading.Event()

    # 创建一个简单的 GUI 界面
    root = tk.Tk()
    root.title("WiFi Scanner")

    tk.Label(root, text="Location Number:").pack()
    location_entry = tk.Entry(root)
    location_entry.pack()

    start_button = tk.Button(root, text="Start Scan", command=start_scan)
    start_button.pack()

    stop_button = tk.Button(root, text="Stop Scan", command=stop_scan)
    stop_button.pack()

    root.mainloop()
