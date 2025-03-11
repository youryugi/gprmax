import sys
from pywifi import PyWiFi, const, Profile
import pandas as pd
import time

wifi = PyWiFi()
ifaces = wifi.interfaces()[0]

target_ssids = ['ytest1-2.4g', 'ytest2-2.4g', 'ytest3-2.4g', 'ytest4-2.4g']
scans = []
ifaces.scan()
time.sleep(2)  # 休眠 2 秒等待扫描结果
results = ifaces.scan_results()

# 初始化 scan_data，确保所有 SSID 顺序一致，未扫描到的默认值为 -50
scan_data = {ssid: -50 for ssid in target_ssids}

# 只更新扫描到的 SSID
for network in results:
    if network.ssid in target_ssids:
        scan_data[network.ssid] = network.signal

scans.append(pd.DataFrame([scan_data]))

# 以 target_ssids 指定的顺序打印 RSSI 值
ordered_scan_data = {ssid: scan_data[ssid] for ssid in target_ssids}
print(ordered_scan_data)
