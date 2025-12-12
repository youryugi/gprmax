import os
from PIL import Image

def crop_single_image(image_path, crop_box):
    """
    裁剪单张图片并保存到当前文件夹（文件名不变）。
    """
    if not os.path.exists(image_path):
        print(f"错误: 文件 '{image_path}' 不存在。")
        return

    try:
        with Image.open(image_path) as img:
            print(f"原始尺寸: {img.size}")
            
            # 裁剪图片
            cropped_img = img.crop(crop_box)
            
            # 1. 获取原文件名 (例如 bscan_0000_v0.png)
            filename = os.path.basename(image_path)
            
            # 2. 构造保存路径：当前文件夹 + 原文件名
            save_path = os.path.join('.', filename)
            
            cropped_img.save(save_path)
            
            print(f"裁剪后尺寸: {cropped_img.size}")
            print(f"已保存至当前文件夹: {save_path}")
            
    except Exception as e:
        print(f"处理图片时出错: {e}")

if __name__ == "__main__":
    # ================= 配置区域 =================
    # 1. 设置源图片路径 (请修改为实际的外部文件夹路径)
    # 例如: '/home/yang/data/bscan_0000_v0.png'
    IMAGE_PATH = '/home/yang/gprmax/gengpr/gen16gpr50/bscan_0000_v0.png'  
    
    # 2. 设置裁剪区域 (左, 上, 右, 下)
    # 提示：(0, 0) 是左上角
    CROP_BOX = (189, 109, 1117, 800)
    # ===========================================

    crop_single_image(IMAGE_PATH, CROP_BOX)