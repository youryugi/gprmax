import os
from PIL import Image

def crop_images_from_folder(source_folder, crop_box):
    """
    遍历源文件夹下的所有PNG图片，裁剪后保存到当前文件夹（文件名不变）。
    """
    if not os.path.exists(source_folder):
        print(f"错误: 源文件夹 '{source_folder}' 不存在。")
        return

    # 获取源文件夹下所有 png 文件
    files = [f for f in os.listdir(source_folder) if f.lower().endswith('.png')]
    
    if not files:
        print(f"在 '{source_folder}' 中未找到 PNG 文件。")
        return

    print(f"找到 {len(files)} 个 PNG 文件，准备开始处理...")
    
    count = 0
    for filename in files:
        source_path = os.path.join(source_folder, filename)
        
        try:
            with Image.open(source_path) as img:
                # 裁剪图片
                cropped_img = img.crop(crop_box)
                
                # 构造保存路径：当前文件夹 + 原文件名
                save_path = os.path.join('.', filename)
                
                cropped_img.save(save_path)
                count += 1
                
                # 每处理50张打印一次进度
                if count % 50 == 0:
                    print(f"已处理 {count}/{len(files)} 张...")
                    
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")

    print(f"全部完成！共裁剪并保存了 {count} 张图片到当前目录。")

if __name__ == "__main__":
    # ================= 配置区域 =================
    # 1. 设置源图片所在的文件夹路径
    SOURCE_FOLDER = '/home/yang/gprmax/gengpr/cutmore16gpr50grey'  
    
    # 2. 设置裁剪区域 (左, 上, 右, 下)
    CROP_BOX = (0, 109, 920, 690)
    # ===========================================

    crop_images_from_folder(SOURCE_FOLDER, CROP_BOX)