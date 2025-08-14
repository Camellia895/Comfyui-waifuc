# 文件名: waifuc_cache_splitter.py
# 放置于: ComfyUI\custom_nodes\Comfyui-waifuc\

import torch
import numpy as np
from PIL import Image
import os
import shutil

try:
    from waifuc.model import ImageItem
    from waifuc.action import ThreeStageSplitAction
    WAIFUC_INSTALLED = True
except ImportError:
    WAIFUC_INSTALLED = False

# --- 核心修改：定义缓存目录 ---
# 使用 __file__ 来获取当前文件的路径，从而定位 Cache 文件夹
# 这样做比硬编码 'G:\...' 要好，因为无论你的 ComfyUI 移动到哪里，它都能正常工作。
NODE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(NODE_DIR, 'Cache')

class WaifucCacheSplitter:
    """
    一个使用 Waifuc 的 ThreeStageSplitAction 来处理图像的节点，
    并将结果保存到磁盘上的缓存文件夹中。
    """
    def __init__(self):
        if not WAIFUC_INSTALLED:
            raise ImportError("Waifuc 库或其依赖项未找到。请在 ComfyUI 环境中安装: ...")
        # 确保基础缓存目录存在
        os.makedirs(CACHE_DIR, exist_ok=True)

    @classmethod
    def INPUT_TYPES(cls):
        # --- 核心修改：动态扫描文件夹来创建下拉菜单 ---
        # 确保基础缓存目录存在
        os.makedirs(CACHE_DIR, exist_ok=True)
        # 获取 Cache 目录下的所有子文件夹
        try:
            subfolders = [d for d in os.listdir(CACHE_DIR) if os.path.isdir(os.path.join(CACHE_DIR, d))]
        except:
            subfolders = []
            
        return {
            "required": {
                "images": ("IMAGE",),
                # 下拉菜单 + 手动输入
                "cache_folder_name": (subfolders + ["_new_folder_"],),
                "clear_cache_folder": ("BOOLEAN", {"default": True}),
            }
        }

    # --- 核心修改：更改返回类型 ---
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("cache_path", "image_count")
    FUNCTION = "process_and_cache"
    CATEGORY = "Waifuc"

    def _tensor_to_pil(self, tensor_images):
        return [Image.fromarray(np.clip(255. * i.cpu().numpy(), 0, 255).astype(np.uint8)) for i in tensor_images]

    def process_and_cache(self, images, cache_folder_name, clear_cache_folder):
        # 检查是否选择了新建文件夹
        if cache_folder_name == "_new_folder_":
             print("Waifuc Cache Splitter: 请在下拉框中输入一个有效的文件夹名称。")
             return ("", 0)

        # 1. 确定并创建最终的输出路径
        output_path = os.path.join(CACHE_DIR, cache_folder_name)
        os.makedirs(output_path, exist_ok=True)

        # 2. 如果用户选择，则清空目标文件夹
        if clear_cache_folder:
            print(f"Waifuc Cache Splitter: 清空文件夹 - {output_path}")
            for filename in os.listdir(output_path):
                file_path = os.path.join(output_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'无法删除 {file_path}. 原因: {e}')

        # 3. 执行图像处理
        input_pils = self._tensor_to_pil(images)
        print(f"Waifuc Cache Splitter: 接收到 {len(input_pils)} 张图片进行处理。")
        image_items = [ImageItem(image=img) for img in input_pils]
        
        action = ThreeStageSplitAction()
        processed_iter = action.iter_from(image_items)
        processed_pils = [item.image for item in processed_iter]
        
        if not processed_pils:
            print("Waifuc Cache Splitter: 没有图像通过处理。")
            return (output_path, 0)

        # 4. 循环保存每一个处理结果
        print(f"Waifuc Cache Splitter: 生成了 {len(processed_pils)} 张图片，正在保存到 {output_path}")
        for i, pil_image in enumerate(processed_pils):
            # 创建一个带序号的文件名
            filename = f"image_{i+1:04d}.png"
            file_path = os.path.join(output_path, filename)
            pil_image.save(file_path)
            print(f"  - 已保存: {file_path}")

        # 5. 返回缓存路径和成功处理的图片数量
        return (output_path, len(processed_pils))

# 注册新节点
NODE_CLASS_MAPPINGS = {
    "WaifucCacheSplitter": WaifucCacheSplitter
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WaifucCacheSplitter": "Waifuc Cache Splitter"
}