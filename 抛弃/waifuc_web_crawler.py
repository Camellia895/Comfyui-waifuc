# -*- coding: utf-8 -*-
# 文件名: waifuc_web_crawler.py (最终验证版)

import torch
import numpy as np
from PIL import Image
import traceback

# ======================================================================
# [核心修改] 直接导入模块，不再使用 try/except 隐藏错误。
# 如果以下任何一个导入失败，控制台将直接显示详细的 ImportError。
# ======================================================================
from waifuc.source import DanbooruSource
from waifuc.action import FirstNSelectAction
from waifuc.source import DanbooruSource
from waifuc.action import ModeConvertAction, FirstNSelectAction, TaggingAction

class WaifucCrawlerValidatorNode:
    """
    一个绝对最小化的 Waifuc 网页爬取验证节点。
    它只做一件事：从 Danbooru 获取指定标签的前N张图片。
    如果这个节点能工作，说明基础的爬取功能是完好的。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tags": ("STRING", {"default": "1girl, solo, silver_hair, highres"}),
                "limit": ("INT", {"default": 3, "min": 1, "max": 50}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "crawl"
    CATEGORY = "Waifuc" # 放在同一个分类下

    def _pil_to_tensor_batch(self, pil_images):
        """一个简化的转换函数，将PIL图像列表转换为Tensor批次"""
        if not pil_images:
            # 如果列表为空，返回一个占位符
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        
        tensors = []
        for img in pil_images:
            img_rgb = img.convert("RGB")
            np_img = np.array(img_rgb).astype(np.float32) / 255.0
            tensor = torch.from_numpy(np_img).unsqueeze(0)
            tensors.append(tensor)
        
        # 注意：这里我们故意不处理尺寸不一的问题，因为waifuc下载的图尺寸可能不同
        # ComfyUI 会在预览时选择显示第一张。这足以验证功能。
        # 要么就用回之前的 padding 函数。为了最小化，先这样。
        # 好吧，为了工作流的连续性，还是加上padding
        return self._pad_and_convert_to_tensor(pil_images)

    def _pad_and_convert_to_tensor(self, pil_images):
        if not pil_images: return torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        max_width = max(img.width for img in pil_images)
        max_height = max(img.height for img in pil_images)
        tensors = []
        for img in pil_images:
            img_rgb = img.convert("RGB")
            padded_img = Image.new("RGB", (max_width, max_height), (0, 0, 0))
            padded_img.paste(img_rgb, ((max_width - img_rgb.width) // 2, (max_height - img_rgb.height) // 2))
            np_img = np.array(padded_img).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(np_img).unsqueeze(0))
        return torch.cat(tensors, 0)

    def crawl(self, tags, limit):
        try:
            tags_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
            print(f"Waifuc Validator: 开始爬取, 标签: {tags_list}, 数量: {limit}")

            # 1. 创建数据源
            source = DanbooruSource(tags_list)
            
            # 2. 只使用最简单的 Action，获取前N项
            pipeline = FirstNSelectAction(limit).iter_from(source)

            # 3. 执行并收集图像
            processed_pils = [item.image for item in pipeline]

            if not processed_pils:
                print("Waifuc Validator: 警告 - 未找到任何图像。")
                return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)

            print(f"Waifuc Validator: 成功爬取 {len(processed_pils)} 张图像。")
            
            # 4. 转换为 Tensor Batch
            return (self._pad_and_convert_to_tensor(processed_pils),)

        except Exception:
            print("Waifuc Validator 发生严重错误:")
            print(traceback.format_exc())
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "WaifucCrawlerValidatorNode": WaifucCrawlerValidatorNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WaifucCrawlerValidatorNode": "Waifuc 爬虫验证器"
}