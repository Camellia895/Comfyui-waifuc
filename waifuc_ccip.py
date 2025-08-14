import torch
import numpy as np
from PIL import Image
import comfy.model_management
from typing import Iterator, Optional, List
import tempfile
import os
import json

from waifuc.action import ProcessAction
from waifuc.model import ImageItem
from waifuc.source import BaseDataSource

from waifuc.action import (CCIPAction, ModeConvertAction)
from waifuc.source import LocalSource

# ================================================================
# 自定义 Action：中断检查器 (共用)
# ================================================================
class ComfyInterruptAction(ProcessAction):
    def process(self, item: ImageItem) -> ImageItem:
        comfy.model_management.throw_exception_if_processing_interrupted()
        return item

# ================================================================
# 自定义 Source：内存中的PIL图像源
# ================================================================
class InMemoryPILSource(BaseDataSource):
    def __init__(self, pil_images: List[Image.Image]):
        super().__init__()
        self.pil_images = pil_images

    def _iter(self) -> Iterator[ImageItem]: # [修正] 将 _iter_data 改为 _iter
        for img in self.pil_images:
            yield ImageItem(image=img)

# ================================================================
# 节点: Waifuc CCIP 图像处理器 (简洁版)
# ================================================================
class WaifucCCIPNode:
    DISPLAY_NAME = "Waifuc CCIP 图像处理器"
    CATEGORY = "图像/Waifuc处理"
    FUNCTION = "process_images_with_ccip"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "padding_info_json": ("STRING", {"forceInput": True}),
            }
        }

    def process_images_with_ccip(self, images: torch.Tensor, padding_info_json: str):
        print(f"Waifuc CCIP 处理器 (简洁版): 开始执行...")
        
        # 1. 还原图像
        try:
            original_dims = json.loads(padding_info_json)
        except (json.JSONDecodeError, TypeError):
            print("Waifuc CCIP 处理器 (简洁版): JSON 解析失败或格式不正确，跳过还原。")
            original_dims = []

        if len(original_dims) != images.shape[0]:
            print("Waifuc CCIP 处理器 (简洁版): JSON 信息与图像批次数量不匹配，跳过还原。")
            restored_pil_images = [Image.fromarray((images[i].cpu().numpy() * 255).astype(np.uint8)) for i in range(images.shape[0])]
        else:
            restored_pil_images = []
            for i, img_tensor in enumerate(images):
                original_h = original_dims[i]["height"]
                original_w = original_dims[i]["width"]
                cropped_img_tensor = img_tensor[:original_h, :original_w, :]
                pil_img = Image.fromarray((cropped_img_tensor.cpu().numpy() * 255).astype(np.uint8))
                restored_pil_images.append(pil_img)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 2. 将还原后的PIL图像保存到临时目录
            for i, pil_img in enumerate(restored_pil_images):
                pil_img.save(os.path.join(temp_dir, f"image_{i}.png"))

            # 3. 使用临时目录的路径初始化 LocalSource
            source = LocalSource(temp_dir)
            
            # 4. 构建和执行流水线
            actions = [ComfyInterruptAction(), ModeConvertAction('RGB', 'white')]
            actions.append(CCIPAction()) # CCIPAction 始终启用
            
            pipeline = source.attach(*actions)
            processed_pil_images = [item.image for item in pipeline]

        if not processed_pil_images:
            print("Waifuc CCIP 处理器 (简洁版) 警告: 没有图像通过筛选，已输出1x1黑色占位图。")
            placeholder_img = Image.new('RGB', (1, 1), 'black')
            tensors = [torch.from_numpy(np.array(placeholder_img).astype(np.float32) / 255.0)[None,]]
        else:
            tensors = [torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,] for img in processed_pil_images]
            
        return (tensors,)

# ================================================================
# 节点: Waifuc CCIP 图像处理器 (高级版)
# ================================================================
class WaifucAdvancedCCIPNode:
    DISPLAY_NAME = "Waifuc CCIP 图像处理器 (高级)"
    CATEGORY = "图像/Waifuc处理"
    FUNCTION = "process_images_with_advanced_ccip"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "padding_info_json": ("STRING", {"forceInput": True}),
                "anchor_images": ("IMAGE", {"label": "锚点图像"}),
                "anchor_padding_info_json": ("STRING", {"forceInput": True, "label": "锚点图像裁剪信息 (JSON)"}),
            },
            "optional": {
                "min_val_count": ("INT", {"default": 15, "min": 1, "max": 100, "step": 1, "label": "最小验证图像数"}),
                "step": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1, "label": "聚类步长"}),
                "ratio_threshold": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01, "label": "聚类比例阈值"}),
                "min_clu_dump_ratio": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "label": "最小聚类倾倒比例"}),
                "cmp_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "label": "比较阈值"}),
                "eps": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "label": "DBSCAN/OPTICS eps (可选)"}),
                "min_samples": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1, "label": "DBSCAN/OPTICS min_samples (可选)"}),
                "model": (["ccip-caformer-24-randaug-pruned"], {"default": "ccip-caformer-24-randaug-pruned", "label": "CCIP模型"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "label": "CCIP相似度阈值 (可选)"}),
            }
        }

    def process_images_with_advanced_ccip(self, images: torch.Tensor, padding_info_json: str,
                                          anchor_images: torch.Tensor, anchor_padding_info_json: str,
                                          min_val_count: int = 15, step: int = 5,
                                          ratio_threshold: float = 0.6, min_clu_dump_ratio: float = 0.3, cmp_threshold: float = 0.5,
                                          eps: Optional[float] = None, min_samples: Optional[int] = None,
                                          model: str = 'ccip-caformer-24-randaug-pruned', threshold: Optional[float] = None):
        print(f"Waifuc CCIP 处理器 (高级): 开始执行...")
        
        # 1. 还原主图像
        try:
            original_dims = json.loads(padding_info_json)
        except (json.JSONDecodeError, TypeError):
            print("Waifuc CCIP 处理器 (高级): 主图像JSON解析失败或格式不正确，跳过还原。")
            original_dims = []

        if len(original_dims) != images.shape[0]:
            print("Waifuc CCIP 处理器 (高级): 主图像JSON信息与图像批次数量不匹配，跳过还原。")
            restored_pil_images = [Image.fromarray((images[i].cpu().numpy() * 255).astype(np.uint8)) for i in range(images.shape[0])]
        else:
            restored_pil_images = []
            for i, img_tensor in enumerate(images):
                original_h = original_dims[i]["height"]
                original_w = original_dims[i]["width"]
                cropped_img_tensor = img_tensor[:original_h, :original_w, :]
                pil_img = Image.fromarray((cropped_img_tensor.cpu().numpy() * 255).astype(np.uint8))
                restored_pil_images.append(pil_img)

        ccip_init_source = None
        
        # 2. 还原锚点图像并设置 ccip_init_source (使用内存源)
        if anchor_images is not None and anchor_images.shape[0] > 0:
            try:
                anchor_original_dims = json.loads(anchor_padding_info_json)
            except (json.JSONDecodeError, TypeError):
                print("Waifuc CCIP 处理器 (高级): 锚点图像JSON解析失败或格式不正确，跳过还原。")
                anchor_original_dims = []

            if len(anchor_original_dims) != anchor_images.shape[0]:
                print("Waifuc CCIP 处理器 (高级): 锚点图像JSON信息与图像批次数量不匹配，跳过还原。")
                restored_anchor_pil_images = [Image.fromarray((anchor_images[i].cpu().numpy() * 255).astype(np.uint8)) for i in range(anchor_images.shape[0])]
            else:
                restored_anchor_pil_images = []
                for i, img_tensor in enumerate(anchor_images):
                    original_h = anchor_original_dims[i]["height"]
                    original_w = anchor_original_dims[i]["width"]
                    cropped_img_tensor = img_tensor[:original_h, :original_w, :]
                    pil_img = Image.fromarray((cropped_img_tensor.cpu().numpy() * 255).astype(np.uint8))
                    restored_anchor_pil_images.append(pil_img)
            
            ccip_init_source = InMemoryPILSource(restored_anchor_pil_images)
            print(f"Waifuc CCIP 处理器 (高级): 使用内存锚点图像源。")
        else:
            print("Waifuc CCIP 处理器 (高级): 未提供锚点图像或锚点图像为空，CCIPAction将以INIT模式启动。")

        # 3. 处理主图像流水线
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, pil_img in enumerate(restored_pil_images):
                pil_img.save(os.path.join(temp_dir, f"image_{i}.png"))

            source = LocalSource(temp_dir)
            
            actions = [ComfyInterruptAction(), ModeConvertAction('RGB', 'white')]
            
            ccip_action_instance = CCIPAction(
                init_source=ccip_init_source,
                min_val_count=min_val_count,
                step=step,
                ratio_threshold=ratio_threshold,
                min_clu_dump_ratio=min_clu_dump_ratio,
                cmp_threshold=cmp_threshold,
                eps=eps,
                min_samples=min_samples,
                model=model,
                threshold=threshold
            )
            actions.append(ccip_action_instance)
            
            pipeline = source.attach(*actions)
            processed_pil_images = [item.image for item in pipeline]

        if not processed_pil_images:
            print("Waifuc CCIP 处理器 (高级) 警告: 没有图像通过筛选，已输出1x1黑色占位图。")
            placeholder_img = Image.new('RGB', (1, 1), 'black')
            tensors = [torch.from_numpy(np.array(placeholder_img).astype(np.float32) / 255.0)[None,]]
        else:
            tensors = [torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,] for img in processed_pil_images]
            
        return (tensors,)

# ================================================================
# ComfyUI 节点注册
# ================================================================
NODE_CLASS_MAPPINGS = { 
    "WaifucCCIPNode": WaifucCCIPNode,
    "WaifucAdvancedCCIPNode": WaifucAdvancedCCIPNode, # 注册高级节点
}
NODE_DISPLAY_NAME_MAPPINGS = { 
    "WaifucCCIPNode": WaifucCCIPNode.DISPLAY_NAME,
    "WaifucAdvancedCCIPNode": WaifucAdvancedCCIPNode.DISPLAY_NAME, # 注册高级节点
}
