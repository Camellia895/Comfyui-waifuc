import torch
import numpy as np
from PIL import Image
import os
import json # 添加这一行
from waifuc.source import LocalSource
from waifuc.model import ImageItem

class WaifucImageBatchToImageList:
    DISPLAY_NAME = "Waifuc 图像批次转列表"
    CATEGORY = "Waifuc/Waifuc辅助"
    FUNCTION = "doit"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), }}

    def doit(self, image):
        images = [image[i:i + 1, ...] for i in range(image.shape[0])]
        return (images,)

class WaifucLoadImagesFromPathNode:
    DISPLAY_NAME = "Waifuc 从路径加载图像列表"
    CATEGORY = "Waifuc/Waifuc源"
    FUNCTION = "load_images_from_path"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "multiline": False, "placeholder": "输入文件夹路径..."}),
            }
        }

    def load_images_from_path(self, folder_path: str):
        if not folder_path or not os.path.isdir(folder_path):
            print(f"WaifucLoadImagesFromPathNode: 提供的路径 '{folder_path}' 无效或不存在。输出 1x1 黑色占位图。")
            placeholder_img = Image.new('RGB', (1, 1), 'black')
            return (torch.from_numpy(np.array(placeholder_img).astype(np.float32) / 255.0)[None,],)

        try:
            source = LocalSource(folder_path)
            
            # 将 ImageItem 转换为 PIL Image，然后转换为 Tensor
            output_tensors = []
            for item in source:
                pil_img = item.image
                img_tensor = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0)[None,]
                output_tensors.append(img_tensor)

            if not output_tensors:
                print(f"WaifucLoadImagesFromPathNode: 路径 '{folder_path}' 中未找到图像。输出 1x1 黑色占位图。")
                placeholder_img = Image.new('RGB', (1, 1), 'black')
                return (torch.from_numpy(np.array(placeholder_img).astype(np.float32) / 255.0)[None,],)
            
            return (output_tensors,)

        except Exception as e:
            print(f"WaifucLoadImagesFromPathNode: 加载图像时发生错误: {e}。输出 1x1 黑色占位图。")
            placeholder_img = Image.new('RGB', (1, 1), 'black')
            return (torch.from_numpy(np.array(placeholder_img).astype(np.float32) / 255.0)[None,],)

class WaifucBatchImagesNode:
    DISPLAY_NAME = "Waifuc 图像批处理"
    CATEGORY = "Waifuc/Waifuc辅助"
    FUNCTION = "batch_images"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (False,) # 输出图像批次，而不是列表

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"forceInput": True, "is_list": True}),
            }
        }

    def batch_images(self, images: list):
        # 检查输入图像列表是否为空或不包含任何图像
        if not images or len(images) == 0:
            print("WaifucBatchImagesNode: 输入图像列表为空。输出 1x1 黑色占位图。")
            placeholder_img = Image.new('RGB', (1, 1), 'black')
            return (torch.from_numpy(np.array(placeholder_img).astype(np.float32) / 255.0)[None,],)

        max_width = 0
        max_height = 0

        # 找到最大宽度和最大高度
        for img_tensor in images:
            # img_tensor 的形状通常是 (1, H, W, C) 或 (H, W, C)
            # 我们需要处理这两种情况
            if img_tensor.dim() == 4: # (B, H, W, C)
                h, w = img_tensor.shape[1:3]
            elif img_tensor.dim() == 3: # (H, W, C)
                h, w = img_tensor.shape[0:2]
            else:
                print(f"WaifucBatchImagesNode: 发现不支持的图像张量维度: {img_tensor.shape}。跳过此图像。")
                continue
            
            max_height = max(max_height, h)
            max_width = max(max_width, w)

        if max_width == 0 or max_height == 0:
            print("WaifucBatchImagesNode: 无法确定图像尺寸。输出 1x1 黑色占位图。")
            placeholder_img = Image.new('RGB', (1, 1), 'black')
            return (torch.from_numpy(np.array(placeholder_img).astype(np.float32) / 255.0)[None,],)

        batched_images = []
        for img_tensor in images:
            if img_tensor.dim() == 4: # (B, H, W, C)
                img_tensor = img_tensor.squeeze(0) # 移除批次维度，变为 (H, W, C)
            
            # 将 Tensor 转换为 PIL Image
            pil_img = Image.fromarray((img_tensor.cpu().numpy() * 255).astype(np.uint8))

            # 创建一个黑色背景的新图像
            new_img = Image.new('RGB', (max_width, max_height), 'black')
            
            # 将原始图像粘贴到新图像的左上角
            new_img.paste(pil_img, (0, 0))
            
            # 将 PIL Image 转换回 Tensor
            img_tensor_padded = torch.from_numpy(np.array(new_img).astype(np.float32) / 255.0)
            batched_images.append(img_tensor_padded)

        # 将所有处理后的图像堆叠成一个批次
        # 确保所有图像都有批次维度 (B, H, W, C)
        final_batch = torch.stack(batched_images, dim=0)
        
        return (final_batch,)

class ImageListToImageBatch:
    DISPLAY_NAME = "图像列表转图像批次"
    CATEGORY = "Waifuc/Waifuc辅助"
    FUNCTION = "doit"
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("IMAGE_BATCH", "PADDING_INFO_JSON",)
    OUTPUT_IS_LIST = (False, False,)
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    def doit(self, images):
        if not images:
            print("ImageListToImageBatch: 输入图像列表为空。输出 1x1 黑色占位图和空 JSON。")
            placeholder_img = Image.new('RGB', (1, 1), 'black')
            return (torch.from_numpy(np.array(placeholder_img).astype(np.float32) / 255.0)[None,], json.dumps([]),)

        max_width = 0
        max_height = 0
        original_dims = []

        for img_tensor in images:
            if img_tensor.dim() == 4:
                h, w = img_tensor.shape[1:3]
            elif img_tensor.dim() == 3:
                h, w = img_tensor.shape[0:2]
            else:
                print(f"WaifucImageBatchPadAndInfoNode: 发现不支持的图像张量维度: {img_tensor.shape}。跳过此图像。")
                continue
            
            original_dims.append({"height": h, "width": w})
            max_height = max(max_height, h)
            max_width = max(max_width, w)

        if max_width == 0 or max_height == 0:
            print("WaifucImageBatchPadAndInfoNode: 无法确定图像尺寸。输出 1x1 黑色占位图和空 JSON。")
            placeholder_img = Image.new('RGB', (1, 1), 'black')
            return (torch.from_numpy(np.array(placeholder_img).astype(np.float32) / 255.0)[None,], json.dumps([]),)

        batched_images = []
        for img_tensor in images:
            if img_tensor.dim() == 4:
                img_tensor = img_tensor.squeeze(0)
            
            pil_img = Image.fromarray((img_tensor.cpu().numpy() * 255).astype(np.uint8))
            new_img = Image.new('RGB', (max_width, max_height), 'black')
            new_img.paste(pil_img, (0, 0))
            
            img_tensor_padded = torch.from_numpy(np.array(new_img).astype(np.float32) / 255.0)
            batched_images.append(img_tensor_padded)

        final_batch = torch.stack(batched_images, dim=0)
        
        padding_info_json = json.dumps(original_dims)
        
        return (final_batch, padding_info_json,)

class WaifucImageBatchToImageListRestoreNode:
    DISPLAY_NAME = "Waifuc 图像批次还原为列表"
    CATEGORY = "Waifuc/Waifuc辅助"
    FUNCTION = "restore_images"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_batch": ("IMAGE",),
                "padding_info_json": ("STRING", {"forceInput": True}),
            }
        }

    def restore_images(self, image_batch: torch.Tensor, padding_info_json: str):
        if image_batch.shape[0] == 0:
            print("WaifucImageBatchToImageListRestoreNode: 输入图像批次为空。输出 1x1 黑色占位图。")
            placeholder_img = Image.new('RGB', (1, 1), 'black')
            return ([torch.from_numpy(np.array(placeholder_img).astype(np.float32) / 255.0)[None,]],)

        try:
            original_dims = json.loads(padding_info_json)
        except json.JSONDecodeError as e:
            print(f"WaifucImageBatchToImageListRestoreNode: 解析 JSON 失败: {e}。输出 1x1 黑色占位图。")
            placeholder_img = Image.new('RGB', (1, 1), 'black')
            return ([torch.from_numpy(np.array(placeholder_img).astype(np.float32) / 255.0)[None,]],)

        if len(original_dims) != image_batch.shape[0]:
            print("WaifucImageBatchToImageListRestoreNode: JSON 信息与图像批次数量不匹配。输出 1x1 黑色占位图。")
            placeholder_img = Image.new('RGB', (1, 1), 'black')
            return ([torch.from_numpy(np.array(placeholder_img).astype(np.float32) / 255.0)[None,]],)

        restored_images = []
        for i, img_tensor in enumerate(image_batch):
            # img_tensor 的形状是 (H, W, C)
            original_h = original_dims[i]["height"]
            original_w = original_dims[i]["width"]

            # 裁剪图像到原始尺寸
            cropped_img_tensor = img_tensor[:original_h, :original_w, :]
            restored_images.append(cropped_img_tensor[None,]) # 添加批次维度

        return (restored_images,)

NODE_CLASS_MAPPINGS = {
    "WaifucLoadImagesFromPathNode": WaifucLoadImagesFromPathNode,
    "WaifucBatchImagesNode": WaifucBatchImagesNode,
    "WaifucImageBatchToImageList": WaifucImageBatchToImageList,
    "ImageListToImageBatch": ImageListToImageBatch, # 更新为新名称
    "WaifucImageBatchToImageListRestoreNode": WaifucImageBatchToImageListRestoreNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaifucLoadImagesFromPathNode": WaifucLoadImagesFromPathNode.DISPLAY_NAME,
    "WaifucBatchImagesNode": WaifucBatchImagesNode.DISPLAY_NAME,
    "WaifucImageBatchToImageList": WaifucImageBatchToImageList.DISPLAY_NAME,
    "ImageListToImageBatch": ImageListToImageBatch.DISPLAY_NAME, # 更新为新名称
    "WaifucImageBatchToImageListRestoreNode": WaifucImageBatchToImageListRestoreNode.DISPLAY_NAME,
}
