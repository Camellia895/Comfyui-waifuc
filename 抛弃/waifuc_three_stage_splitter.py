# 文件名: waifuc_three_stage_splitter.py
# 放置于: ComfyUI\custom_nodes\Comfyui-waifuc\

import torch
import numpy as np
from PIL import Image

try:
    from waifuc.model import ImageItem
    # 导入新的 Action
    from waifuc.action import ThreeStageSplitAction
    WAIFUC_INSTALLED = True
except ImportError:
    WAIFUC_INSTALLED = False

class WaifucThreeStageSplitter:
    """
    一个使用 Waifuc 的 ThreeStageSplitAction 来处理图像的节点。
    这是专门为动漫角色设计的、更先进的人物分割方法。
    """
    def __init__(self):
        if not WAIFUC_INSTALLED:
            raise ImportError("Waifuc 库或其依赖项未找到。请确保已在 ComfyUI 的 Python 环境中安装它:\n"
                              "在 ComfyUI 目录运行: python_embeded\\python.exe -m pip install \"waifuc[all]\"")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "keep_multiple_results": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("processed_images", "success_flag")
    FUNCTION = "process"
    CATEGORY = "Waifuc" # 归到同一个类别下

    def _tensor_to_pil(self, tensor_images):
        return [Image.fromarray(np.clip(255. * i.cpu().numpy(), 0, 255).astype(np.uint8)) for i in tensor_images]

    def _pad_and_convert_to_tensor(self, pil_images):
        """
        将不同尺寸的 PIL 图像列表，通过填充统一尺寸后，转换为一个 Tensor Batch。
        """
        if not pil_images:
            return torch.zeros((1, 1, 1, 3))

        max_width = max(img.width for img in pil_images)
        max_height = max(img.height for img in pil_images)

        padded_tensors = []
        for img in pil_images:
            img = img.convert("RGB")
            padded_tensor = torch.zeros((1, max_height, max_width, 3), dtype=torch.float32)
            np_img = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(np_img)
            padded_tensor[0, :img.height, :img.width, :] = img_tensor
            padded_tensors.append(padded_tensor)

        return torch.cat(padded_tensors, 0)

    def process(self, images, keep_multiple_results):
        input_pils = self._tensor_to_pil(images)
        print(f"Waifuc ThreeStageSplitter: Received {len(input_pils)} image(s).")

        image_items = [ImageItem(image=img) for img in input_pils]

        # --- 核心逻辑变更 ---
        # 直接使用 ThreeStageSplitAction，它内置了完整的检测和裁剪流程。
        action = ThreeStageSplitAction()
        processed_iter = action.iter_from(image_items)
        # --------------------

        processed_pils = [item.image for item in processed_iter]
        
        if not processed_pils:
            print("Waifuc ThreeStageSplitter: No images passed the filter. Outputting placeholder.")
            return (torch.zeros((1, 1, 1, 3)), 0)

        print(f"Waifuc ThreeStageSplitter: Produced {len(processed_pils)} processed image(s).")
        for i, img in enumerate(processed_pils):
            print(f"  - Result {i+1}: {img.width}x{img.height}")

        if not keep_multiple_results and processed_pils:
            final_pils = [processed_pils[0]]
        else:
            final_pils = processed_pils
            
        output_tensor = self._pad_and_convert_to_tensor(final_pils)
        
        return (output_tensor, 1)

# 注册新节点到 ComfyUI
NODE_CLASS_MAPPINGS = {
    # 确保这里的类名和上面的 class 定义一致
    "WaifucThreeStageSplitter": WaifucThreeStageSplitter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # 这是节点在菜单中显示的名字
    "WaifucThreeStageSplitter": "Waifuc Three-Stage Splitter"
}