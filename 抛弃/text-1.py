# 文件名: waifuc_loader.py
# 放置于: ComfyUI\custom_nodes\Comfyui-waifuc\

import torch
import numpy as np
from PIL import Image

# 检查 waifuc 是否已安装
try:
    from waifuc.source import DanbooruSource
    from waifuc.action import ModeConvertAction, FirstNSelectAction, TaggingAction
    WAIFUC_INSTALLED = True
except ImportError:
    WAIFUC_INSTALLED = False

class WaifucSimpleLoader:
    """
    一个简单的 Waifuc 测试节点，用于从 Danbooru 加载单张图片及其标签。
    """
    def __init__(self):
        if not WAIFUC_INSTALLED:
            # 如果库未安装，在初始化时抛出异常
            raise ImportError("Waifuc 库未找到。请先在 ComfyUI 的 Python 环境中安装它：\n"
                              "在 ComfyUI 目录运行: python_embeded\\python.exe -m pip install \"waifuc[all]\"")

    @classmethod
    def INPUT_TYPES(cls):
        """定义节点的输入参数。"""
        return {
            "required": {
                "tags": ("STRING", {"default": "1girl, solo, silver_hair, looking_at_viewer, highres"}),
                "char_tags": ("STRING", {"default": "surtr_(arknights)"}),
            }
        }

    # 定义节点的返回类型和名称
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "caption")
    
    # 定义节点的主函数和类别
    FUNCTION = "process"
    CATEGORY = "Waifuc"

    def process(self, tags, char_tags):
        """节点执行的核心逻辑。"""
        # 1. 组合并处理标签
        # 将逗号分隔的字符串转换为列表，并去除首尾空格
        all_tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
        if char_tags.strip():
            all_tags.insert(0, char_tags.strip()) # 将角色标签放在最前面，权重更高

        print(f"Waifuc: Searching with tags - {all_tags}")

        # 2. 创建数据源
        source = DanbooruSource(all_tags)

        # 3. 创建一个最小化的处理流水线
        #    - ModeConvertAction: 确保图片是 'RGB' 格式，背景为白色
        #    - FirstNSelectAction(1): 只处理找到的第一张图片就停止
        #    - TaggingAction: 使用模型为图片生成标签 (caption)
        pipeline = source.attach(
            ModeConvertAction('RGB', 'white'),
            FirstNSelectAction(1),
            TaggingAction(force=True),
        )

        # 4. 从流水线中获取处理好的第一项数据
        #    iter(pipeline) 创建一个迭代器, next(..., None) 获取第一项，如果没有则返回 None
        item = next(iter(pipeline), None)

        if item is None:
            raise RuntimeError(f"Waifuc 未能找到任何符合标签的图片: {all_tags}")

        print("Waifuc: Image found and processed.")

        # 5. 将 PIL.Image 对象转换为 ComfyUI 需要的 Torch Tensor
        pil_image = item.image
        
        # 从 PIL -> NumPy 数组, 并将像素值归一化到 [0, 1]
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        
        # 从 NumPy -> Torch Tensor, 并增加一个批处理维度 (batch_size=1)
        # 格式: [batch_size, height, width, channels]
        tensor_image = torch.from_numpy(np_image)[None,]

        # 6. 从元数据中获取生成的标签
        caption = item.meta.get('tags', '')

        # 7. 返回 Tensor 和标签字符串，必须是元组格式
        return (tensor_image, caption)

# ComfyUI 的节点注册样板代码
NODE_CLASS_MAPPINGS = {
    "WaifucSimpleLoader": WaifucSimpleLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaifucSimpleLoader": "Waifuc Simple Loader (Test)"
}