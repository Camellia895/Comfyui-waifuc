# -*- coding: utf-8 -*-
# 文件名: waifuc_executor.py
# 放置于: ComfyUI\custom_nodes\waifuc_node\

import torch
import numpy as np
from PIL import Image
import os
import shutil
import traceback

# ---------------------------------
# 检查 waifuc 是否已安装
# ---------------------------------
try:
    from waifuc.source import LocalSource, EmptySource
    from waifuc.action import (
        NoMonochromeAction, FilterSimilarAction, TaggingAction, PersonSplitAction, 
        FaceCountAction, FirstNSelectAction, CCIPAction, ModeConvertAction, 
        ClassFilterAction, RandomFilenameAction, AlignMinSizeAction
    )
    from waifuc.export import SaveExporter
    from waifuc.model import ImageItem
    WAIFUC_INSTALLED = True
except ImportError:
    WAIFUC_INSTALLED = False

# ---------------------------------
# 默认的 waifuc 操作代码示例
# ---------------------------------
DEFAULT_ACTION_CODE = """# 这是一个 waifuc 操作链的示例代码。
# 'source' 变量是预设的图像来源，你无需定义它。
# 你需要做的就是不断在 'source' 后面追加 .iter_from(...) 即可。
# 下面的例子是：分割出所有人像，并确保最终每张图只有1个正面。

from waifuc.action import PersonSplitAction, FaceCountAction

# 步骤1: 从源中分割出所有人像
source = PersonSplitAction().iter_from(source)

# 步骤2: 筛选出只有1张脸的图像
source = FaceCountAction(1).iter_from(source)
"""

class WaifucExecutorNode:
    """
    Waifuc 核心执行节点
    一个高度灵活的 waifuc 图像处理节点，支持两种操作模式和自定义处理流程。
    """
    def __init__(self):
        if not WAIFUC_INSTALLED:
            raise ImportError("Waifuc 库或其依赖项未找到。请在 ComfyUI 的 Python 环境中安装它:\n"
                              "1. 打开 ComfyUI 目录下的 'portable_embeded' 或类似文件夹。\n"
                              "2. 在该目录地址栏输入 'cmd' 并回车，打开命令行。\n"
                              "3. 运行: python.exe -m pip install \"waifuc[all]\"")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["流水线模式", "批处理模式"],),
                "action_code": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_ACTION_CODE
                }),
            },
            "optional": {
                "images": ("IMAGE",),
                "input_directory": ("STRING", {"default": "G:\\ComfyUI_windows_portable\\ComfyUI\\input"}),
                "output_directory": ("STRING", {"default": "G:\\ComfyUI_windows_portable\\ComfyUI\\custom_nodes\\Comfyui-waifuc\\Cache"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("processed_images", "success_flag")
    FUNCTION = "execute"
    CATEGORY = "Waifuc"

    def _tensor_to_pil(self, tensor_images):
        return [Image.fromarray(np.clip(255. * i.cpu().numpy(), 0, 255).astype(np.uint8)) for i in tensor_images]

    def _pad_and_convert_to_tensor(self, pil_images):
        if not pil_images:
            return torch.zeros((1, 1, 1, 3), dtype=torch.float32)

        max_width = max(img.width for img in pil_images)
        max_height = max(img.height for img in pil_images)

        padded_tensors = []
        for img in pil_images:
            img_rgb = img.convert("RGB")
            
            padded_img = Image.new("RGB", (max_width, max_height), (0, 0, 0))
            paste_x = (max_width - img_rgb.width) // 2
            paste_y = (max_height - img_rgb.height) // 2
            padded_img.paste(img_rgb, (paste_x, paste_y))
            
            np_img = np.array(padded_img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(np_img).unsqueeze(0)
            padded_tensors.append(img_tensor)

        return torch.cat(padded_tensors, 0)
        
    def execute(self, mode, action_code, images=None, input_directory=None, output_directory=None):
        print(f"Waifuc Executor: 以 '{mode}' 模式启动。")
        
        # --- 准备执行环境 ---
        # 准备一个安全的全局命名空间来执行 action_code
        execution_globals = {
            '__builtins__': __builtins__,
        }
        
        # 动态导入所有 waifuc.action 中可用的类，使其在 action_code 中可直接使用
        try:
            from waifuc.action import __all__ as all_actions
            action_module = __import__('waifuc.action', fromlist=all_actions)
            for action_name in all_actions:
                if hasattr(action_module, action_name):
                    execution_globals[action_name] = getattr(action_module, action_name)
        except (ImportError, AttributeError):
             print("警告：无法动态导入所有 waifuc.action。代码框中可能需要手动 from waifuc.action import ...")


        try:
            if mode == "批处理模式":
                # (批处理模式逻辑保持不变，因为它是正确的)
                if not input_directory or not os.path.isdir(input_directory):
                    raise FileNotFoundError(f"输入目录无效或未提供: {input_directory}")
                if not output_directory:
                    raise ValueError("批处理模式下必须提供输出目录")

                print(f"批处理模式: 从 '{input_directory}' 读取图像。")
                source = LocalSource(input_directory)
                execution_globals['source'] = source

                exec(action_code, execution_globals)
                final_source = execution_globals.get('source')
                
                if os.path.exists(output_directory):
                    shutil.rmtree(output_directory)
                os.makedirs(output_directory, exist_ok=True)
                
                exporter = SaveExporter(output_directory)
                exporter.export_from(final_source)
                
                print(f"批处理模式: 处理完成，文件已保存到 '{output_directory}'。")
                
                output_files = [os.path.join(output_directory, f) for f in os.listdir(output_directory) if os.path.isfile(os.path.join(output_directory, f))]
                processed_pil_images = [Image.open(f).convert("RGB") for f in output_files]
                
                if not processed_pil_images:
                    print("警告: 批处理未产生任何图像。")
                    return (self._pad_and_convert_to_tensor([]), False)
                
                output_tensor = self._pad_and_convert_to_tensor(processed_pil_images)
                return (output_tensor, True)

            # --- 这是修正后的流水线模式逻辑 ---
            elif mode == "流水线模式":
                if images is None:
                    print("警告: 流水线模式需要连接 'images' 输入。")
                    return (self._pad_and_convert_to_tensor([]), False)

                input_pils = self._tensor_to_pil(images)
                print(f"流水线模式: 收到 {len(input_pils)} 张图像。")
                
                # 1. 将输入的PIL图像列表转换为一个 ImageItem 列表
                source_items = [ImageItem(p_img, {'filename': f'input_{i}.png'}) for i, p_img in enumerate(input_pils)]
                
                # 2. 将这个列表作为'source'变量直接传递给执行环境
                execution_globals['source'] = source_items

                # 3. 执行用户代码，代码中的 'source' 就是上面的列表
                exec(action_code, execution_globals)
                final_source = execution_globals.get('source')

                # 4. 从最终的迭代器中收集结果
                processed_pil_images = [item.image for item in final_source]

                if not processed_pil_images:
                    print("警告: 流水线处理后没有图像通过筛选。")
                    return (self._pad_and_convert_to_tensor([]), False)

                print(f"流水线模式: 成功处理得到 {len(processed_pil_images)} 张图像。")
                output_tensor = self._pad_and_convert_to_tensor(processed_pil_images)
                return (output_tensor, True)

        except Exception as e:
            print(f"Waifuc Executor Error:\n{traceback.format_exc()}")
            return (self._pad_and_convert_to_tensor([]), False)
NODE_CLASS_MAPPINGS = {
    "WaifucExecutorNode": WaifucExecutorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaifucExecutorNode": "Waifuc 核心执行器"
}