# 文件名: waifuc_executor_node_pure.py (纯净版)
# 描述: 按照用户的明确要求编写。所有 Action 在文件顶部直接导入，无任何动态检查或 try-except 块。
#      这要求运行环境必须已完整安装所有相关依赖。

import torch
import numpy as np
from PIL import Image
import os
import shutil
import json
import time
import glob

# --------------------------------------------------------------------------------
# 核心依赖: 按照官方示例风格，直接、明确地导入所有 Action
# --------------------------------------------------------------------------------
from waifuc.action import (
    NoMonochromeAction, SimilarFilterAction, TaggingAction, PersonSplitAction,
    FirstNSelectAction, CCIPAction, ModeConvertAction, ClassFilterAction,
    RandomFilenameAction, AlignMinSizeAction, HeadCropAction, FaceCropAction,
    ThreeStageSplitAction, MinSizeFilterAction, MaxSizeFilterAction, HeadCountAction,
    RandomSelectAction
)
from waifuc.model import ImageItem


# --------------------------------------------------------------------------------
# 静态功能清单 (Action Mapping)
# --------------------------------------------------------------------------------
# 直接根据上面成功导入的类来创建功能字典。
ACTION_MAPPING = {
    'NoMonochromeAction': NoMonochromeAction,
    'SimilarFilterAction': SimilarFilterAction,
    'TaggingAction': TaggingAction,
    'PersonSplitAction': PersonSplitAction,
    'FirstNSelectAction': FirstNSelectAction,
    'CCIPAction': CCIPAction,
    'ModeConvertAction': ModeConvertAction,
    'ClassFilterAction': ClassFilterAction,
    'RandomFilenameAction': RandomFilenameAction,
    'AlignMinSizeAction': AlignMinSizeAction,
    'HeadCropAction': HeadCropAction,
    'FaceCropAction': FaceCropAction,
    'ThreeStageSplitAction': ThreeStageSplitAction,
    'MinSizeFilterAction': MinSizeFilterAction,
    'MaxSizeFilterAction': MaxSizeFilterAction,
    'HeadCountAction': HeadCountAction,
    'RandomSelectAction': RandomSelectAction,
}

# --------------------------------------------------------------------------------
# 工具函数 (带填充功能)
# --------------------------------------------------------------------------------
def tensor_to_pil(tensor_images):
    return [Image.fromarray(np.clip(255. * i.cpu().numpy(), 0, 255).astype(np.uint8)) for i in tensor_images]

def pil_to_tensor(pil_images):
    valid_images = [img for img in pil_images if img is not None]
    if not valid_images: return torch.empty(0, 64, 64, 3)
    rgb_images = [img.convert('RGB') for img in valid_images]
    max_width = max(img.width for img in rgb_images)
    max_height = max(img.height for img in rgb_images)
    padded_tensors = []
    for img in rgb_images:
        padded_img = Image.new('RGB', (max_width, max_height), (0, 0, 0))
        paste_x, paste_y = (max_width - img.width) // 2, (max_height - img.height) // 2
        padded_img.paste(img, (paste_x, paste_y))
        tensor_img = torch.from_numpy(np.array(padded_img).astype(np.float32) / 255.0)
        padded_tensors.append(tensor_img)
    return torch.stack(padded_tensors)


# --------------------------------------------------------------------------------
# 主节点类
# --------------------------------------------------------------------------------
class WaifucExecutorNodePure:
    def __init__(self):
        self.log_messages = []

    @classmethod
    def INPUT_TYPES(cls):
        NODE_DIR = os.path.dirname(os.path.abspath(__file__))
        CACHE_DIR = os.path.join(NODE_DIR, 'Cache')
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            subfolders = [d for d in os.listdir(CACHE_DIR) if os.path.isdir(os.path.join(CACHE_DIR, d))]
        except Exception: subfolders = []
        
        CREATE_NEW_FOLDER_ID = "[ 新建文件夹 ]"
        ANY_TYPE_LIST = ["IMAGE", "MASK", "LATENT", "MODEL", "CLIP", "VAE", "CONDITIONING", "CONTROL_NET", "STRING", "INT", "FLOAT", "BOOLEAN", "STYLE_MODEL"]

        return {
            "required": {
                "actions_config": ("STRING", {"multiline": True, "default": '{\n    "actions": []\n}'}),
                "mode_selection": (["批处理模式 (文件夹)", "流水线模式 (图像)"],),
            },
            "optional": {
                "input_folder_name": (subfolders if subfolders else ["无可用文件夹"],),
                "output_folder_name": (subfolders + [CREATE_NEW_FOLDER_ID],),
                "new_output_folder_name": ("STRING", {"default": "new_dataset"}),
                "clear_output_folder": ("BOOLEAN", {"default": True}),
                "input_image": ("IMAGE",),
                "trigger": (ANY_TYPE_LIST,), 
            }
        }
    
    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING", "STRING", "INT")
    RETURN_NAMES = ("输出图像", "处理成功", "日志", "输出路径", "图像数量")
    FUNCTION = "execute"
    CATEGORY = "Waifuc (汉化)"

    def _log(self, message, is_error=False):
        prefix = "错误: " if is_error else ""
        timestamp = time.strftime("%H:%M:%S")
        msg = f"[{timestamp}] {prefix}{message}"
        print(f"Waifuc执行节点: {msg}")
        self.log_messages.append(msg)
        
    def _parse_and_run_actions(self, image_items, config_str):
        try:
            config = json.loads(config_str)
            actions_config = config.get("actions", [])
        except json.JSONDecodeError as e:
            self._log(f"动作配置不是有效的JSON格式。 {e}", is_error=True)
            return None
        if not actions_config:
            self._log("警告: 动作配置为空，将直接返回原始图像。")
            return [item.image for item in image_items]
        self._log(f"开始执行 {len(actions_config)} 个动作...")
        processed_iter = iter(image_items)
        for i, action_conf in enumerate(actions_config):
            action_name = action_conf.get("name")
            action_params = action_conf.get("params", {})
            self._log(f"  -> 步骤 {i+1}: 应用动作 '{action_name}'")
            action_class = ACTION_MAPPING.get(action_name)
            if not action_class:
                self._log(f"未知的动作名称 '{action_name}'。请检查 JSON 配置中的拼写是否与导入列表一致。", is_error=True)
                return None
            try:
                action_instance = action_class(**action_params)
                processed_iter = action_instance.iter_from(processed_iter)
            except Exception as e:
                self._log(f"执行动作 '{action_name}' 时失败。参数: {action_params}。原因: {e}", is_error=True)
                return None
        final_pils = [item.image for item in processed_iter]
        self._log("所有动作执行完毕。")
        return final_pils

    def execute(self, **kwargs):
        self.log_messages = []
        # (完整的 execute 方法逻辑，与之前版本相同)
        actions_config = kwargs.get('actions_config')
        mode_selection = kwargs.get('mode_selection')
        is_pipeline_mode = "流水线模式" in mode_selection
        
        if is_pipeline_mode:
            self._log("进入流水线模式。")
            input_image = kwargs.get('input_image')
            if input_image is None:
                self._log("流水线模式下 'input_image' 是必须的。", is_error=True)
                return (torch.empty(0, 64, 64, 3), False, "\n".join(self.log_messages), "N/A", 0)
            input_pils = tensor_to_pil(input_image)
        else:
            self._log("进入批处理模式。")
            input_folder_name = kwargs.get('input_folder_name')
            if not input_folder_name or input_folder_name == "无可用文件夹":
                self._log("必须选择一个有效的输入文件夹。", is_error=True)
                return (torch.empty(0, 64, 64, 3), False, "\n".join(self.log_messages), "N/A", 0)
            NODE_DIR = os.path.dirname(os.path.abspath(__file__))
            CACHE_DIR = os.path.join(NODE_DIR, 'Cache')
            input_path = os.path.join(CACHE_DIR, input_folder_name)
            image_paths = glob.glob(os.path.join(input_path, '*.*'))
            input_pils = [Image.open(p) for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            self._log(f"从 '{input_path}' 加载了 {len(input_pils)} 张图像。")

        image_items = [ImageItem(image=img) for img in input_pils]
        processed_pils = self._parse_and_run_actions(image_items, actions_config)
        
        if processed_pils is None:
            return (torch.empty(0, 64, 64, 3), False, "\n".join(self.log_messages), "N/A", 0)

        num_processed = len(processed_pils)
        self._log(f"处理完成，共生成 {num_processed} 张图像。")

        if is_pipeline_mode:
            output_tensor = pil_to_tensor(processed_pils)
            return (output_tensor, True, "\n".join(self.log_messages), "流水线模式无输出路径", num_processed)
        else:
            output_folder_name = kwargs.get('output_folder_name')
            new_output_folder_name = kwargs.get('new_output_folder_name')
            NODE_DIR = os.path.dirname(os.path.abspath(__file__))
            CACHE_DIR = os.path.join(NODE_DIR, 'Cache')
            final_output_folder_name = new_output_folder_name.strip() if output_folder_name == "[ 新建文件夹 ]" else output_folder_name
            output_path = os.path.join(CACHE_DIR, final_output_folder_name)
            os.makedirs(output_path, exist_ok=True)
            if kwargs.get('clear_output_folder'):
                shutil.rmtree(output_path)
                os.makedirs(output_path)
            for i, pil_image in enumerate(processed_pils):
                pil_image.save(os.path.join(output_path, f"image_{i+1:05d}.png"))
            return (torch.empty(0, 64, 64, 3), True, "\n".join(self.log_messages), output_path, num_processed)

# --------------------------------------------------------------------------------
# ComfyUI 节点映射
# --------------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "WaifucExecutorNode_Pure": WaifucExecutorNodePure
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaifucExecutorNode_Pure": "Waifuc 执行节点 (纯净版)"
}