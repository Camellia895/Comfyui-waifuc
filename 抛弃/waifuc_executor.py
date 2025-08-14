# -*- coding: utf-8 -*-
# 文件名: waifuc_executor.py (V17)
# 新增 Preset Loader 节点，并优化主节点布局。

import torch
import numpy as np
from PIL import Image
import os
import shutil
import traceback
import json

# ---------------------------------
# 预设文件加载器节点 (新功能)
# ---------------------------------
# 获取此文件所在的目录
NODE_DIRECTORY = os.path.dirname(__file__)
PRESET_DIR = os.path.join(NODE_DIRECTORY, "Preset")

# 如果 Preset 目录不存在，则创建它
if not os.path.exists(PRESET_DIR):
    os.makedirs(PRESET_DIR)

def get_preset_files():
    """扫描 Preset 目录并返回 .txt 文件列表"""
    if not os.path.isdir(PRESET_DIR):
        return []
    return [f for f in os.listdir(PRESET_DIR) if f.endswith(".txt")]

class WaifucPresetLoaderNode:
    """
    Waifuc 预设加载器
    从 .../Comfyui-waifuc/Preset/ 目录加载文本预设文件。
    """
    @classmethod
    def INPUT_TYPES(cls):
        # 创建一个“无”选项，以便可以轻松地手动输入代码
        preset_files = ["None"] + get_preset_files()
        return {
            "required": {
                "preset_file": (preset_files, ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("preset_code",)
    FUNCTION = "load_preset"
    CATEGORY = "Waifuc"

    def load_preset(self, preset_file):
        if preset_file == "None":
            return ("",) # 如果选择 "None"，返回空字符串

        file_path = os.path.join(PRESET_DIR, preset_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"Waifuc Preset: 已成功加载 '{preset_file}'。")
            return (content,)
        except Exception as e:
            print(f"Waifuc Preset Error: 加载预设文件 '{preset_file}'失败: {e}")
            return ("",)


# ---------------------------------
# 核心执行器节点 (已修改)
# ---------------------------------

# [V16] 保持详细注释，作为备忘录
DEFAULT_ACTION_CODE_V16 = """# Waifuc 操作代码 (V16 - 带详细备忘录)
# ====================================================
# 参考文档: https://deepghs.github.io/waifuc/main/tutorials-CN/index.html
# 在 '流水线' 和 '批处理' 模式下, source 已由节点自动提供。
# 你可以用 .attach() 将你的处理动作(Action)附加到它上面。
# 最终的处理流程对象必须命名为 'input'。

input = source.attach(
    AlignMinSizeAction(800),
    # PersonSplitAction(),
)
# input.export(SaveExporter(output))
"""

class WaifucExecutorNode:
    """Waifuc 核心执行器 (V17)"""
    def __init__(self):
        self.execution_globals = self._prepare_waifuc_globals()

    def _prepare_waifuc_globals(self):
        # 此部分与 V15 相同，保持不变
        g = {'__builtins__': __builtins__}
        for type_name, import_list, module_path in [
            ("Action", ACTIONS_TO_IMPORT, "waifuc.action"),
            ("Source", SOURCES_TO_IMPORT, "waifuc.source"),
            ("Exporter", EXPORTERS_TO_IMPORT, "waifuc.export")]:
            for class_name in import_list:
                try: g[class_name] = getattr(__import__(module_path, fromlist=[class_name]), class_name)
                except (ImportError, AttributeError): print(f"[Waifuc] Warning: {type_name} '{class_name}' unavailable.")
        try:
            from waifuc.model import ImageItem
            g['ImageItem'] = ImageItem
            from waifuc.source.base import BaseDataSource
            g['BaseDataSource'] = BaseDataSource
        except ImportError: pass
        return g

    @classmethod
    def INPUT_TYPES(cls):
        # [FIX] 重新排序输入项以优化UI布局
        return {
            "required": {
                "mode": (["流水线模式", "批处理模式", "网页爬取模式"], {"default": "流水线模式"}),
                "output_path": ("STRING", {"default": "ComfyUI/output/waifuc_export"}),
            },
            "optional": {
                 "images": ("IMAGE",),
                 "input_directory": ("STRING", {"default": "ComfyUI/input"}),
                 # 将 action_code 移动到 optional 中，使其出现在UI底部
                 # 它仍然可以作为输入连接点，接收来自 Preset Loader 的文本
                 "action_code": ("STRING", {"multiline": True, "default": DEFAULT_ACTION_CODE_V16}),
            }
        }

    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = ("IMAGE", "BOOLEAN", "STRING"), ("processed_images", "success_flag", "padding_data"), "execute", "Waifuc"

    # _pad_and_convert_to_tensor 方法保持不变
    def _pad_and_convert_to_tensor(self, pil_images):
        if not pil_images:
            print("警告: 流水线未产生任何有效图像，将输出一个占位符。")
            return (torch.zeros((1, 1, 1, 3), dtype=torch.float32), "[]")
        max_width, max_height = max(img.width for img in pil_images), max(img.height for img in pil_images)
        tensors, metadata = [], []
        for img in pil_images:
            img_rgb, paste_x, paste_y = img.convert("RGB"), (max_width - img.width) // 2, (max_height - img.height) // 2
            metadata.append({"original_width": img.width, "original_height": img.height, "paste_x": paste_x, "paste_y": paste_y})
            padded_img = Image.new("RGB", (max_width, max_height), (0, 0, 0))
            padded_img.paste(img_rgb, (paste_x, paste_y))
            tensors.append(torch.from_numpy(np.array(padded_img).astype(np.float32) / 255.0).unsqueeze(0))
        return (torch.cat(tensors, 0), json.dumps(metadata))

    # execute 方法的主体逻辑保持不变
    def execute(self, mode, output_path, images=None, input_directory=None, action_code=""):
        # 如果 action_code 为空（例如从空预设加载），则使用默认代码以防出错
        if not action_code.strip():
            action_code = DEFAULT_ACTION_CODE_V16
            print("提示: action_code 为空，已加载默认操作代码。")

        print(f"Waifuc Executor: 以 '{mode}' 模式启动。")
        exec_globals = self.execution_globals.copy()
        
        sanitized_output_path = os.path.normpath(os.path.abspath(output_path))
        exec_globals['output'] = sanitized_output_path
        exec_globals['mode'] = mode
        
        source_obj = None
        if mode == "流水线模式":
            if images is None:
                print("警告: 流水线模式需要连接 'images' 输入。将使用空数据源。")
                source_obj = exec_globals['EmptySource']()
            else:
                class ListSource(exec_globals['BaseDataSource']):
                    def __init__(self, items_list): self._items = items_list
                    def _iter(self): yield from self._items
                pils = [Image.fromarray(np.clip(255. * i.cpu().numpy(), 0, 255).astype(np.uint8)) for i in images]
                image_items = [exec_globals['ImageItem'](p, {'filename': f'input_{i}.png'}) for i, p in enumerate(pils)]
                source_obj = ListSource(image_items)
                print(f"流水线模式: 已加载 {len(pils)} 张上游图像。")
        elif mode == "批处理模式":
            if not input_directory or not os.path.isdir(input_directory): raise FileNotFoundError(f"输入目录无效: {input_directory}")
            source_obj = exec_globals['LocalSource'](input_directory)
            print(f"批处理模式: 已加载来自 '{input_directory}' 的数据源。")
        elif mode == "网页爬取模式":
            print("网页爬取模式: 等待用户代码创建数据源...")
            pass
        
        exec_globals['source'] = source_obj

        try:
            if os.path.exists(sanitized_output_path): shutil.rmtree(sanitized_output_path)
            os.makedirs(sanitized_output_path, exist_ok=True)
            exec(action_code, exec_globals)
            final_obj = exec_globals.get('input') or exec_globals.get('source')
            if final_obj is None: raise ValueError("未能从用户代码中获取 'input' 或 'source' 变量。")
            processed_pil_images = [item.image for item in final_obj]
            print(f"Waifuc 流程执行完毕，成功处理得到 {len(processed_pil_images)} 张图像。")
            output_tensor, padding_data_json = self._pad_and_convert_to_tensor(processed_pil_images)
            return (output_tensor, True, padding_data_json)
        except Exception:
            print("Waifuc Executor 执行代码时发生严重错误:"); print(traceback.format_exc())
            return (torch.zeros((1, 1, 1, 3)), False, "[]")

# WaifucImageUnpadAndSplitNode 保持不变...
class WaifucImageUnpadAndSplitNode:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"padded_images": ("IMAGE",),"padding_data": ("STRING", {"multiline": True, "default":"[]"}),}}
    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = ("IMAGE", "IMAGE", "IMAGE", "BOOLEAN"), ("image_1_largest", "image_2_medium", "image_3_smallest", "success_flag"), "unpad_and_split", "Waifuc"
    def _pil_to_tensor(self, p): return torch.from_numpy(np.array(p.convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0) if p else torch.zeros((1, 1, 1, 3))
    def unpad_and_split(self, padded_images, padding_data):
        try:
            metadata_list = json.loads(padding_data)
            if not metadata_list: return (self._pil_to_tensor(None),)*3 + (True,)
        except (json.JSONDecodeError, TypeError):
             print(f"Waifuc Unpad: 无效的 padding_data: {padding_data}"); return (self._pil_to_tensor(None),)*3 + (False,)
        input_pils = [Image.fromarray(np.clip(255. * i.cpu().numpy(), 0, 255).astype(np.uint8)) for i in padded_images]
        if len(input_pils) != len(metadata_list):
            print(f"Waifuc Unpad: 图像数量 ({len(input_pils)}) 与元数据数量 ({len(metadata_list)}) 不匹配。"); return (self._pil_to_tensor(None),)*3 + (False,)
        cropped_pils = [img.crop((d['paste_x'], d['paste_y'], d['paste_x'] + d['original_width'], d['paste_y'] + d['original_height'])) for img, d in zip(input_pils, metadata_list)]
        cropped_pils.sort(key=lambda img: img.width * img.height, reverse=True)
        outputs = [self._pil_to_tensor(cropped_pils[i] if i < len(cropped_pils) else None) for i in range(3)]
        return tuple(outputs) + (True,)


# ---------------------------------
# 注册所有节点到 ComfyUI
# ---------------------------------
NODE_CLASS_MAPPINGS = {
    "WaifucExecutorNode": WaifucExecutorNode,
    "WaifucPresetLoaderNode": WaifucPresetLoaderNode, # 注册新节点
    "WaifucImageUnpadAndSplitNode": WaifucImageUnpadAndSplitNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaifucExecutorNode": "Waifuc 核心执行器 (V17)",
    "WaifucPresetLoaderNode": "Waifuc 预设加载器", # 为新节点添加显示名称
    "WaifucImageUnpadAndSplitNode": "Waifuc 图像反填充与分割器",
}

# 需要导入的 Action 和 Source 列表保持不变
ACTIONS_TO_IMPORT = ['AlignMaxSizeAction', 'AlignMinSizeAction', 'PaddingAlignAction', 'AlignMaxAreaAction', 'RandomFilenameAction', 'RandomChoiceAction', 'MirrorAction', 'CharacterEnhanceAction', 'BackgroundRemovalAction', 'ModeConvertAction', 'CCIPAction', 'SliceSelectAction', 'FirstNSelectAction', 'ArrivalAction', 'FileExtAction', 'FileOrderAction', 'NoMonochromeAction', 'OnlyMonochromeAction', 'ClassFilterAction', 'RatingFilterAction', 'FaceCountAction', 'HeadCountAction', 'PersonRatioAction', 'MinSizeFilterAction', 'MinAreaFilterAction', 'FrameSplitAction', 'HeadCoverAction', 'HeadCutOutAction', 'FilterSimilarAction', 'SafetyAction', 'PersonSplitAction', 'ThreeStageSplitAction', 'TaggingAction', 'TagFilterAction', 'TagOverlapDropAction', 'TagDropAction', 'BlacklistedTagDropAction', 'TagRemoveUnderlineAction']
SOURCES_TO_IMPORT = ['ATFBooruSource', 'AnimePicturesSource', 'DanbooruSource', 'DerpibooruSource', 'DuitangSource', 'E621Source', 'E926Source', 'FurbooruSource', 'GelbooruSource', 'Huashi6Source', 'HypnoHubSource', 'KonachanNetSource', 'KonachanSource', 'LolibooruSource', 'PahealSource', 'PixivRankingSource', 'PixivSearchSource', 'PixivUserSource', 'Rule34Source', 'SafebooruOrgSource', 'SafebooruSource', 'SankakuSource', 'TBIBSource', 'WallHavenSource', 'XbooruSource', 'YandeSource', 'ZerochanSource', 'LocalSource', 'EmptySource']
EXPORTERS_TO_IMPORT = ['SaveExporter', 'TextualInversionExporter']