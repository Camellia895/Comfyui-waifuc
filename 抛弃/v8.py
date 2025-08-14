# -*- coding: utf-8 -*-
# 文件名: waifuc_executor.py (V8 - 终极修正版)
# 导入列表完全基于用户提供的、经验证的正确模块列表。

import torch
import numpy as np
from PIL import Image
import os
import shutil
import traceback
import json

# ======================================================================
# [核心] 基于用户反馈修正的、正确的 Action 列表
# ======================================================================
ACTIONS_TO_IMPORT = [
    'AlignMaxSizeAction', 'AlignMinSizeAction', 'PaddingAlignAction', 'AlignMaxAreaAction',
    'RandomFilenameAction', 'RandomChoiceAction', 'MirrorAction', 'CharacterEnhanceAction',
    'BackgroundRemovalAction',
    'ModeConvertAction', 'CCIPAction', 'SliceSelectAction', 'FirstNSelectAction',
    'ArrivalAction', 'FileExtAction', 'FileOrderAction',
    'NoMonochromeAction', 'OnlyMonochromeAction', 'ClassFilterAction', 'RatingFilterAction',
    'FaceCountAction', 'HeadCountAction', 'PersonRatioAction', 'MinSizeFilterAction',
    'MinAreaFilterAction', 'FrameSplitAction', 'HeadCoverAction', 'HeadCutOutAction',
    'FilterSimilarAction', 'SafetyAction', 'PersonSplitAction', 'ThreeStageSplitAction',
    'TaggingAction', 'TagFilterAction', 'TagOverlapDropAction', 'TagDropAction',
    'BlacklistedTagDropAction', 'TagRemoveUnderlineAction'
]

# 用户提供的 Source 列表
SOURCES_TO_IMPORT = [
    'ATFBooruSource', 'AnimePicturesSource', 'DanbooruSource', 'DerpibooruSource',
    'DuitangSource', 'E621Source', 'E926Source', 'FurbooruSource', 'GelbooruSource',
    'Huashi6Source', 'HypnoHubSource', 'KonachanNetSource', 'KonachanSource',
    'LolibooruSource', 'PahealSource', 'PixivRankingSource', 'PixivSearchSource',
    'PixivUserSource', 'Rule34Source', 'SafebooruOrgSource', 'SafebooruSource',
    'SankakuSource', 'TBIBSource', 'WallHavenSource', 'XbooruSource', 'YandeSource',
    'ZerochanSource', 'LocalSource', 'EmptySource'
]

# ======================================================================
# [核心] 使用修正后的 Action 重新编写的默认代码
# ======================================================================
DEFAULT_ACTION_CODE_V8 = """# Waifuc 操作代码 (V8 - 终极修正版)
# 所有可用的 Actions 和 Sources 已被自动预导入。
# 你无需手动编写 import 语句。
# ====================================================

# 步骤 1: 定义数据源 (仅限“网页爬取模式”)
source = DanbooruSource(['genshin_impact', '1girl', 'solo', 'highres'])

# 步骤 2: 组合处理链
# 为了演示，我们使用一些你已验证存在的 Action
source = FirstNSelectAction(10).iter_from(source)
source = MinSizeFilterAction(800, 800).iter_from(source)
source = PersonSplitAction().iter_from(source)
source = FaceCountAction(1).iter_from(source)
# 注意: ScoreFilterAction 不存在, 如果需要按评分筛选, 请使用 RatingFilterAction
# 例如: source = RatingFilterAction('s').iter_from(source) # 's' for safe
"""

# ======================================================================
# 节点 1: Waifuc 核心执行器 (V8)
# ======================================================================
class WaifucExecutorNode:
    """Waifuc 核心执行器 (V8) - 导入列表已根据用户环境修正"""
    def __init__(self):
        self.execution_globals = self._prepare_waifuc_globals()

    def _prepare_waifuc_globals(self):
        g = {'__builtins__': __builtins__}
        for module_type, import_list, module_path in [
            ("Action", ACTIONS_TO_IMPORT, "waifuc.action"),
            ("Source", SOURCES_TO_IMPORT, "waifuc.source")
        ]:
            for class_name in import_list:
                try:
                    module = __import__(module_path, fromlist=[class_name])
                    g[class_name] = getattr(module, class_name)
                except (ImportError, AttributeError):
                    print(f"[Waifuc Executor] Warning: {module_type} '{class_name}' not found and will be unavailable.")
        try: from waifuc.model import ImageItem; g['ImageItem'] = ImageItem
        except: pass
        return g

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["网页爬取模式", "流水线模式", "批处理模式"],),
                "action_code": ("STRING", {"multiline": True, "default": DEFAULT_ACTION_CODE_V8}),
            },
            "optional": {"images": ("IMAGE",), "input_directory": ("STRING", {"default": "ComfyUI/input"}), "output_directory": ("STRING", {"default": "ComfyUI/output/waifuc_cache"}),}
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING")
    RETURN_NAMES = ("processed_images", "success_flag", "padding_data")
    FUNCTION = "execute"
    CATEGORY = "Waifuc"

    def _pad_and_convert_to_tensor(self, pil_images):
        if not pil_images: return (torch.zeros((1, 1, 1, 3), dtype=torch.float32), "[]")
        max_width, max_height = max(img.width for img in pil_images), max(img.height for img in pil_images)
        tensors, metadata = [], []
        for img in pil_images:
            img_rgb, paste_x, paste_y = img.convert("RGB"), (max_width - img.width) // 2, (max_height - img.height) // 2
            metadata.append({"original_width": img.width, "original_height": img.height, "paste_x": paste_x, "paste_y": paste_y})
            padded_img = Image.new("RGB", (max_width, max_height), (0, 0, 0))
            padded_img.paste(img_rgb, (paste_x, paste_y))
            tensors.append(torch.from_numpy(np.array(padded_img).astype(np.float32) / 255.0).unsqueeze(0))
        return (torch.cat(tensors, 0), json.dumps(metadata))

    def execute(self, mode, action_code, images=None, input_directory=None, output_directory=None):
        print(f"Waifuc Executor: 以 '{mode}' 模式启动。")
        exec_globals = self.execution_globals.copy()
        
        try:
            source_obj = None
            if mode == "流水线模式" and images is not None:
                pils = [Image.fromarray(np.clip(255. * i.cpu().numpy(), 0, 255).astype(np.uint8)) for i in images]
                exec_globals['source'] = [exec_globals['ImageItem'](p, {'filename': f'input_{i}.png'}) for i, p in enumerate(pils)]
            elif mode == "批处理模式":
                if not input_directory or not os.path.isdir(input_directory): raise FileNotFoundError(f"输入目录无效: {input_directory}")
                exec_globals['source'] = exec_globals['LocalSource'](input_directory)

            exec(action_code, exec_globals)
            source_obj = exec_globals.get('source')
            if source_obj is None: raise ValueError("未能从用户代码中获取 'source' 变量。")

            print("正在处理图像...")
            processed_pil_images = [item.image for item in source_obj]

            if not processed_pil_images:
                print("警告: 未产生任何有效图像。"); return (torch.zeros((1, 1, 1, 3)), False, "[]")

            print(f"成功处理得到 {len(processed_pil_images)} 张图像。")
            output_tensor, padding_data_json = self._pad_and_convert_to_tensor(processed_pil_images)
            return (output_tensor, True, padding_data_json)

        except Exception:
            print("Waifuc Executor 执行代码时发生严重错误:"); print(traceback.format_exc())
            return (torch.zeros((1, 1, 1, 3)), False, "[]")


# ======================================================================
# 节点 2: Waifuc 图像反填充与分割器 (保持不变)
# ======================================================================
class WaifucImageUnpadAndSplitNode:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"padded_images": ("IMAGE",),"padding_data": ("STRING", {"multiline": True}),}}
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image_1_largest", "image_2_medium", "image_3_smallest", "success_flag")
    FUNCTION = "unpad_and_split"
    CATEGORY = "Waifuc"
    def _pil_to_tensor(self, p): return torch.from_numpy(np.array(p.convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0) if p else torch.zeros((1, 1, 1, 3))
    def unpad_and_split(self, padded_images, padding_data):
        try: metadata_list = json.loads(padding_data)
        except: return (self._pil_to_tensor(None),)*3 + (False,)
        input_pils = [Image.fromarray(np.clip(255. * i.cpu().numpy(), 0, 255).astype(np.uint8)) for i in padded_images]
        if len(input_pils) != len(metadata_list): return (self._pil_to_tensor(None),)*3 + (False,)
        cropped_pils = [img.crop((d['paste_x'], d['paste_y'], d['paste_x'] + d['original_width'], d['paste_y'] + d['original_height'])) for img, d in zip(input_pils, metadata_list)]
        cropped_pils.sort(key=lambda img: img.width * img.height, reverse=True)
        outputs = [self._pil_to_tensor(cropped_pils[i] if i < len(cropped_pils) else None) for i in range(3)]
        return tuple(outputs) + (True,)

# ======================================================================
# 注册所有节点
# ======================================================================
NODE_CLASS_MAPPINGS = {"WaifucExecutorNodev8": WaifucExecutorNode, "WaifucImageUnpadAndSplitNode": WaifucImageUnpadAndSplitNode}
NODE_DISPLAY_NAME_MAPPINGS = {"WaifucExecutorNode": "Waifuc 核心执行器 (V8-终极版)", "WaifucImageUnpadAndSplitNode": "Waifuc 图像反填充与分割器"}