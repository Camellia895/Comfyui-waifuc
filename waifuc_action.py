# action.py
import torch
import numpy as np
from PIL import Image, ImageDraw
import comfy.model_management
import tempfile
import os
from typing import List
import random
from waifuc.source import LocalSource
from waifuc.action.ccip import ccip_default_threshold # 用于获取默认阈值
# ================================================================
# Waifuc 核心模块导入
# ================================================================
from waifuc.action import (
    ActionStop,
    AlignMaxAreaAction,
    AlignMaxSizeAction,
    AlignMinSizeAction,
    ArrivalAction,
    BackgroundRemovalAction,
    BaseAction,
    BaseRandomAction,
    BlacklistedTagDropAction,
    CCIPAction,
    CharacterEnhanceAction,
    ClassFilterAction,
    FaceCountAction,
    FileExtAction,
    FileOrderAction,
    FilterAction,
    FilterSimilarAction,
    FirstNSelectAction,
    FrameSplitAction,
    HeadCountAction,
    HeadCoverAction,
    HeadCutOutAction,
    MinAreaFilterAction,
    MinSizeFilterAction,
    MirrorAction,
    ModeConvertAction,
    NoMonochromeAction,
    OnlyMonochromeAction,
    PaddingAlignAction,
    PersonRatioAction,
    PersonSplitAction,
    ProcessAction,
    ProgressBarAction,
    RandomChoiceAction,
    RandomFilenameAction,
    RatingFilterAction,
    SafetyAction,
    SliceSelectAction,
    TagDropAction,
    TagFilterAction,
    TagOverlapDropAction,
    TagRemoveUnderlineAction,
    TaggingAction,
    ThreeStageSplitAction
)
from waifuc.model import ImageItem
from waifuc.source import LocalSource

# ================================================================
# 通用辅助模块
# ================================================================
VALID_FACE_MODELS = [
    'face_detect_v1.4_s', 'face_detect_v1.4_n',
    'face_detect_v1.3_s', 'face_detect_v1.3_n',
    'face_detect_v1.2_s',
    'face_detect_v1.1_s', 'face_detect_v1.1_n',
    'face_detect_v1_s', 'face_detect_v1_n',
    'face_detect_v0_s', 'face_detect_v0_n',
]

class ComfyInterruptAction(ProcessAction):
    """在 waifuc 流水线中检查 ComfyUI 的中断信号"""
    def process(self, item: ImageItem) -> ImageItem:
        comfy.model_management.throw_exception_if_processing_interrupted()
        return item

class WaifucActionHelper:
    """
    通用辅助类，封装了所有 Action 节点的共同逻辑。
    将 ComfyUI 的 Tensor 输入转换为 waifuc 可处理的 LocalSource，
    执行指定的 Action，然后将结果转换回 Tensor。
    """
    @staticmethod
    def process_with_actions(images: torch.Tensor, actions: List[ProcessAction]):
        """
        核心处理流程: Tensor -> 临时文件 -> LocalSource -> Pipeline -> PIL -> Tensor
        """
        # 1. 将输入的 Tensor 转换为 PIL 图像列表
        input_pil_images = [
            Image.fromarray((images[i].cpu().numpy() * 255).astype(np.uint8))
            for i in range(images.shape[0])
        ]

        processed_pil_images = []
        with tempfile.TemporaryDirectory() as temp_dir:
            # 2. 将 PIL 图像保存到临时目录
            for i, pil_img in enumerate(input_pil_images):
                pil_img.save(os.path.join(temp_dir, f"image_{i:04d}.png"), "PNG")

            # 3. 使用临时目录初始化 LocalSource
            source = LocalSource(temp_dir)
            
            # 4. 构建并执行流水线
            # 在用户指定的 action 前后加入必要的通用 action
            full_actions = [ComfyInterruptAction(), *actions]
            pipeline = source.attach(*full_actions)
            processed_pil_images = [item.image for item in pipeline]

        # 5. 处理空输出，返回占位符
        if not processed_pil_images:
            print("Waifuc Action 警告: 没有图像通过筛选，已输出1x1黑色占位图。")
            placeholder_img = Image.new('RGB', (1, 1), 'black')
            tensors = [torch.from_numpy(np.array(placeholder_img).astype(np.float32) / 255.0)[None,]]
        else:
            # 6. 将处理后的 PIL 图像转换回 Tensor
            tensors = [
                torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]
                for img in processed_pil_images
            ]
            
        return (tensors,)
#ccip


class WaifucCCIPNode:
    DISPLAY_NAME = "Waifuc AI角色校验器 (CCIP)"
    CATEGORY = "Waifuc/Waifuc筛选"
    FUNCTION = "filter_character"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    def __init__(self):
        # 初始化时，不创建 Action 实例，因为需要等待运行时的参数
        self.action_instance: CCIPAction | None = None

    @classmethod
    def INPUT_TYPES(cls):
        # 获取支持的 CCIP 模型列表
        try:
            from imgutils.metrics.ccip import CCIP_MODELS
            model_list = CCIP_MODELS
        except (ImportError, ModuleNotFoundError):
            model_list = ['ccip-caformer-24-randaug-pruned'] # 提供一个备用默认值

        return {
            "required": {
                "images": ("IMAGE",),
                "model": (model_list, {"default": 'ccip-caformer-24-randaug-pruned'}),
                "threshold": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01, "label": "相似度阈值"}),
                "force_reset": ("BOOLEAN", {"default": False, "label_on": "立即重置学习状态", "label_off": "保持学习状态"}),
            },
            "optional": {
                "anchor_images": ("IMAGE", {"label": "锚点图像 (可选)"}),
                "ADVANCED_OPTIONS": ("*", {"label": "显示高级参数", "visible": False}), # 占位符
                "min_val_count": ("INT", {"default": 15, "min": 2, "max": 100, "label": "最小学习数量"}),
                "cmp_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "label": "簇内比较阈值"}),
            }
        }

    def filter_character(self, images: torch.Tensor, model: str, threshold: float, force_reset: bool, 
                         min_val_count: int, cmp_threshold: float, anchor_images: torch.Tensor = None):
        
        # 1. 重置逻辑
        if force_reset:
            if self.action_instance:
                print("WaifucCCIP: 强制重置学习状态。")
                self.action_instance.reset()
            self.action_instance = None # 强制在下一次运行时重新创建实例
        
        # 检查参数是否变化，如果变化也需要重置
        if self.action_instance:
            if (self.action_instance.model != model or 
                self.action_instance.threshold != threshold or
                self.action_instance.min_val_count != min_val_count or
                self.action_instance.cmp_threshold != cmp_threshold):
                print("WaifucCCIP: 参数变更，重置学习状态。")
                self.action_instance.reset()
                self.action_instance = None

        # 2. 初始化 Action 实例 (仅在首次运行或重置后执行)
        if self.action_instance is None:
            print("WaifucCCIP: 正在初始化新的角色学习实例...")
            init_source = None
            temp_dir_manager = None # 用于管理临时目录的生命周期
            
            # 如果连接了锚点图像，则创建 LocalSource
            if anchor_images is not None and anchor_images.shape[0] > 0:
                print(f"WaifucCCIP: 检测到 {anchor_images.shape[0]} 张锚点图像，将用于预学习。")
                temp_dir_manager = tempfile.TemporaryDirectory()
                temp_dir = temp_dir_manager.name
                
                anchor_pil_images = [Image.fromarray((anchor_images[i].cpu().numpy() * 255).astype(np.uint8)) for i in range(anchor_images.shape[0])]
                for i, pil_img in enumerate(anchor_pil_images):
                    pil_img.save(os.path.join(temp_dir, f"anchor_{i:04d}.png"), "PNG")
                
                init_source = LocalSource(temp_dir)
            
            # 创建 Action 实例
            self.action_instance = CCIPAction(
                init_source=init_source,
                model=model,
                threshold=threshold or ccip_default_threshold(model), # 如果用户未指定，则使用模型默认值
                min_val_count=min_val_count,
                cmp_threshold=cmp_threshold
            )
            # 注意：如果使用了临时目录，我们不能在这里关闭它，因为 Action 可能还没有读取完。
            # 这是一个简化处理，在实际应用中可能需要更复杂的生命周期管理。
            # 但对于 ComfyUI 的单次工作流执行来说，这通常是可行的。
        
        # 3. 逐个处理输入图像
        input_pil_images = [Image.fromarray((images[i].cpu().numpy() * 255).astype(np.uint8)) for i in range(images.shape[0])]
        output_pil_images = []

        print(f"WaifucCCIP: 当前状态: {self.action_instance.status.name}. "
              f"已学习 {len(self.action_instance.feats)} 个特征。")

        for pil_img in input_pil_images:
            comfy.model_management.throw_exception_if_processing_interrupted()
            item = ImageItem(pil_img)
            
            # 调用 action.iter() 并收集所有产出的图像
            for processed_item in self.action_instance.iter(item):
                output_pil_images.append(processed_item.image)
        
        print(f"WaifucCCIP: 本次输入 {len(input_pil_images)} 张，输出 {len(output_pil_images)} 张。")

        # 4. 格式化输出
        if not output_pil_images:
            # 在学习阶段，返回一个空的 IMAGE 张量，而不是占位符，以免污染下游
            return (torch.zeros((0, 1, 1, 3)),) 

        tensors = [torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,] for img in output_pil_images]
        return (torch.cat(tensors, dim=0),)
    
# ================================================================
# 图像处理节点 (Processing Nodes)
# ================================================================

class WaifucModeConvertNode:
    DISPLAY_NAME = "Waifuc 图像模式转换"
    CATEGORY = "Waifuc/Waifuc处理"
    FUNCTION = "convert_mode"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (['RGB', 'RGBA', 'L'], {"default": 'RGB'}),
                "background": ("STRING", {"default": "white", "label": "背景色 (例如: white, black, #RRGGBB)"}),
            }
        }

    def convert_mode(self, images: torch.Tensor, mode: str, background: str):
        actions = [ModeConvertAction(mode, background)]
        return WaifucActionHelper.process_with_actions(images, actions)

class WaifucAlignMinSizeNode:
    DISPLAY_NAME = "Waifuc 对齐最小边"
    CATEGORY = "Waifuc/Waifuc处理"
    FUNCTION = "align_size"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "min_size": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 64, "label": "最小边尺寸"}),
            }
        }

    def align_size(self, images: torch.Tensor, min_size: int):
        actions = [AlignMinSizeAction(min_size)]
        return WaifucActionHelper.process_with_actions(images, actions)


# ================================================================
# 图像筛选节点 (Filter Nodes)
# ================================================================


class WaifucOnlyMonochromeNode:
    DISPLAY_NAME = "Waifuc 仅保留单色图像"
    CATEGORY = "Waifuc/Waifuc筛选"
    FUNCTION = "filter_monochrome"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return { "required": { "images": ("IMAGE",) } }

    def filter_monochrome(self, images: torch.Tensor):
        return WaifucActionHelper.process_with_actions(images, [OnlyMonochromeAction()])

class WaifucRatingFilterNode:
    DISPLAY_NAME = "Waifuc 按评级过滤#safe, r15, r18"
    CATEGORY = "Waifuc/Waifuc筛选"
    FUNCTION = "filter_rating"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "allowed_ratings": ("STRING", {"default": "safe", "label": "允许的评级 (safe, r15, r18)"}),
                "threshold": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05, "label": "置信度阈值"}),
            }
        }
    
    def filter_rating(self, images: torch.Tensor, allowed_ratings: str, threshold: float):
        rating_list = [r.strip() for r in allowed_ratings.split(',') if r.strip()]
        action = RatingFilterAction(rating_list, threshold=threshold)
        return WaifucActionHelper.process_with_actions(images, [action])

class WaifucHeadCountNode:
    DISPLAY_NAME = "Waifuc 按头部数量过滤"
    CATEGORY = "Waifuc/Waifuc筛选"
    FUNCTION = "filter_head_count"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "min_count": ("INT", {"default": 1, "min": 0, "max": 20, "step": 1, "label": "最少头部数"}),
                "max_count": ("INT", {"default": 1, "min": 0, "max": 20, "step": 1, "label": "最多头部数 (0为不限)"}),
                "conf_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "label": "置信度阈值"}),
            }
        }

    def filter_head_count(self, images: torch.Tensor, min_count: int, max_count: int, conf_threshold: float):
        max_count_or_none = max_count if max_count > 0 else None
        action = HeadCountAction(
            min_count=min_count,
            max_count=max_count_or_none,
            conf_threshold=conf_threshold
        )
        return WaifucActionHelper.process_with_actions(images, [action])

class WaifucPersonRatioNode:
    DISPLAY_NAME = "Waifuc 按人物占比过滤"
    CATEGORY = "Waifuc/Waifuc筛选"
    FUNCTION = "filter_person_ratio"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "min_ratio": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05, "label": "最小人物面积占比"}),
                "conf_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "label": "置信度阈值"}),
            }
        }

    def filter_person_ratio(self, images: torch.Tensor, min_ratio: float, conf_threshold: float):
        action = PersonRatioAction(ratio=min_ratio, conf_threshold=conf_threshold)
        return WaifucActionHelper.process_with_actions(images, [action])

class WaifucMinSizeFilterNode:
    DISPLAY_NAME = "Waifuc 按最小边过滤"
    CATEGORY = "Waifuc/Waifuc筛选"
    FUNCTION = "filter_min_size"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "min_size": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 64, "label": "最小边长 (像素)"}),
            }
        }

    def filter_min_size(self, images: torch.Tensor, min_size: int):
        action = MinSizeFilterAction(min_size)
        return WaifucActionHelper.process_with_actions(images, [action])

class WaifucMinAreaFilterNode:
    DISPLAY_NAME = "Waifuc 按最小面积过滤"
    CATEGORY = "Waifuc/Waifuc筛选"
    FUNCTION = "filter_min_area"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "min_equivalent_size": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 64, "label": "最小等效尺寸"}),
            }
        }
    
    def filter_min_area(self, images: torch.Tensor, min_equivalent_size: int):
        # 源码中使用 min_size 参数名
        action = MinAreaFilterAction(min_size=min_equivalent_size)
        return WaifucActionHelper.process_with_actions(images, [action])

class WaifucFilterSimilarNode:
    DISPLAY_NAME = "Waifuc 过滤相似图像"
    CATEGORY = "Waifuc/Waifuc筛选"
    FUNCTION = "filter_similar"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    def __init__(self):
        # 在节点初始化时，创建一个 Action 实例并持有它
        self.action = FilterSimilarAction(mode='all')

    @classmethod
    def INPUT_TYPES(cls):
        # 增加一个 force_reset 开关，用于手动清空过滤器的记忆
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (['all', 'group'], {"default": 'all'}),
                "threshold": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                "force_reset": ("BOOLEAN", {"default": False, "label_on": "立即重置", "label_off": "不重置"}),
            }
        }

    def filter_similar(self, images: torch.Tensor, mode: str, threshold: float, force_reset: bool):
        # 每次执行时，检查用户设置的参数是否与持有的 Action 实例参数一致
        # 或者用户是否强制要求重置
        if force_reset or self.action.mode != mode or self.action.threshold != threshold:
            print("WaifucFilterSimilar: 参数变更或强制重置，正在重新创建/重置 Action。")
            # 如果不一致或需要重置，则更新或重新创建 Action 实例
            # 并调用其 reset 方法清空内部的特征桶
            self.action = FilterSimilarAction(mode=mode, threshold=threshold)
            self.action.reset()

        # 使用节点持有的、有记忆的 action 实例去处理图像
        return WaifucActionHelper.process_with_actions(images, [self.action])


class WaifucNoMonochromeNode:
    DISPLAY_NAME = "Waifuc 过滤单色图像"
    CATEGORY = "Waifuc/Waifuc筛选"
    FUNCTION = "filter_monochrome"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return { "required": { "images": ("IMAGE",) } }

    def filter_monochrome(self, images: torch.Tensor):
        actions = [NoMonochromeAction()]
        return WaifucActionHelper.process_with_actions(images, actions)

class WaifucClassFilterNode:
    DISPLAY_NAME = "Waifuc 按分类过滤"
    CATEGORY = "Waifuc/Waifuc筛选"
    FUNCTION = "filter_class"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "allowed_classes": ("STRING", {
                    "multiline": True,
                    "default": "illustration, bangumi",
                    "label": "允许的分类 (英文逗号隔开)"
                }),
                "threshold": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05, "label": "置信度阈值"}),
            }
        }

    def filter_class(self, images: torch.Tensor, allowed_classes: str, threshold: float):
        class_list = [c.strip() for c in allowed_classes.split(',') if c.strip()]
        action = ClassFilterAction(class_list, threshold=threshold)
        return WaifucActionHelper.process_with_actions(images, [action])

class WaifucPersonSplitNode:
    DISPLAY_NAME = "Waifuc 拆分单个人物" # 名称微调以区分
    CATEGORY = "Waifuc/Waifuc裁切"
    FUNCTION = "split_person"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "level": (['n', 's', 'm', 'l', 'x'], {"default": 'm', "label": "检测模型等级"}),
                "conf_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "label": "置信度阈值"}),
                "iou_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "label": "IOU阈值"}),
                "keep_original": ("BOOLEAN", {"default": False, "label_on": "保留原图", "label_off": "不保留原图"}),
                "keep_origin_tags": ("BOOLEAN", {"default": False, "label_on": "保留原标签", "label_off": "不保留原标签"}),
            }
        }

    def split_person(self, images: torch.Tensor, level: str, conf_threshold: float, iou_threshold: float, keep_original: bool, keep_origin_tags: bool):
        action = PersonSplitAction(
            level=level,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            keep_original=keep_original,
            keep_origin_tags=keep_origin_tags
        )
        return WaifucActionHelper.process_with_actions(images, [action])
    

class WaifucThreeStageSplitNode:
    DISPLAY_NAME = "Waifuc 三段式裁切"
    CATEGORY = "Waifuc/Waifuc裁切"
    FUNCTION = "split_three_stage"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "split_person": ("BOOLEAN", {"default": True, "label_on": "启用人物裁切", "label_off": "禁用人物裁切"}),
                "person_conf_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "label": "人物置信度"}),
                "head_conf_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "label": "头部置信度"}),
                "head_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1, "label": "头部缩放"}),
                "split_eyes": ("BOOLEAN", {"default": False, "label_on": "启用眼睛裁切", "label_off": "禁用眼睛裁切"}),
                "eye_scale": ("FLOAT", {"default": 2.4, "min": 1.0, "max": 5.0, "step": 0.1, "label": "眼睛缩放"}),
                "keep_origin_tags": ("BOOLEAN", {"default": False, "label_on": "保留原标签", "label_off": "不保留原标签"}),
            }
        }

    def split_three_stage(self, images: torch.Tensor, split_person: bool, person_conf_threshold: float,
                          head_conf_threshold: float, head_scale: float, split_eyes: bool, eye_scale: float,
                          keep_origin_tags: bool):
        # 根据源码，可以将部分参数组织成字典传入
        person_conf = {'conf_threshold': person_conf_threshold}
        head_conf = {'conf_threshold': head_conf_threshold}
        
        action = ThreeStageSplitAction(
            split_person=split_person,
            person_conf=person_conf,
            head_conf=head_conf,
            head_scale=head_scale,
            split_eyes=split_eyes,
            eye_scale=eye_scale,
            keep_origin_tags=keep_origin_tags
        )
        return WaifucActionHelper.process_with_actions(images, [action])

class WaifucHeadCutOutNode:
    DISPLAY_NAME = "Waifuc 裁切出身体"
    CATEGORY = "图像/Waifuc裁切"
    FUNCTION = "cut_out_head"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    _MODELS = ['face_detect_v1.4_s', 'face_detect_v1.4_n', 'face_detect_v1.3_s', 'face_detect_v1.3_n', 
               'face_detect_v1.2_s', 'face_detect_v1.1_s', 'face_detect_v1.1_n', 'face_detect_v1_s', 
               'face_detect_v1_n', 'face_detect_v0_s', 'face_detect_v0_n']

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model_name": (cls._MODELS, {"default": "face_detect_v1.4_s"}),
                "kp_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "label": "姿态关键点阈值"}),
                "conf_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05, "label": "人脸置信度阈值"}),
            }
        }

    def cut_out_head(self, images: torch.Tensor, model_name: str, kp_threshold: float, conf_threshold: float):
        # 导入所需的模块
        from waifuc.action.head import HeadCutOutAction
        from waifuc.model import ImageItem
        from imgutils.pose import dwpose_estimate
        from imgutils.detect import detect_faces
        import types
        from typing import Iterator

        action = HeadCutOutAction(
            kp_threshold=kp_threshold,
            conf_threshold=conf_threshold
        )
        action.level = model_name
        
        def fixed_iter(self, item: ImageItem) -> Iterator[ImageItem]:
            poses = dwpose_estimate(item.image)
            if not poses:
                return

            pose = poses[0]
            points = pose.body

            # --- 核心修正点 (最终版) ---
            # 移除了不被接受的 `max_infer_size` 关键字参数
            faces = detect_faces(item.image, model_name=self.level,
                                 conf_threshold=self.conf_threshold, iou_threshold=self.iou_threshold)
            
            if not faces:
                return

            (x0, y0, x1, y1), _, _ = faces[0]
            crop_areas = [
                (0, 0, x0, item.image.height),
                (0, 0, item.image.width, y0),
                (x1, 0, item.image.width, item.image.height),
                (0, y1, item.image.width, item.image.height),
            ]

            maxi, maxcnt = None, None
            for i in range(len(crop_areas)):
                cx0, cy0, cx1, cy1 = crop_areas[i]
                cnt = sum([
                    1 for x, y, score in points
                    if score >= self.kp_threshold and cx0 <= x <= cx1 and cy0 <= y <= cy1
                ])
                if maxcnt is None or cnt > maxcnt:
                    maxi, maxcnt = i, cnt

            if maxcnt > 0:
                yield ImageItem(item.image.crop(crop_areas[maxi]), item.meta)

        action.iter = types.MethodType(fixed_iter, action)
        
        return WaifucActionHelper.process_with_actions(images, [action])


class WaifucHeadCoverNode:
    DISPLAY_NAME = "Waifuc 头部打码并生成遮罩"
    CATEGORY = "Waifuc/Waifuc遮罩" # 放到一个新的分类下
    FUNCTION = "cover_head"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("censored_image", "mask")
    
    _MODELS = ['head_detect_v1_s', 'head_detect_v1_n'] # 假设的模型，请根据实际情况调整

    @classmethod
    def INPUT_TYPES(cls):
        # waifuc 的 head detect 似乎没有很多模型可选，这里先用占位
        # 如果有更多模型，可以像上面一样列出
        return {
            "required": {
                "images": ("IMAGE",),
                "color": ("STRING", {"default": "random", "label": "颜色 (random, white, #RRGGBB)"}),
                "scale": ("FLOAT", {"default": 1.1, "min": 0.1, "max": 5.0, "step": 0.1, "label": "覆盖范围缩放"}),
                "conf_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "label": "置信度阈值"}),
            }
        }

    def cover_head(self, images: torch.Tensor, color: str, scale: float, conf_threshold: float):
        from imgutils.detect import detect_heads # 局部导入
        from imgutils.operate import censor_areas

        batch_size = images.shape[0]
        result_images = []
        result_masks = []

        for i in range(batch_size):
            # 将 Tensor 转为 PIL Image
            pil_image = Image.fromarray((images[i].cpu().numpy() * 255).astype(np.uint8))
            
            # 1. 检测头部区域
            head_areas = []
            for (x0, y0, x1, y1), _, _ in detect_heads(pil_image, conf_threshold=conf_threshold):
                # 根据 scale 参数调整区域大小
                width, height = x1 - x0, y1 - y0
                xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
                width, height = width * scale, height * scale
                nx0, ny0 = int(xc - width / 2), int(yc - height / 2)
                nx1, ny1 = int(xc + width / 2), int(yc + height / 2)
                head_areas.append((nx0, ny0, nx1, ny1))

            # 2. 生成打码后的图像
            if color == 'random':
                actual_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            else:
                actual_color = color
            censored_image = censor_areas(pil_image, 'color', head_areas, color=actual_color)
            
            # 3. 生成对应的 Mask
            mask = Image.new('L', pil_image.size, 0) # 创建黑色背景的灰度图
            if head_areas:
                draw = ImageDraw.Draw(mask)
                for area in head_areas:
                    draw.rectangle(area, fill=255) # 在头部区域画白色方块
            
            # 4. 将 PIL 转回 Tensor
            # 处理后的图像转回 Tensor
            img_tensor = torch.from_numpy(np.array(censored_image).astype(np.float32) / 255.0)[None,]
            result_images.append(img_tensor)

            # Mask 转为 Tensor
            mask_tensor = torch.from_numpy(np.array(mask).astype(np.float32) / 255.0).unsqueeze(0)
            result_masks.append(mask_tensor)

        # 将列表中的所有 Tensor 合并成一个批次
        final_images = torch.cat(result_images, dim=0)
        final_masks = torch.cat(result_masks, dim=0)

        return (final_images, final_masks)
    
class WaifucFaceCountNode:
    DISPLAY_NAME = "Waifuc 按人脸数量过滤"
    CATEGORY = "Waifuc/Waifuc筛选"
    FUNCTION = "filter_face_count"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "min_count": ("INT", {"default": 1, "min": 0, "max": 20, "step": 1, "label": "最少人脸数"}),
                "max_count": ("INT", {"default": 1, "min": 0, "max": 20, "step": 1, "label": "最多人脸数 (0为不限)"}),
                "conf_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05, "label": "置信度阈值"}),
            }
        }

    def filter_face_count(self, images: torch.Tensor, min_count: int, max_count: int, conf_threshold: float):
        # 如果 max_count 设置为0，则代表不设上限 (None)
        max_count_or_none = max_count if max_count > 0 else None
        
        action = FaceCountAction(
            min_count=min_count,
            max_count=max_count_or_none,
            conf_threshold=conf_threshold
        )
        return WaifucActionHelper.process_with_actions(images, [action])
    
# tagger 打标部分


class WaifucTaggerNode:
    DISPLAY_NAME = "Waifuc 自动打标"
    CATEGORY = "Waifuc/Waifuc打标"
    FUNCTION = "tag_images"
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "tags_text",)

    # 从 tagging.py 源码中获取可用的模型方法
    _METHODS = [
        'wd14_v3_swinv2', 'wd14_v3_convnext', 'wd14_v3_vit', 'wd14_moat',
        'wd14_swinv2', 'wd14_convnextv2', 'wd14_convnext', 'wd14_vit',
        'mldanbooru', 'deepdanbooru'
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "method": (cls._METHODS, {"default": 'wd14_v3_swinv2'}),
                "general_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "character_threshold": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05}),
                "force_rerun": ("BOOLEAN", {"default": True, "label_on": "强制重新打标", "label_off": "使用已有标签"}),
            }
        }

    def tag_images(self, images: torch.Tensor, method: str, general_threshold: float, character_threshold: float, force_rerun: bool):
        # 此节点逻辑特殊，需要手动实现，因为它要分离出 meta['tags']
        all_tags = set()
        
        # 准备 kwargs 字典给 Action
        kwargs = {
            'general_threshold': general_threshold,
            'character_threshold': character_threshold
        }
        action = TaggingAction(method=method, force=force_rerun, **kwargs)

        for i in range(images.shape[0]):
            pil_image = Image.fromarray((images[i].cpu().numpy() * 255).astype(np.uint8))
            item = ImageItem(pil_image)
            
            # 使用 waifuc 的 Action 处理单个 ImageItem
            processed_item = action.process(item)
            
            # 从处理后的 item 中提取标签
            if 'tags' in processed_item.meta:
                all_tags.update(processed_item.meta['tags'].keys())
        
        # 将所有不重复的标签排序后，用逗号连接成字符串
        tags_text = ', '.join(sorted(list(all_tags)))
        
        print(f"Waifuc Tagger: Generated {len(all_tags)} unique tags.")
        
        # 返回原始图像和生成的标签文本
        return (images, tags_text)



class WaifucTagRemoveUnderlineNode:
    DISPLAY_NAME = "Waifuc 标签去下划线"
    CATEGORY = "Waifuc/Waifuc打标/处理"
    FUNCTION = "remove_underline"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags_text",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"tags_text": ("STRING", {"multiline": True})}}

    def remove_underline(self, tags_text: str):
        from imgutils.tagging import remove_underline
        tags = [remove_underline(t.strip()) for t in tags_text.split(',') if t.strip()]
        return (', '.join(tags),)

class WaifucTagDropOverlapNode:
    DISPLAY_NAME = "Waifuc 移除重叠标签"
    CATEGORY = "Waifuc/Waifuc打标/处理"
    FUNCTION = "drop_overlap"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags_text",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"tags_text": ("STRING", {"multiline": True})}}
        
    def drop_overlap(self, tags_text: str):
        from imgutils.tagging import drop_overlap_tags
        if not tags_text.strip():
            return ("",)
        # Action 需要字典，我们创建一个虚拟的
        tags_dict = {t.strip(): 1.0 for t in tags_text.split(',') if t.strip()}
        processed_dict = drop_overlap_tags(tags_dict)
        return (', '.join(processed_dict.keys()),)

class WaifucTagDropBlacklistedNode:
    DISPLAY_NAME = "Waifuc 移除黑名单标签"
    CATEGORY = "Waifuc/Waifuc打标/处理"
    FUNCTION = "drop_blacklisted"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags_text",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"tags_text": ("STRING", {"multiline": True})}}

    def drop_blacklisted(self, tags_text: str):
        from imgutils.tagging import is_blacklisted
        tags = [t.strip() for t in tags_text.split(',') if t.strip() and not is_blacklisted(t.strip())]
        return (', '.join(tags),)

class WaifucTagDropCustomNode:
    DISPLAY_NAME = "Waifuc 移除指定标签"
    CATEGORY = "Waifuc/Waifuc打标/处理"
    FUNCTION = "drop_custom"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags_text",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tags_text": ("STRING", {"multiline": True}),
                "tags_to_drop": ("STRING", {"multiline": True, "placeholder": "输入要移除的标签, 用逗号隔开..."}),
            }
        }

    def drop_custom(self, tags_text: str, tags_to_drop: str):
        drop_set = {t.strip() for t in tags_to_drop.split(',') if t.strip()}
        tags = [t.strip() for t in tags_text.split(',') if t.strip() and t.strip() not in drop_set]
        return (', '.join(tags),)

class WaifucTagFilterNode:
    DISPLAY_NAME = "Waifuc 按标签筛选图像"
    CATEGORY = "Waifuc/Waifuc筛选"
    FUNCTION = "filter_by_tag"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "tags_to_check": ("STRING", {"multiline": True, "placeholder": "必需包含的标签, 用逗号隔开..."}),
                "method": (WaifucTaggerNode._METHODS, {"default": 'wd14_v3_swinv2'}),
                "reversed": ("BOOLEAN", {"default": False, "label_on": "反向 (排除)", "label_off": "正向 (包含)"}),
            }
        }

    def filter_by_tag(self, images: torch.Tensor, tags_to_check: str, method: str, reversed: bool):
        # 这个 Action 可以直接用我们的通用 Helper
        tag_list = [t.strip() for t in tags_to_check.split(',') if t.strip()]
        if not tag_list:
            return (images,) # 如果没有提供标签，则不过滤，直接返回原图

        action = TagFilterAction(tags=tag_list, method=method, reversed=reversed)
        return WaifucActionHelper.process_with_actions(images, [action])



# ================================================================
# ComfyUI 节点注册
# ================================================================
NODE_CLASS_MAPPINGS = {
    "WaifucModeConvertNode": WaifucModeConvertNode,
    "WaifucAlignMinSizeNode": WaifucAlignMinSizeNode,
    "WaifucFilterSimilarNode": WaifucFilterSimilarNode,
    "WaifucNoMonochromeNode": WaifucNoMonochromeNode,
    "WaifucClassFilterNode": WaifucClassFilterNode, 
    "WaifucFaceCountNode": WaifucFaceCountNode,     
    "WaifucPersonSplitNode": WaifucPersonSplitNode,
    "WaifucThreeStageSplitNode": WaifucThreeStageSplitNode,
    "WaifucHeadCutOutNode": WaifucHeadCutOutNode,
    "WaifucHeadCoverNode": WaifucHeadCoverNode,
    "WaifucOnlyMonochromeNode": WaifucOnlyMonochromeNode,
    "WaifucRatingFilterNode": WaifucRatingFilterNode,
    "WaifucHeadCountNode": WaifucHeadCountNode,
    "WaifucPersonRatioNode": WaifucPersonRatioNode,
    "WaifucMinSizeFilterNode": WaifucMinSizeFilterNode,
    "WaifucMinAreaFilterNode": WaifucMinAreaFilterNode,
    "WaifucCCIPNode": WaifucCCIPNode,
    "WaifucTaggerNode": WaifucTaggerNode,
    "WaifucTagRemoveUnderlineNode": WaifucTagRemoveUnderlineNode,
    "WaifucTagDropOverlapNode": WaifucTagDropOverlapNode,
    "WaifucTagDropBlacklistedNode": WaifucTagDropBlacklistedNode,
    "WaifucTagDropCustomNode": WaifucTagDropCustomNode,
    "WaifucTagFilterNode": WaifucTagFilterNode,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaifucModeConvertNode": WaifucModeConvertNode.DISPLAY_NAME,
    "WaifucAlignMinSizeNode": WaifucAlignMinSizeNode.DISPLAY_NAME,
    "WaifucFilterSimilarNode": WaifucFilterSimilarNode.DISPLAY_NAME,
    "WaifucNoMonochromeNode": WaifucNoMonochromeNode.DISPLAY_NAME,
    "WaifucClassFilterNode": WaifucClassFilterNode.DISPLAY_NAME, # 已修改
    "WaifucFaceCountNode": WaifucFaceCountNode.DISPLAY_NAME,     # 已修改
    "WaifucPersonSplitNode": WaifucPersonSplitNode.DISPLAY_NAME,
    "WaifucThreeStageSplitNode": WaifucThreeStageSplitNode.DISPLAY_NAME,
    "WaifucHeadCutOutNode": WaifucHeadCutOutNode.DISPLAY_NAME,
    "WaifucHeadCoverNode": WaifucHeadCoverNode.DISPLAY_NAME,
    "WaifucCCIPNode": WaifucCCIPNode.DISPLAY_NAME,
    "WaifucOnlyMonochromeNode": WaifucOnlyMonochromeNode.DISPLAY_NAME,
    "WaifucRatingFilterNode": WaifucRatingFilterNode.DISPLAY_NAME,
    "WaifucHeadCountNode": WaifucHeadCountNode.DISPLAY_NAME,
    "WaifucPersonRatioNode": WaifucPersonRatioNode.DISPLAY_NAME,
    "WaifucMinSizeFilterNode": WaifucMinSizeFilterNode.DISPLAY_NAME,
    "WaifucMinAreaFilterNode": WaifucMinAreaFilterNode.DISPLAY_NAME,
    "WaifucTaggerNode": WaifucTaggerNode.DISPLAY_NAME,
    "WaifucTagRemoveUnderlineNode": WaifucTagRemoveUnderlineNode.DISPLAY_NAME,
    "WaifucTagDropOverlapNode": WaifucTagDropOverlapNode.DISPLAY_NAME,
    "WaifucTagDropBlacklistedNode": WaifucTagDropBlacklistedNode.DISPLAY_NAME,
    "WaifucTagDropCustomNode": WaifucTagDropCustomNode.DISPLAY_NAME,
    "WaifucTagFilterNode": WaifucTagFilterNode.DISPLAY_NAME,
}