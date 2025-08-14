# waifuc_node.py
import torch
import numpy as np
from PIL import Image
import comfy.model_management
from typing import Iterator
import tempfile # [新增] 导入临时文件/目录模块
import os       # [新增] 导入操作系统路径模块

from waifuc.action import ProcessAction
from waifuc.model import ImageItem

from waifuc.action import (NoMonochromeAction, FilterSimilarAction, TaggingAction,
                           PersonSplitAction, FaceCountAction, FirstNSelectAction,
                           CCIPAction, ModeConvertAction, ClassFilterAction,
                           AlignMinSizeAction)
from waifuc.source import (DanbooruSource, ZerochanSource, GelbooruSource, 
                           AnimePicturesSource, LocalSource)

# ================================================================
# 自定义 Action：中断检查器 (共用)
# ================================================================
class ComfyInterruptAction(ProcessAction):
    def process(self, item: ImageItem) -> ImageItem:
        comfy.model_management.throw_exception_if_processing_interrupted()
        return item

# ================================================================
# 节点 1: Waifuc 图像加载器 (之前版本已正确，保持不变)
# ================================================================
class WaifucLoader:
    DISPLAY_NAME = "Waifuc 图像加载器"
    CATEGORY = "加载器/Waifuc"
    FUNCTION = "load_and_process"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        SUPPORTED_SITES = ["Danbooru", "Zerochan", "Gelbooru", "AnimePictures"]
        return {
            "required": {
                "source_site": (SUPPORTED_SITES, {"label": "数据源站点"}),
                "tags": ("STRING", {"multiline": True, "default": "surtr_(arknights), solo", "label": "搜索标签 (英文逗号隔开)"}),
                "max_images": ("INT", {"default": 8, "min": 1, "max": 200, "step": 1, "label": "最大图像数"}),
            },
            # 将通用参数提取出来，方便复用
            "optional": WaifucProcessor.get_common_inputs()
        }
    
    def load_and_process(self, source_site, tags, max_images, **kwargs):
        print(f"Waifuc加载器: 开始执行...")
        tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
        if source_site == "Danbooru": source = DanbooruSource(tag_list)
        elif source_site == "Zerochan": source = ZerochanSource(tag_list[0] if tag_list else "")
        elif source_site == "Gelbooru": source = GelbooruSource(tag_list)
        elif source_site == "AnimePictures": source = AnimePicturesSource(tag_list)
        else: raise ValueError(f"不支持的数据源: {source_site}")
        
        actions = WaifucProcessor.build_actions(**kwargs)
        actions.append(FirstNSelectAction(max_images))
        pipeline = source.attach(*actions)
        
        pil_images = [item.image for item in pipeline]

        if not pil_images:
            print("Waifuc加载器警告: 未找到或处理任何图像。")
            return ([],)

        tensors = [torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,] for img in pil_images]
        return (tensors,)

# ================================================================
# 节点 2: Waifuc 图像处理器 (已修正)
# ================================================================
class WaifucProcessor:
    DISPLAY_NAME = "Waifuc 图像处理器"
    CATEGORY = "图像/Waifuc处理"
    FUNCTION = "process_images"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def get_common_inputs(cls):
        """将两个节点共用的输入参数提取出来，避免代码重复"""
        return {
            "align_min_size": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64, "label": "对齐最小边尺寸"}),
            "processing_toggles": ("BOOLEAN", {"default": True, "label_on": "高级处理 (已启用)", "label_off": "高级处理 (已禁用)"}),
            "enable_monochrome_filter": ("BOOLEAN", {"default": True, "label_on": "过滤单色图 (是)", "label_off": "过滤单色图 (否)"}),
            "enable_class_filter": ("BOOLEAN", {"default": True, "label_on": "过滤分类 (是)", "label_off": "过滤分类 (否)"}),
            "enable_similar_filter": ("BOOLEAN", {"default": True, "label_on": "过滤相似图 (是)", "label_off": "过滤相似图 (否)"}),
            "enable_person_split": ("BOOLEAN", {"default": True, "label_on": "拆分多人图 (是)", "label_off": "拆分多人图 (否)"}),
            "face_count_filter": ("INT", {"default": 1, "min": 0, "max": 10, "step": 1, "label": "人脸数量过滤 (0为禁用)"}),
            "enable_ccip_filter": ("BOOLEAN", {"default": True, "label_on": "AI角色校验 (是)", "label_off": "AI角色校验 (否)"}),
            "enable_tagging": ("BOOLEAN", {"default": True, "label_on": "自动打标 (是)", "label_off": "自动打标 (否)"}),
        }

    @classmethod
    def INPUT_TYPES(cls):
        return { "required": { "images": ("IMAGE",), }, "optional": cls.get_common_inputs() }

    @staticmethod
    def build_actions(align_min_size, processing_toggles=True, **kwargs):
        actions = [ComfyInterruptAction(), ModeConvertAction('RGB', 'white')]
        if processing_toggles:
            # 使用kwargs.get()安全地访问可选参数
            if kwargs.get('enable_similar_filter', True): actions.append(FilterSimilarAction('all'))
            if kwargs.get('enable_monochrome_filter', True): actions.append(NoMonochromeAction())
            if kwargs.get('enable_class_filter', True): actions.append(ClassFilterAction(['illustration', 'bangumi']))
            
            enable_person_split = kwargs.get('enable_person_split', True)
            face_count_filter = kwargs.get('face_count_filter', 1)
            if enable_person_split: actions.extend([PersonSplitAction(), FaceCountAction(1)])
            elif face_count_filter > 0: actions.append(FaceCountAction(face_count_filter))
            
            if kwargs.get('enable_ccip_filter', True): actions.append(CCIPAction())
        
        actions.append(AlignMinSizeAction(align_min_size))
        if processing_toggles and kwargs.get('enable_tagging', True): actions.append(TaggingAction(force=True))
        if processing_toggles and kwargs.get('enable_similar_filter', True): actions.append(FilterSimilarAction('all'))
        return actions

    def process_images(self, images: torch.Tensor, **kwargs):
        print(f"Waifuc处理器: 开始执行...")
        input_pil_images = [(Image.fromarray((images[i].cpu().numpy() * 255).astype(np.uint8))) for i in range(images.shape[0])]
        
        # [核心修正] 使用临时目录来存放待处理的图像
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. 将接收到的PIL图像保存到临时目录
            for i, pil_img in enumerate(input_pil_images):
                # 使用PNG格式以保证无损保存
                pil_img.save(os.path.join(temp_dir, f"image_{i}.png"))

            # 2. 使用临时目录的路径初始化 LocalSource
            source = LocalSource(temp_dir)
            
            # 3. 构建和执行流水线 (与之前相同)
            actions = self.build_actions(**kwargs)
            pipeline = source.attach(*actions)
            processed_pil_images = [item.image for item in pipeline]

        if not processed_pil_images:
            print("Waifuc处理器警告: 没有图像通过筛选，已输出1x1黑色占位图。")
            placeholder_img = Image.new('RGB', (1, 1), 'black')
            tensors = [torch.from_numpy(np.array(placeholder_img).astype(np.float32) / 255.0)[None,]]
        else:
            tensors = [torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,] for img in processed_pil_images]
            
        return (tensors,)

# ================================================================
# ComfyUI 节点注册
# ================================================================
NODE_CLASS_MAPPINGS = { "WaifucLoader": WaifucLoader, "WaifucProcessor": WaifucProcessor, }
NODE_DISPLAY_NAME_MAPPINGS = { "WaifucLoader": WaifucLoader.DISPLAY_NAME, "WaifucProcessor": WaifucProcessor.DISPLAY_NAME, }