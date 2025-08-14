# waifuc_source.py (v3 - 稳定重构版)
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError
import comfy.model_management
import os
import warnings
from typing import List, Set, Iterator, Tuple, Union, Dict
import httpx
from hbutils.system import urlsplit, TemporaryDirectory

# ================================================================
# [新] 自包含的 Waifuc 核心功能，减少对外部库瑕疵的依赖
# ================================================================
from waifuc.utils import get_requests_session, download_file
from waifuc.source import LocalSource
from waifuc.action import ProcessAction, FirstNSelectAction, ModeConvertAction
from waifuc.model import ImageItem
# 导入 waifuc 的原始类，仅用于类型提示和继承，核心逻辑将被重写
from waifuc.source.danbooru import DanbooruLikeSource, E621LikeSource


# ================================================================
# 通用辅助模块
# ================================================================
class ComfyInterruptAction(ProcessAction):
    def process(self, item: ImageItem) -> ImageItem:
        comfy.model_management.throw_exception_if_processing_interrupted()
        return item

class WaifucSourceHelper:
    @staticmethod
    def process_items_to_output(items_iterator: Iterator[ImageItem]):
        """
        [核心修正] 这个函数现在只负责将 ImageItem 迭代器转换为 ComfyUI 输出。
        不再包含任何 action pipeline，因为核心逻辑已移至节点内部。
        """
        processed_pil_images: List[Image.Image] = []
        all_tags: Set[str] = set()

        for item in items_iterator:
            comfy.model_management.throw_exception_if_processing_interrupted()
            # 统一转为RGB
            if item.image.mode != 'RGB':
                item.image = item.image.convert('RGB')
            processed_pil_images.append(item.image)
            if 'tags' in item.meta and isinstance(item.meta['tags'], dict):
                all_tags.update(item.meta['tags'].keys())
        
        print(f"WaifucSourceHelper: 流水线执行完毕，成功获取 {len(processed_pil_images)} 张图像。")

        if not processed_pil_images:
            warnings.warn("未能从数据源获取任何满足条件的图像 (可能是 min_size 过高或标签无结果)。将输出空列表。")
            return ([], "")

        image_tensors = [torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,] for img in processed_pil_images]
        tags_text = ', '.join(sorted(list(all_tags)))
        print(f"WaifucSourceHelper: 共提取出 {len(all_tags)} 个唯一标签。")

        return (image_tensors, tags_text)


# ================================================================
# [新] 自定义的、健壮的图源节点基类
# ================================================================
class WaifucBooruSourceNodeBase:
    # 这些属性应在子类中被覆盖
    SITE_URL = ""
    SITE_NAME = ""
    IS_E621_LIKE = False # E621的API和Danbooru有差异

    @classmethod
    def INPUT_TYPES(cls):
        # 通用输入参数
        return {
            "required": {
                "tags": ("STRING", {"multiline": True, "default": "1girl, solo, masterpiece", "label": "搜索标签"}),
                "max_images": ("INT", {"default": 8, "min": 1, "max": 1000, "step": 1, "label": "最大图像数"}),
                "randomize": ("BOOLEAN", {"default": True, "label_on": "随机获取", "label_off": "顺序获取"}),
            },
            "optional": {
                "min_size": ("INT", {"default": 800, "min": 0, "max": 8192, "step": 64, "label": "最小边长 (0为不限)"}),
                "username": ("STRING", {"multiline": False, "label": "用户名 (可选)"}),
                "api_key": ("STRING", {"multiline": False, "label": "API Key (可选)"}),
            }
        }

    def _get_booru_items(self, tags: List[str], max_images: int, randomize: bool, min_size: int, username: str, api_key: str) -> Iterator[ImageItem]:
        """
        [核心修正] 重写的、自包含的核心网络请求逻辑
        """
        session = get_requests_session()
        auth = (username, api_key) if username and api_key else None
        page = 1
        collected_count = 0

        # 创建一个临时的 waifuc source 实例，只为了借用它的 _select_url 和 _get_tags 方法
        # 这是比完全重写这两个复杂方法更简单的做法
        source_delegator_cls = E621LikeSource if self.IS_E621_LIKE else DanbooruLikeSource
        source_delegator = source_delegator_cls(tags, min_size=min_size)
        
        while collected_count < max_images:
            # [核心修正] 解决了硬编码 limit 问题
            limit = min(max_images - collected_count + 50, 200 if not self.IS_E621_LIKE else 320) # 多取一些以备筛选
            
            params = {"page": page, "limit": limit, "tags": ' '.join(tags)}
            
            # [核心修正] 正确处理随机化
            if randomize:
                if self.IS_E621_LIKE:
                    params["tags"] += ' order:random'
                else:
                    params['random'] = 'true'

            try:
                resp = session.get(f'{self.SITE_URL}/posts.json', params=params, auth=auth)
                resp.raise_for_status()
                json_data = resp.json()
            except (httpx.HTTPStatusError, httpx.RequestError) as err:
                warnings.warn(f"请求 {self.SITE_NAME} API失败: {err!r}")
                break
            
            posts = json_data.get('posts') if self.IS_E621_LIKE else json_data
            if not posts:
                print(f"{self.SITE_NAME}: 在第 {page} 页没有更多结果了。")
                break

            for data in posts:
                if collected_count >= max_images:
                    break
                
                try:
                    # 借用 waifuc 的原生方法来筛选URL和解析TAGS
                    url = source_delegator._select_url(data)
                    tags_dict = {key: 1.0 for key in source_delegator._get_tags(data)}
                except Exception: # 捕获 NoURL 等所有可能的错误
                    continue
                
                with TemporaryDirectory() as td:
                    _, ext_name = os.path.splitext(urlsplit(url).filename)
                    filename = f'{self.SITE_NAME}_{data["id"]}{ext_name}'
                    file_path = os.path.join(td, filename)
                    try:
                        download_file(url, file_path, session=session, silent=True)
                        image = Image.open(file_path)
                        image.load()
                        
                        meta = {
                            self.SITE_NAME: data, 'url': url, 'filename': filename, 'tags': tags_dict
                        }
                        yield ImageItem(image, meta)
                        collected_count += 1

                    except (httpx.HTTPError, UnidentifiedImageError, DecompressionBombError, IOError) as err:
                        # warnings.warn(f'下载或加载文件 {filename} 时跳过: {err!r}')
                        continue
            
            page += 1

# ================================================================
# Source 节点实现 (已重构)
# ================================================================

class WaifucDanbooruSourceNode(WaifucBooruSourceNodeBase):
    DISPLAY_NAME = "Waifuc Danbooru图源"
    CATEGORY = "Waifuc/Waifuc图源"
    FUNCTION = "load_images"
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "tags_text",)
    OUTPUT_IS_LIST = (True, False) 

    # --- Site Specific Config ---
    SITE_URL = "https://danbooru.donmai.us"
    SITE_NAME = "danbooru"
    IS_E621_LIKE = False

    def load_images(self, tags: str, max_images: int, randomize: bool, min_size: int, 
                    username: str = "", api_key: str = ""):
        tag_list = [t.strip() for t in tags.split(',') if t.strip()]
        items_iterator = self._get_booru_items(tag_list, max_images, randomize, min_size, username, api_key)
        return WaifucSourceHelper.process_items_to_output(items_iterator)

class WaifucE621SourceNode(WaifucBooruSourceNodeBase):
    DISPLAY_NAME = "Waifuc E621图源"
    CATEGORY = "Waifuc/Waifuc图源"
    FUNCTION = "load_images"
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "tags_text",)
    OUTPUT_IS_LIST = (True, False)

    # --- Site Specific Config ---
    SITE_URL = "https://e621.net"
    SITE_NAME = "e621"
    IS_E621_LIKE = True
    
    def load_images(self, tags: str, max_images: int, randomize: bool, min_size: int, 
                    username: str = "", api_key: str = ""):
        tag_list = [t.strip() for t in tags.split(',') if t.strip()]
        items_iterator = self._get_booru_items(tag_list, max_images, randomize, min_size, username, api_key)
        return WaifucSourceHelper.process_items_to_output(items_iterator)
        
# 其他节点可以类似地继承 WaifucBooruSourceNodeBase

class WaifucLocalSourceNode:
    DISPLAY_NAME = "Waifuc 本地文件夹图源"
    CATEGORY = "Waifuc/Waifuc图源"
    FUNCTION = "load_images"
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "tags_text",)
    OUTPUT_IS_LIST = (True, False)
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"directory": ("STRING", {"default": "D:/path/to/your/images"}), "max_images": ("INT", {"default": 8, "min": 1, "max": 10000}), "recursive": ("BOOLEAN", {"default": True})}}
    def load_images(self, directory: str, max_images: int, recursive: bool):
        if not os.path.isdir(directory): raise FileNotFoundError(f"指定的本地文件夹路径不存在: {directory}")
        source = LocalSource(directory, recursive=recursive)
        # 本地源使用 FirstNSelectAction 即可
        pipeline = source.attach(ComfyInterruptAction(), FirstNSelectAction(max_images))
        return WaifucSourceHelper.process_items_to_output(pipeline)

# ================================================================
# ComfyUI 节点注册
# ================================================================
NODE_CLASS_MAPPINGS = {
    "WaifucLocalSourceNode": WaifucLocalSourceNode,
    "WaifucDanbooruSourceNode": WaifucDanbooruSourceNode,
    "WaifucE621SourceNode": WaifucE621SourceNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaifucLocalSourceNode": WaifucLocalSourceNode.DISPLAY_NAME,
    "WaifucDanbooruSourceNode": WaifucDanbooruSourceNode.DISPLAY_NAME,
    "WaifucE621SourceNode": WaifucE621SourceNode.DISPLAY_NAME,
}