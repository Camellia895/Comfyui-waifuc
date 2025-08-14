# image_combine_node.py
import torch

class CombineImagesToList:
    DISPLAY_NAME = "组合图像为列表"
    CATEGORY = "图像/列表操作"
    FUNCTION = "combine"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,) # 关键：声明输出是一个列表

    @classmethod
    def INPUT_TYPES(cls):
        # 提供3个可选的图像输入接口
        return {
            "optional": {
                "images_1": ("IMAGE",),
                "images_2": ("IMAGE",),
                "images_3": ("IMAGE",),
            }
        }
    
    def combine(self, images_1=None, images_2=None, images_3=None):
        """
        将所有传入的图像输入（单个图像或图像列表）合并为一个统一的图像列表。
        """
        combined_tensors = []
        
        # 依次处理每个输入参数
        for image_input in [images_1, images_2, images_3]:
            if image_input is None:
                continue

            # 检查输入是单个图像 (Tensor) 还是图像列表 (list)
            if isinstance(image_input, list):
                # 如果是列表，直接扩展到总列表
                combined_tensors.extend(image_input)
            elif isinstance(image_input, torch.Tensor):
                # 如果是单个图像，则将其添加到总列表
                combined_tensors.append(image_input)

        if not combined_tensors:
            print("组合图像为列表警告: 没有有效的图像输入。")
            # 返回一个包含空列表的元组，这是ComfyUI期望的格式
            return ([],)
        
        # 返回包含所有组合图像的列表的元组
        return (combined_tensors,)

# ================================================================
# ComfyUI 节点注册
# ================================================================
NODE_CLASS_MAPPINGS = {
    "CombineImagesToList": CombineImagesToList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CombineImagesToList": CombineImagesToList.DISPLAY_NAME,
}