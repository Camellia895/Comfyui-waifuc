目前只实现了action部分的节点化改造（ccip正在尝试支持）。我尽可能地暴露了代码中可用的参数。上传了action部分的工作流。*基本上就是个堆料场*

网络爬取的节点完成了理论验证。可以正常运行，但是只支持无登录的网站。理论上可以是登录的，但这得等待我把source部分做出来。（鸽好久了混蛋）

随机爬虫问题解决了一半。但没有应用在这个理论验证节点上。所以这个验证节点总是能输出一致的图像（没有改变节点参数的话）（注意，dandooru只支持两个tagger搜索）

<img width="889" height="480" alt="image" src="https://github.com/user-attachments/assets/87c07113-e6dc-4d1e-86e9-e23f12f314a2" />
<img width="1120" height="777" alt="image" src="https://github.com/user-attachments/assets/2e81e393-d0aa-4701-b42c-ec5300c1d2b4" />

# 临时节点。

ccip的图像输入问题一直得不到解决，因为问题来自comfyui的接口和图像list的特性。我正在寻找一个方法解决它。

<img width="1411" height="528" alt="image" src="https://github.com/user-attachments/assets/6f0b302f-097e-4ff7-a719-0d01302abc85" />

筛选后的图像通常为黑色占位图像。你可以使用我朋友的图像保存节点。 它在保存图像的时候能够忽略节点产生的占位图像。 https://github.com/zml-w/ComfyUI-ZML-Image 
