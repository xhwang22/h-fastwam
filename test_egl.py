import os
# 关键：彻底禁用 Vulkan 探测
os.environ["SAPIEN_NO_VULKAN"] = "1" 
os.environ["SAPIEN_RENDER_MODE"] = "egl"
os.environ["GALLIUM_DRIVER"] = "llvmpipe"

import sapien.core as sapien
import numpy as np

def test():
    try:
        engine = sapien.Engine()
        # 强制创建非 Vulkan 渲染器
        renderer = sapien.OptifuserRenderer() 
        engine.set_renderer(renderer)
        print("成功启动 Optifuser (EGL) 渲染器！")
    except Exception as e:
        print(f"EGL 渲染启动失败: {e}")

if __name__ == "__main__":
    test()
