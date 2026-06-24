import sapien
import numpy as np
import os

# 强制使用 CPU 软件渲染环境变量
os.environ["GALLIUM_DRIVER"] = "llvmpipe"
os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"

def test_v3():
    try:
        # Sapien 3.0 写法
        scene = sapien.Scene()
        
        # 添加相机
        cam = scene.add_camera(
            name="cam",
            width=256,
            height=256,
            fovy=1.0,
            near=0.1,
            far=100
        )
        
        # 渲染
        scene.update_render()
        cam.take_picture()
        
        # 获取图片数据
        rgba = cam.get_float_texture("Color")
        print(f"Sapien 3.0.0b1 CPU 渲染成功！形状: {rgba.shape}")
        
    except Exception as e:
        print(f"渲染失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_v3()
