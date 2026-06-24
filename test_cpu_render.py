import sapien.core as sapien
import numpy as np

def test_cpu():
    try:
        # 创建场景
        scene = sapien.Scene()
        
        # 添加一个相机
        cam = scene.add_camera(
            name="cpu_cam",
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
        img = cam.get_float_texture('Color')
        print(f"CPU 渲染成功！图片数据形状: {img.shape}")
        
    except Exception as e:
        print(f"CPU 渲染失败: {e}")

if __name__ == "__main__":
    test_cpu()
