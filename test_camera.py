import sapien.core as sapien
import numpy as np

def test():
    try:
        # 1. 创建场景
        scene = sapien.Scene()
        
        # 2. 添加一个相机 (这是 RoboTwin 最核心的操作)
        cam = scene.add_camera(
            name="test_camera",
            width=128,
            height=128,
            fovy=np.deg2rad(35),
            near=0.1,
            far=100
        )
        
        # 3. 尝试更新渲染器
        scene.update_render()
        cam.take_picture()
        
        # 4. 获取一张图片数据来确认渲染器工作了
        rgba = cam.get_float_texture('Color')
        print(f"渲染成功！图片形状: {rgba.shape}")
        print(">>> 恭喜，你的环境已经完全支持离屏渲染了！")
        
    except Exception as e:
        print(f">>> 渲染失败，详细错误: {e}")

if __name__ == "__main__":
    test()
