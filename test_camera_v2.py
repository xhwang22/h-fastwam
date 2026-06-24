import sapien.core as sapien
import numpy as np

def test():
    try:
        # Sapien 2.x 的初始化方式
        engine = sapien.Engine()
        # 显式指定使用库文件进行渲染
        renderer = sapien.SapienRenderer()
        engine.set_renderer(renderer)
        
        scene = engine.create_scene()
        
        cam = scene.add_camera(
            name="test_camera",
            width=128,
            height=128,
            fovy=np.deg2rad(35),
            near=0.1,
            far=100
        )
        
        scene.step() # 步进物理
        scene.update_render()
        cam.take_picture()
        
        rgba = cam.get_float_texture('Color')
        print(f"渲染成功！图片形状: {rgba.shape}")
        print(">>> 恭喜！Sapien 2.2.1 在软件渲染模式下工作正常！")
        
    except Exception as e:
        print(f">>> 还是失败了: {e}")

if __name__ == "__main__":
    test()
