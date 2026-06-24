import os
import ctypes

# 1. 检查环境变量
print(f"当前 LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '未设置')}")

# 2. 尝试加载 Conda 路径下的 libEGL
try:
    # 动态加载测试
    egl = ctypes.CDLL("libEGL.so.1")
    print("成功加载 libEGL.so.1")
except Exception as e:
    print(f"加载 libEGL 失败: {e}")

# 3. SAPIEN 探测
try:
    import sapien.core as sapien
    # 尝试初始化一个无头场景
    engine = sapien.Engine()
    print("SAPIEN 引擎启动成功！")
except Exception as e:
    print(f"SAPIEN 启动失败 (可能是正常的，如果还没配好渲染器): {e}")


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
